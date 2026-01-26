"""Publication-ready image grid visualization for ICIP 2026.

Creates a grid layout with:
- Y-axis: Z-bins (rows)
- X-axis: Experiments grouped by prediction type (columns)
  - Within each prediction type: sub-columns for Lp norms
- Real data as leftmost column group
- IEEE publication style with Paul Tol colorblind-friendly palette

Usage:
    python -m src.classification.plotting.image_grid \
        --patches-dir /path/to/full_images \
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# Import settings from similarity_metrics plotting module directly
# Uses importlib to avoid the __init__.py chain which imports heavy dependencies
import importlib.util
from pathlib import Path as _Path

def _load_settings_module():
    """Load settings module directly without going through __init__.py chain."""
    settings_path = (
        _Path(__file__).parent.parent.parent
        / "diffusion" / "scripts" / "similarity_metrics" / "plotting" / "settings.py"
    )
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings

_settings = _load_settings_module()

PLOT_SETTINGS = _settings.PLOT_SETTINGS
PREDICTION_TYPE_COLORS = _settings.PREDICTION_TYPE_COLORS
PREDICTION_TYPE_LABELS_SHORT = _settings.PREDICTION_TYPE_LABELS_SHORT
LP_NORM_LABELS = _settings.LP_NORM_LABELS
LP_NORM_MARKERS = _settings.LP_NORM_MARKERS
LP_NORM_HATCHES = _settings.LP_NORM_HATCHES
apply_ieee_style = _settings.apply_ieee_style

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading (reused from dataset_analysis)
# =============================================================================


def load_patches_npz(npz_path: Path) -> dict:
    """Load patches from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "patches": data["patches"],
        "z_bins": data["z_bins"],
    }


def detect_lesion_samples(patches: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Detect which samples contain lesions based on mask channel."""
    masks = patches[:, 1, :, :]
    lesion_counts = (masks > threshold).sum(axis=(1, 2))
    return lesion_counts > 0


def to_display_range(x: np.ndarray) -> np.ndarray:
    """Convert from [-1, 1] to [0, 1] for display."""
    return np.clip((x + 1) / 2, 0, 1)


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Create image with lesion mask overlay."""
    if image.ndim == 3:
        image = image.squeeze()
    if mask.ndim == 3:
        mask = mask.squeeze()

    rgb = np.stack([image, image, image], axis=-1).astype(np.float32)
    binary_mask = mask > 0
    color_norm = np.array(color, dtype=np.float32) / 255.0

    if binary_mask.any():
        for c in range(3):
            rgb[:, :, c] = np.where(
                binary_mask,
                (1 - alpha) * rgb[:, :, c] + alpha * color_norm[c],
                rgb[:, :, c],
            )

    return rgb


def list_experiments(patches_dir: Path) -> list[str]:
    """List all experiment directories in patches_dir."""
    experiments = []
    for subdir in sorted(patches_dir.iterdir()):
        if subdir.is_dir() and (subdir / "synthetic_patches.npz").exists():
            experiments.append(subdir.name)
    return experiments


def select_representative_zbins(z_bins: np.ndarray, n_bins: int = 5) -> list[int]:
    """Select evenly-spaced representative z-bins."""
    unique_zbins = sorted(np.unique(z_bins))
    if len(unique_zbins) <= n_bins:
        return unique_zbins
    indices = np.linspace(0, len(unique_zbins) - 1, n_bins, dtype=int)
    return [unique_zbins[i] for i in indices]


# =============================================================================
# Experiment Parsing
# =============================================================================


def parse_experiment_name(exp_name: str) -> tuple[str, float] | None:
    """Parse experiment name to extract prediction type and Lp norm.

    Args:
        exp_name: Experiment name like "epsilon_lp_1.5" or "velocity_lp_2.0".

    Returns:
        Tuple of (prediction_type, lp_norm) or None if parsing fails.
    """
    match = re.match(r"(\w+)_lp_(\d+\.?\d*)", exp_name)
    if match:
        pred_type = match.group(1)
        lp_norm = float(match.group(2))
        return pred_type, lp_norm
    return None


def group_experiments_by_prediction_type(
    experiments: list[str],
) -> dict[str, list[tuple[str, float]]]:
    """Group experiments by prediction type.

    Args:
        experiments: List of experiment names.

    Returns:
        Dict mapping prediction_type -> list of (exp_name, lp_norm) sorted by lp_norm.
    """
    groups: dict[str, list[tuple[str, float]]] = {}

    for exp_name in experiments:
        parsed = parse_experiment_name(exp_name)
        if parsed:
            pred_type, lp_norm = parsed
            if pred_type not in groups:
                groups[pred_type] = []
            groups[pred_type].append((exp_name, lp_norm))

    # Sort each group by lp_norm
    for pred_type in groups:
        groups[pred_type].sort(key=lambda x: x[1])

    return groups


# =============================================================================
# Publication-Ready Image Grid
# =============================================================================


def plot_publication_image_grid(
    patches_dir: Path,
    output_path: Path,
    experiments: list[str] | None = None,
    n_zbins: int = 6,
    show_lesion: bool = True,
    overlay_alpha: float = 0.4,
    overlay_color: tuple[int, int, int] = (255, 100, 100),
    cell_size: float | None = None,
    formats: list[str] = ["pdf", "png"],
) -> None:
    """Create publication-ready image grid with IEEE single-column style.

    Portrait layout (taller than wide) for IEEE ICIP single-column:
    - X-axis (columns): Z-bins
    - Y-axis (rows): Experiments (Real + synthetic grouped by prediction type)
    - Left header columns: Prediction type (color) | Lp norm (lighter)
    - Top header row: Z-bin labels

    Args:
        patches_dir: Directory containing experiment subdirectories.
        output_path: Base path for output (extension will be replaced).
        experiments: List of experiment names. If None, uses all found.
        n_zbins: Number of representative z-bins to show.
        show_lesion: If True, prefer samples with lesions.
        overlay_alpha: Alpha for lesion overlay.
        overlay_color: RGB color for lesion overlay.
        cell_size: Size of each image cell in inches. If None, auto-calculated
                   to fit single column width.
        formats: Output formats (pdf, png).
    """
    apply_ieee_style()

    patches_dir = Path(patches_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if experiments is None:
        experiments = list_experiments(patches_dir)

    if not experiments:
        logger.warning("No experiments found in %s", patches_dir)
        return

    # Load real data
    first_exp_dir = patches_dir / experiments[0]
    real_data = load_patches_npz(first_exp_dir / "real_patches.npz")

    # Select representative z-bins
    zbins = select_representative_zbins(real_data["z_bins"], n_zbins)
    logger.info("Selected z-bins: %s", zbins)

    # Group experiments by prediction type
    exp_groups = group_experiments_by_prediction_type(experiments)

    # Define prediction type order
    pred_type_order = ["epsilon", "velocity", "x0"]
    pred_types = [pt for pt in pred_type_order if pt in exp_groups]

    # Build ordered list of all datasets: Real + synthetic experiments
    # Each entry: (pred_type, lp_norm, exp_name, data) - pred_type=None for Real
    all_datasets = [("Real", None, None, real_data)]
    for pred_type in pred_types:
        for exp_name, lp_norm in exp_groups[pred_type]:
            all_datasets.append((pred_type, lp_norm, exp_name, None))

    # Calculate layout dimensions
    # Portrait: Z-bins as columns, experiments as rows
    n_data_cols = len(zbins)
    n_data_rows = len(all_datasets)

    # Header dimensions based on font size
    font_size_inches = PLOT_SETTINGS["font_size"] / 72.0
    annotation_font_size_inches = PLOT_SETTINGS["annotation_fontsize"] / 72.0

    # Left header widths (prediction type + Lp norm columns)
    # Tighter widths to maximize space for brain images
    header_col1_width = font_size_inches * 2.0   # Prediction type symbols (horizontal)
    header_col2_width = annotation_font_size_inches * 3.2  # Lp labels

    # Top header height (z-bin labels)
    header_row_height = annotation_font_size_inches * 2.5

    # Calculate cell size to fit IEEE single column width
    available_width = PLOT_SETTINGS["figure_width_single"] - header_col1_width - header_col2_width - 0.1
    if cell_size is None:
        cell_size = available_width / n_data_cols

    # Figure dimensions (single column, portrait)
    fig_width = PLOT_SETTINGS["figure_width_single"]
    fig_height = header_row_height + n_data_rows * cell_size + 0.15

    # Ensure we don't exceed max height
    max_height = PLOT_SETTINGS.get("figure_height_max", 9.0)
    if fig_height > max_height:
        # Scale down cell size to fit
        cell_size = (max_height - header_row_height - 0.15) / n_data_rows
        fig_height = max_height

    logger.info(f"Figure size: {fig_width:.2f} x {fig_height:.2f} inches, cell size: {cell_size:.3f} inches")

    fig = plt.figure(figsize=(fig_width, fig_height))

    # GridSpec: 2 header columns + data columns, 1 header row + data rows
    n_cols_total = 2 + n_data_cols  # pred_type, lp_norm, zbins...
    n_rows_total = 1 + n_data_rows  # header row, data rows...

    # Width ratios
    width_ratios = [header_col1_width / cell_size, header_col2_width / cell_size] + [1] * n_data_cols

    # Height ratios
    height_ratios = [header_row_height / cell_size] + [1] * n_data_rows

    gs = GridSpec(
        n_rows_total,
        n_cols_total,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.02,
        wspace=0.02,
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.02,
    )

    # =========================================================================
    # Top Header Row: Z-bin Labels
    # =========================================================================

    # Empty corner cells (top-left)
    for col in range(2):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#F5F5F5")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(PLOT_SETTINGS["spine_linewidth"] * 0.5)
            spine.set_edgecolor("#DDDDDD")

    # Z-bin labels
    for col_idx, zbin in enumerate(zbins):
        ax = fig.add_subplot(gs[0, col_idx + 2])
        ax.set_facecolor("#E8E8E8")
        ax.text(
            0.5, 0.5, f"z={zbin}",
            ha="center", va="center",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(PLOT_SETTINGS["spine_linewidth"] * 0.5)
            spine.set_edgecolor("#CCCCCC")

    # =========================================================================
    # Left Header Columns + Image Grid
    # =========================================================================

    # Load all synthetic datasets
    synth_datasets = {}
    for exp_name in experiments:
        synth_data = load_patches_npz(patches_dir / exp_name / "synthetic_patches.npz")
        synth_data["has_lesion"] = detect_lesion_samples(synth_data["patches"])
        synth_datasets[exp_name] = synth_data

    real_data["has_lesion"] = detect_lesion_samples(real_data["patches"])

    # Track which rows belong to which prediction type for grouping
    row_idx = 0
    current_pred_type = None
    pred_type_start_row = {}
    pred_type_end_row = {}

    for dataset_info in all_datasets:
        pred_type, lp_norm, exp_name, _ = dataset_info
        data_row = row_idx + 1  # +1 for header row

        # Track prediction type row spans
        if pred_type != current_pred_type:
            if current_pred_type is not None:
                pred_type_end_row[current_pred_type] = row_idx
            pred_type_start_row[pred_type] = row_idx
            current_pred_type = pred_type
        pred_type_end_row[pred_type] = row_idx + 1

        row_idx += 1

    # Now render the grid
    row_idx = 0
    for dataset_info in all_datasets:
        pred_type, lp_norm, exp_name, _ = dataset_info
        data_row = row_idx + 1  # +1 for header row

        # Get data
        if pred_type == "Real":
            data = real_data
        else:
            data = synth_datasets[exp_name]

        # Column 0: Prediction type (grouped with rowspan)
        start_row = pred_type_start_row[pred_type]
        end_row = pred_type_end_row[pred_type]

        if row_idx == start_row:
            # Create merged cell for this prediction type group
            # For "Real", span both columns 0 and 1
            if pred_type == "Real":
                ax_pred = fig.add_subplot(gs[start_row + 1 : end_row + 1, 0:2])
            else:
                ax_pred = fig.add_subplot(gs[start_row + 1 : end_row + 1, 0])

            if pred_type == "Real":
                ax_pred.set_facecolor("#E0E0E0")
                label = "Real"
                text_color = "black"
            else:
                color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")
                ax_pred.set_facecolor(color)
                label = PREDICTION_TYPE_LABELS_SHORT.get(pred_type, pred_type)
                text_color = "white"

            ax_pred.text(
                0.5, 0.5, label,
                ha="center", va="center",
                fontsize=PLOT_SETTINGS["font_size"],
                fontweight="bold",
                color=text_color,
                rotation=0,  # Horizontal text
                transform=ax_pred.transAxes,
            )
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            for spine in ax_pred.spines.values():
                spine.set_linewidth(PLOT_SETTINGS["spine_linewidth"])
                if pred_type != "Real":
                    spine.set_edgecolor("white")

        # Column 1: Lp norm (skip for Real since it spans both columns)
        if pred_type != "Real":
            ax_lp = fig.add_subplot(gs[data_row, 1])

            color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")
            ax_lp.set_facecolor(_lighten_color(color, 0.7))
            lp_label = LP_NORM_LABELS.get(lp_norm, f"L{lp_norm}")

            ax_lp.text(
                0.5, 0.5, lp_label,
                ha="center", va="center",
                fontsize=PLOT_SETTINGS["annotation_fontsize"],
                fontweight="bold",
                transform=ax_lp.transAxes,
            )
            ax_lp.set_xticks([])
            ax_lp.set_yticks([])
            for spine in ax_lp.spines.values():
                spine.set_linewidth(PLOT_SETTINGS["spine_linewidth"] * 0.5)
                spine.set_edgecolor("#CCCCCC")

        # Data columns: Images for each z-bin
        for col_idx, zbin in enumerate(zbins):
            ax = fig.add_subplot(gs[data_row, col_idx + 2])
            _plot_image_cell(
                ax, data, zbin, show_lesion, overlay_alpha, overlay_color
            )

        row_idx += 1

    # =========================================================================
    # Save
    # =========================================================================

    base_path = output_path.with_suffix("")
    for fmt in formats:
        out_file = base_path.with_suffix(f".{fmt}")
        fig.savefig(
            out_file,
            dpi=PLOT_SETTINGS["dpi_print"],
            bbox_inches="tight",
            pad_inches=0.02,
        )
        logger.info("Saved: %s", out_file)

    plt.close(fig)


def _plot_image_cell(
    ax: plt.Axes,
    data: dict,
    zbin: int,
    show_lesion: bool,
    overlay_alpha: float,
    overlay_color: tuple[int, int, int],
) -> None:
    """Plot a single image cell."""
    patches = data["patches"]
    z_bins_arr = data["z_bins"]
    has_lesion = data["has_lesion"]

    # Find samples for this z-bin
    zbin_mask = z_bins_arr == zbin

    if show_lesion:
        # Prefer lesion samples if available
        lesion_mask = zbin_mask & has_lesion
        if lesion_mask.any():
            candidate_indices = np.where(lesion_mask)[0]
        else:
            candidate_indices = np.where(zbin_mask)[0]
    else:
        candidate_indices = np.where(zbin_mask)[0]

    if len(candidate_indices) > 0:
        idx = candidate_indices[0]
        image = patches[idx, 0]
        mask = patches[idx, 1]

        image_disp = to_display_range(image)

        if has_lesion[idx]:
            rgb = create_overlay(image_disp, mask, alpha=overlay_alpha, color=overlay_color)
        else:
            rgb = np.stack([image_disp] * 3, axis=-1)

        ax.imshow(rgb, aspect="equal")
    else:
        ax.set_facecolor("#F8F8F8")
        ax.text(
            0.5, 0.5, "-",
            ha="center", va="center",
            fontsize=PLOT_SETTINGS["font_size"],
            color="#CCCCCC",
            transform=ax.transAxes,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_SETTINGS["spine_linewidth"] * 0.3)
        spine.set_edgecolor("#E0E0E0")


def _lighten_color(hex_color: str, factor: float = 0.5) -> str:
    """Lighten a hex color by blending with white.

    Args:
        hex_color: Color in hex format (e.g., "#4477AA").
        factor: Blend factor (0 = original, 1 = white).

    Returns:
        Lightened color in hex format.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)

    return f"#{r:02x}{g:02x}{b:02x}"


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready image grid for ICIP 2026.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--patches-dir",
        type=str,
        required=True,
        help="Directory containing experiment subdirectories with patches.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output. Default: patches_dir/analysis",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to include. Default: all found.",
    )
    parser.add_argument(
        "--n-zbins",
        type=int,
        default=6,
        help="Number of representative z-bins (default: 6).",
    )
    parser.add_argument(
        "--show-lesion",
        action="store_true",
        default=True,
        help="Prefer samples with lesions (default: True).",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["pdf", "png"],
        help="Output formats (default: pdf png).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    patches_dir = Path(args.patches_dir)
    output_dir = Path(args.output_dir) if args.output_dir else patches_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_publication_image_grid(
        patches_dir=patches_dir,
        output_path=output_dir / "image_grid_publication",
        experiments=args.experiments,
        n_zbins=args.n_zbins,
        show_lesion=args.show_lesion,
        formats=args.formats,
    )


if __name__ == "__main__":
    main()
