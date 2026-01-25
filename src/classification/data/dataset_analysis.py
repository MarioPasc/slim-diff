"""Dataset analysis and visualization for extracted patches.

Generates publication-quality visualizations comparing real vs synthetic data:
1. Z-bin distribution histograms showing lesion/no-lesion breakdown
2. Image comparison grid with representative samples across z-bins and experiments

Usage:
    python -m src.classification.data.dataset_analysis \
        --patches-dir /path/to/full_images \
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================


def load_patches_npz(npz_path: Path) -> dict:
    """Load patches from NPZ file.

    Args:
        npz_path: Path to patches.npz file.

    Returns:
        Dict with patches, z_bins, and other metadata.
    """
    data = np.load(npz_path, allow_pickle=True)
    return {
        "patches": data["patches"],
        "z_bins": data["z_bins"],
        "subject_ids": data.get("subject_ids", None),
        "sources": data.get("sources", None),
        "replica_ids": data.get("replica_ids", None),
        "sample_indices": data.get("sample_indices", None),
    }


def detect_lesion_samples(patches: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Detect which samples contain lesions based on mask channel.

    Args:
        patches: Array of shape (N, 2, H, W) where channel 1 is the mask.
        threshold: Threshold for considering a pixel as lesion (in [-1, 1] range).

    Returns:
        Boolean array of shape (N,) indicating lesion presence.
    """
    # Channel 1 is mask, values > threshold indicate lesion
    masks = patches[:, 1, :, :]
    # Count pixels above threshold for each sample
    lesion_counts = (masks > threshold).sum(axis=(1, 2))
    # Consider it a lesion if there are any positive pixels
    return lesion_counts > 0


def list_experiments(patches_dir: Path) -> list[str]:
    """List all experiment directories in patches_dir.

    Args:
        patches_dir: Base directory containing experiment subdirectories.

    Returns:
        Sorted list of experiment names.
    """
    experiments = []
    for subdir in sorted(patches_dir.iterdir()):
        if subdir.is_dir() and (subdir / "synthetic_patches.npz").exists():
            experiments.append(subdir.name)
    return experiments


# =============================================================================
# Distribution Visualization
# =============================================================================


def plot_zbin_distribution(
    patches_dir: Path,
    output_path: Path,
    experiments: list[str] | None = None,
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 150,
) -> None:
    """Plot z-bin distribution comparing real vs synthetic across experiments.

    Creates a multi-panel figure showing:
    - Top row: Lesion samples per z-bin
    - Bottom row: No-lesion samples per z-bin

    Args:
        patches_dir: Directory containing experiment subdirectories.
        output_path: Path to save the figure.
        experiments: List of experiment names. If None, uses all found.
        figsize: Figure size (width, height).
        dpi: DPI for saved figure.
    """
    patches_dir = Path(patches_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover experiments if not provided
    if experiments is None:
        experiments = list_experiments(patches_dir)

    if not experiments:
        logger.warning("No experiments found in %s", patches_dir)
        return

    # Load real data from first experiment (same for all)
    first_exp_dir = patches_dir / experiments[0]
    real_data = load_patches_npz(first_exp_dir / "real_patches.npz")
    real_lesion = detect_lesion_samples(real_data["patches"])

    # Get z-bin range
    all_zbins = np.unique(real_data["z_bins"])
    n_zbins = len(all_zbins)

    # Prepare data for plotting
    n_experiments = len(experiments)
    n_cols = min(3, n_experiments + 1)  # +1 for real
    n_rows = int(np.ceil((n_experiments + 1) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle("Z-bin Distribution: Lesion vs No-Lesion Samples", fontsize=14, fontweight="bold")

    # Color scheme
    lesion_color = "#e74c3c"
    nolesion_color = "#3498db"
    bar_width = 0.8

    # Plot real data first
    ax = axes.flat[0]
    _plot_single_distribution(
        ax,
        real_data["z_bins"],
        real_lesion,
        all_zbins,
        "Real Data",
        lesion_color,
        nolesion_color,
        bar_width,
    )

    # Plot each experiment
    for i, exp_name in enumerate(experiments):
        ax = axes.flat[i + 1]
        exp_dir = patches_dir / exp_name
        synth_data = load_patches_npz(exp_dir / "synthetic_patches.npz")
        synth_lesion = detect_lesion_samples(synth_data["patches"])

        _plot_single_distribution(
            ax,
            synth_data["z_bins"],
            synth_lesion,
            all_zbins,
            exp_name,
            lesion_color,
            nolesion_color,
            bar_width,
        )

    # Hide unused axes
    for i in range(n_experiments + 1, len(axes.flat)):
        axes.flat[i].set_visible(False)

    # Add legend
    lesion_patch = mpatches.Patch(color=lesion_color, label="Lesion")
    nolesion_patch = mpatches.Patch(color=nolesion_color, label="No Lesion")
    fig.legend(
        handles=[lesion_patch, nolesion_patch],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info("Saved z-bin distribution to %s", output_path)


def _plot_single_distribution(
    ax: plt.Axes,
    z_bins: np.ndarray,
    has_lesion: np.ndarray,
    all_zbins: np.ndarray,
    title: str,
    lesion_color: str,
    nolesion_color: str,
    bar_width: float,
) -> None:
    """Plot stacked bar chart for a single dataset."""
    # Count lesion and no-lesion per z-bin
    lesion_counts = np.zeros(len(all_zbins))
    nolesion_counts = np.zeros(len(all_zbins))

    for i, zbin in enumerate(all_zbins):
        mask = z_bins == zbin
        lesion_counts[i] = (mask & has_lesion).sum()
        nolesion_counts[i] = (mask & ~has_lesion).sum()

    x = np.arange(len(all_zbins))

    ax.bar(x, nolesion_counts, bar_width, label="No Lesion", color=nolesion_color, alpha=0.8)
    ax.bar(x, lesion_counts, bar_width, bottom=nolesion_counts, label="Lesion", color=lesion_color, alpha=0.8)

    ax.set_xlabel("Z-bin")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Show every 5th z-bin label
    step = max(1, len(all_zbins) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(all_zbins[::step])

    # Add total count annotation
    total = int(lesion_counts.sum() + nolesion_counts.sum())
    lesion_pct = 100 * lesion_counts.sum() / total if total > 0 else 0
    ax.annotate(
        f"N={total}\n({lesion_pct:.1f}% lesion)",
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def plot_zbin_comparison_bars(
    patches_dir: Path,
    output_path: Path,
    experiments: list[str] | None = None,
    figsize: tuple[float, float] = (16, 6),
    dpi: int = 150,
) -> None:
    """Plot side-by-side comparison of z-bin distributions.

    Creates grouped bar chart comparing real vs all synthetic experiments.

    Args:
        patches_dir: Directory containing experiment subdirectories.
        output_path: Path to save the figure.
        experiments: List of experiment names. If None, uses all found.
        figsize: Figure size (width, height).
        dpi: DPI for saved figure.
    """
    patches_dir = Path(patches_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if experiments is None:
        experiments = list_experiments(patches_dir)

    if not experiments:
        logger.warning("No experiments found")
        return

    # Load real data
    first_exp_dir = patches_dir / experiments[0]
    real_data = load_patches_npz(first_exp_dir / "real_patches.npz")
    real_lesion = detect_lesion_samples(real_data["patches"])
    all_zbins = np.unique(real_data["z_bins"])

    # Compute counts for all datasets
    n_datasets = len(experiments) + 1  # +1 for real
    lesion_counts = np.zeros((n_datasets, len(all_zbins)))
    nolesion_counts = np.zeros((n_datasets, len(all_zbins)))
    dataset_names = ["Real"] + experiments

    # Real data counts
    for i, zbin in enumerate(all_zbins):
        mask = real_data["z_bins"] == zbin
        lesion_counts[0, i] = (mask & real_lesion).sum()
        nolesion_counts[0, i] = (mask & ~real_lesion).sum()

    # Synthetic data counts
    for d, exp_name in enumerate(experiments, start=1):
        synth_data = load_patches_npz(patches_dir / exp_name / "synthetic_patches.npz")
        synth_lesion = detect_lesion_samples(synth_data["patches"])
        for i, zbin in enumerate(all_zbins):
            mask = synth_data["z_bins"] == zbin
            lesion_counts[d, i] = (mask & synth_lesion).sum()
            nolesion_counts[d, i] = (mask & ~synth_lesion).sum()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Z-bin Distribution Comparison", fontsize=14, fontweight="bold")

    x = np.arange(len(all_zbins))
    width = 0.8 / n_datasets
    colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))

    # Plot lesion samples
    for d in range(n_datasets):
        offset = (d - n_datasets / 2 + 0.5) * width
        ax1.bar(x + offset, lesion_counts[d], width, label=dataset_names[d], color=colors[d], alpha=0.8)

    ax1.set_xlabel("Z-bin")
    ax1.set_ylabel("Count")
    ax1.set_title("Lesion Samples", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)
    step = max(1, len(all_zbins) // 10)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels(all_zbins[::step])

    # Plot no-lesion samples
    for d in range(n_datasets):
        offset = (d - n_datasets / 2 + 0.5) * width
        ax2.bar(x + offset, nolesion_counts[d], width, label=dataset_names[d], color=colors[d], alpha=0.8)

    ax2.set_xlabel("Z-bin")
    ax2.set_ylabel("Count")
    ax2.set_title("No-Lesion Samples", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels(all_zbins[::step])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info("Saved z-bin comparison to %s", output_path)


# =============================================================================
# Image Grid Visualization
# =============================================================================


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

    # Convert to RGB
    rgb = np.stack([image, image, image], axis=-1).astype(np.float32)

    # Binarize mask (values > 0 in [-1, 1] space)
    binary_mask = mask > 0

    # Normalize color
    color_norm = np.array(color, dtype=np.float32) / 255.0

    # Apply overlay
    if binary_mask.any():
        for c in range(3):
            rgb[:, :, c] = np.where(
                binary_mask,
                (1 - alpha) * rgb[:, :, c] + alpha * color_norm[c],
                rgb[:, :, c],
            )

    return rgb


def select_representative_zbins(z_bins: np.ndarray, n_bins: int = 5) -> list[int]:
    """Select evenly-spaced representative z-bins."""
    unique_zbins = sorted(np.unique(z_bins))
    if len(unique_zbins) <= n_bins:
        return unique_zbins
    indices = np.linspace(0, len(unique_zbins) - 1, n_bins, dtype=int)
    return [unique_zbins[i] for i in indices]


def plot_image_comparison_grid(
    patches_dir: Path,
    output_path: Path,
    experiments: list[str] | None = None,
    n_zbins: int = 6,
    n_samples: int = 1,
    show_lesion: bool = True,
    overlay_alpha: float = 0.4,
    overlay_color: tuple[int, int, int] = (255, 0, 0),
    cell_size: float = 1.2,
    dpi: int = 150,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot image comparison grid with real vs synthetic experiments.

    Creates a grid where:
    - X-axis: Representative z-bins
    - Y-axis: Real | Experiment_0 | Experiment_1 | ... | Experiment_n

    Args:
        patches_dir: Directory containing experiment subdirectories.
        output_path: Path to save the figure.
        experiments: List of experiment names. If None, uses all found.
        n_zbins: Number of representative z-bins to show.
        n_samples: Number of sample images per cell.
        show_lesion: If True, prefer samples with lesions.
        overlay_alpha: Alpha for lesion overlay.
        overlay_color: RGB color for lesion overlay.
        cell_size: Size of each image cell in inches.
        dpi: DPI for saved figure.
        figsize: Figure size. If None, auto-calculated.
    """
    patches_dir = Path(patches_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if experiments is None:
        experiments = list_experiments(patches_dir)

    if not experiments:
        logger.warning("No experiments found")
        return

    # Load real data
    first_exp_dir = patches_dir / experiments[0]
    real_data = load_patches_npz(first_exp_dir / "real_patches.npz")

    # Select representative z-bins
    zbins = select_representative_zbins(real_data["z_bins"], n_zbins)
    logger.info("Selected z-bins: %s", zbins)

    # Load all datasets
    datasets = [("Real", real_data)]
    for exp_name in experiments:
        synth_data = load_patches_npz(patches_dir / exp_name / "synthetic_patches.npz")
        datasets.append((exp_name, synth_data))

    # Layout
    n_rows = len(datasets)
    n_cols = len(zbins) * n_samples

    if figsize is None:
        figsize = (n_cols * cell_size + 2, n_rows * cell_size + 1)

    fig = plt.figure(figsize=figsize)

    # Create gridspec with header row
    gs = GridSpec(
        n_rows + 1,
        n_cols,
        figure=fig,
        height_ratios=[0.15] + [1] * n_rows,
        hspace=0.03,
        wspace=0.03,
        left=0.12,
        right=0.98,
        top=0.95,
        bottom=0.02,
    )

    fig.suptitle(
        "Image Comparison: Real vs Synthetic Experiments",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    # Header row with z-bin labels
    for col_idx, zbin in enumerate(zbins):
        for sample_idx in range(n_samples):
            ax = fig.add_subplot(gs[0, col_idx * n_samples + sample_idx])
            ax.set_facecolor("#e0e0e0")
            if sample_idx == n_samples // 2:
                ax.text(
                    0.5, 0.5, f"z={zbin}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    transform=ax.transAxes,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Plot images
    for row_idx, (name, data) in enumerate(datasets):
        patches = data["patches"]
        z_bins_arr = data["z_bins"]
        has_lesion = detect_lesion_samples(patches)

        for col_idx, zbin in enumerate(zbins):
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

            for sample_idx in range(n_samples):
                ax = fig.add_subplot(gs[row_idx + 1, col_idx * n_samples + sample_idx])

                if sample_idx < len(candidate_indices):
                    idx = candidate_indices[sample_idx]
                    image = patches[idx, 0]  # Channel 0 is image
                    mask = patches[idx, 1]   # Channel 1 is mask

                    # Convert to display range
                    image_disp = to_display_range(image)

                    # Create overlay if lesion present
                    if has_lesion[idx]:
                        rgb = create_overlay(image_disp, mask, alpha=overlay_alpha, color=overlay_color)
                    else:
                        rgb = np.stack([image_disp] * 3, axis=-1)

                    ax.imshow(rgb)
                else:
                    # No sample available
                    ax.set_facecolor("#f5f5f5")
                    ax.text(
                        0.5, 0.5, "-",
                        ha="center", va="center",
                        fontsize=12, color="#999",
                        transform=ax.transAxes,
                    )

                ax.set_xticks([])
                ax.set_yticks([])

                # Add row label on first column
                if col_idx == 0 and sample_idx == 0:
                    ax.set_ylabel(
                        name,
                        fontsize=9,
                        fontweight="bold",
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=10,
                    )

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info("Saved image comparison grid to %s", output_path)


def plot_lesion_nolesion_grid(
    patches_dir: Path,
    output_path: Path,
    experiments: list[str] | None = None,
    n_zbins: int = 5,
    overlay_alpha: float = 0.4,
    overlay_color: tuple[int, int, int] = (255, 0, 0),
    cell_size: float = 1.2,
    dpi: int = 150,
) -> None:
    """Plot grid showing lesion and no-lesion samples side by side.

    Creates a grid where:
    - X-axis: Representative z-bins, with Lesion | No-Lesion sub-columns
    - Y-axis: Real | Experiment_0 | ... | Experiment_n

    Args:
        patches_dir: Directory containing experiment subdirectories.
        output_path: Path to save the figure.
        experiments: List of experiment names.
        n_zbins: Number of representative z-bins.
        overlay_alpha: Alpha for lesion overlay.
        overlay_color: RGB color for lesion overlay.
        cell_size: Size of each cell in inches.
        dpi: DPI for saved figure.
    """
    patches_dir = Path(patches_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if experiments is None:
        experiments = list_experiments(patches_dir)

    if not experiments:
        logger.warning("No experiments found")
        return

    # Load real data
    first_exp_dir = patches_dir / experiments[0]
    real_data = load_patches_npz(first_exp_dir / "real_patches.npz")

    # Select representative z-bins
    zbins = select_representative_zbins(real_data["z_bins"], n_zbins)

    # Load all datasets
    datasets = [("Real", real_data)]
    for exp_name in experiments:
        synth_data = load_patches_npz(patches_dir / exp_name / "synthetic_patches.npz")
        datasets.append((exp_name, synth_data))

    # Layout: 2 images per z-bin (lesion + no-lesion)
    n_rows = len(datasets)
    n_cols = len(zbins) * 2

    figsize = (n_cols * cell_size + 2, n_rows * cell_size + 1.5)
    fig = plt.figure(figsize=figsize)

    # Create gridspec with 2 header rows
    gs = GridSpec(
        n_rows + 2,
        n_cols,
        figure=fig,
        height_ratios=[0.12, 0.12] + [1] * n_rows,
        hspace=0.03,
        wspace=0.03,
        left=0.10,
        right=0.98,
        top=0.94,
        bottom=0.02,
    )

    fig.suptitle(
        "Lesion vs No-Lesion Comparison Across Z-bins",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    # Top header: z-bin labels
    for col_idx, zbin in enumerate(zbins):
        ax = fig.add_subplot(gs[0, col_idx * 2 : col_idx * 2 + 2])
        ax.set_facecolor("#d0d0d0")
        ax.text(
            0.5, 0.5, f"z={zbin}",
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Second header: Lesion / No-Lesion labels
    for col_idx in range(len(zbins)):
        # Lesion
        ax_l = fig.add_subplot(gs[1, col_idx * 2])
        ax_l.set_facecolor("#ffcccc")
        ax_l.text(
            0.5, 0.5, "L",
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="#990000",
            transform=ax_l.transAxes,
        )
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        for spine in ax_l.spines.values():
            spine.set_visible(False)

        # No-Lesion
        ax_nl = fig.add_subplot(gs[1, col_idx * 2 + 1])
        ax_nl.set_facecolor("#ccccff")
        ax_nl.text(
            0.5, 0.5, "NL",
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="#000099",
            transform=ax_nl.transAxes,
        )
        ax_nl.set_xticks([])
        ax_nl.set_yticks([])
        for spine in ax_nl.spines.values():
            spine.set_visible(False)

    # Plot images
    for row_idx, (name, data) in enumerate(datasets):
        patches = data["patches"]
        z_bins_arr = data["z_bins"]
        has_lesion = detect_lesion_samples(patches)

        for col_idx, zbin in enumerate(zbins):
            zbin_mask = z_bins_arr == zbin

            # Lesion sample
            lesion_mask = zbin_mask & has_lesion
            ax_l = fig.add_subplot(gs[row_idx + 2, col_idx * 2])

            if lesion_mask.any():
                idx = np.where(lesion_mask)[0][0]
                image = to_display_range(patches[idx, 0])
                mask = patches[idx, 1]
                rgb = create_overlay(image, mask, alpha=overlay_alpha, color=overlay_color)
                ax_l.imshow(rgb)
            else:
                ax_l.set_facecolor("#fff5f5")
                ax_l.text(0.5, 0.5, "-", ha="center", va="center", color="#ccc", transform=ax_l.transAxes)

            ax_l.set_xticks([])
            ax_l.set_yticks([])

            # No-lesion sample
            nolesion_mask = zbin_mask & ~has_lesion
            ax_nl = fig.add_subplot(gs[row_idx + 2, col_idx * 2 + 1])

            if nolesion_mask.any():
                idx = np.where(nolesion_mask)[0][0]
                image = to_display_range(patches[idx, 0])
                rgb = np.stack([image] * 3, axis=-1)
                ax_nl.imshow(rgb)
            else:
                ax_nl.set_facecolor("#f5f5ff")
                ax_nl.text(0.5, 0.5, "-", ha="center", va="center", color="#ccc", transform=ax_nl.transAxes)

            ax_nl.set_xticks([])
            ax_nl.set_yticks([])

            # Row label
            if col_idx == 0:
                ax_l.set_ylabel(
                    name,
                    fontsize=9,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=10,
                )

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info("Saved lesion/no-lesion grid to %s", output_path)


# =============================================================================
# Main Analysis Runner
# =============================================================================


def run_dataset_analysis(
    patches_dir: Path,
    output_dir: Path,
    experiments: list[str] | None = None,
    dpi: int = 150,
) -> None:
    """Run full dataset analysis and save all visualizations.

    Args:
        patches_dir: Directory containing experiment subdirectories.
        output_dir: Directory to save analysis outputs.
        experiments: List of experiments. If None, uses all found.
        dpi: DPI for saved figures.
    """
    patches_dir = Path(patches_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running dataset analysis...")
    logger.info("Patches directory: %s", patches_dir)
    logger.info("Output directory: %s", output_dir)

    if experiments is None:
        experiments = list_experiments(patches_dir)

    logger.info("Experiments found: %s", experiments)

    if not experiments:
        logger.warning("No experiments found, exiting")
        return

    # 1. Z-bin distribution per dataset
    logger.info("Generating z-bin distribution plots...")
    plot_zbin_distribution(
        patches_dir,
        output_dir / "zbin_distribution.png",
        experiments=experiments,
        dpi=dpi,
    )

    # 2. Z-bin comparison bars
    logger.info("Generating z-bin comparison bars...")
    plot_zbin_comparison_bars(
        patches_dir,
        output_dir / "zbin_comparison.png",
        experiments=experiments,
        dpi=dpi,
    )

    # 3. Image comparison grid (lesion samples)
    logger.info("Generating image comparison grid (lesion)...")
    plot_image_comparison_grid(
        patches_dir,
        output_dir / "image_grid_lesion.png",
        experiments=experiments,
        n_zbins=6,
        n_samples=2,
        show_lesion=True,
        dpi=dpi,
    )

    # 4. Image comparison grid (any samples)
    logger.info("Generating image comparison grid (all)...")
    plot_image_comparison_grid(
        patches_dir,
        output_dir / "image_grid_all.png",
        experiments=experiments,
        n_zbins=6,
        n_samples=2,
        show_lesion=False,
        dpi=dpi,
    )

    # 5. Lesion vs no-lesion grid
    logger.info("Generating lesion vs no-lesion grid...")
    plot_lesion_nolesion_grid(
        patches_dir,
        output_dir / "lesion_nolesion_grid.png",
        experiments=experiments,
        n_zbins=5,
        dpi=dpi,
    )

    # Save PDF versions
    logger.info("Generating PDF versions...")
    for png_name in ["zbin_distribution", "zbin_comparison", "image_grid_lesion", "image_grid_all", "lesion_nolesion_grid"]:
        png_path = output_dir / f"{png_name}.png"
        pdf_path = output_dir / f"{png_name}.pdf"
        if png_path.exists():
            # Re-generate as PDF
            if "zbin_distribution" in png_name:
                plot_zbin_distribution(patches_dir, pdf_path, experiments, dpi=dpi)
            elif "zbin_comparison" in png_name:
                plot_zbin_comparison_bars(patches_dir, pdf_path, experiments, dpi=dpi)
            elif "image_grid_lesion" in png_name:
                plot_image_comparison_grid(patches_dir, pdf_path, experiments, n_zbins=6, n_samples=2, show_lesion=True, dpi=dpi)
            elif "image_grid_all" in png_name:
                plot_image_comparison_grid(patches_dir, pdf_path, experiments, n_zbins=6, n_samples=2, show_lesion=False, dpi=dpi)
            elif "lesion_nolesion" in png_name:
                plot_lesion_nolesion_grid(patches_dir, pdf_path, experiments, n_zbins=5, dpi=dpi)

    logger.info("Dataset analysis complete. Output saved to %s", output_dir)


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Dataset analysis and visualization for extracted patches.",
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
        help="Directory to save analysis outputs. Default: patches_dir/analysis",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to analyze. Default: all found.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures (default: 150).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    patches_dir = Path(args.patches_dir)
    output_dir = Path(args.output_dir) if args.output_dir else patches_dir / "analysis"

    run_dataset_analysis(
        patches_dir=patches_dir,
        output_dir=output_dir,
        experiments=args.experiments,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
