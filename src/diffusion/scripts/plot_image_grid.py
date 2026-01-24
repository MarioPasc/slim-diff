#!/usr/bin/env python3
"""Standalone script for generating image grid visualizations.

Creates publication-quality image grids comparing real vs synthetic images
across z-bins and conditions (control/epilepsy, lesion/no-lesion).

Features:
- Configurable z-bins to display (specific list, range, or automatic selection)
- Configurable number of images per condition
- Optional title display
- Configurable DPI and figure size
- Support for showing only synthetic, only real, or both
- Optional lesion overlay with configurable color and alpha

Usage:
    # Basic usage with automatic z-bin selection
    python -m src.diffusion.scripts.plot_image_grid \
        --replicas-dir /path/to/replicas \
        --test-slices-csv /path/to/test.csv \
        --output-path output.png

    # Full control over parameters
    python -m src.diffusion.scripts.plot_image_grid \
        --replicas-dir /path/to/replicas \
        --test-slices-csv /path/to/test.csv \
        --output-path output.png \
        --zbins 5 10 15 20 25 \
        --n-images 4 \
        --no-title \
        --dpi 300 \
        --figsize 20,12

    # Show only synthetic images
    python -m src.diffusion.scripts.plot_image_grid \
        --replicas-dir /path/to/replicas \
        --output-path output.png \
        --synthetic-only

Example:
    python -m src.diffusion.scripts.plot_image_grid \
        --replicas-dir /media/mpascual/Sandisk2TB/research/jsddpm/results/epilepsy/icip2026/runs/velocity_lp_1.5/replicas \
        --test-slices-csv /media/mpascual/Sandisk2TB/research/jsddpm/data/epilepsy/slice_cache/test.csv \
        --output-path /media/mpascual/Sandisk2TB/research/jsddpm/results/epilepsy/icip2026/runs/velocity_lp_1.5/replicas/quality_check/image_grid.pdf \
        --zbins 5 10 15 20 25 \
        --min-lesion-pixels 25 \
        --no-title \
        --n-images 3 \
        --format pdf \
        --dpi 300
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================


def to_display_range(x: np.ndarray) -> np.ndarray:
    """Convert from [-1, 1] to [0, 1] for display.

    Args:
        x: Array in [-1, 1].

    Returns:
        Array in [0, 1].
    """
    return np.clip((x + 1) / 2, 0, 1)


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
    threshold: float = 0.0,
) -> np.ndarray:
    """Create image with mask overlay.

    Args:
        image: Grayscale image in [0, 1], shape (H, W).
        mask: Mask in [-1, 1] or [0, 1], shape (H, W).
        alpha: Overlay transparency.
        color: RGB color for overlay (0-255).
        threshold: Threshold for binarizing mask.

    Returns:
        RGB image with overlay, shape (H, W, 3).
    """
    # Ensure 2D
    if image.ndim == 3:
        image = image.squeeze()
    if mask.ndim == 3:
        mask = mask.squeeze()

    # Convert to RGB
    rgb = np.stack([image, image, image], axis=-1)

    # Binarize mask
    if mask.min() < 0:
        mask = to_display_range(mask)
    binary_mask = mask > (threshold + 1) / 2

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


def count_lesion_pixels(mask: np.ndarray, threshold: float = 0.0) -> int:
    """Count the number of lesion pixels in a mask.

    Args:
        mask: Mask array in [-1, 1] or [0, 1], shape (H, W).
        threshold: Threshold for binarizing mask.

    Returns:
        Number of pixels above threshold.
    """
    # Ensure 2D
    if mask.ndim == 3:
        mask = mask.squeeze()

    # Convert to [0, 1] if needed
    if mask.min() < 0:
        mask = to_display_range(mask)

    # Binarize and count
    binary_mask = mask > (threshold + 1) / 2
    return int(binary_mask.sum())


# =============================================================================
# Data Loading
# =============================================================================


def load_replica(npz_path: Path) -> dict[str, Any] | None:
    """Load single replica NPZ file.

    Args:
        npz_path: Path to replica_XXX.npz file.

    Returns:
        Dict with keys: replica_id, images, masks, zbin, lesion_present, domain.
        Returns None if loading fails.
    """
    try:
        # Parse replica ID from filename (e.g., replica_001.npz -> 1)
        match = re.search(r"replica_(\d+)\.npz$", npz_path.name)
        if not match:
            logger.warning(f"Could not parse replica ID from {npz_path.name}")
            return None
        replica_id = int(match.group(1))

        # Load NPZ
        data = np.load(npz_path, allow_pickle=True)

        # Extract arrays
        replica = {
            "replica_id": replica_id,
            "images": data["images"],
            "masks": data["masks"],
            "zbin": data["zbin"],
            "lesion_present": data["lesion_present"],
            "domain": data["domain"],
        }

        logger.debug(f"Loaded replica {replica_id}: {len(replica['images'])} samples")
        return replica

    except Exception as e:
        logger.warning(f"Failed to load {npz_path.name}: {e}")
        return None


def load_all_replicas(
    replicas_dir: Path,
    max_replicas: int | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Load all replicas from directory.

    Args:
        replicas_dir: Directory containing replica_*.npz files.
        max_replicas: Maximum number of replicas to load (None = all).
        verbose: Show progress bar.

    Returns:
        List of replica dicts.
    """
    # Find all replica files
    replica_files = sorted(replicas_dir.glob("replica_*.npz"))

    if not replica_files:
        raise FileNotFoundError(f"No replica_*.npz files found in {replicas_dir}")

    if max_replicas is not None:
        replica_files = replica_files[:max_replicas]

    logger.info(f"Loading {len(replica_files)} replicas from {replicas_dir}")

    # Load each replica
    replicas = []
    iterator = tqdm(replica_files, desc="Loading replicas", disable=not verbose)

    for npz_path in iterator:
        replica = load_replica(npz_path)
        if replica is not None:
            replicas.append(replica)

    if not replicas:
        raise ValueError("No replicas could be loaded successfully")

    logger.info(f"Successfully loaded {len(replicas)} replicas")
    return replicas


def load_real_slices(
    csv_path: Path,
    verbose: bool = True,
) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    """Load real slices from CSV.

    Args:
        csv_path: Path to CSV with columns: filepath, z_bin, has_lesion, source.
        verbose: Show progress.

    Returns:
        Dict mapping (zbin, lesion_present, domain) -> list of slice dicts with
        keys: image, mask.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Slices CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Map source to domain int
    source_to_domain = {"control": 0, "epilepsy": 1}

    # Base directory for slice files (same directory as CSV)
    base_dir = csv_path.parent

    # Group slices by condition
    slices_by_condition: dict[tuple[int, int, int], list[dict[str, Any]]] = {}

    iterator = tqdm(
        df.iterrows(), total=len(df), desc="Loading real slices", disable=not verbose
    )

    for _, row in iterator:
        zbin = int(row["z_bin"])
        lesion_present = 1 if row["has_lesion"] else 0
        domain = source_to_domain.get(row["source"], 0)

        key = (zbin, lesion_present, domain)

        # Load the NPZ file
        npz_path = base_dir / row["filepath"]
        if not npz_path.exists():
            continue

        try:
            data = np.load(npz_path)
            slice_dict = {
                "image": data["image"],
                "mask": data["mask"] if "mask" in data else np.zeros_like(data["image"]),
            }

            if key not in slices_by_condition:
                slices_by_condition[key] = []
            slices_by_condition[key].append(slice_dict)

        except Exception as e:
            logger.debug(f"Failed to load {npz_path}: {e}")
            continue

    total_slices = sum(len(v) for v in slices_by_condition.values())
    logger.info(f"Loaded {total_slices} real slices across {len(slices_by_condition)} conditions")

    return slices_by_condition


# =============================================================================
# Z-bin Selection
# =============================================================================


def select_representative_zbins(all_zbins: list[int], n_bins: int = 5) -> list[int]:
    """Select evenly-spaced z-bins.

    Args:
        all_zbins: List of all z-bin values.
        n_bins: Number of bins to select.

    Returns:
        List of selected z-bin values.
    """
    sorted_zbins = sorted(all_zbins)
    if len(sorted_zbins) <= n_bins:
        return sorted_zbins

    indices = np.linspace(0, len(sorted_zbins) - 1, n_bins, dtype=int)
    return [sorted_zbins[i] for i in indices]


def get_available_zbins(replicas: list[dict[str, Any]]) -> list[int]:
    """Get all unique z-bins from replicas.

    Args:
        replicas: List of replica dicts.

    Returns:
        Sorted list of unique z-bins.
    """
    all_zbins = set()
    for replica in replicas:
        all_zbins.update(replica["zbin"].tolist())
    return sorted(all_zbins)


# =============================================================================
# Visualization
# =============================================================================


def plot_image_grid(
    replicas: list[dict[str, Any]],
    zbins_to_show: list[int],
    n_images_per_condition: int,
    output_path: Path,
    real_slices: dict[tuple[int, int, int], list[dict[str, Any]]] | None = None,
    dpi: int = 150,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    overlay_alpha: float = 0.5,
    overlay_color: tuple[int, int, int] = (255, 0, 0),
    conditions_to_show: list[str] | None = None,
    show_zbin_labels: bool = True,
    cell_size: float = 1.5,
    min_lesion_pixels: int = 0,
    output_format: str = "png",
) -> None:
    """Create image grid visualization comparing real vs synthetic images.

    Args:
        replicas: List of replica dicts.
        zbins_to_show: Z-bins to visualize.
        n_images_per_condition: Images per condition.
        output_path: Path to save PNG.
        real_slices: Optional dict mapping (zbin, lesion, domain) -> list of real slice dicts.
        dpi: DPI for saved figure.
        title: Figure title. If None, no title is shown.
        figsize: Figure size (width, height). If None, auto-calculated.
        overlay_alpha: Alpha for lesion overlay.
        overlay_color: RGB color for lesion overlay.
        conditions_to_show: List of conditions to show. Options:
            "control_no_lesion", "control_lesion", "epilepsy_no_lesion", "epilepsy_lesion".
            If None, shows all conditions with data.
        show_zbin_labels: Whether to show z-bin labels on the left.
        cell_size: Size of each image cell in inches.
        min_lesion_pixels: Minimum number of lesion pixels for lesion conditions.
            Samples with fewer lesion pixels will be skipped. Default: 0 (no filtering).
        output_format: Output format, either 'png' or 'pdf'. Default: 'png'.
    """
    # Define all possible conditions: (lesion_present, domain_str, domain_int, label, color)
    all_conditions = [
        (0, "control", 0, "Control (No Lesion)", "steelblue", "control_no_lesion"),
        (1, "control", 0, "Control (Lesion)", "darkblue", "control_lesion"),
        (0, "epilepsy", 1, "Epilepsy (No Lesion)", "lightcoral", "epilepsy_no_lesion"),
        (1, "epilepsy", 1, "Epilepsy (Lesion)", "darkred", "epilepsy_lesion"),
    ]

    # Filter conditions if specified
    if conditions_to_show is not None:
        all_conditions = [c for c in all_conditions if c[5] in conditions_to_show]

    # Check which conditions actually have synthetic data
    valid_conditions = []
    for lesion_present, domain_str, domain, label, color, cond_key in all_conditions:
        has_data = False
        for replica in replicas:
            for zbin in zbins_to_show:
                mask = (
                    (replica["zbin"] == zbin)
                    & (replica["lesion_present"] == lesion_present)
                    & (replica["domain"] == domain)
                )
                if mask.any():
                    has_data = True
                    break
            if has_data:
                break
        if has_data:
            valid_conditions.append((lesion_present, domain_str, domain, label, color, cond_key))

    if not valid_conditions:
        logger.warning("No valid conditions found for image grid")
        return

    n_rows = len(zbins_to_show)
    show_real = real_slices is not None

    # Columns: for each condition, show real images then synthetic images
    if show_real:
        n_cols = len(valid_conditions) * n_images_per_condition * 2  # real + synthetic
    else:
        n_cols = len(valid_conditions) * n_images_per_condition

    # Calculate figure size if not provided
    if figsize is None:
        width = n_cols * cell_size + 1
        height = n_rows * cell_size + 2
        figsize = (width, height)

    # Create figure with extra space for headers
    fig = plt.figure(figsize=figsize)

    # Create gridspec for better control (2 header rows if showing real)
    n_header_rows = 2 if show_real else 1
    gs = fig.add_gridspec(
        n_rows + n_header_rows,
        n_cols,
        height_ratios=[0.12] * n_header_rows + [1] * n_rows,
        hspace=0.03,
        wspace=0.02,
        left=0.06,
        right=0.99,
        top=0.93 if title else 0.98,
        bottom=0.02,
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # Add headers
    if show_real:
        # Top-level headers: Real | Synthetic for each condition group
        for cond_idx, (lesion_present, domain_str, domain, label, color, _) in enumerate(
            valid_conditions
        ):
            base_col = cond_idx * n_images_per_condition * 2

            # Condition header spanning real + synthetic
            ax_cond = fig.add_subplot(gs[0, base_col : base_col + n_images_per_condition * 2])
            ax_cond.set_facecolor(color)
            ax_cond.text(
                0.5,
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                transform=ax_cond.transAxes,
            )
            ax_cond.set_xticks([])
            ax_cond.set_yticks([])
            for spine in ax_cond.spines.values():
                spine.set_visible(False)

            # Real sub-header
            ax_real = fig.add_subplot(gs[1, base_col : base_col + n_images_per_condition])
            ax_real.set_facecolor("#e8e8e8")
            ax_real.text(
                0.5,
                0.5,
                "Real",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#333",
                transform=ax_real.transAxes,
            )
            ax_real.set_xticks([])
            ax_real.set_yticks([])
            for spine in ax_real.spines.values():
                spine.set_visible(False)

            # Synthetic sub-header
            ax_synth = fig.add_subplot(
                gs[1, base_col + n_images_per_condition : base_col + n_images_per_condition * 2]
            )
            ax_synth.set_facecolor("#d0d0d0")
            ax_synth.text(
                0.5,
                0.5,
                "Synthetic",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#333",
                transform=ax_synth.transAxes,
            )
            ax_synth.set_xticks([])
            ax_synth.set_yticks([])
            for spine in ax_synth.spines.values():
                spine.set_visible(False)
    else:
        # Simple headers for synthetic only
        for cond_idx, (lesion_present, domain_str, domain, label, color, _) in enumerate(
            valid_conditions
        ):
            start_col = cond_idx * n_images_per_condition
            end_col = start_col + n_images_per_condition

            ax_header = fig.add_subplot(gs[0, start_col:end_col])
            ax_header.set_facecolor(color)
            ax_header.text(
                0.5,
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                transform=ax_header.transAxes,
            )
            ax_header.set_xticks([])
            ax_header.set_yticks([])
            for spine in ax_header.spines.values():
                spine.set_visible(False)

    # Helper function to plot a single image
    def plot_single_image(ax, image, mask_arr, lesion_present):
        """Plot a single image with optional lesion overlay."""
        # Convert to display range and ensure float32 for matplotlib
        image_disp = to_display_range(image).astype(np.float32)

        # Ensure 2D for stacking
        if image_disp.ndim == 3:
            image_disp = image_disp.squeeze()

        # Create overlay if lesion
        if lesion_present == 1 and mask_arr is not None:
            rgb = create_overlay(
                image_disp, mask_arr, alpha=overlay_alpha, color=overlay_color
            )
        else:
            rgb = np.stack([image_disp, image_disp, image_disp], axis=-1).astype(np.float32)

        ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot images
    for row_idx, zbin in enumerate(zbins_to_show):
        col_idx = 0

        for lesion_present, domain_str, domain, label, color, _ in valid_conditions:
            condition_key = (zbin, lesion_present, domain)

            if show_real:
                # Plot real images first
                real_samples = real_slices.get(condition_key, [])

                # Filter by minimum lesion pixels for lesion conditions
                if lesion_present == 1 and min_lesion_pixels > 0:
                    real_samples = [
                        s for s in real_samples
                        if s.get("mask") is not None
                        and count_lesion_pixels(s["mask"]) >= min_lesion_pixels
                    ]

                for i in range(n_images_per_condition):
                    ax = fig.add_subplot(gs[row_idx + n_header_rows, col_idx])
                    if i < len(real_samples):
                        slice_data = real_samples[i]
                        plot_single_image(
                            ax, slice_data["image"], slice_data.get("mask"), lesion_present
                        )
                    else:
                        ax.set_facecolor("#f5f5f5")
                        ax.text(
                            0.5,
                            0.5,
                            "\u2014",
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="#999",
                            transform=ax.transAxes,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Add z-bin label on leftmost column
                    if col_idx == 0 and show_zbin_labels:
                        ax.set_ylabel(
                            f"z={zbin}",
                            fontsize=8,
                            fontweight="bold",
                            rotation=0,
                            ha="right",
                            va="center",
                            labelpad=8,
                        )
                    col_idx += 1

            # Plot synthetic images
            samples_found = 0
            for replica in replicas:
                if samples_found >= n_images_per_condition:
                    break

                mask = (
                    (replica["zbin"] == zbin)
                    & (replica["lesion_present"] == lesion_present)
                    & (replica["domain"] == domain)
                )

                indices = np.where(mask)[0]

                # Filter indices by minimum lesion pixels for lesion conditions
                if lesion_present == 1 and min_lesion_pixels > 0:
                    indices = [
                        idx for idx in indices
                        if count_lesion_pixels(replica["masks"][idx]) >= min_lesion_pixels
                    ]

                for idx in indices[: n_images_per_condition - samples_found]:
                    ax = fig.add_subplot(gs[row_idx + n_header_rows, col_idx])
                    plot_single_image(
                        ax, replica["images"][idx], replica["masks"][idx], lesion_present
                    )

                    # Add z-bin label on leftmost column (only if not showing real)
                    if col_idx == 0 and not show_real and show_zbin_labels:
                        ax.set_ylabel(
                            f"z={zbin}",
                            fontsize=8,
                            fontweight="bold",
                            rotation=0,
                            ha="right",
                            va="center",
                            labelpad=8,
                        )

                    col_idx += 1
                    samples_found += 1

                if samples_found >= n_images_per_condition:
                    break

            # Fill empty synthetic cells if not enough samples
            while samples_found < n_images_per_condition:
                ax = fig.add_subplot(gs[row_idx + n_header_rows, col_idx])
                ax.set_facecolor("#f0f0f0")
                ax.text(
                    0.5,
                    0.5,
                    "\u2014",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#999",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                col_idx += 1
                samples_found += 1

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", format=output_format)
    plt.close()

    logger.info(f"Saved image grid to {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate image grid visualization comparing real vs synthetic images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with automatic z-bin selection
    python -m src.diffusion.scripts.plot_image_grid \\
        --replicas-dir /path/to/replicas \\
        --test-slices-csv /path/to/test.csv \\
        --output-path output.png

    # Specific z-bins with 4 images per condition
    python -m src.diffusion.scripts.plot_image_grid \\
        --replicas-dir /path/to/replicas \\
        --test-slices-csv /path/to/test.csv \\
        --output-path output.png \\
        --zbins 5 10 15 20 25 \\
        --n-images 4

    # Z-bin range
    python -m src.diffusion.scripts.plot_image_grid \\
        --replicas-dir /path/to/replicas \\
        --output-path output.png \\
        --zbin-range 5 25 \\
        --n-zbins 6

    # Synthetic only, no title
    python -m src.diffusion.scripts.plot_image_grid \\
        --replicas-dir /path/to/replicas \\
        --output-path output.png \\
        --synthetic-only \\
        --no-title
        """,
    )

    # Required arguments
    parser.add_argument(
        "--replicas-dir",
        type=str,
        required=True,
        help="Directory containing replica_*.npz files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for the image grid (PNG/PDF)",
    )

    # Optional real data
    parser.add_argument(
        "--test-slices-csv",
        type=str,
        default=None,
        help="Path to test.csv with real slice filepaths (optional)",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Show only synthetic images (ignore --test-slices-csv)",
    )

    # Z-bin selection
    zbin_group = parser.add_mutually_exclusive_group()
    zbin_group.add_argument(
        "--zbins",
        type=int,
        nargs="+",
        default=None,
        help="Specific z-bins to show (e.g., --zbins 5 10 15 20 25)",
    )
    zbin_group.add_argument(
        "--zbin-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Range of z-bins to show (e.g., --zbin-range 5 25)",
    )
    parser.add_argument(
        "--n-zbins",
        type=int,
        default=5,
        help="Number of z-bins to select (used with --zbin-range or automatic selection, default: 5)",
    )

    # Image configuration
    parser.add_argument(
        "--n-images",
        type=int,
        default=3,
        help="Number of images per condition (default: 3)",
    )
    parser.add_argument(
        "--max-replicas",
        type=int,
        default=None,
        help="Maximum number of replicas to load (default: all)",
    )

    # Conditions to show
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        choices=["control_no_lesion", "control_lesion", "epilepsy_no_lesion", "epilepsy_lesion"],
        default=None,
        help="Specific conditions to show (default: all with data)",
    )

    # Title and labels
    parser.add_argument(
        "--title",
        type=str,
        default="Real vs Synthetic Comparison",
        help="Figure title (default: 'Real vs Synthetic Comparison')",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Hide the figure title",
    )
    parser.add_argument(
        "--no-zbin-labels",
        action="store_true",
        help="Hide z-bin labels on the left",
    )

    # Figure appearance
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure (default: 150)",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default=None,
        help="Figure size as 'width,height' in inches (e.g., '20,12'). Default: auto-calculated.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=1.5,
        help="Size of each image cell in inches (default: 1.5)",
    )

    # Overlay configuration
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Alpha for lesion overlay (default: 0.5)",
    )
    parser.add_argument(
        "--overlay-color",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=[255, 0, 0],
        help="RGB color for lesion overlay (default: 255 0 0 = red)",
    )

    # Output format
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf"],
        default="png",
        help="Output format: 'png' or 'pdf' (default: png)",
    )

    # Lesion filtering
    parser.add_argument(
        "--min-lesion-pixels",
        type=int,
        default=0,
        help="Minimum number of lesion pixels for lesion conditions. "
        "Samples with fewer lesion pixels will be skipped. (default: 0 = no filtering)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Update logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Convert paths
    replicas_dir = Path(args.replicas_dir)
    output_path = Path(args.output_path)

    # Validate replicas directory
    if not replicas_dir.exists():
        raise FileNotFoundError(f"Replicas directory not found: {replicas_dir}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load replicas
    replicas = load_all_replicas(replicas_dir, max_replicas=args.max_replicas)

    # Load real slices if provided and not synthetic-only mode
    real_slices = None
    if args.test_slices_csv and not args.synthetic_only:
        test_csv = Path(args.test_slices_csv)
        if test_csv.exists():
            real_slices = load_real_slices(test_csv)
        else:
            logger.warning(f"Test slices CSV not found: {test_csv}")

    # Determine z-bins to show
    available_zbins = get_available_zbins(replicas)
    logger.info(f"Available z-bins in replicas: {available_zbins}")

    if args.zbins:
        # User specified specific z-bins
        zbins_to_show = [z for z in args.zbins if z in available_zbins]
        if not zbins_to_show:
            logger.warning(f"None of specified z-bins {args.zbins} found in replicas")
            zbins_to_show = select_representative_zbins(available_zbins, args.n_zbins)
    elif args.zbin_range:
        # User specified range
        min_z, max_z = args.zbin_range
        range_zbins = [z for z in available_zbins if min_z <= z <= max_z]
        zbins_to_show = select_representative_zbins(range_zbins, args.n_zbins)
    else:
        # Automatic selection
        zbins_to_show = select_representative_zbins(available_zbins, args.n_zbins)

    logger.info(f"Selected z-bins: {zbins_to_show}")

    # Parse figsize if provided
    figsize = None
    if args.figsize:
        try:
            parts = args.figsize.split(",")
            figsize = (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError):
            logger.warning(f"Invalid figsize format: {args.figsize}. Using auto-calculated size.")

    # Determine title
    title = None if args.no_title else args.title
    if args.synthetic_only and title == "Real vs Synthetic Comparison":
        title = "Synthetic Replicas"

    # Log lesion filtering
    if args.min_lesion_pixels > 0:
        logger.info(
            f"Filtering lesion samples: only showing masks with >= {args.min_lesion_pixels} pixels"
        )

    # Generate image grid
    plot_image_grid(
        replicas=replicas,
        zbins_to_show=zbins_to_show,
        n_images_per_condition=args.n_images,
        output_path=output_path,
        real_slices=real_slices,
        dpi=args.dpi,
        title=title,
        figsize=figsize,
        overlay_alpha=args.overlay_alpha,
        overlay_color=tuple(args.overlay_color),
        conditions_to_show=args.conditions,
        show_zbin_labels=not args.no_zbin_labels,
        cell_size=args.cell_size,
        min_lesion_pixels=args.min_lesion_pixels,
        output_format=args.format,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
