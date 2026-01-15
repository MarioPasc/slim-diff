#!/usr/bin/env python3
"""Visualize extracted audition patches with Real vs Synthetic comparison.

This script creates a grid visualization of extracted patches showing:
- Z-bins on the y-axis
- Two groups per row: Real and Synthetic patches
- Optional lesion overlay highlighting

Usage:
    python -m src.diffusion.audition.scripts.visualize_patches --config path/to/audition.yaml

Example:
    python -m src.diffusion.audition.scripts.visualize_patches \
        --config src/diffusion/audition/config/audition.yaml \
        --n-per-group 5 \
        --overlay
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def to_display_range(x: np.ndarray) -> np.ndarray:
    """Convert from [-1, 1] to [0, 1] for display.

    Args:
        x: Array in [-1, 1] range.

    Returns:
        Array in [0, 1] range.
    """
    return np.clip((x + 1) / 2, 0, 1)


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    color: tuple[int, int, int] = (255, 0, 0),
    threshold: float = 0.0,
) -> np.ndarray:
    """Create image with lesion mask overlay.

    Args:
        image: Grayscale image in [0, 1] range, shape (H, W).
        mask: Lesion mask in [-1, 1] range, shape (H, W).
        alpha: Overlay transparency (0=invisible, 1=opaque).
        color: RGB color tuple for overlay.
        threshold: Mask threshold for lesion pixels.

    Returns:
        RGB image with overlay, shape (H, W, 3), values in [0, 1].
    """
    # Convert grayscale to RGB
    rgb = np.stack([image, image, image], axis=-1)

    # Normalize color to [0, 1]
    color_normalized = np.array(color) / 255.0

    # Create mask for lesion pixels
    lesion_mask = mask > threshold

    # Apply overlay where lesion is present
    for c in range(3):
        rgb[..., c] = np.where(
            lesion_mask,
            (1 - alpha) * rgb[..., c] + alpha * color_normalized[c],
            rgb[..., c],
        )

    return rgb


def load_patches(patches_dir: Path) -> tuple[dict, dict]:
    """Load extracted patches from NPZ files.

    Args:
        patches_dir: Directory containing real_patches.npz and synthetic_patches.npz.

    Returns:
        Tuple of (real_data, synthetic_data) dictionaries with 'images', 'masks', 'z_bins' keys.
    """
    real_path = patches_dir / "real_patches.npz"
    synthetic_path = patches_dir / "synthetic_patches.npz"

    if not real_path.exists():
        raise FileNotFoundError(f"Real patches not found: {real_path}")
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Synthetic patches not found: {synthetic_path}")

    logger.info(f"Loading real patches from {real_path}")
    real_npz = np.load(real_path)
    # Patches are stored as (N, 2, H, W): channel 0 = image, channel 1 = mask
    real_data = {
        "images": real_npz["patches"][:, 0],  # Extract image channel
        "masks": real_npz["patches"][:, 1],   # Extract mask channel
        "z_bins": real_npz["z_bins"],
    }

    logger.info(f"Loading synthetic patches from {synthetic_path}")
    synth_npz = np.load(synthetic_path)
    synthetic_data = {
        "images": synth_npz["patches"][:, 0],  # Extract image channel
        "masks": synth_npz["patches"][:, 1],   # Extract mask channel
        "z_bins": synth_npz["z_bins"],
    }

    return real_data, synthetic_data


def get_zbin_samples(
    data: dict,
    z_bin: int,
    n_samples: int,
    rng: np.random.Generator,
) -> list[int]:
    """Get random sample indices for a specific z-bin.

    Args:
        data: Patch data dictionary with 'z_bins' key.
        z_bin: Target z-bin.
        n_samples: Number of samples to select.
        rng: Random number generator.

    Returns:
        List of sample indices.
    """
    z_bins = data["z_bins"]
    indices = np.where(z_bins == z_bin)[0]

    if len(indices) == 0:
        return []

    # Sample with replacement if needed
    if len(indices) < n_samples:
        selected = rng.choice(indices, size=n_samples, replace=True)
    else:
        selected = rng.choice(indices, size=n_samples, replace=False)

    return selected.tolist()


def visualize_patches(
    patches_dir: Path,
    output_path: Path,
    n_per_group: int = 5,
    show_overlay: bool = True,
    overlay_alpha: float = 0.3,
    seed: int = 42,
    figsize_per_patch: float = 1.5,
    dpi: int = 150,
    zbins_filter: list[int] | None = None,
) -> None:
    """Create visualization grid of Real vs Synthetic patches.

    Args:
        patches_dir: Directory containing extracted patches.
        output_path: Path to save the figure.
        n_per_group: Number of patches per group (Real/Synthetic) per row.
        show_overlay: Whether to show lesion overlay.
        overlay_alpha: Alpha value for lesion overlay.
        seed: Random seed for reproducibility.
        figsize_per_patch: Figure size multiplier per patch.
        dpi: Figure DPI.
        zbins_filter: Optional list of specific z-bins to visualize. If None, show all common bins.
    """
    rng = np.random.default_rng(seed)

    # Load patches
    real_data, synthetic_data = load_patches(patches_dir)

    # Get unique z-bins present in both datasets
    real_zbins = set(real_data["z_bins"])
    synthetic_zbins = set(synthetic_data["z_bins"])
    common_zbins = sorted(real_zbins & synthetic_zbins)

    if not common_zbins:
        raise ValueError("No common z-bins between real and synthetic patches")

    # Filter to requested z-bins if specified
    if zbins_filter is not None:
        requested_zbins = set(zbins_filter)
        available_zbins = set(common_zbins)
        missing_zbins = requested_zbins - available_zbins
        if missing_zbins:
            logger.warning(f"Requested z-bins not available: {sorted(missing_zbins)}")
        common_zbins = [z for z in zbins_filter if z in available_zbins]
        if not common_zbins:
            raise ValueError(f"None of the requested z-bins {zbins_filter} are available")

    logger.info(f"Visualizing {len(common_zbins)} z-bins: {common_zbins}")

    # Calculate figure dimensions
    n_rows = len(common_zbins)
    n_cols = 2 * n_per_group + 1  # Real patches + separator + Synthetic patches
    fig_width = n_cols * figsize_per_patch
    fig_height = n_rows * figsize_per_patch

    # Create figure
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    # Add title
    title = "Audition Patches: Real vs Synthetic"
    if show_overlay:
        title += " (with lesion overlay)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # Process each z-bin
    for row_idx, z_bin in enumerate(common_zbins):
        # Get samples for this z-bin
        real_indices = get_zbin_samples(real_data, z_bin, n_per_group, rng)
        synthetic_indices = get_zbin_samples(synthetic_data, z_bin, n_per_group, rng)

        # Plot Real patches (left side)
        for col_idx in range(n_per_group):
            ax = axes[row_idx, col_idx]

            if col_idx < len(real_indices):
                idx = real_indices[col_idx]
                image = to_display_range(real_data["images"][idx])
                mask = real_data["masks"][idx]

                if show_overlay:
                    display = create_overlay(image, mask, alpha=overlay_alpha)
                else:
                    display = image

                ax.imshow(display, cmap="gray" if not show_overlay else None)
            else:
                ax.set_facecolor("lightgray")

            ax.axis("off")

            # Add column headers on first row
            if row_idx == 0 and col_idx == n_per_group // 2:
                ax.set_title("Real", fontsize=12, fontweight="bold", color="blue")

        # Separator column
        sep_idx = n_per_group
        axes[row_idx, sep_idx].axis("off")
        axes[row_idx, sep_idx].set_facecolor("white")

        # Add z-bin label in separator
        axes[row_idx, sep_idx].text(
            0.5,
            0.5,
            f"z={z_bin}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=axes[row_idx, sep_idx].transAxes,
        )

        # Plot Synthetic patches (right side)
        for col_offset in range(n_per_group):
            col_idx = sep_idx + 1 + col_offset
            ax = axes[row_idx, col_idx]

            if col_offset < len(synthetic_indices):
                idx = synthetic_indices[col_offset]
                image = to_display_range(synthetic_data["images"][idx])
                mask = synthetic_data["masks"][idx]

                if show_overlay:
                    display = create_overlay(image, mask, alpha=overlay_alpha)
                else:
                    display = image

                ax.imshow(display, cmap="gray" if not show_overlay else None)
            else:
                ax.set_facecolor("lightgray")

            ax.axis("off")

            # Add column headers on first row
            if row_idx == 0 and col_offset == n_per_group // 2:
                ax.set_title("Synthetic", fontsize=12, fontweight="bold", color="green")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved visualization to {output_path}")


def main() -> None:
    """Main entry point for patch visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize extracted audition patches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.diffusion.audition.scripts.visualize_patches \\
        --config src/diffusion/audition/config/audition.yaml

    python -m src.diffusion.audition.scripts.visualize_patches \\
        --config src/diffusion/audition/config/audition.yaml \\
        --n-per-group 3 \\
        --no-overlay

    python -m src.diffusion.audition.scripts.visualize_patches \\
        --patches-dir outputs/audition/patches \\
        --output patches_grid.png \\
        --overlay-alpha 0.2
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to audition configuration YAML file",
    )
    parser.add_argument(
        "--patches-dir",
        type=str,
        default=None,
        help="Override patches directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path",
    )
    parser.add_argument(
        "--n-per-group",
        type=int,
        default=5,
        help="Number of patches per group (Real/Synthetic) per row",
    )
    parser.add_argument(
        "--zbins",
        type=str,
        default=None,
        help="Comma-separated list of z-bins to visualize (e.g., '5,10,15,20,25'). If not specified, all common bins are shown.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        default=True,
        help="Show lesion overlay (default: on)",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable lesion overlay",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.3,
        help="Overlay transparency (0=invisible, 1=opaque)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine patches directory
    if args.patches_dir:
        patches_dir = Path(args.patches_dir)
    elif args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg = OmegaConf.load(config_path)
        patches_dir = Path(cfg.output.patches_dir)
    else:
        raise ValueError("Either --config or --patches-dir must be specified")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif args.config:
        cfg = OmegaConf.load(args.config)
        figures_dir = Path(cfg.output.figures_dir)
        output_path = figures_dir / "audition_patches_grid.png"
    else:
        output_path = patches_dir / "audition_patches_grid.png"

    # Handle overlay flag
    show_overlay = args.overlay and not args.no_overlay

    # Parse zbins filter
    zbins_filter = None
    if args.zbins:
        try:
            zbins_filter = [int(z.strip()) for z in args.zbins.split(",")]
        except ValueError as e:
            raise ValueError(f"Invalid --zbins format. Use comma-separated integers: {e}")

    logger.info(f"Patches directory: {patches_dir}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Patches per group: {args.n_per_group}")
    logger.info(f"Show overlay: {show_overlay}")
    if zbins_filter:
        logger.info(f"Z-bins filter: {zbins_filter}")

    # Generate visualization
    visualize_patches(
        patches_dir=patches_dir,
        output_path=output_path,
        n_per_group=args.n_per_group,
        show_overlay=show_overlay,
        overlay_alpha=args.overlay_alpha,
        seed=args.seed,
        dpi=args.dpi,
        zbins_filter=zbins_filter,
    )

    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
