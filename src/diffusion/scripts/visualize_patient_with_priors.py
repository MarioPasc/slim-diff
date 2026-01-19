#!/usr/bin/env python
"""
Visualize a patient from the cache with z-bin priors overlayed.

This script is automatically called after cache building to create a visualization
showing how z-bin priors align with actual patient data.
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


def _get_scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    """Extract scalar value from NPZ data."""
    if key in data:
        val = data[key]
        return val.item() if hasattr(val, 'item') and val.ndim == 0 else val
    return default


def _load_sample_data(sample_file: Path) -> dict | None:
    """Load and parse a sample file, returning extracted data or None on failure."""
    try:
        data = np.load(sample_file, allow_pickle=True)

        # Handle both old format (sample) and new format (image/mask)
        if "sample" in data:
            sample = data["sample"]
            image_slice = sample[0]
            mask_slice = sample[1]
        else:
            image_slice = data["image"]
            mask_slice = data["mask"]

        return {
            "image": image_slice,
            "mask": mask_slice,
            "z_bin": _get_scalar(data, "z_bin", 0),
            "z_idx": _get_scalar(data, "z_idx", 0),
            "subject_id": _get_scalar(data, "subject_id", "unknown"),
        }
    except Exception as e:
        logger.warning(f"Failed to load sample {sample_file}: {e}")
        return None


def visualize_all_zbins(
    cache_dir: Path,
    output_dir: Path | None = None,
    alpha: float = 0.3,
) -> bool:
    """Create one visualization per z-bin in the cache.

    Args:
        cache_dir: Path to cache directory
        output_dir: Output directory for visualizations (defaults to cache_dir/viz_zbin_priors/)
        alpha: Alpha value for prior overlay

    Returns:
        True if at least one visualization was created, False otherwise
    """
    cache_dir = Path(cache_dir)
    slice_dir = cache_dir / "slices"
    priors_file = cache_dir / "zbin_priors_brain_roi.npz"

    # Check if priors exist
    if not priors_file.exists():
        logger.warning(f"Z-bin priors not found: {priors_file}")
        logger.warning("Skipping visualization (priors disabled in config)")
        return False

    # Load priors
    try:
        priors_data = np.load(priors_file, allow_pickle=True)
    except Exception as e:
        logger.error(f"Failed to load priors file: {e}")
        return False

    # Get metadata to determine number of bins
    if "metadata" not in priors_data:
        logger.error("Invalid priors file: missing metadata")
        return False

    metadata = priors_data["metadata"].item()
    z_bins = metadata.get("z_bins", 30)

    # Reconstruct priors dict from individual bin keys
    priors = {}
    for i in range(z_bins):
        key = f"bin_{i}"
        if key in priors_data:
            priors[i] = priors_data[key].astype(np.bool_)

    logger.info(f"Loaded {len(priors)} z-bin priors")

    # Find all samples and group by z-bin
    all_samples = sorted(slice_dir.glob("*.npz"))
    if not all_samples:
        logger.error("No samples found in cache")
        return False

    # Group samples by z-bin, preferring lesion samples
    samples_by_zbin: dict[int, list[Path]] = defaultdict(list)
    lesion_samples_by_zbin: dict[int, list[Path]] = defaultdict(list)

    for sample_file in all_samples:
        sample_data = _load_sample_data(sample_file)
        if sample_data is None:
            continue

        z_bin = sample_data["z_bin"]
        samples_by_zbin[z_bin].append(sample_file)

        # Check if it's a lesion sample (has non-zero mask)
        if sample_data["mask"].any():
            lesion_samples_by_zbin[z_bin].append(sample_file)

    logger.info(f"Found samples in {len(samples_by_zbin)} z-bins")

    # Setup output directory
    if output_dir is None:
        output_dir = cache_dir / "viz_zbin_priors"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization for each z-bin
    created_count = 0
    for z_bin in sorted(priors.keys()):
        # Prefer lesion samples, fall back to any sample
        if z_bin in lesion_samples_by_zbin and lesion_samples_by_zbin[z_bin]:
            candidates = lesion_samples_by_zbin[z_bin]
        elif z_bin in samples_by_zbin and samples_by_zbin[z_bin]:
            candidates = samples_by_zbin[z_bin]
        else:
            logger.warning(f"No samples found for z-bin {z_bin}, skipping")
            continue

        # Pick middle sample for better representation
        sample_file = candidates[len(candidates) // 2]
        sample_data = _load_sample_data(sample_file)

        if sample_data is None:
            continue

        prior_slice = priors[z_bin]

        # Create visualization
        fig = create_four_panel_visualization(
            image_slice=sample_data["image"],
            mask_slice=sample_data["mask"],
            prior_slice=prior_slice,
            z_idx=sample_data["z_idx"],
            z_bin=z_bin,
            subject_id=sample_data["subject_id"],
            alpha=alpha,
        )

        # Save with zero-padded bin number for proper sorting
        output_path = output_dir / f"zbin_{z_bin:02d}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        created_count += 1
        logger.debug(f"Created visualization for z-bin {z_bin}")

    logger.info(f"Created {created_count} z-bin visualizations in: {output_dir}")
    return created_count > 0


def visualize_cache_sample_with_priors(
    cache_dir: Path,
    output_path: Path | None = None,
    alpha: float = 0.3,
) -> bool:
    """Create visualization of cache sample with z-bin priors.

    Args:
        cache_dir: Path to cache directory
        output_path: Optional output path (defaults to cache_dir/viz_zbin_priors.png)
        alpha: Alpha value for prior overlay

    Returns:
        True if successful, False otherwise
    """
    cache_dir = Path(cache_dir)
    slice_dir = cache_dir / "slices"
    priors_file = cache_dir / "zbin_priors_brain_roi.npz"

    # Check if priors exist
    if not priors_file.exists():
        logger.warning(f"Z-bin priors not found: {priors_file}")
        logger.warning("Skipping visualization (priors disabled in config)")
        return False

    # Find a lesion sample
    lesion_samples = sorted(slice_dir.glob("*_lesion_*.npz"))
    if not lesion_samples:
        logger.warning("No lesion samples found, using any sample")
        lesion_samples = sorted(slice_dir.glob("*.npz"))

    if not lesion_samples:
        logger.error("No samples found in cache")
        return False

    # Load sample (pick middle one for better chance of good visualization)
    sample_file = lesion_samples[len(lesion_samples) // 2]
    logger.info(f"Loading sample: {sample_file.name}")

    sample_data = _load_sample_data(sample_file)
    if sample_data is None:
        logger.error("Failed to load sample data")
        return False

    try:
        # Load priors
        priors_data = np.load(priors_file, allow_pickle=True)

        # Get metadata to determine number of bins
        if "metadata" not in priors_data:
            logger.error("Invalid priors file: missing metadata")
            return False

        metadata = priors_data["metadata"].item()
        z_bins = metadata.get("z_bins", 30)

        # Reconstruct priors dict from individual bin keys
        priors = {}
        for i in range(z_bins):
            key = f"bin_{i}"
            if key in priors_data:
                priors[i] = priors_data[key].astype(np.bool_)

        z_bin = sample_data["z_bin"]
        if z_bin not in priors:
            logger.error(f"Z-bin {z_bin} not found in priors")
            return False

        prior_slice = priors[z_bin]

        # Create visualization
        fig = create_four_panel_visualization(
            image_slice=sample_data["image"],
            mask_slice=sample_data["mask"],
            prior_slice=prior_slice,
            z_idx=sample_data["z_idx"],
            z_bin=z_bin,
            subject_id=sample_data["subject_id"],
            alpha=alpha,
        )

        # Save
        if output_path is None:
            output_path = cache_dir / "viz_zbin_priors.png"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved z-bin prior visualization to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        return False


def create_four_panel_visualization(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    prior_slice: np.ndarray,
    z_idx: int,
    z_bin: int,
    subject_id: str,
    alpha: float = 0.3,
) -> plt.Figure:
    """Create 4-panel visualization: [image, mask, prior, combined].

    Args:
        image_slice: 2D image slice (H, W)
        mask_slice: 2D mask slice (H, W)
        prior_slice: 2D prior slice (H, W)
        z_idx: Z-index of the slice
        z_bin: Z-bin index
        subject_id: Subject identifier
        alpha: Alpha value for overlays

    Returns:
        matplotlib Figure
    """
    # Normalize image to [0, 1]
    image_norm = (image_slice - image_slice.min()) / (
        image_slice.max() - image_slice.min() + 1e-8
    )

    fig = plt.figure(figsize=(16, 4.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.05)

    # Color schemes
    prior_color = (0, 255, 255)  # Cyan
    lesion_color = (255, 0, 0)  # Red

    # Panel 1: Raw image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_norm, cmap="gray", vmin=0, vmax=1)
    ax1.set_title(f"Image\n{subject_id}", fontsize=11, fontweight="bold")
    ax1.text(
        0.05,
        0.95,
        f"z={z_idx}\nbin={z_bin}",
        transform=ax1.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax1.axis("off")

    # Panel 2: Mask overlay
    ax2 = fig.add_subplot(gs[0, 1])
    rgb_mask = _create_overlay(image_norm, mask_slice > 0, alpha=0.5, color=lesion_color)
    ax2.imshow(rgb_mask)
    ax2.set_title("Lesion Mask", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # Panel 3: Prior overlay
    ax3 = fig.add_subplot(gs[0, 2])
    rgb_prior = _create_overlay(image_norm, prior_slice > 0, alpha=alpha, color=prior_color)
    ax3.imshow(rgb_prior)
    ax3.set_title("Z-Bin Prior", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # Panel 4: Combined (prior + lesion)
    ax4 = fig.add_subplot(gs[0, 3])
    rgb_combined = _create_overlay(
        image_norm, prior_slice > 0, alpha=alpha, color=prior_color
    )
    # Add lesion overlay on top
    if (mask_slice > 0).any():
        rgb_combined = _blend_overlay(
            rgb_combined, mask_slice > 0, alpha=0.4, color=lesion_color
        )
    ax4.imshow(rgb_combined)
    ax4.set_title("Combined", fontsize=11, fontweight="bold")
    ax4.axis("off")

    # Add legend
    fig.text(
        0.5,
        0.02,
        f"Prior overlay (cyan) shows anatomical ROI for z-bin {z_bin} | "
        f"Lesion overlay (red) shows ground truth | Alpha: {alpha}",
        ha="center",
        fontsize=10,
        style="italic",
    )

    plt.suptitle(
        "Z-Bin Prior Visualization: Anatomical Conditioning Alignment",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    return fig


def _create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Create RGB overlay of mask on grayscale image."""
    # Convert grayscale to RGB
    rgb = np.stack([image, image, image], axis=-1)

    # Create colored overlay
    color_norm = np.array(color) / 255.0
    overlay = np.zeros_like(rgb)
    overlay[mask] = color_norm

    # Blend
    result = rgb * (1 - alpha * mask[..., None]) + overlay * alpha

    return np.clip(result, 0, 1)


def _blend_overlay(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Blend an additional overlay onto an existing RGB image."""
    # Create colored overlay
    color_norm = np.array(color) / 255.0
    overlay = np.zeros_like(base_rgb)
    overlay[mask] = color_norm

    # Blend
    result = base_rgb * (1 - alpha * mask[..., None]) + overlay * alpha

    return np.clip(result, 0, 1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize cache sample with z-bin priors overlay"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to cache directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for single visualization (default: <cache_dir>/viz_zbin_priors.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for all z-bin visualizations (default: <cache_dir>/viz_zbin_priors/)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Alpha value for prior overlay (default: 0.3)",
    )
    parser.add_argument(
        "--all-zbins",
        action="store_true",
        help="Create one visualization per z-bin (saved to output-dir)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    # Create visualization(s)
    if args.all_zbins:
        success = visualize_all_zbins(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            alpha=args.alpha,
        )
    else:
        success = visualize_cache_sample_with_priors(
            cache_dir=args.cache_dir,
            output_path=args.output,
            alpha=args.alpha,
        )

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
