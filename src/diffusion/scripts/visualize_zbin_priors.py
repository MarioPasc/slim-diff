"""Visualize z-bin anatomical ROI priors.

Creates comprehensive visualizations showing:
1. Overview grid of all z-bin priors
2. Per-bin patient slices with prior overlays
3. Statistical analysis of priors
4. Comparison with cache distribution

Uses configuration from src/diffusion/config/jsddpm.yaml.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from src.diffusion.data.transforms import get_volume_transforms
from src.diffusion.model.embeddings.zpos import quantize_z
from src.diffusion.utils.io import get_image_path, get_label_path
from src.diffusion.utils.logging import setup_logger
from src.diffusion.utils.zbin_priors import load_zbin_priors


def visualize_priors_overview(
    priors: dict[int, np.ndarray],
    z_bins: int,
    z_range: tuple[int, int],
    output_dir: Path,
) -> None:
    """Create overview visualization of all z-bin priors.

    Args:
        priors: Dict mapping z-bin to prior ROI.
        z_bins: Total number of bins.
        z_range: (min_z, max_z) range.
        output_dir: Output directory for visualization.
    """
    # Create grid layout
    n_cols = 10
    n_rows = (z_bins + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2, n_rows * 2),
        squeeze=False
    )

    for z_bin in range(z_bins):
        row = z_bin // n_cols
        col = z_bin % n_cols
        ax = axes[row, col]

        if z_bin in priors:
            prior = priors[z_bin]
            ax.imshow(prior, cmap="gray", vmin=0, vmax=1)

            # Compute coverage
            coverage = prior.sum() / prior.size
            ax.set_title(f"Bin {z_bin}\n{coverage:.1%}", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center")
            ax.set_title(f"Bin {z_bin}", fontsize=8)

        ax.axis("off")

    # Hide unused subplots
    for z_bin in range(z_bins, n_rows * n_cols):
        row = z_bin // n_cols
        col = z_bin % n_cols
        axes[row, col].axis("off")

    fig.suptitle(
        f"Z-Bin Brain Mask Priors (z_range={z_range}, {z_bins} bins)",
        fontsize=14,
        fontweight="bold"
    )
    fig.tight_layout()

    # Save
    output_path = output_dir / "zbin_priors_overview.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved priors overview to: {output_path}")


def visualize_priors_with_patient(
    priors: dict[int, np.ndarray],
    config,
    output_dir: Path,
    subject_id: str = "MRIe_087",
) -> None:
    """Visualize priors overlaid on actual patient images.

    Args:
        priors: Dict mapping z-bin to prior ROI.
        config: Configuration object.
        output_dir: Output directory for visualization.
        subject_id: Patient ID to visualize (default: MRIe_087).
    """
    z_bins = config.conditioning.z_bins
    z_range = tuple(config.data.slice_sampling.z_range)
    min_z, max_z = z_range

    # Load patient volume
    data_root = Path(config.data.root_dir)
    epilepsy_dataset = data_root / config.data.epilepsy.name

    image_path = get_image_path(
        epilepsy_dataset,
        subject_id,
        modality_index=config.data.epilepsy.modality_index,
        image_dir="imagesTr"
    )
    label_path = get_label_path(
        epilepsy_dataset,
        subject_id,
        label_dir="labelsTr"
    )

    if not image_path.exists():
        print(f"⚠ Patient image not found: {image_path}")
        return

    # Load and transform volume
    has_label = label_path.exists()
    transforms = get_volume_transforms(config, has_label=has_label)

    data_dict = {"image": str(image_path)}
    if has_label:
        data_dict["seg"] = str(label_path)

    try:
        transformed = transforms(data_dict)
        image_vol = transformed["image"]
        mask_vol = transformed["seg"] if has_label else None
    except Exception as e:
        print(f"⚠ Failed to load volume: {e}")
        return

    # Select representative z-bins to visualize (evenly spaced)
    bins_to_show = [0, z_bins // 4, z_bins // 2, 3 * z_bins // 4, z_bins - 1]
    bins_to_show = [b for b in bins_to_show if b in priors]
    # Override, show all available bins 
    bins_to_show = list(priors.keys())

    if not bins_to_show:
        print("⚠ No valid bins to visualize")
        return

    # Create output directory for this patient
    patient_output_dir = output_dir / subject_id
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize each bin
    for z_bin in bins_to_show:
        # Find a representative z_index for this bin
        z_indices_in_bin = []
        for z_idx in range(min_z, max_z + 1):
            if quantize_z(z_idx, z_range, z_bins) == z_bin:
                z_indices_in_bin.append(z_idx)

        if not z_indices_in_bin:
            continue

        # Use middle slice of bin
        z_idx = z_indices_in_bin[len(z_indices_in_bin) // 2]

        # Extract slice
        image_slice = image_vol[0, :, :, z_idx].cpu().numpy()
        if mask_vol is not None:
            mask_slice = mask_vol[0, :, :, z_idx].cpu().numpy()
        else:
            mask_slice = np.zeros_like(image_slice)

        prior = priors[z_bin]

        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 1. Original image
        axes[0].imshow(image_slice, cmap="gray", vmin=-1, vmax=1)
        axes[0].set_title(f"Original Image\nz={z_idx}, bin={z_bin}")
        axes[0].axis("off")

        # 2. Lesion mask
        axes[1].imshow(image_slice, cmap="gray", vmin=-1, vmax=1)
        if mask_slice.max() > 0:
            # Overlay lesion in red
            mask_rgb = np.zeros((*mask_slice.shape, 3))
            mask_rgb[..., 0] = mask_slice > 0
            axes[1].imshow(mask_rgb, alpha=0.5)
        axes[1].set_title("Lesion Mask")
        axes[1].axis("off")

        # 3. Prior ROI
        axes[2].imshow(prior, cmap="Blues", vmin=0, vmax=1)
        coverage = prior.sum() / prior.size
        axes[2].set_title(f"Prior ROI\n{coverage:.1%} coverage")
        axes[2].axis("off")

        # 4. Overlay prior on image
        axes[3].imshow(image_slice, cmap="gray", vmin=-1, vmax=1)
        # Show prior boundary in green
        prior_boundary = np.zeros((*prior.shape, 3))
        prior_boundary[..., 1] = prior  # Green channel
        axes[3].imshow(prior_boundary, alpha=0.3)
        axes[3].set_title("Image + Prior Overlay")
        axes[3].axis("off")

        fig.suptitle(
            f"{subject_id} - Slice {z_idx} (z-bin {z_bin})",
            fontsize=14,
            fontweight="bold"
        )
        fig.tight_layout()

        # Save to bin folder
        bin_dir = patient_output_dir / f"bin_{z_bin:02d}"
        bin_dir.mkdir(parents=True, exist_ok=True)
        output_path = bin_dir / f"slice_{z_idx:03d}_bin_{z_bin:02d}_with_prior.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"✓ Saved: {output_path}")

    print(f"\n✓ Completed visualization for {subject_id}")
    print(f"  Output directory: {patient_output_dir}")


def visualize_prior_statistics(
    priors: dict[int, np.ndarray],
    z_bins: int,
    output_dir: Path,
) -> None:
    """Visualize statistics about the priors.

    Shows:
    - Coverage per z-bin (% of pixels marked as brain)
    - Prior consistency (how much priors vary across bins)

    Args:
        priors: Dict mapping z-bin to prior ROI.
        z_bins: Total number of bins.
        output_dir: Output directory for visualization.
    """
    # Compute statistics
    coverages = []
    for z_bin in range(z_bins):
        if z_bin in priors:
            prior = priors[z_bin]
            coverage = prior.sum() / prior.size
            coverages.append(coverage)
        else:
            coverages.append(0.0)

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 1. Coverage by z-bin
    axes[0].bar(range(z_bins), coverages, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Z-bin")
    axes[0].set_ylabel("Brain Coverage (%)")
    axes[0].set_title("Brain ROI Coverage per Z-bin")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Add mean line
    mean_coverage = np.mean(coverages)
    axes[0].axhline(mean_coverage, color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {mean_coverage:.1%}")
    axes[0].legend()

    # 2. Spatial variation (Dice similarity between consecutive bins)
    variations = []
    for z_bin in range(z_bins - 1):
        if z_bin in priors and z_bin + 1 in priors:
            prior_curr = priors[z_bin]
            prior_next = priors[z_bin + 1]
            # Dice similarity
            intersection = (prior_curr & prior_next).sum()
            union = (prior_curr | prior_next).sum()
            if union > 0:
                dice = 2 * intersection / (prior_curr.sum() + prior_next.sum())
            else:
                dice = 0.0
            variations.append(dice)
        else:
            variations.append(0.0)

    axes[1].plot(range(len(variations)), variations, marker="o",
                color="darkorange", linewidth=2, markersize=4)
    axes[1].set_xlabel("Z-bin")
    axes[1].set_ylabel("Dice Similarity with Next Bin")
    axes[1].set_title("Prior Consistency Across Z-bins")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)

    # Add mean line
    mean_dice = np.mean(variations)
    axes[1].axhline(mean_dice, color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {mean_dice:.2f}")
    axes[1].legend()

    fig.tight_layout()

    # Save
    output_path = output_dir / "zbin_priors_statistics.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved prior statistics to: {output_path}")
    print(f"  Mean coverage: {mean_coverage:.1%}")
    print(f"  Mean consistency (Dice): {mean_dice:.2f}")


def compare_priors_with_cache_distribution(
    priors: dict[int, np.ndarray],
    z_bins: int,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Compare prior coverage with actual slice distribution in cache.

    Args:
        priors: Dict mapping z-bin to prior ROI.
        z_bins: Total number of bins.
        cache_dir: Path to cache directory.
        output_dir: Output directory for visualization.
    """
    # Read cache statistics
    train_csv = cache_dir / "train.csv"
    if not train_csv.exists():
        print(f"⚠ Train CSV not found: {train_csv}")
        return

    # Count slices per bin
    bin_slice_counts = {}
    with open(train_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z_bin = int(row["z_bin"])
            bin_slice_counts[z_bin] = bin_slice_counts.get(z_bin, 0) + 1

    # Compute prior coverages
    prior_coverages = []
    slice_counts = []

    for z_bin in range(z_bins):
        if z_bin in priors:
            prior = priors[z_bin]
            coverage = prior.sum() / prior.size
        else:
            coverage = 0.0
        prior_coverages.append(coverage)

        count = bin_slice_counts.get(z_bin, 0)
        slice_counts.append(count)

    # Normalize slice counts to [0, 1] for comparison
    max_count = max(slice_counts) if slice_counts else 1
    normalized_counts = [c / max_count for c in slice_counts]

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(z_bins)
    width = 0.35

    bars1 = ax.bar(
        x - width/2, prior_coverages, width,
        label="Prior Coverage", color="steelblue", edgecolor="black"
    )
    bars2 = ax.bar(
        x + width/2, normalized_counts, width,
        label="Slice Count (normalized)", color="coral", edgecolor="black"
    )

    ax.set_xlabel("Z-bin", fontsize=12)
    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.set_title(
        "Prior Coverage vs. Actual Slice Distribution",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    # Save
    output_path = output_dir / "zbin_priors_vs_distribution.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved comparison to: {output_path}")

    # Print statistics
    print("\nZ-bin statistics:")
    print(f"  Total bins: {z_bins}")
    print(f"  Bins with data: {len([c for c in slice_counts if c > 0])}")
    print(f"  Mean prior coverage: {np.mean(prior_coverages):.1%}")
    print(f"  Min/Max coverage: {min(prior_coverages):.1%} / {max(prior_coverages):.1%}")


def main() -> None:
    """Main visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize z-bin anatomical ROI priors"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/diffusion/config/jsddpm.yaml",
        help="Path to configuration YAML file (default: src/diffusion/config/jsddpm.yaml)",
    )
    parser.add_argument(
        "--subject_id",
        type=str,
        default="MRIe_087",
        help="Patient ID for per-slice visualization (default: MRIe_087)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/zbin_priors_visualization",
        help="Output directory for visualizations (default: docs/zbin_priors_visualization)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger("jsddpm", level="INFO")

    # Load config
    config = OmegaConf.load(args.config)
    print(f"Loaded config from: {args.config}")

    # Check if priors are enabled
    pp_cfg = config.postprocessing.zbin_priors
    if not pp_cfg.get("enabled", False):
        print("⚠ Z-bin priors not enabled in config. Set postprocessing.zbin_priors.enabled=true")
        return

    # Load priors
    cache_dir = Path(config.data.cache_dir)
    if not cache_dir.exists():
        print(f"⚠ Cache directory not found: {cache_dir}")
        return

    priors_path = cache_dir / pp_cfg.priors_filename
    if not priors_path.exists():
        print(f"⚠ Priors file not found: {priors_path}")
        print("  Run: jsddpm-cache --config <config_path>")
        return

    z_bins = config.conditioning.z_bins
    z_range = tuple(config.data.slice_sampling.z_range)

    print(f"Loading z-bin priors from: {priors_path}")
    priors = load_zbin_priors(cache_dir, pp_cfg.priors_filename, z_bins)
    print(f"Loaded {len(priors)} z-bin priors")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60 + "\n")

    # 1. Overview grid
    print("[1/4] Creating priors overview grid...")
    visualize_priors_overview(priors, z_bins, z_range, output_dir)

    # 2. Per-bin patient slices
    print(f"\n[2/4] Creating patient-specific visualizations for {args.subject_id}...")
    visualize_priors_with_patient(priors, config, output_dir, args.subject_id)

    # 3. Statistics
    print("\n[3/4] Creating statistical analysis...")
    visualize_prior_statistics(priors, z_bins, output_dir)

    # 4. Cache distribution comparison
    print("\n[4/4] Creating cache distribution comparison...")
    compare_priors_with_cache_distribution(priors, z_bins, cache_dir, output_dir)

    print("\n" + "=" * 60)
    print("✓ All visualizations complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
