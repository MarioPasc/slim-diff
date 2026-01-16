#!/usr/bin/env python3
"""Analyze impact of lesion area filtering thresholds (dry-run mode).

Quickly computes sample counts at different thresholds without running full k-fold planning.
This helps choose an appropriate value for the `lesions_over_pixel_area` config option.

Usage:
    python -m src.segmentation.scripts.dryrun_lesion_area_filter \
        --real-cache-dir /media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache \
        --synthetic-dir /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/replicas \
        --replicas replica_000.npz replica_001.npz replica_002.npz replica_003.npz replica_004.npz replica_005.npz \
        --thresholds 0 50 60 75 80 90 100 \
        --output-dir outputs/lesion_area_analysis

Output:
    - Console table with sample counts per (source, split, threshold)
    - threshold_analysis_summary.csv
    - lesion_area_distribution.png (histogram of lesion areas)
    - threshold_examples_grid.png (example images per threshold range)
    - zbin_threshold_counts.png (per-zbin counts over thresholds)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Cache for loaded synthetic replicas (to avoid reloading)
_synthetic_replica_cache: dict[str, dict] = {}


def compute_lesion_areas_real(
    cache_dir: Path,
    csv_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute lesion areas for all real samples.

    Args:
        cache_dir: Path to slice_cache directory containing CSVs and NPZ files.
        csv_names: List of CSV filenames to process (default: train, val, test).

    Returns:
        List of dicts with: source, split, subject_id, z_bin, has_lesion, lesion_area,
        filepath, cache_dir (for loading images later)
    """
    if csv_names is None:
        csv_names = ["train.csv", "val.csv", "test.csv"]

    results = []

    for csv_name in csv_names:
        csv_path = cache_dir / csv_name
        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}")
            continue

        split = csv_name.replace(".csv", "")

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        logger.info(f"Processing {len(rows)} samples from {csv_name}...")

        for row in rows:
            npz_path = cache_dir / row["filepath"]
            has_lesion = row["has_lesion"].lower() == "true"

            lesion_area = 0
            if has_lesion and npz_path.exists():
                try:
                    with np.load(npz_path) as data:
                        mask = data["mask"]
                        lesion_area = int((mask > 0.0).sum())
                except Exception as e:
                    logger.warning(f"Error loading {npz_path}: {e}")

            results.append({
                "source": "real",
                "split": split,
                "subject_id": row["subject_id"],
                "z_bin": int(row["z_bin"]),
                "has_lesion": has_lesion,
                "lesion_area": lesion_area,
                "filepath": row["filepath"],
                "cache_dir": str(cache_dir),
            })

    return results


def compute_lesion_areas_synthetic(
    synthetic_dir: Path,
    replicas: list[str],
) -> list[dict[str, Any]]:
    """Compute lesion areas for synthetic samples.

    Args:
        synthetic_dir: Path to directory containing replica NPZ files.
        replicas: List of replica NPZ filenames to process.

    Returns:
        List of dicts with: source, split, replica, index, z_bin, has_lesion, lesion_area,
        synthetic_dir (for loading images later)
    """
    results = []

    for replica_name in replicas:
        replica_path = synthetic_dir / replica_name
        if not replica_path.exists():
            logger.warning(f"Replica not found: {replica_path}")
            continue

        logger.info(f"Processing replica {replica_name}...")

        # Cache the loaded replica for later image loading
        data = np.load(replica_path, allow_pickle=True)
        _synthetic_replica_cache[str(replica_path)] = {
            "images": data["images"],
            "masks": data["masks"],
        }
        masks = data["masks"]
        zbins = data["zbin"]

        for idx in range(len(masks)):
            mask = masks[idx]
            lesion_area = int((mask > 0.0).sum())
            has_lesion = lesion_area > 10  # Same threshold as KFoldPlanner

            results.append({
                "source": "synthetic",
                "split": "train",  # Synthetic typically used for training
                "replica": replica_name,
                "index": idx,
                "z_bin": int(zbins[idx]),
                "has_lesion": has_lesion,
                "lesion_area": lesion_area,
                "synthetic_dir": str(synthetic_dir),
            })

    return results


def load_sample_image_mask(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """Load image and mask for a sample.

    Args:
        sample: Sample dict with source-specific path info.

    Returns:
        Tuple of (image, mask) arrays, or None if loading fails.
    """
    try:
        if sample["source"] == "real":
            cache_dir = Path(sample["cache_dir"])
            npz_path = cache_dir / sample["filepath"]
            with np.load(npz_path) as data:
                return data["image"], data["mask"]
        else:
            # Synthetic sample
            synthetic_dir = Path(sample["synthetic_dir"])
            replica_path = synthetic_dir / sample["replica"]
            idx = sample["index"]

            # Check cache first
            cache_key = str(replica_path)
            if cache_key in _synthetic_replica_cache:
                cached = _synthetic_replica_cache[cache_key]
                return cached["images"][idx], cached["masks"][idx]

            # Load from disk if not cached
            with np.load(replica_path, allow_pickle=True) as data:
                return data["images"][idx], data["masks"][idx]
    except Exception as e:
        logger.warning(f"Error loading sample: {e}")
        return None


def analyze_thresholds(
    samples: list[dict[str, Any]],
    thresholds: list[int],
) -> list[dict[str, Any]]:
    """Analyze sample counts at different thresholds.

    Args:
        samples: List of sample dicts with lesion_area field.
        thresholds: List of pixel area thresholds to analyze.

    Returns:
        List of summary rows with: source, split, threshold, total, n_lesion, n_over_threshold
    """
    rows = []

    # Group by (source, split)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        key = (sample["source"], sample["split"])
        groups[key].append(sample)

    for (source, split), group_samples in sorted(groups.items()):
        for threshold in thresholds:
            n_total = len(group_samples)
            n_lesion = sum(1 for s in group_samples if s["has_lesion"])
            n_over_threshold = sum(
                1 for s in group_samples
                if s["has_lesion"] and s["lesion_area"] >= threshold
            )

            rows.append({
                "source": source,
                "split": split,
                "threshold": threshold,
                "total_samples": n_total,
                "n_lesion": n_lesion,
                "n_over_threshold": n_over_threshold,
                "pct_retained": f"{n_over_threshold / n_lesion * 100:.1f}%" if n_lesion > 0 else "N/A",
            })

    return rows


def analyze_by_zbin(
    samples: list[dict[str, Any]],
    thresholds: list[int],
) -> list[dict[str, Any]]:
    """Analyze sample counts by z-bin at different thresholds.

    Args:
        samples: List of sample dicts with lesion_area and z_bin fields.
        thresholds: List of pixel area thresholds to analyze.

    Returns:
        List of rows with: source, split, z_bin, threshold, n_lesion, n_over_threshold
    """
    rows = []

    # Group by (source, split, z_bin)
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for sample in samples:
        key = (sample["source"], sample["split"], sample["z_bin"])
        groups[key].append(sample)

    for (source, split, z_bin), group_samples in sorted(groups.items()):
        for threshold in thresholds:
            n_lesion = sum(1 for s in group_samples if s["has_lesion"])
            n_over_threshold = sum(
                1 for s in group_samples
                if s["has_lesion"] and s["lesion_area"] >= threshold
            )

            rows.append({
                "source": source,
                "split": split,
                "z_bin": z_bin,
                "threshold": threshold,
                "n_lesion": n_lesion,
                "n_over_threshold": n_over_threshold,
            })

    return rows


def print_summary_table(summary: list[dict[str, Any]], thresholds: list[int]) -> None:
    """Print a formatted summary table to console."""
    print("\n" + "=" * 90)
    print("LESION AREA THRESHOLD ANALYSIS")
    print("=" * 90)

    # Header
    header = f"{'Source':<12} {'Split':<8}"
    for thresh in thresholds:
        header += f" {f'>={thresh}px':>12}"
    print(header)
    print("-" * 90)

    # Group by (source, split)
    current_source = None
    for row in summary:
        if row["threshold"] != thresholds[0]:
            continue  # Only print first threshold row per group

        source = row["source"]
        split = row["split"]

        if source != current_source:
            if current_source is not None:
                print("-" * 90)
            current_source = source

        # Find all threshold values for this (source, split)
        line = f"{source:<12} {split:<8}"
        for thresh in thresholds:
            matching = [r for r in summary
                       if r["source"] == source and r["split"] == split and r["threshold"] == thresh]
            if matching:
                r = matching[0]
                line += f" {r['n_over_threshold']:>6}/{r['n_lesion']:<5}"
            else:
                line += f" {'N/A':>12}"
        print(line)

    print("=" * 90)
    print("Format: n_over_threshold/n_total_lesion")
    print()


def plot_lesion_area_distribution(
    samples: list[dict[str, Any]],
    output_path: Path,
    thresholds: list[int],
) -> None:
    """Plot histogram of lesion areas with threshold lines.

    Args:
        samples: List of sample dicts with lesion_area field.
        output_path: Path to save the plot.
        thresholds: List of thresholds to mark on the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
        return

    lesion_areas_real = [
        s["lesion_area"] for s in samples
        if s["source"] == "real" and s["has_lesion"] and s["lesion_area"] > 0
    ]
    lesion_areas_synth = [
        s["lesion_area"] for s in samples
        if s["source"] == "synthetic" and s["has_lesion"] and s["lesion_area"] > 0
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Real data
    ax = axes[0]
    if lesion_areas_real:
        ax.hist(lesion_areas_real, bins=50, alpha=0.7, color="#2ecc71", edgecolor="black")
        for thresh in thresholds[1:]:  # Skip 0
            ax.axvline(thresh, color="red", linestyle="--", alpha=0.7, label=f">={thresh}px")
        ax.set_xlabel("Lesion Area (pixels)")
        ax.set_ylabel("Count")
        ax.set_title(f"Real Data - Lesion Area Distribution (n={len(lesion_areas_real)})")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "No lesion samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Real Data - No lesion samples")

    # Synthetic data
    ax = axes[1]
    if lesion_areas_synth:
        ax.hist(lesion_areas_synth, bins=50, alpha=0.7, color="#9b59b6", edgecolor="black")
        for thresh in thresholds[1:]:
            ax.axvline(thresh, color="red", linestyle="--", alpha=0.7, label=f">={thresh}px")
        ax.set_xlabel("Lesion Area (pixels)")
        ax.set_ylabel("Count")
        ax.set_title(f"Synthetic Data - Lesion Area Distribution (n={len(lesion_areas_synth)})")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "No lesion samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Synthetic Data - No lesion samples")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved lesion area distribution plot to {output_path}")


def plot_threshold_examples_grid(
    samples: list[dict[str, Any]],
    output_path: Path,
    thresholds: list[int],
) -> None:
    """Plot a grid of example images per threshold range and source.

    Creates a grid where:
    - Y-axis (rows): threshold ranges (e.g., 0-50, 50-100, 100-200, ...)
    - X-axis (columns): real vs synthetic
    - Each cell: one example slice with mask contour overlayed on image

    Args:
        samples: List of sample dicts with lesion_area and path info.
        output_path: Path to save the plot.
        thresholds: List of thresholds defining the ranges.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        logger.warning("matplotlib not available, skipping threshold examples grid")
        return

    # Create threshold ranges (pairs of lower, upper bounds)
    threshold_ranges = []
    for i in range(len(thresholds)):
        lower = thresholds[i]
        upper = thresholds[i + 1] if i + 1 < len(thresholds) else float("inf")
        threshold_ranges.append((lower, upper))

    sources = ["real", "synthetic"]
    n_rows = len(threshold_ranges)
    n_cols = len(sources)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    # Ensure axes is 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, (lower, upper) in enumerate(threshold_ranges):
        for col_idx, source in enumerate(sources):
            ax = axes[row_idx, col_idx]

            # Find samples in this threshold range and source
            matching_samples = [
                s for s in samples
                if s["source"] == source
                and s["has_lesion"]
                and s["lesion_area"] >= lower
                and (upper == float("inf") or s["lesion_area"] < upper)
            ]

            if not matching_samples:
                ax.text(
                    0.5, 0.5, "No samples",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray"
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Pick a sample (middle of the range by area)
                matching_samples.sort(key=lambda s: s["lesion_area"])
                sample = matching_samples[len(matching_samples) // 2]

                # Load image and mask
                result = load_sample_image_mask(sample)
                if result is None:
                    ax.text(
                        0.5, 0.5, "Load error",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=10, color="red"
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    image, mask = result

                    # Normalize image for display
                    img_display = (image - image.min()) / (image.max() - image.min() + 1e-8)

                    # Show image
                    ax.imshow(img_display, cmap="gray", vmin=0, vmax=1)

                    # Overlay mask contour
                    mask_binary = (mask > 0.0).astype(float)
                    ax.contour(mask_binary, levels=[0.5], colors=["red"], linewidths=1.5)

                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Add area annotation
                    ax.text(
                        0.02, 0.98, f"area={sample['lesion_area']}px",
                        ha="left", va="top", transform=ax.transAxes,
                        fontsize=8, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7)
                    )

            # Row labels (threshold range)
            if col_idx == 0:
                upper_str = f"{int(upper)}" if upper != float("inf") else "∞"
                ax.set_ylabel(f"{lower}-{upper_str} px", fontsize=10)

            # Column labels (source)
            if row_idx == 0:
                ax.set_title(source.capitalize(), fontsize=12, fontweight="bold")

    plt.suptitle("Example Lesions by Threshold Range", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved threshold examples grid to {output_path}")


def plot_zbin_threshold_counts(
    samples: list[dict[str, Any]],
    output_path: Path,
    thresholds: list[int],
) -> None:
    """Plot per-zbin count of lesion images over thresholds, stratified by source.

    Creates a multi-panel figure showing how many lesion samples remain at each
    threshold for each z-bin, with separate lines/bars for real vs synthetic.

    Args:
        samples: List of sample dicts with lesion_area, z_bin, and source fields.
        output_path: Path to save the plot.
        thresholds: List of thresholds to analyze.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping z-bin threshold counts plot")
        return

    # Get all z-bins
    all_zbins = sorted(set(s["z_bin"] for s in samples if s["has_lesion"]))

    if not all_zbins:
        logger.warning("No z-bins with lesions found, skipping z-bin plot")
        return

    # Compute counts per (source, z_bin, threshold)
    counts: dict[tuple[str, int, int], int] = defaultdict(int)
    for sample in samples:
        if not sample["has_lesion"]:
            continue
        source = sample["source"]
        z_bin = sample["z_bin"]
        area = sample["lesion_area"]
        for thresh in thresholds:
            if area >= thresh:
                counts[(source, z_bin, thresh)] += 1

    # Create figure with one subplot per threshold
    n_thresholds = len(thresholds)
    n_cols = min(3, n_thresholds)
    n_rows = (n_thresholds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    bar_width = 0.35
    x = np.arange(len(all_zbins))

    for idx, thresh in enumerate(thresholds):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get counts for each source
        real_counts = [counts[("real", zb, thresh)] for zb in all_zbins]
        synth_counts = [counts[("synthetic", zb, thresh)] for zb in all_zbins]

        # Plot bars
        bars1 = ax.bar(x - bar_width / 2, real_counts, bar_width, label="Real", color="#2ecc71", alpha=0.8)
        bars2 = ax.bar(x + bar_width / 2, synth_counts, bar_width, label="Synthetic", color="#9b59b6", alpha=0.8)

        ax.set_xlabel("Z-bin")
        ax.set_ylabel("Count")
        ax.set_title(f"Threshold ≥ {thresh} px")
        ax.set_xticks(x)
        ax.set_xticklabels(all_zbins, rotation=45, ha="right", fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Hide empty subplots
    for idx in range(n_thresholds, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.suptitle("Lesion Counts by Z-bin at Different Thresholds", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved z-bin threshold counts plot to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze impact of lesion area filtering thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze real data only
    python -m src.segmentation.scripts.dryrun_lesion_area_filter \\
        --real-cache-dir /path/to/slice_cache \\
        --output-dir outputs/lesion_analysis

    # Analyze real + synthetic data
    python -m src.segmentation.scripts.dryrun_lesion_area_filter \\
        --real-cache-dir /path/to/slice_cache \\
        --synthetic-dir /path/to/replicas \\
        --replicas replica_000.npz replica_001.npz \\
        --thresholds 0 50 100 200 500 1000 \\
        --output-dir outputs/lesion_analysis
""",
    )

    parser.add_argument(
        "--real-cache-dir",
        type=str,
        required=True,
        help="Path to slice_cache directory containing real data CSVs and NPZs",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default=None,
        help="Path to directory containing synthetic replica NPZ files (optional)",
    )
    parser.add_argument(
        "--replicas",
        type=str,
        nargs="*",
        default=[],
        help="List of replica NPZ filenames to include (e.g., replica_000.npz)",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[0, 50, 100, 200, 500, 1000],
        help="Lesion area thresholds to analyze (default: 0 50 100 200 500 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all samples
    logger.info("Computing lesion areas for real data...")
    all_samples = compute_lesion_areas_real(Path(args.real_cache_dir))

    if args.synthetic_dir and args.replicas:
        logger.info("Computing lesion areas for synthetic data...")
        synth_samples = compute_lesion_areas_synthetic(
            Path(args.synthetic_dir), args.replicas
        )
        all_samples.extend(synth_samples)

    logger.info(f"Total samples: {len(all_samples)}")

    # Analyze thresholds
    logger.info("Analyzing thresholds...")
    summary = analyze_thresholds(all_samples, args.thresholds)

    # Print summary table
    print_summary_table(summary, args.thresholds)

    # Save summary CSV
    summary_csv_path = output_dir / "threshold_analysis_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    logger.info(f"Saved summary to {summary_csv_path}")

    # Analyze by z-bin
    zbin_analysis = analyze_by_zbin(all_samples, args.thresholds)
    zbin_csv_path = output_dir / "threshold_analysis_by_zbin.csv"
    with open(zbin_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=zbin_analysis[0].keys())
        writer.writeheader()
        writer.writerows(zbin_analysis)
    logger.info(f"Saved z-bin analysis to {zbin_csv_path}")

    # Save JSON summary
    summary_json = {
        "thresholds_analyzed": args.thresholds,
        "total_samples": len(all_samples),
        "real_samples": len([s for s in all_samples if s["source"] == "real"]),
        "synthetic_samples": len([s for s in all_samples if s["source"] == "synthetic"]),
        "summary": summary,
    }
    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    logger.info(f"Saved JSON summary to {json_path}")

    # Plot distribution
    plot_path = output_dir / "lesion_area_distribution.png"
    plot_lesion_area_distribution(all_samples, plot_path, args.thresholds)

    # Plot threshold examples grid (image + mask contour per threshold range)
    examples_path = output_dir / "threshold_examples_grid.png"
    plot_threshold_examples_grid(all_samples, examples_path, args.thresholds)

    # Plot per-zbin counts over thresholds
    zbin_plot_path = output_dir / "zbin_threshold_counts.png"
    plot_zbin_threshold_counts(all_samples, zbin_plot_path, args.thresholds)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
