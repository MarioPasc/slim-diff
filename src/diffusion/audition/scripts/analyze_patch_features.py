#!/usr/bin/env python3
"""Analyze features that differentiate real from synthetic lesion patches.

This script extracts comprehensive statistics from both real and synthetic patches
to identify discriminating factors. Features include:
- Intensity statistics (lesion, background, contrast)
- Morphological metrics (area, circularity, solidity, eccentricity)
- Boundary characteristics (edge sharpness, boundary regularity)
- Texture features (entropy, homogeneity, gradient statistics)

Usage:
    python -m src.diffusion.audition.scripts.analyze_patch_features \
        --config src/diffusion/audition/config/audition.yaml
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import ndimage, stats
from skimage import measure, morphology

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class PatchFeatures:
    """Container for extracted patch features."""

    # Identification
    source: str  # "real" or "synthetic"
    z_bin: int
    index: int

    # Intensity features (image channel)
    lesion_mean_intensity: float = np.nan
    lesion_std_intensity: float = np.nan
    lesion_median_intensity: float = np.nan
    background_mean_intensity: float = np.nan
    background_std_intensity: float = np.nan
    contrast_ratio: float = np.nan  # lesion_mean / background_mean
    intensity_range_lesion: float = np.nan  # max - min within lesion

    # Morphological features (mask channel)
    lesion_area: float = np.nan  # number of pixels
    lesion_perimeter: float = np.nan
    circularity: float = np.nan  # 4*pi*area / perimeter^2
    solidity: float = np.nan  # area / convex_hull_area
    eccentricity: float = np.nan  # 0=circle, 1=line
    extent: float = np.nan  # area / bounding_box_area
    aspect_ratio: float = np.nan  # bbox width / height
    equivalent_diameter: float = np.nan

    # Boundary features
    edge_sharpness_mean: float = np.nan  # gradient magnitude at boundary
    edge_sharpness_std: float = np.nan
    boundary_irregularity: float = np.nan  # std of radial distances

    # Texture features (within lesion)
    texture_entropy: float = np.nan
    texture_homogeneity: float = np.nan  # inverse of local std
    gradient_mean: float = np.nan  # internal gradient magnitude
    gradient_std: float = np.nan

    # Spatial features
    centroid_x_normalized: float = np.nan  # relative to image center
    centroid_y_normalized: float = np.nan


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using Sobel filters."""
    grad_x = ndimage.sobel(image, axis=1)
    grad_y = ndimage.sobel(image, axis=0)
    return np.sqrt(grad_x**2 + grad_y**2)


def compute_entropy(values: np.ndarray, bins: int = 50) -> float:
    """Compute entropy of a distribution."""
    if len(values) < 2:
        return np.nan
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zeros
    if len(hist) == 0:
        return np.nan
    return -np.sum(hist * np.log2(hist + 1e-10))


def compute_boundary_mask(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Extract boundary region from mask."""
    binary = mask > 0
    eroded = morphology.binary_erosion(binary, morphology.disk(width))
    dilated = morphology.binary_dilation(binary, morphology.disk(width))
    return dilated & ~eroded


def extract_features(
    image: np.ndarray,
    mask: np.ndarray,
    source: str,
    z_bin: int,
    index: int,
) -> PatchFeatures:
    """Extract all features from a single patch.

    Args:
        image: Image array in [-1, 1] range, shape (H, W).
        mask: Mask array in [-1, 1] range, shape (H, W).
        source: "real" or "synthetic".
        z_bin: Z-bin of the patch.
        index: Sample index.

    Returns:
        PatchFeatures dataclass with all computed features.
    """
    features = PatchFeatures(source=source, z_bin=z_bin, index=index)

    # Binarize mask (threshold at 0 for [-1, 1] encoding)
    binary_mask = mask > 0
    lesion_pixels = binary_mask.sum()

    if lesion_pixels < 5:
        # Not enough lesion pixels for meaningful analysis
        return features

    # === INTENSITY FEATURES ===
    lesion_intensities = image[binary_mask]
    background_mask = ~binary_mask
    background_intensities = image[background_mask]

    features.lesion_mean_intensity = float(np.mean(lesion_intensities))
    features.lesion_std_intensity = float(np.std(lesion_intensities))
    features.lesion_median_intensity = float(np.median(lesion_intensities))
    features.intensity_range_lesion = float(
        np.max(lesion_intensities) - np.min(lesion_intensities)
    )

    if len(background_intensities) > 0:
        features.background_mean_intensity = float(np.mean(background_intensities))
        features.background_std_intensity = float(np.std(background_intensities))
        if features.background_mean_intensity != 0:
            features.contrast_ratio = (
                features.lesion_mean_intensity / features.background_mean_intensity
            )

    # === MORPHOLOGICAL FEATURES ===
    # Label connected components
    labeled = measure.label(binary_mask)
    regions = measure.regionprops(labeled)

    if len(regions) == 0:
        return features

    # Use largest region
    largest_region = max(regions, key=lambda r: r.area)

    features.lesion_area = float(largest_region.area)
    features.lesion_perimeter = float(largest_region.perimeter)
    features.eccentricity = float(largest_region.eccentricity)
    features.extent = float(largest_region.extent)
    features.equivalent_diameter = float(largest_region.equivalent_diameter)
    features.solidity = float(largest_region.solidity)

    # Circularity: 4*pi*area / perimeter^2
    if features.lesion_perimeter > 0:
        features.circularity = (
            4 * np.pi * features.lesion_area / (features.lesion_perimeter**2)
        )

    # Aspect ratio from bounding box
    bbox = largest_region.bbox  # (min_row, min_col, max_row, max_col)
    height = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]
    if height > 0:
        features.aspect_ratio = float(width / height)

    # === BOUNDARY FEATURES ===
    gradient_mag = compute_gradient_magnitude(image)
    boundary_mask = compute_boundary_mask(binary_mask, width=2)
    boundary_gradients = gradient_mag[boundary_mask]

    if len(boundary_gradients) > 0:
        features.edge_sharpness_mean = float(np.mean(boundary_gradients))
        features.edge_sharpness_std = float(np.std(boundary_gradients))

    # Boundary irregularity: std of distances from centroid to boundary
    boundary_coords = np.argwhere(boundary_mask)
    if len(boundary_coords) > 5:
        centroid = np.array([largest_region.centroid])
        distances = np.sqrt(np.sum((boundary_coords - centroid) ** 2, axis=1))
        features.boundary_irregularity = float(np.std(distances) / (np.mean(distances) + 1e-6))

    # === TEXTURE FEATURES ===
    features.texture_entropy = compute_entropy(lesion_intensities)

    # Homogeneity: inverse of local variance
    if lesion_pixels > 10:
        # Compute local std within lesion using a small window
        local_std = ndimage.generic_filter(
            image, np.std, size=3, mode="constant", cval=0
        )
        lesion_local_stds = local_std[binary_mask]
        features.texture_homogeneity = float(1.0 / (np.mean(lesion_local_stds) + 0.01))

    # Internal gradients
    internal_gradients = gradient_mag[binary_mask]
    features.gradient_mean = float(np.mean(internal_gradients))
    features.gradient_std = float(np.std(internal_gradients))

    # === SPATIAL FEATURES ===
    h, w = image.shape
    cy, cx = largest_region.centroid
    features.centroid_x_normalized = float((cx - w / 2) / (w / 2))
    features.centroid_y_normalized = float((cy - h / 2) / (h / 2))

    return features


def load_and_extract_all_features(patches_dir: Path) -> pd.DataFrame:
    """Load patches and extract features for all samples.

    Args:
        patches_dir: Directory containing real_patches.npz and synthetic_patches.npz.

    Returns:
        DataFrame with all features for all samples.
    """
    real_path = patches_dir / "real_patches.npz"
    synthetic_path = patches_dir / "synthetic_patches.npz"

    if not real_path.exists():
        raise FileNotFoundError(f"Real patches not found: {real_path}")
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Synthetic patches not found: {synthetic_path}")

    all_features = []

    # Process real patches
    logger.info(f"Loading real patches from {real_path}")
    real_npz = np.load(real_path)
    real_patches = real_npz["patches"]  # (N, 2, H, W)
    real_zbins = real_npz["z_bins"]

    logger.info(f"Extracting features from {len(real_patches)} real patches...")
    for i in range(len(real_patches)):
        image = real_patches[i, 0]  # Image channel
        mask = real_patches[i, 1]   # Mask channel
        features = extract_features(image, mask, "real", int(real_zbins[i]), i)
        all_features.append(features)

        if (i + 1) % 100 == 0:
            logger.debug(f"  Processed {i + 1}/{len(real_patches)} real patches")

    # Process synthetic patches
    logger.info(f"Loading synthetic patches from {synthetic_path}")
    synth_npz = np.load(synthetic_path)
    synth_patches = synth_npz["patches"]  # (N, 2, H, W)
    synth_zbins = synth_npz["z_bins"]

    logger.info(f"Extracting features from {len(synth_patches)} synthetic patches...")
    for i in range(len(synth_patches)):
        image = synth_patches[i, 0]
        mask = synth_patches[i, 1]
        features = extract_features(image, mask, "synthetic", int(synth_zbins[i]), i)
        all_features.append(features)

        if (i + 1) % 100 == 0:
            logger.debug(f"  Processed {i + 1}/{len(synth_patches)} synthetic patches")

    # Convert to DataFrame
    df = pd.DataFrame([vars(f) for f in all_features])
    logger.info(f"Extracted features for {len(df)} total patches")

    return df


def compute_statistics(df: pd.DataFrame, feature: str) -> dict:
    """Compute comparison statistics for a feature.

    Args:
        df: DataFrame with features.
        feature: Feature column name.

    Returns:
        Dictionary with statistics.
    """
    real_values = df[df["source"] == "real"][feature].dropna()
    synth_values = df[df["source"] == "synthetic"][feature].dropna()

    if len(real_values) < 5 or len(synth_values) < 5:
        return {"valid": False}

    # Mann-Whitney U test (non-parametric)
    try:
        statistic, p_value = stats.mannwhitneyu(
            real_values, synth_values, alternative="two-sided"
        )
    except ValueError:
        p_value = np.nan
        statistic = np.nan

    # Effect size: Cohen's d
    pooled_std = np.sqrt(
        (real_values.std() ** 2 + synth_values.std() ** 2) / 2
    )
    if pooled_std > 0:
        cohens_d = (real_values.mean() - synth_values.mean()) / pooled_std
    else:
        cohens_d = 0

    return {
        "valid": True,
        "real_mean": real_values.mean(),
        "real_std": real_values.std(),
        "real_median": real_values.median(),
        "synth_mean": synth_values.mean(),
        "synth_std": synth_values.std(),
        "synth_median": synth_values.median(),
        "p_value": p_value,
        "cohens_d": cohens_d,
        "n_real": len(real_values),
        "n_synth": len(synth_values),
    }


def create_comparison_panel(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> pd.DataFrame:
    """Create comprehensive panel plot comparing real vs synthetic features.

    Args:
        df: DataFrame with extracted features.
        output_path: Path to save the figure.
        dpi: Figure DPI.

    Returns:
        DataFrame with statistical summary.
    """
    # Define feature groups for organized visualization
    feature_groups = {
        "Intensity Statistics": [
            ("lesion_mean_intensity", "Lesion Mean Intensity"),
            ("lesion_std_intensity", "Lesion Intensity Std"),
            ("background_mean_intensity", "Background Mean Intensity"),
            ("contrast_ratio", "Contrast Ratio (Lesion/BG)"),
            ("intensity_range_lesion", "Intensity Range in Lesion"),
        ],
        "Morphological Features": [
            ("lesion_area", "Lesion Area (pixels)"),
            ("circularity", "Circularity"),
            ("solidity", "Solidity"),
            ("eccentricity", "Eccentricity"),
            ("aspect_ratio", "Aspect Ratio"),
        ],
        "Boundary Characteristics": [
            ("edge_sharpness_mean", "Edge Sharpness (Mean)"),
            ("edge_sharpness_std", "Edge Sharpness (Std)"),
            ("boundary_irregularity", "Boundary Irregularity"),
        ],
        "Texture Features": [
            ("texture_entropy", "Texture Entropy"),
            ("texture_homogeneity", "Texture Homogeneity"),
            ("gradient_mean", "Internal Gradient (Mean)"),
            ("gradient_std", "Internal Gradient (Std)"),
        ],
    }

    # Count total features
    total_features = sum(len(v) for v in feature_groups.values())
    n_cols = 5
    n_rows = (total_features + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    # Collect statistics
    stats_records = []

    # Plot each feature
    plot_idx = 0
    for group_name, features in feature_groups.items():
        for feature_col, feature_label in features:
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]

            # Get data
            real_data = df[df["source"] == "real"][feature_col].dropna()
            synth_data = df[df["source"] == "synthetic"][feature_col].dropna()

            if len(real_data) < 5 or len(synth_data) < 5:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                       transform=ax.transAxes)
                ax.set_title(feature_label, fontsize=10)
                plot_idx += 1
                continue

            # Compute statistics
            stat = compute_statistics(df, feature_col)
            stats_records.append({
                "group": group_name,
                "feature": feature_col,
                "label": feature_label,
                **stat,
            })

            # Create box plot
            box_data = [real_data, synth_data]
            bp = ax.boxplot(
                box_data,
                labels=["Real", "Synthetic"],
                patch_artist=True,
                widths=0.6,
            )

            # Color boxes
            colors = ["#3498db", "#e74c3c"]  # Blue for real, red for synthetic
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add individual points (jittered)
            for i, (data, color) in enumerate(zip(box_data, colors)):
                x = np.random.normal(i + 1, 0.04, size=len(data))
                ax.scatter(x, data, alpha=0.3, s=10, c=color, edgecolors="none")

            # Add p-value annotation
            p_val = stat["p_value"]
            if p_val < 0.001:
                p_text = "p < 0.001"
            elif p_val < 0.01:
                p_text = f"p = {p_val:.3f}"
            elif p_val < 0.05:
                p_text = f"p = {p_val:.3f}"
            else:
                p_text = f"p = {p_val:.2f}"

            # Color p-value by significance
            if p_val < 0.001:
                p_color = "darkred"
                significance = "***"
            elif p_val < 0.01:
                p_color = "red"
                significance = "**"
            elif p_val < 0.05:
                p_color = "orange"
                significance = "*"
            else:
                p_color = "gray"
                significance = "ns"

            ax.set_title(f"{feature_label}\n{p_text} {significance}", fontsize=9)

            # Add effect size annotation
            d = stat["cohens_d"]
            effect_text = f"d={d:.2f}"
            ax.text(0.95, 0.95, effect_text, transform=ax.transAxes,
                   fontsize=8, ha="right", va="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

            ax.tick_params(axis="both", labelsize=8)
            plot_idx += 1

    # Hide unused axes
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis("off")

    # Add overall title
    fig.suptitle(
        "Real vs Synthetic Lesion Patch Feature Comparison\n"
        "(*** p<0.001, ** p<0.01, * p<0.05, ns=not significant, d=Cohen's d effect size)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved comparison panel to {output_path}")

    # Create stats DataFrame
    stats_df = pd.DataFrame(stats_records)
    return stats_df


def create_distribution_panel(
    df: pd.DataFrame,
    output_path: Path,
    top_n: int = 8,
    dpi: int = 150,
) -> None:
    """Create detailed distribution plots for top discriminating features.

    Args:
        df: DataFrame with extracted features.
        output_path: Path to save the figure.
        top_n: Number of top features to show.
        dpi: Figure DPI.
    """
    # Find most discriminating features (by p-value and effect size)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ["z_bin", "index"]]

    rankings = []
    for col in feature_cols:
        stat = compute_statistics(df, col)
        if stat.get("valid", False):
            # Combine p-value and effect size for ranking
            # Lower p-value and higher |d| = more discriminating
            score = -np.log10(stat["p_value"] + 1e-10) * abs(stat["cohens_d"])
            rankings.append({
                "feature": col,
                "p_value": stat["p_value"],
                "cohens_d": stat["cohens_d"],
                "score": score,
            })

    rankings = sorted(rankings, key=lambda x: x["score"], reverse=True)[:top_n]

    if len(rankings) == 0:
        logger.warning("No valid features for distribution panel")
        return

    # Create figure
    n_cols = 2
    n_rows = (len(rankings) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for idx, rank_info in enumerate(rankings):
        ax = axes[idx]
        feature = rank_info["feature"]

        real_data = df[df["source"] == "real"][feature].dropna()
        synth_data = df[df["source"] == "synthetic"][feature].dropna()

        # Histogram comparison
        bins = np.histogram_bin_edges(
            np.concatenate([real_data, synth_data]), bins=30
        )

        ax.hist(real_data, bins=bins, alpha=0.6, label="Real",
               color="#3498db", density=True, edgecolor="black", linewidth=0.5)
        ax.hist(synth_data, bins=bins, alpha=0.6, label="Synthetic",
               color="#e74c3c", density=True, edgecolor="black", linewidth=0.5)

        # Add vertical lines for means
        ax.axvline(real_data.mean(), color="#2980b9", linestyle="--",
                  linewidth=2, label=f"Real μ={real_data.mean():.3f}")
        ax.axvline(synth_data.mean(), color="#c0392b", linestyle="--",
                  linewidth=2, label=f"Synth μ={synth_data.mean():.3f}")

        # Title with statistics
        p_val = rank_info["p_value"]
        d = rank_info["cohens_d"]
        ax.set_title(f"{feature}\np={p_val:.2e}, Cohen's d={d:.2f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)

    # Hide unused axes
    for idx in range(len(rankings), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"Top {len(rankings)} Most Discriminating Features\n(Distributions)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved distribution panel to {output_path}")


def create_zbin_analysis(
    df: pd.DataFrame,
    output_path: Path,
    top_features: list[str] | None = None,
    dpi: int = 150,
) -> None:
    """Analyze feature differences across z-bins.

    Args:
        df: DataFrame with extracted features.
        output_path: Path to save the figure.
        top_features: List of features to analyze. If None, uses defaults.
        dpi: Figure DPI.
    """
    if top_features is None:
        top_features = [
            "edge_sharpness_mean",
            "circularity",
            "lesion_mean_intensity",
            "texture_homogeneity",
        ]

    # Filter to features that exist
    top_features = [f for f in top_features if f in df.columns]

    if len(top_features) == 0:
        logger.warning("No valid features for z-bin analysis")
        return

    n_features = len(top_features)
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
    if n_features == 1:
        axes = [axes]

    z_bins = sorted(df["z_bin"].unique())

    for ax, feature in zip(axes, top_features):
        real_means = []
        real_stds = []
        synth_means = []
        synth_stds = []

        for zb in z_bins:
            real_vals = df[(df["source"] == "real") & (df["z_bin"] == zb)][feature].dropna()
            synth_vals = df[(df["source"] == "synthetic") & (df["z_bin"] == zb)][feature].dropna()

            real_means.append(real_vals.mean() if len(real_vals) > 0 else np.nan)
            real_stds.append(real_vals.std() if len(real_vals) > 0 else np.nan)
            synth_means.append(synth_vals.mean() if len(synth_vals) > 0 else np.nan)
            synth_stds.append(synth_vals.std() if len(synth_vals) > 0 else np.nan)

        real_means = np.array(real_means)
        real_stds = np.array(real_stds)
        synth_means = np.array(synth_means)
        synth_stds = np.array(synth_stds)

        # Plot with error bands
        ax.fill_between(
            z_bins, real_means - real_stds, real_means + real_stds,
            alpha=0.3, color="#3498db"
        )
        ax.plot(z_bins, real_means, "-o", color="#3498db", label="Real", markersize=4)

        ax.fill_between(
            z_bins, synth_means - synth_stds, synth_means + synth_stds,
            alpha=0.3, color="#e74c3c"
        )
        ax.plot(z_bins, synth_means, "-o", color="#e74c3c", label="Synthetic", markersize=4)

        ax.set_xlabel("Z-bin", fontsize=10)
        ax.set_ylabel(feature, fontsize=10)
        ax.set_title(feature.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Feature Comparison Across Z-bins\n(Mean ± Std)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved z-bin analysis to {output_path}")


def create_localization_heatmap(
    cfg: "OmegaConf",
    output_path: Path,
    min_frequency: int = 3,
    max_bins_per_row: int = 10,
    dpi: int = 150,
) -> None:
    """Create lesion localization heatmap comparing real vs synthetic positions.

    This function uses the ORIGINAL full-resolution slices (not centered patches)
    to show where lesions actually appear in the brain.

    For each z-bin, this function:
    1. Accumulates all lesion masks to create a frequency map
    2. Overlays real (blue) and synthetic (red) heatmaps on a representative slice
    3. Only shows pixels where lesion frequency >= min_frequency

    Args:
        cfg: Configuration with data paths.
        output_path: Path to save the figure.
        min_frequency: Minimum number of lesion occurrences to display a pixel.
        max_bins_per_row: Maximum number of z-bins per row in the figure.
        dpi: Figure DPI.
    """
    import csv

    # === Load Real Data from slice_cache ===
    cache_dir = Path(cfg.data.real.cache_dir)

    logger.info(f"Loading real slices from {cache_dir}")

    # Collect all real lesion slices
    real_data_by_zbin: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

    for csv_file in cfg.data.real.csv_files:
        csv_path = cache_dir / csv_file
        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            continue

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only include slices with lesions
                if row.get("has_lesion", "False").lower() != "true":
                    continue

                # filepath column already includes 'slices/' prefix
                slice_path = cache_dir / row["filepath"]
                if not slice_path.exists():
                    continue

                data = np.load(slice_path)
                image = data["image"]
                mask = data["mask"]
                z_bin = int(data["z_bin"])

                # Only include if mask actually has lesion pixels
                if (mask > 0).sum() < 5:
                    continue

                if z_bin not in real_data_by_zbin:
                    real_data_by_zbin[z_bin] = []
                real_data_by_zbin[z_bin].append((image, mask))

    logger.info(f"Loaded real data for {len(real_data_by_zbin)} z-bins")

    # === Load Synthetic Data from replicas ===
    replicas_dir = Path(cfg.data.synthetic.replicas_dir)
    replica_ids = cfg.data.synthetic.replica_ids

    logger.info(f"Loading synthetic slices from {replicas_dir}")

    synth_data_by_zbin: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

    for replica_id in replica_ids:
        replica_path = replicas_dir / f"replica_{replica_id:03d}.npz"
        if not replica_path.exists():
            logger.warning(f"Replica not found: {replica_path}")
            continue

        replica_data = np.load(replica_path)
        images = replica_data["images"]
        masks = replica_data["masks"]
        zbins = replica_data["zbin"]
        lesion_present = replica_data.get("lesion_present", np.ones(len(images)))

        # Only include samples with lesions
        for i in range(len(images)):
            if lesion_present[i] != 1:
                continue

            mask = masks[i]
            # Only include if mask actually has lesion pixels
            if (mask > 0).sum() < 5:
                continue

            z_bin = int(zbins[i])
            if z_bin not in synth_data_by_zbin:
                synth_data_by_zbin[z_bin] = []
            synth_data_by_zbin[z_bin].append((images[i], mask))

    logger.info(f"Loaded synthetic data for {len(synth_data_by_zbin)} z-bins")

    # === Compute frequency maps ===
    all_zbins = sorted(set(real_data_by_zbin.keys()) | set(synth_data_by_zbin.keys()))

    if len(all_zbins) == 0:
        logger.warning("No z-bins found for localization analysis")
        return

    # Determine image size from first available sample
    sample_shape = None
    for zb in all_zbins:
        if zb in real_data_by_zbin and len(real_data_by_zbin[zb]) > 0:
            sample_shape = real_data_by_zbin[zb][0][0].shape
            break
        if zb in synth_data_by_zbin and len(synth_data_by_zbin[zb]) > 0:
            sample_shape = synth_data_by_zbin[zb][0][0].shape
            break

    if sample_shape is None:
        logger.warning("Could not determine image shape")
        return

    H, W = sample_shape
    logger.info(f"Image size: {H}x{W}")

    zbin_data = {}
    for zb in all_zbins:
        # Real frequency map
        real_samples = real_data_by_zbin.get(zb, [])
        if len(real_samples) > 0:
            real_masks = np.stack([m for _, m in real_samples], axis=0)
            real_freq = (real_masks > 0).sum(axis=0).astype(float)
            real_imgs = np.stack([img for img, _ in real_samples], axis=0)
            real_img = real_imgs.mean(axis=0)
        else:
            real_freq = np.zeros((H, W))
            real_img = None

        # Synthetic frequency map
        synth_samples = synth_data_by_zbin.get(zb, [])
        if len(synth_samples) > 0:
            synth_masks = np.stack([m for _, m in synth_samples], axis=0)
            synth_freq = (synth_masks > 0).sum(axis=0).astype(float)
            synth_imgs = np.stack([img for img, _ in synth_samples], axis=0)
            synth_img = synth_imgs.mean(axis=0)
        else:
            synth_freq = np.zeros((H, W))
            synth_img = None

        # Use whichever representative image is available (prefer real)
        if real_img is not None:
            rep_img = real_img
        elif synth_img is not None:
            rep_img = synth_img
        else:
            rep_img = np.zeros((H, W))

        zbin_data[zb] = {
            "real_freq": real_freq,
            "synth_freq": synth_freq,
            "rep_img": rep_img,
            "n_real": len(real_samples),
            "n_synth": len(synth_samples),
        }

    # Calculate figure layout
    n_zbins = len(all_zbins)
    n_cols = min(max_bins_per_row, n_zbins)
    n_rows = (n_zbins + n_cols - 1) // n_cols

    fig_width = 2.5 * n_cols
    fig_height = 2.8 * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Create custom colormaps with transparency
    from matplotlib.colors import LinearSegmentedColormap

    # Blues colormap (for real) - transparent to blue
    blues_colors = [(0, 0, 1, 0), (0, 0, 1, 0.7)]  # RGBA: transparent to blue
    blues_cmap = LinearSegmentedColormap.from_list("custom_blues", blues_colors)

    # Reds colormap (for synthetic) - transparent to red
    reds_colors = [(1, 0, 0, 0), (1, 0, 0, 0.7)]  # RGBA: transparent to red
    reds_cmap = LinearSegmentedColormap.from_list("custom_reds", reds_colors)

    # Plot each z-bin
    for idx, zb in enumerate(all_zbins):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        data = zbin_data[zb]

        # Convert representative image to display range [0, 1]
        rep_img = (data["rep_img"] + 1) / 2
        rep_img = np.clip(rep_img, 0, 1)

        # Show background image in grayscale
        ax.imshow(rep_img, cmap="gray", vmin=0, vmax=1)

        # Prepare frequency maps with threshold
        real_freq = data["real_freq"].copy()
        synth_freq = data["synth_freq"].copy()

        # Apply minimum frequency threshold (set below-threshold to NaN for transparency)
        real_freq_masked = np.where(real_freq >= min_frequency, real_freq, np.nan)
        synth_freq_masked = np.where(synth_freq >= min_frequency, synth_freq, np.nan)

        # Normalize frequencies for colormap (0 to max)
        real_max = np.nanmax(real_freq_masked) if np.any(~np.isnan(real_freq_masked)) else 1
        synth_max = np.nanmax(synth_freq_masked) if np.any(~np.isnan(synth_freq_masked)) else 1

        if real_max > 0:
            real_freq_norm = real_freq_masked / real_max
        else:
            real_freq_norm = real_freq_masked

        if synth_max > 0:
            synth_freq_norm = synth_freq_masked / synth_max
        else:
            synth_freq_norm = synth_freq_masked

        # Overlay real heatmap (blues)
        ax.imshow(real_freq_norm, cmap=blues_cmap, vmin=0, vmax=1, alpha=0.8)

        # Overlay synthetic heatmap (reds)
        ax.imshow(synth_freq_norm, cmap=reds_cmap, vmin=0, vmax=1, alpha=0.8)

        # Title with z-bin and sample counts
        ax.set_title(f"z={zb}\nR:{data['n_real']} S:{data['n_synth']}", fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n_zbins, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="blue", alpha=0.6, label="Real lesions"),
        Patch(facecolor="red", alpha=0.6, label="Synthetic lesions"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=10,
    )

    # Add title
    fig.suptitle(
        f"Lesion Localization by Z-bin\n"
        f"(Heatmap shows lesion frequency, threshold ≥ {min_frequency} occurrences)\n"
        f"Blue = Real, Red = Synthetic",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved localization heatmap to {output_path}")


def create_summary_report(stats_df: pd.DataFrame, output_path: Path) -> None:
    """Create text summary of most discriminating features.

    Args:
        stats_df: DataFrame with statistical results.
        output_path: Path to save the report.
    """
    # Sort by effect size (absolute)
    stats_df = stats_df[stats_df["valid"] == True].copy()
    stats_df["abs_d"] = stats_df["cohens_d"].abs()
    stats_df = stats_df.sort_values("abs_d", ascending=False)

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("REAL VS SYNTHETIC PATCH FEATURE ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("INTERPRETATION GUIDE:\n")
        f.write("-" * 40 + "\n")
        f.write("- Cohen's d effect size:\n")
        f.write("  |d| < 0.2  = negligible\n")
        f.write("  |d| 0.2-0.5 = small\n")
        f.write("  |d| 0.5-0.8 = medium\n")
        f.write("  |d| > 0.8  = large\n")
        f.write("- Negative d: Real > Synthetic\n")
        f.write("- Positive d: Synthetic > Real\n\n")

        f.write("TOP DISCRIMINATING FEATURES (by effect size):\n")
        f.write("-" * 70 + "\n")

        for idx, row in stats_df.head(10).iterrows():
            f.write(f"\n{row['label']} ({row['feature']})\n")
            f.write(f"  Group: {row['group']}\n")
            f.write(f"  Real:      mean={row['real_mean']:.4f}, std={row['real_std']:.4f}\n")
            f.write(f"  Synthetic: mean={row['synth_mean']:.4f}, std={row['synth_std']:.4f}\n")
            f.write(f"  p-value:   {row['p_value']:.2e}\n")
            f.write(f"  Cohen's d: {row['cohens_d']:.3f}\n")

            # Interpretation
            d = row["cohens_d"]
            if abs(d) < 0.2:
                effect = "negligible"
            elif abs(d) < 0.5:
                effect = "small"
            elif abs(d) < 0.8:
                effect = "medium"
            else:
                effect = "large"

            direction = "Real > Synthetic" if d < 0 else "Synthetic > Real"
            f.write(f"  Effect: {effect} ({direction})\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("SUMMARY OF FINDINGS:\n")
        f.write("-" * 70 + "\n")

        # Count significant features
        sig_features = stats_df[stats_df["p_value"] < 0.05]
        large_effect = stats_df[stats_df["abs_d"] >= 0.8]
        medium_effect = stats_df[(stats_df["abs_d"] >= 0.5) & (stats_df["abs_d"] < 0.8)]

        f.write(f"Total features analyzed: {len(stats_df)}\n")
        f.write(f"Statistically significant (p < 0.05): {len(sig_features)}\n")
        f.write(f"Large effect size (|d| >= 0.8): {len(large_effect)}\n")
        f.write(f"Medium effect size (0.5 <= |d| < 0.8): {len(medium_effect)}\n")

        if len(large_effect) > 0:
            f.write(f"\nFeatures with LARGE effect:\n")
            for _, row in large_effect.iterrows():
                direction = "↓" if row["cohens_d"] < 0 else "↑"
                f.write(f"  - {row['label']}: d={row['cohens_d']:.2f} {direction}\n")

    logger.info(f"Saved summary report to {output_path}")


def main() -> None:
    """Main entry point for feature analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze features differentiating real from synthetic patches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.diffusion.audition.scripts.analyze_patch_features \\
        --config src/diffusion/audition/config/audition.yaml

    python -m src.diffusion.audition.scripts.analyze_patch_features \\
        --patches-dir outputs/audition/patches \\
        --output-dir outputs/audition/analysis
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
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=3,
        help="Minimum lesion frequency to show in localization heatmap",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine paths
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

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.config:
        cfg = OmegaConf.load(args.config)
        output_dir = Path(cfg.output.results_dir) / "feature_analysis"
    else:
        output_dir = patches_dir / "analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Patches directory: {patches_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Extract features
    logger.info("Extracting features from all patches...")
    df = load_and_extract_all_features(patches_dir)

    # Save raw features
    features_csv = output_dir / "extracted_features.csv"
    df.to_csv(features_csv, index=False)
    logger.info(f"Saved raw features to {features_csv}")

    # Create comparison panel
    logger.info("Creating comparison panel...")
    stats_df = create_comparison_panel(
        df,
        output_dir / "feature_comparison_panel.png",
        dpi=args.dpi,
    )

    # Save statistics
    stats_csv = output_dir / "feature_statistics.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info(f"Saved statistics to {stats_csv}")

    # Create distribution panel for top features
    logger.info("Creating distribution panel...")
    create_distribution_panel(
        df,
        output_dir / "top_features_distributions.png",
        dpi=args.dpi,
    )

    # Create z-bin analysis
    logger.info("Creating z-bin analysis...")
    create_zbin_analysis(
        df,
        output_dir / "zbin_feature_analysis.png",
        dpi=args.dpi,
    )

    # Create localization heatmap (uses original full-resolution slices, not patches)
    logger.info("Creating localization heatmap from original slices...")
    if args.config:
        cfg = OmegaConf.load(args.config)
        create_localization_heatmap(
            cfg,
            output_dir / "lesion_localization_heatmap.png",
            min_frequency=args.min_frequency,
            dpi=args.dpi,
        )
    else:
        logger.warning("Skipping localization heatmap: requires --config to access original data")

    # Create summary report
    logger.info("Creating summary report...")
    create_summary_report(stats_df, output_dir / "analysis_report.txt")

    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("Generated files:")
    logger.info("  - extracted_features.csv (raw feature data)")
    logger.info("  - feature_statistics.csv (statistical summary)")
    logger.info("  - feature_comparison_panel.png (box plots)")
    logger.info("  - top_features_distributions.png (histograms)")
    logger.info("  - zbin_feature_analysis.png (z-bin trends)")
    logger.info("  - lesion_localization_heatmap.png (spatial localization)")
    logger.info("  - analysis_report.txt (text summary)")


if __name__ == "__main__":
    main()
