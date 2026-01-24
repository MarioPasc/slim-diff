"""Aggregation and visualization utilities for Grad-CAM results.

Provides functions to aggregate per-sample Grad-CAM heatmaps by class and
z-bin, compute attention difference metrics between real and synthetic classes,
and generate comprehensive visualization figures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from omegaconf import DictConfig
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr

from src.classification.diagnostics.utils import save_figure

logger = logging.getLogger(__name__)


@dataclass
class AggregatedHeatmaps:
    """Aggregated Grad-CAM heatmaps for a group of samples.

    Attributes:
        mean_heatmap: Mean heatmap across samples, shape (H, W).
        std_heatmap: Standard deviation heatmap, shape (H, W).
        n_samples: Number of samples aggregated.
        class_label: Class label for this group (0=real, 1=synthetic), or -1 if mixed.
        z_bin: Z-bin for this group, or -1 if aggregated across all z-bins.
    """

    mean_heatmap: np.ndarray
    std_heatmap: np.ndarray
    n_samples: int
    class_label: int
    z_bin: int


def aggregate_heatmaps(
    results: list,
    group_by: str = "both",
) -> dict[str, AggregatedHeatmaps]:
    """Aggregate Grad-CAM heatmaps by class, z-bin, or both.

    Groups the per-sample results and computes mean and standard deviation
    heatmaps for each group.

    Args:
        results: List of GradCAMResult instances to aggregate.
        group_by: Grouping strategy:
            - "class": Aggregate separately for each class label.
            - "z_bin": Aggregate separately for each z-bin.
            - "both": Aggregate by (class, z_bin) pairs.

    Returns:
        Dictionary mapping group keys to AggregatedHeatmaps. Keys are formatted
        as 'class_{label}', 'zbin_{bin}', or 'class_{label}_zbin_{bin}'.

    Raises:
        ValueError: If group_by is not one of the valid options.
    """
    if group_by not in ("class", "z_bin", "both"):
        raise ValueError(f"group_by must be 'class', 'z_bin', or 'both', got '{group_by}'")

    if not results:
        return {}

    # Group results
    groups: dict[str, list] = {}
    for r in results:
        if group_by == "class":
            key = f"class_{r.label}"
        elif group_by == "z_bin":
            key = f"zbin_{r.z_bin}"
        else:  # both
            key = f"class_{r.label}_zbin_{r.z_bin}"
        groups.setdefault(key, []).append(r)

    # Compute aggregated statistics
    aggregated: dict[str, AggregatedHeatmaps] = {}
    for key, group_results in groups.items():
        heatmaps = np.stack([r.heatmap for r in group_results], axis=0)
        mean_heatmap = heatmaps.mean(axis=0)
        std_heatmap = heatmaps.std(axis=0)

        # Parse class and z_bin from key
        class_label = -1
        z_bin = -1
        if "class_" in key:
            parts = key.split("_")
            class_idx = parts.index("class") + 1
            class_label = int(parts[class_idx])
        if "zbin_" in key:
            parts = key.split("_")
            zbin_idx = parts.index("zbin") + 1
            z_bin = int(parts[zbin_idx])

        aggregated[key] = AggregatedHeatmaps(
            mean_heatmap=mean_heatmap,
            std_heatmap=std_heatmap,
            n_samples=len(group_results),
            class_label=class_label,
            z_bin=z_bin,
        )

    return aggregated


def compute_attention_difference(
    real_agg: AggregatedHeatmaps,
    synth_agg: AggregatedHeatmaps,
) -> dict:
    """Compute spatial attention differences between real and synthetic classes.

    Analyzes how the classifier's attention differs between real and synthetic
    samples, providing both a difference map and scalar similarity metrics.

    Args:
        real_agg: Aggregated heatmaps for real samples (class 0).
        synth_agg: Aggregated heatmaps for synthetic samples (class 1).

    Returns:
        Dictionary containing:
            - "difference_map": Signed difference (synth - real), shape (H, W).
            - "abs_difference_map": Absolute difference, shape (H, W).
            - "cosine_similarity": Cosine similarity between flattened mean heatmaps.
            - "spatial_correlation": Pearson correlation coefficient.
            - "spatial_correlation_pvalue": P-value for the correlation test.
            - "mean_abs_difference": Mean of absolute difference map.
            - "max_abs_difference": Maximum of absolute difference map.
    """
    real_map = real_agg.mean_heatmap
    synth_map = synth_agg.mean_heatmap

    # Signed difference: positive = more attention for synthetic
    difference_map = synth_map - real_map
    abs_difference_map = np.abs(difference_map)

    # Cosine similarity between flattened heatmaps
    real_flat = real_map.flatten()
    synth_flat = synth_map.flatten()

    dot_product = np.dot(real_flat, synth_flat)
    norm_real = np.linalg.norm(real_flat)
    norm_synth = np.linalg.norm(synth_flat)
    if norm_real > 1e-8 and norm_synth > 1e-8:
        cosine_sim = float(dot_product / (norm_real * norm_synth))
    else:
        cosine_sim = 0.0

    # Pearson spatial correlation
    if real_flat.std() > 1e-8 and synth_flat.std() > 1e-8:
        corr, pvalue = pearsonr(real_flat, synth_flat)
        spatial_corr = float(corr)
        spatial_pvalue = float(pvalue)
    else:
        spatial_corr = 0.0
        spatial_pvalue = 1.0

    return {
        "difference_map": difference_map,
        "abs_difference_map": abs_difference_map,
        "cosine_similarity": cosine_sim,
        "spatial_correlation": spatial_corr,
        "spatial_correlation_pvalue": spatial_pvalue,
        "mean_abs_difference": float(abs_difference_map.mean()),
        "max_abs_difference": float(abs_difference_map.max()),
    }


def radial_attention_profile(
    heatmap: np.ndarray,
    center: Optional[tuple[int, int]] = None,
    n_bins: int = 20,
) -> np.ndarray:
    """Compute the radial attention profile from a center point.

    Averages heatmap values in concentric annular bins around a center point,
    producing a 1D profile of attention vs. distance from center. Useful for
    detecting whether the classifier focuses on patch centers or edges.

    Args:
        heatmap: 2D heatmap array of shape (H, W), values in [0, 1].
        center: (row, col) center point. If None, uses the heatmap center.
        n_bins: Number of radial distance bins.

    Returns:
        Array of shape (n_bins,) with mean attention at each radial distance.
        Bins with no pixels are assigned 0.
    """
    h, w = heatmap.shape

    if center is None:
        center = (h // 2, w // 2)

    # Compute distance from center for each pixel
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt(
        (y_coords - center[0]) ** 2 + (x_coords - center[1]) ** 2
    )

    # Maximum possible distance (corner to center)
    max_dist = np.sqrt(center[0] ** 2 + center[1] ** 2)
    max_dist = max(max_dist, np.sqrt((h - center[0]) ** 2 + (w - center[1]) ** 2))

    # Bin edges
    bin_edges = np.linspace(0, max_dist, n_bins + 1)

    # Compute mean attention in each bin
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        if mask.any():
            profile[i] = heatmap[mask].mean()

    return profile


def _plot_sample_grid(
    results: list,
    output_dir: Path,
    n_samples: int = 16,
    n_cols: int = 4,
) -> None:
    """Generate a grid of sample heatmaps overlaid on input patches.

    Selects representative samples from both classes (real and synthetic)
    and visualizes the Grad-CAM overlay.

    Args:
        results: List of GradCAMResult instances.
        output_dir: Directory to save the figure.
        n_samples: Total number of samples to display (half per class).
        n_cols: Number of columns in the grid.
    """
    real_results = [r for r in results if r.label == 0]
    synth_results = [r for r in results if r.label == 1]

    # Select samples evenly from each class
    n_per_class = min(n_samples // 2, len(real_results), len(synth_results))
    if n_per_class == 0:
        logger.warning("Not enough samples for sample grid plot.")
        return

    rng = np.random.default_rng(42)
    real_indices = rng.choice(len(real_results), n_per_class, replace=False)
    synth_indices = rng.choice(len(synth_results), n_per_class, replace=False)

    selected = (
        [real_results[i] for i in real_indices]
        + [synth_results[i] for i in synth_indices]
    )

    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for idx, ax in enumerate(axes.flat):
        if idx >= len(selected):
            ax.axis("off")
            continue

        result = selected[idx]
        heatmap = result.heatmap
        class_name = "Real" if result.label == 0 else "Synth"
        pred_str = f"p={result.prediction:.2f}"

        ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"{class_name} (z={result.z_bin}) {pred_str}", fontsize=8)
        ax.axis("off")

    plt.suptitle("Grad-CAM Sample Heatmaps", fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "gradcam_sample_grid")
    plt.close(fig)


def _plot_mean_heatmaps(
    aggregated: dict[str, AggregatedHeatmaps],
    output_dir: Path,
) -> None:
    """Plot mean Grad-CAM heatmaps for real and synthetic classes.

    Shows the class-level mean and standard deviation heatmaps side-by-side.

    Args:
        aggregated: Dictionary of aggregated results (must contain class keys).
        output_dir: Directory to save the figure.
    """
    # Find class-level aggregations
    real_key = None
    synth_key = None
    for key, agg in aggregated.items():
        if agg.class_label == 0 and agg.z_bin == -1:
            real_key = key
        elif agg.class_label == 1 and agg.z_bin == -1:
            synth_key = key

    # Fall back to any class-level key
    if real_key is None:
        for key, agg in aggregated.items():
            if agg.class_label == 0:
                real_key = key
                break
    if synth_key is None:
        for key, agg in aggregated.items():
            if agg.class_label == 1:
                synth_key = key
                break

    if real_key is None or synth_key is None:
        logger.warning("Cannot find class-level aggregations for mean heatmap plot.")
        return

    real_agg = aggregated[real_key]
    synth_agg = aggregated[synth_key]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Mean heatmaps
    im0 = axes[0, 0].imshow(real_agg.mean_heatmap, cmap="hot", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Real Mean (n={real_agg.n_samples})")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(synth_agg.mean_heatmap, cmap="hot", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Synthetic Mean (n={synth_agg.n_samples})")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Std heatmaps
    vmax_std = max(real_agg.std_heatmap.max(), synth_agg.std_heatmap.max())
    im2 = axes[1, 0].imshow(real_agg.std_heatmap, cmap="viridis", vmin=0, vmax=vmax_std)
    axes[1, 0].set_title("Real Std Dev")
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(synth_agg.std_heatmap, cmap="viridis", vmin=0, vmax=vmax_std)
    axes[1, 1].set_title("Synthetic Std Dev")
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle("Mean Grad-CAM Heatmaps by Class", fontsize=13)
    plt.tight_layout()
    save_figure(fig, output_dir, "gradcam_mean_heatmaps")
    plt.close(fig)


def _plot_difference_map(
    aggregated: dict[str, AggregatedHeatmaps],
    output_dir: Path,
) -> None:
    """Plot the attention difference map between synthetic and real classes.

    Shows signed difference (synth - real) with a diverging colormap.

    Args:
        aggregated: Dictionary of aggregated results.
        output_dir: Directory to save the figure.
    """
    # Extract class-level aggregations
    real_agg = None
    synth_agg = None
    for agg in aggregated.values():
        if agg.class_label == 0 and real_agg is None:
            real_agg = agg
        elif agg.class_label == 1 and synth_agg is None:
            synth_agg = agg

    if real_agg is None or synth_agg is None:
        logger.warning("Cannot compute difference map: missing class aggregations.")
        return

    diff_results = compute_attention_difference(real_agg, synth_agg)
    diff_map = diff_results["difference_map"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Signed difference
    vmax = max(abs(diff_map.min()), abs(diff_map.max()))
    im0 = axes[0].imshow(diff_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Difference (Synth - Real)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Absolute difference
    im1 = axes[1].imshow(diff_results["abs_difference_map"], cmap="hot", vmin=0)
    axes[1].set_title("Absolute Difference")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Annotation with metrics
    axes[2].axis("off")
    metrics_text = (
        f"Cosine Similarity: {diff_results['cosine_similarity']:.4f}\n"
        f"Spatial Correlation: {diff_results['spatial_correlation']:.4f}\n"
        f"  (p={diff_results['spatial_correlation_pvalue']:.2e})\n"
        f"Mean |Diff|: {diff_results['mean_abs_difference']:.4f}\n"
        f"Max |Diff|: {diff_results['max_abs_difference']:.4f}"
    )
    axes[2].text(
        0.1, 0.5, metrics_text, transform=axes[2].transAxes,
        fontsize=11, verticalalignment="center", fontfamily="monospace",
    )
    axes[2].set_title("Attention Difference Metrics")

    plt.suptitle("Classifier Attention: Synthetic vs Real", fontsize=13)
    plt.tight_layout()
    save_figure(fig, output_dir, "gradcam_difference_map")
    plt.close(fig)


def _plot_per_zbin_grid(
    aggregated: dict[str, AggregatedHeatmaps],
    output_dir: Path,
) -> None:
    """Plot per-z-bin mean heatmaps for both classes.

    Creates a grid where rows are z-bins and columns are [real, synthetic].

    Args:
        aggregated: Dictionary of aggregated results (grouped by both class and z_bin).
        output_dir: Directory to save the figure.
    """
    # Collect available z-bins
    zbins_with_data: set[int] = set()
    for agg in aggregated.values():
        if agg.z_bin >= 0:
            zbins_with_data.add(agg.z_bin)

    if not zbins_with_data:
        logger.info("No per-z-bin data available for grid plot.")
        return

    sorted_zbins = sorted(zbins_with_data)
    n_zbins = len(sorted_zbins)

    fig, axes = plt.subplots(n_zbins, 2, figsize=(6, 2.5 * n_zbins))
    if n_zbins == 1:
        axes = axes[np.newaxis, :]

    for row, zbin in enumerate(sorted_zbins):
        for col, class_label in enumerate([0, 1]):
            key = f"class_{class_label}_zbin_{zbin}"
            ax = axes[row, col]

            if key in aggregated:
                agg = aggregated[key]
                ax.imshow(agg.mean_heatmap, cmap="hot", vmin=0, vmax=1)
                class_name = "Real" if class_label == 0 else "Synth"
                ax.set_title(f"{class_name}, z={zbin} (n={agg.n_samples})", fontsize=9)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
            ax.axis("off")

    plt.suptitle("Per-Z-Bin Mean Grad-CAM Heatmaps", fontsize=12, y=1.01)
    plt.tight_layout()
    save_figure(fig, output_dir, "gradcam_per_zbin_grid")
    plt.close(fig)


def _plot_radial_profiles(
    aggregated: dict[str, AggregatedHeatmaps],
    output_dir: Path,
    n_bins: int = 20,
) -> None:
    """Plot radial attention profiles comparing real and synthetic classes.

    Computes radial profiles from the heatmap center and plots them for
    both classes, revealing whether attention is concentrated at the center
    or periphery.

    Args:
        aggregated: Dictionary of aggregated results.
        output_dir: Directory to save the figure.
        n_bins: Number of radial bins.
    """
    real_agg = None
    synth_agg = None
    for agg in aggregated.values():
        if agg.class_label == 0 and real_agg is None:
            real_agg = agg
        elif agg.class_label == 1 and synth_agg is None:
            synth_agg = agg

    if real_agg is None or synth_agg is None:
        logger.warning("Cannot compute radial profiles: missing class aggregations.")
        return

    real_profile = radial_attention_profile(real_agg.mean_heatmap, n_bins=n_bins)
    synth_profile = radial_attention_profile(synth_agg.mean_heatmap, n_bins=n_bins)

    # Normalize distances to [0, 1]
    distances = np.linspace(0, 1, n_bins)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(distances, real_profile, "b-o", markersize=4, label="Real", linewidth=1.5)
    ax.plot(distances, synth_profile, "r-s", markersize=4, label="Synthetic", linewidth=1.5)
    ax.fill_between(distances, real_profile, synth_profile, alpha=0.15, color="gray")

    ax.set_xlabel("Normalized Distance from Center")
    ax.set_ylabel("Mean Attention")
    ax.set_title("Radial Attention Profile")
    ax.legend(loc="best")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, "gradcam_radial_profiles")
    plt.close(fig)


def plot_gradcam_results(
    aggregated: dict[str, AggregatedHeatmaps],
    results: list,
    output_dir: Path,
    cfg: DictConfig,
) -> None:
    """Generate all Grad-CAM visualization figures.

    Produces a comprehensive set of plots:
      1. Sample grid: Individual heatmaps overlaid on patches.
      2. Mean heatmaps: Class-level mean and std.
      3. Difference map: Signed and absolute attention difference.
      4. Per-z-bin grid: Mean heatmaps stratified by axial position.
      5. Radial profiles: Attention vs. distance from center.

    Args:
        aggregated: Dictionary of AggregatedHeatmaps keyed by group.
        results: List of individual GradCAMResult instances (for sample grid).
        output_dir: Directory to save all figures.
        cfg: Master configuration (currently unused, reserved for plot options).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating Grad-CAM visualizations in {output_dir}")

    # 1. Sample grid
    _plot_sample_grid(results, output_dir)

    # 2. Mean heatmaps for real and synthetic
    _plot_mean_heatmaps(aggregated, output_dir)

    # 3. Difference map
    _plot_difference_map(aggregated, output_dir)

    # 4. Per-z-bin heatmap grid
    _plot_per_zbin_grid(aggregated, output_dir)

    # 5. Radial profile comparison
    _plot_radial_profiles(aggregated, output_dir)

    logger.info("Grad-CAM visualization generation complete.")
