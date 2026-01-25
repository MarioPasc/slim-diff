"""Publication-quality visualization panels for confusion-stratified XAI analysis.

This module generates comprehensive visualization figures suitable for Q1 journal
publications, demonstrating that synthetic images are indistinguishable from real.

Key figures:
1. XAI Panel: Comprehensive 4x4 grid showing all XAI techniques
2. GradCAM Comparison: Input images alongside activation maps
3. FP vs TN Methodology: The key publication figure
4. Feature Space: t-SNE/PCA embeddings with category coloring
5. Z-bin Analysis: Per-anatomical-level breakdown
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.classification.diagnostics.utils import save_figure
from src.classification.diagnostics.xai.confusion_stratified import (
    ConfusionStratifiedResults,
    CategoryAnalysisResult,
)

logger = logging.getLogger(__name__)

# Color scheme for categories
CATEGORY_COLORS = {
    "TP": "#2ecc71",  # Green - real correctly classified
    "TN": "#e74c3c",  # Red - synthetic correctly classified
    "FP": "#3498db",  # Blue - synthetic classified as real (best quality)
    "FN": "#f39c12",  # Orange - real classified as synthetic
}

CATEGORY_LABELS = {
    "TP": "True Positive (Real\u2192Real)",
    "TN": "True Negative (Synth\u2192Synth)",
    "FP": "False Positive (Synth\u2192Real)",
    "FN": "False Negative (Real\u2192Synth)",
}


def _create_diverging_cmap() -> LinearSegmentedColormap:
    """Create a diverging colormap for difference maps."""
    colors = ["#2166ac", "#f7f7f7", "#b2182b"]  # Blue - White - Red
    return LinearSegmentedColormap.from_list("diverging", colors)


def plot_gradcam_comparison_panel(
    results: ConfusionStratifiedResults,
    samples_by_category: dict,
    output_dir: Path,
    n_samples_per_category: int = 4,
) -> None:
    """Generate GradCAM comparison panel showing inputs and activations.

    Creates a figure with one row per sample, showing:
    - Column 1: FLAIR image
    - Column 2: Lesion mask
    - Column 3: GradCAM overlay on image
    - Column 4: GradCAM heatmap

    Args:
        results: ConfusionStratifiedResults with GradCAM data.
        samples_by_category: Dict with (patches, z_bins, indices) per category.
        output_dir: Output directory for figures.
        n_samples_per_category: Number of samples to show per category.
    """
    output_dir = Path(output_dir)

    for cat_name in ["TP", "TN", "FP", "FN"]:
        cat_result = results.categories.get(cat_name)
        if cat_result is None or cat_result.n_samples == 0:
            continue

        patches, z_bins, _ = samples_by_category.get(cat_name, (None, None, None))
        if patches is None or len(patches) == 0:
            continue

        gradcam_results = cat_result.gradcam_results
        if not gradcam_results:
            continue

        # Select samples (evenly spaced or first N)
        # Use minimum of gradcam_results and patches to avoid index out of bounds
        n_available = min(len(gradcam_results), len(patches))
        n_show = min(n_samples_per_category, n_available)
        if n_show < 1:
            continue

        indices = np.linspace(0, n_available - 1, n_show, dtype=int)

        fig, axes = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
        if n_show == 1:
            axes = axes[np.newaxis, :]

        for row, idx in enumerate(indices):
            patch = patches[idx]  # (2, H, W)
            heatmap = gradcam_results[idx].heatmap
            z_bin = z_bins[idx]
            prob = gradcam_results[idx].prediction

            image = patch[0]  # FLAIR
            mask = patch[1]   # Lesion mask

            # Column 0: FLAIR image
            axes[row, 0].imshow(image, cmap="gray", vmin=-1, vmax=1)
            axes[row, 0].set_title(f"FLAIR (z={z_bin})", fontsize=10)
            axes[row, 0].axis("off")

            # Column 1: Lesion mask
            axes[row, 1].imshow(mask, cmap="gray", vmin=-1, vmax=1)
            axes[row, 1].set_title("Mask", fontsize=10)
            axes[row, 1].axis("off")

            # Column 2: GradCAM overlay
            axes[row, 2].imshow(image, cmap="gray", vmin=-1, vmax=1)
            axes[row, 2].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            axes[row, 2].set_title(f"GradCAM (p={prob:.2f})", fontsize=10)
            axes[row, 2].axis("off")

            # Column 3: Heatmap only
            im = axes[row, 3].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
            axes[row, 3].set_title("Attention", fontsize=10)
            axes[row, 3].axis("off")

            if row == 0:
                divider = make_axes_locatable(axes[row, 3])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

        plt.suptitle(
            f"{CATEGORY_LABELS[cat_name]} Samples",
            fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        save_figure(fig, output_dir, f"gradcam_samples_{cat_name.lower()}")
        plt.close(fig)

    logger.info(f"Generated GradCAM comparison panels in {output_dir}")


def plot_mean_gradcam_by_category(
    results: ConfusionStratifiedResults,
    output_dir: Path,
) -> None:
    """Plot mean GradCAM heatmaps for all four categories.

    Creates a 2x2 grid showing mean attention patterns for each category.

    Args:
        results: ConfusionStratifiedResults with computed mean heatmaps.
        output_dir: Output directory.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    category_order = ["TP", "TN", "FP", "FN"]

    for ax, cat_name in zip(axes, category_order):
        cat_result = results.categories.get(cat_name)
        if cat_result is None or cat_result.mean_gradcam is None:
            ax.text(0.5, 0.5, f"No data\n({cat_name})",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        im = ax.imshow(cat_result.mean_gradcam, cmap="hot", vmin=0, vmax=1)
        ax.set_title(
            f"{CATEGORY_LABELS[cat_name]}\n(n={cat_result.n_samples})",
            fontsize=11, color=CATEGORY_COLORS[cat_name]
        )
        ax.axis("off")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.suptitle("Mean GradCAM Attention by Category", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "mean_gradcam_by_category")
    plt.close(fig)


def plot_attention_difference(
    results: ConfusionStratifiedResults,
    output_dir: Path,
) -> None:
    """Plot attention difference maps between key category pairs.

    Shows TN - FP difference (what artifacts TN samples have that FP lack).

    Args:
        results: ConfusionStratifiedResults with computed comparisons.
        output_dir: Output directory.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # FP vs TN comparison (primary)
    comp = results.comparisons.get("FP_vs_TN")
    if comp is None or comp.attention_difference is None:
        logger.warning("No FP vs TN comparison available for attention difference plot")
        plt.close(fig)
        return

    diff_map = comp.attention_difference
    vmax = max(abs(diff_map.min()), abs(diff_map.max()))

    # Signed difference
    cmap_div = _create_diverging_cmap()
    im0 = axes[0].imshow(diff_map, cmap=cmap_div, vmin=-vmax, vmax=vmax)
    axes[0].set_title("Attention Difference\n(FP - TN)", fontsize=12)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Absolute difference
    im1 = axes[1].imshow(comp.abs_attention_difference, cmap="hot", vmin=0)
    axes[1].set_title("Absolute Difference", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Metrics text
    axes[2].axis("off")
    metrics_text = (
        f"Comparison Metrics\n"
        f"{'=' * 30}\n\n"
        f"Cosine Similarity: {comp.cosine_similarity:.4f}\n"
        f"Spatial Correlation: {comp.spatial_correlation:.4f}\n"
        f"  (p = {comp.spatial_correlation_pvalue:.2e})\n\n"
        f"Feature Distance: {comp.feature_distance:.4f}\n"
        f"KS Statistic: {comp.ks_statistic:.4f}\n"
        f"  (p = {comp.ks_pvalue:.2e})\n\n"
        f"Channel Contrib. Delta: {comp.channel_contribution_delta:+.4f}\n"
        f"  (positive = more image reliance)"
    )
    axes[2].text(0.1, 0.5, metrics_text, transform=axes[2].transAxes,
                 fontsize=11, verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle(
        "Classifier Attention: FP (Synthetic\u2192Real) vs TN (Synthetic\u2192Synthetic)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_figure(fig, output_dir, "attention_difference_fp_vs_tn")
    plt.close(fig)


def plot_radial_profiles(
    results: ConfusionStratifiedResults,
    output_dir: Path,
) -> None:
    """Plot radial attention profiles for all categories.

    Shows how attention varies with distance from patch center.

    Args:
        results: ConfusionStratifiedResults with radial profiles.
        output_dir: Output directory.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    has_data = False
    for cat_name in ["TP", "TN", "FP", "FN"]:
        cat_result = results.categories.get(cat_name)
        if cat_result is None or cat_result.radial_profile is None:
            continue

        profile = cat_result.radial_profile
        distances = np.linspace(0, 1, len(profile))
        ax.plot(distances, profile, color=CATEGORY_COLORS[cat_name],
                label=CATEGORY_LABELS[cat_name], linewidth=2, marker="o", markersize=4)
        has_data = True

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Normalized Distance from Center", fontsize=12)
    ax.set_ylabel("Mean Attention", fontsize=12)
    ax.set_title("Radial Attention Profiles by Category", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, "radial_attention_profiles")
    plt.close(fig)


def plot_channel_contributions(
    results: ConfusionStratifiedResults,
    output_dir: Path,
) -> None:
    """Plot channel contribution comparison across categories.

    Shows stacked bar chart of image vs mask channel contributions.

    Args:
        results: ConfusionStratifiedResults with channel contributions.
        output_dir: Output directory.
    """
    categories = []
    image_fractions = []
    mask_fractions = []

    for cat_name in ["TP", "TN", "FP", "FN"]:
        cat_result = results.categories.get(cat_name)
        if cat_result is None or not cat_result.channel_contributions:
            continue
        categories.append(cat_name)
        image_fractions.append(cat_result.channel_contributions.get("image_fraction", 0.5))
        mask_fractions.append(cat_result.channel_contributions.get("mask_fraction", 0.5))

    if not categories:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    x = np.arange(len(categories))
    width = 0.6

    colors_bar = [CATEGORY_COLORS[c] for c in categories]

    # Stacked bars
    bars1 = ax.bar(x, image_fractions, width, label="Image Channel",
                   color=[c for c in colors_bar], alpha=0.8)
    bars2 = ax.bar(x, mask_fractions, width, bottom=image_fractions,
                   label="Mask Channel", color=[c for c in colors_bar], alpha=0.4,
                   hatch="//")

    ax.set_ylabel("Contribution Fraction", fontsize=12)
    ax.set_title("Channel Contributions by Category", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c].split("(")[0].strip() for c in categories],
                       fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Add value labels
    for i, (img, mask) in enumerate(zip(image_fractions, mask_fractions)):
        ax.text(i, img / 2, f"{img:.2f}", ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")
        ax.text(i, img + mask / 2, f"{mask:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold")

    # Legend
    legend_elements = [
        Patch(facecolor="gray", alpha=0.8, label="Image Channel"),
        Patch(facecolor="gray", alpha=0.4, hatch="//", label="Mask Channel"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    save_figure(fig, output_dir, "channel_contributions")
    plt.close(fig)


def plot_feature_space(
    results: ConfusionStratifiedResults,
    output_dir: Path,
) -> None:
    """Plot feature space embeddings (t-SNE and PCA).

    Shows how samples from different categories cluster in the learned
    feature space. Key insight: FP samples should cluster with TP (real).

    Args:
        results: ConfusionStratifiedResults with embeddings.
        output_dir: Output directory.
    """
    if results.all_features is None or results.all_labels is None:
        logger.warning("No feature embeddings available for feature space plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (embedding, title) in zip(axes, [
        (results.pca_embedding, "PCA Projection"),
        (results.tsne_embedding, "t-SNE Projection"),
    ]):
        if embedding is None:
            ax.text(0.5, 0.5, f"{title}\n(Not computed)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.axis("off")
            continue

        for cat_name in ["TP", "TN", "FP", "FN"]:
            mask = results.all_labels == cat_name
            if mask.sum() == 0:
                continue
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=CATEGORY_COLORS[cat_name], label=cat_name,
                       alpha=0.6, s=30, edgecolors="white", linewidth=0.5)

        ax.set_xlabel("Component 1", fontsize=11)
        ax.set_ylabel("Component 2", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Feature Space Embeddings by Confusion Category",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    save_figure(fig, output_dir, "feature_space_embeddings")
    plt.close(fig)


def plot_zbin_analysis(
    results: ConfusionStratifiedResults,
    output_dir: Path,
) -> None:
    """Plot per-z-bin breakdown of confusion categories.

    Shows distribution of categories across anatomical levels.

    Args:
        results: ConfusionStratifiedResults with per-zbin counts.
        output_dir: Output directory.
    """
    # Collect all z-bins across categories
    all_zbins = set()
    for cat_result in results.categories.values():
        all_zbins.update(cat_result.per_zbin_counts.keys())

    if not all_zbins:
        return

    zbins = sorted(all_zbins)
    n_zbins = len(zbins)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Stacked bar of counts
    category_order = ["TP", "TN", "FP", "FN"]
    bottom = np.zeros(n_zbins)

    for cat_name in category_order:
        cat_result = results.categories.get(cat_name)
        if cat_result is None:
            continue

        counts = [cat_result.per_zbin_counts.get(zb, 0) for zb in zbins]
        axes[0].bar(range(n_zbins), counts, bottom=bottom,
                    color=CATEGORY_COLORS[cat_name], label=cat_name)
        bottom += np.array(counts)

    axes[0].set_xlabel("Z-bin", fontsize=11)
    axes[0].set_ylabel("Sample Count", fontsize=11)
    axes[0].set_title("Category Distribution by Z-bin", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(0, n_zbins, max(1, n_zbins // 10)))
    axes[0].set_xticklabels([zbins[i] for i in range(0, n_zbins, max(1, n_zbins // 10))])
    axes[0].legend(loc="best", fontsize=9)

    # Right: FP rate per z-bin
    fp_rates = []
    for zb in zbins:
        n_fp = results.categories.get("FP", CategoryAnalysisResult(category="FP", n_samples=0)).per_zbin_counts.get(zb, 0)
        n_tn = results.categories.get("TN", CategoryAnalysisResult(category="TN", n_samples=0)).per_zbin_counts.get(zb, 0)
        total_synth = n_fp + n_tn
        fp_rate = n_fp / total_synth if total_synth > 0 else 0
        fp_rates.append(fp_rate)

    axes[1].bar(range(n_zbins), fp_rates, color=CATEGORY_COLORS["FP"], alpha=0.8)
    axes[1].axhline(results.fp_rate, color="red", linestyle="--",
                    linewidth=2, label=f"Overall FP rate: {results.fp_rate:.3f}")
    axes[1].set_xlabel("Z-bin", fontsize=11)
    axes[1].set_ylabel("FP Rate (Synth\u2192Real)", fontsize=11)
    axes[1].set_title("False Positive Rate by Z-bin", fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(0, n_zbins, max(1, n_zbins // 10)))
    axes[1].set_xticklabels([zbins[i] for i in range(0, n_zbins, max(1, n_zbins // 10))])
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="best", fontsize=10)

    plt.tight_layout()
    save_figure(fig, output_dir, "zbin_analysis")
    plt.close(fig)


def plot_fp_vs_tn_methodology(
    results: ConfusionStratifiedResults,
    samples_by_category: dict,
    output_dir: Path,
    n_examples: int = 3,
) -> None:
    """Generate the key methodology figure for publication.

    This is the main figure demonstrating synthetic image quality, showing:
    - Panel A: Example FP samples (synthetic that fooled classifier)
    - Panel B: Example TN samples (synthetic correctly identified)
    - Panel C: GradCAM difference map
    - Panel D: Feature space visualization
    - Panel E: Statistical summary

    Args:
        results: ConfusionStratifiedResults.
        samples_by_category: Dict with patches per category.
        output_dir: Output directory.
        n_examples: Number of example samples per category.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.8],
                           hspace=0.3, wspace=0.3)

    # Panel A: FP examples (top-left 2x2)
    fp_patches, fp_zbins, _ = samples_by_category.get("FP", (None, None, None))
    fp_gradcam = results.categories.get("FP", CategoryAnalysisResult(category="FP", n_samples=0)).gradcam_results

    for i in range(min(n_examples, 2)):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            if fp_patches is not None and i < len(fp_patches):
                idx = i
                if j == 0:
                    ax.imshow(fp_patches[idx, 0], cmap="gray", vmin=-1, vmax=1)
                    ax.set_title(f"FP Sample {idx + 1}\n(z={fp_zbins[idx]})", fontsize=10,
                                 color=CATEGORY_COLORS["FP"])
                else:
                    if fp_gradcam and idx < len(fp_gradcam):
                        ax.imshow(fp_patches[idx, 0], cmap="gray", vmin=-1, vmax=1)
                        ax.imshow(fp_gradcam[idx].heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
                        ax.set_title(f"GradCAM (p={fp_gradcam[idx].prediction:.2f})", fontsize=10)
            ax.axis("off")

    # Panel B: TN examples (top-right 2x2)
    tn_patches, tn_zbins, _ = samples_by_category.get("TN", (None, None, None))
    tn_gradcam = results.categories.get("TN", CategoryAnalysisResult(category="TN", n_samples=0)).gradcam_results

    for i in range(min(n_examples, 2)):
        for j in range(2, 4):
            ax = fig.add_subplot(gs[i, j])
            if tn_patches is not None and i < len(tn_patches):
                idx = i
                if j == 2:
                    ax.imshow(tn_patches[idx, 0], cmap="gray", vmin=-1, vmax=1)
                    ax.set_title(f"TN Sample {idx + 1}\n(z={tn_zbins[idx]})", fontsize=10,
                                 color=CATEGORY_COLORS["TN"])
                else:
                    if tn_gradcam and idx < len(tn_gradcam):
                        ax.imshow(tn_patches[idx, 0], cmap="gray", vmin=-1, vmax=1)
                        ax.imshow(tn_gradcam[idx].heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
                        ax.set_title(f"GradCAM (p={tn_gradcam[idx].prediction:.2f})", fontsize=10)
            ax.axis("off")

    # Panel C: Attention difference (middle-left)
    ax_diff = fig.add_subplot(gs[2, 0:2])
    comp = results.comparisons.get("FP_vs_TN")
    if comp is not None and comp.attention_difference is not None:
        diff_map = comp.attention_difference
        vmax = max(abs(diff_map.min()), abs(diff_map.max()))
        im = ax_diff.imshow(diff_map, cmap=_create_diverging_cmap(), vmin=-vmax, vmax=vmax)
        ax_diff.set_title("Attention Difference (FP - TN)", fontsize=12, fontweight="bold")
        ax_diff.axis("off")
        plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)

    # Panel D: Feature space (middle-right, first half)
    ax_feat = fig.add_subplot(gs[2, 2])
    if results.tsne_embedding is not None and results.all_labels is not None:
        for cat_name in ["TP", "TN", "FP", "FN"]:
            mask = results.all_labels == cat_name
            if mask.sum() > 0:
                ax_feat.scatter(results.tsne_embedding[mask, 0],
                                results.tsne_embedding[mask, 1],
                                c=CATEGORY_COLORS[cat_name], label=cat_name,
                                alpha=0.6, s=20)
        ax_feat.set_title("Feature Space (t-SNE)", fontsize=11, fontweight="bold")
        ax_feat.legend(loc="best", fontsize=8)
        ax_feat.set_xticks([])
        ax_feat.set_yticks([])

    # Panel E: Summary statistics (middle-right, second half)
    ax_stats = fig.add_subplot(gs[2, 3])
    ax_stats.axis("off")

    summary_text = (
        f"Summary Statistics\n"
        f"{'=' * 25}\n\n"
        f"FP Rate: {results.fp_rate:.1%}\n"
        f"({results.categories.get('FP', CategoryAnalysisResult(category='FP', n_samples=0)).n_samples} of "
        f"{results.categories.get('FP', CategoryAnalysisResult(category='FP', n_samples=0)).n_samples + results.categories.get('TN', CategoryAnalysisResult(category='TN', n_samples=0)).n_samples} synthetic)\n\n"
    )

    if comp is not None:
        summary_text += (
            f"FP-TN Comparison:\n"
            f"  Feature Dist.: {comp.feature_distance:.3f}\n"
            f"  Attn. Corr.: {comp.spatial_correlation:.3f}\n"
            f"  KS Stat.: {comp.ks_statistic:.3f}\n"
        )

    ax_stats.text(0.1, 0.9, summary_text, transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Main title
    plt.suptitle(
        f"Confusion-Stratified XAI Analysis: {results.experiment_name}\n"
        f"FP samples (synthetic classified as real) vs TN samples (correctly classified synthetic)",
        fontsize=14, fontweight="bold", y=0.98
    )

    save_figure(fig, output_dir, "fp_vs_tn_methodology")
    plt.close(fig)

    logger.info(f"Generated FP vs TN methodology figure in {output_dir}")


def generate_xai_panel(
    results: ConfusionStratifiedResults,
    samples_by_category: dict,
    output_dir: Path,
    figure_format: str = "pdf",
) -> None:
    """Generate comprehensive XAI panel for publication.

    Creates a 4x4 grid showing all XAI analysis results.

    Args:
        results: ConfusionStratifiedResults with all analysis.
        samples_by_category: Dict with patches per category.
        output_dir: Output directory.
        figure_format: Output format (pdf, png).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating XAI visualization panels in {output_dir}")

    # Generate all individual plots
    plot_mean_gradcam_by_category(results, output_dir)
    plot_attention_difference(results, output_dir)
    plot_radial_profiles(results, output_dir)
    plot_channel_contributions(results, output_dir)
    plot_feature_space(results, output_dir)
    plot_zbin_analysis(results, output_dir)
    plot_gradcam_comparison_panel(results, samples_by_category, output_dir)
    plot_fp_vs_tn_methodology(results, samples_by_category, output_dir)

    logger.info("XAI panel generation complete")
