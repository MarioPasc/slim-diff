"""Texture analysis for real vs synthetic MRI comparison.

Compares texture characteristics between real and synthetic FLAIR images
and lesion masks using Gray-Level Co-occurrence Matrices (GLCM),
Local Binary Patterns (LBP), and gradient magnitude statistics.

References:
    Haralick, R.M., Shanmugam, K., & Dinstein, I. (1973). "Textural Features
        for Image Classification." IEEE Trans. Systems, Man, Cybernetics.
    Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). "Multiresolution
        Gray-Scale and Rotation Invariant Texture Classification with Local
        Binary Patterns." IEEE Trans. PAMI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from scipy import ndimage
from scipy.stats import ks_2samp
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    load_patches,
    save_csv,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)

CHANNEL_NAMES = {0: "image", 1: "mask"}

GLCM_PROPERTIES = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
]


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size between two samples.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        Cohen's d (positive means x > y).
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def _quantize_to_levels(image: np.ndarray, n_levels: int) -> np.ndarray:
    """Quantize a float image to integer levels for GLCM computation.

    Maps image values from their range to [0, n_levels-1].

    Args:
        image: 2D float array.
        n_levels: Number of quantization levels.

    Returns:
        2D uint8 array with values in [0, n_levels-1].
    """
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min < 1e-10:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - img_min) / (img_max - img_min)
    quantized = np.clip(
        (normalized * (n_levels - 1)).astype(np.int32), 0, n_levels - 1
    )
    return quantized.astype(np.uint8)


def compute_glcm_features(
    image: np.ndarray,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
    n_levels: int = 64,
) -> dict[str, float]:
    """Compute GLCM-based texture features for a single image.

    Extracts Haralick features averaged over all distance-angle combinations,
    plus Shannon entropy of the GLCM.

    Args:
        image: 2D float array (single-channel image).
        distances: List of pixel distances for co-occurrence (default: [1, 2, 4]).
        angles: List of angles in radians (default: 4 canonical angles).
        n_levels: Number of gray levels for quantization.

    Returns:
        Dictionary with keys: contrast, dissimilarity, homogeneity,
        energy, correlation, entropy.
    """
    if distances is None:
        distances = [1, 2, 4]
    if angles is None:
        angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    quantized = _quantize_to_levels(image, n_levels)

    glcm = graycomatrix(
        quantized,
        distances=distances,
        angles=angles,
        levels=n_levels,
        symmetric=True,
        normed=True,
    )

    features: dict[str, float] = {}
    for prop in GLCM_PROPERTIES:
        values = graycoprops(glcm, prop)
        features[prop] = float(np.mean(values))

    # Compute entropy from the normalized GLCM
    glcm_flat = glcm.astype(np.float64)
    glcm_sum = glcm_flat.sum()
    if glcm_sum > 0:
        glcm_norm = glcm_flat / glcm_sum
        nonzero = glcm_norm > 0
        entropy = -np.sum(glcm_norm[nonzero] * np.log2(glcm_norm[nonzero]))
    else:
        entropy = 0.0
    features["entropy"] = float(entropy)

    return features


def compute_lbp_histogram(
    image: np.ndarray,
    radius: int = 1,
    n_points: int = 8,
) -> np.ndarray:
    """Compute the normalized Local Binary Pattern histogram.

    Uses the 'uniform' LBP method which groups non-uniform patterns into
    a single bin, resulting in n_points + 2 bins.

    Args:
        image: 2D float array (single-channel image).
        radius: Radius of the circular LBP neighborhood.
        n_points: Number of sampling points on the circle.

    Returns:
        Normalized histogram of LBP codes, shape (n_points + 2,).
    """
    # LBP expects uint8 or similar; scale to [0, 255]
    img_min, img_max = image.min(), image.max()
    if img_max - img_min < 1e-10:
        scaled = np.zeros_like(image, dtype=np.float64)
    else:
        scaled = (image - img_min) / (img_max - img_min) * 255.0

    lbp = local_binary_pattern(scaled, n_points, radius, method="uniform")

    # Uniform LBP has n_points + 2 unique codes
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float64)


def compute_gradient_magnitude_stats(images: np.ndarray) -> dict[str, Any]:
    """Compute gradient magnitude distribution statistics across images.

    Uses Sobel operators to compute image gradients and characterizes the
    distribution of gradient magnitudes.

    Args:
        images: Array of shape (N, H, W), single-channel images.

    Returns:
        Dictionary with:
            - mean, std, median, p95, p99: Summary statistics.
            - per_sample_means: Mean gradient magnitude per sample.
    """
    per_sample_means = np.zeros(images.shape[0], dtype=np.float64)

    all_magnitudes = []
    for i in range(images.shape[0]):
        gx = ndimage.sobel(images[i], axis=1).astype(np.float64)
        gy = ndimage.sobel(images[i], axis=0).astype(np.float64)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        per_sample_means[i] = np.mean(mag)
        all_magnitudes.append(mag.ravel())

    all_mags = np.concatenate(all_magnitudes)
    stats = {
        "mean": float(np.mean(all_mags)),
        "std": float(np.std(all_mags)),
        "median": float(np.median(all_mags)),
        "p95": float(np.percentile(all_mags, 95)),
        "p99": float(np.percentile(all_mags, 99)),
        "per_sample_means": per_sample_means,
    }
    return stats


def texture_comparison(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    channel_idx: int = 0,
    cfg: DictConfig | None = None,
) -> dict[str, Any]:
    """Run full texture comparison between real and synthetic patches.

    Computes GLCM features, LBP histograms, and gradient statistics for
    both sets, then performs KS tests and computes Cohen's d effect sizes.

    Args:
        real_patches: Real patches of shape (N, 2, H, W).
        synth_patches: Synthetic patches of shape (M, 2, H, W).
        channel_idx: Which channel to analyze (0=image, 1=mask).
        cfg: Optional feature_probes.texture config for parameters.

    Returns:
        Dictionary with GLCM, LBP, and gradient comparison results
        including KS statistics, p-values, and effect sizes.
    """
    channel_name = CHANNEL_NAMES.get(channel_idx, f"channel_{channel_idx}")
    logger.info(
        f"Running texture analysis on {channel_name} channel: "
        f"real={real_patches.shape[0]}, synth={synth_patches.shape[0]}"
    )

    # Extract parameters from config
    if cfg is not None:
        distances = list(cfg.get("glcm_distances", [1, 2, 4]))
        angles = list(cfg.get("glcm_angles", [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]))
        n_levels = int(cfg.get("glcm_levels", 64))
        lbp_radii = list(cfg.get("lbp_radii", [1, 2, 3]))
        lbp_points = list(cfg.get("lbp_points", [8, 16, 24]))
    else:
        distances = [1, 2, 4]
        angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        n_levels = 64
        lbp_radii = [1, 2, 3]
        lbp_points = [8, 16, 24]

    real_imgs = real_patches[:, channel_idx, :, :]
    synth_imgs = synth_patches[:, channel_idx, :, :]

    # --- GLCM Features ---
    logger.info("  Computing GLCM features...")
    real_glcm_features = []
    for i in range(real_imgs.shape[0]):
        feats = compute_glcm_features(real_imgs[i], distances, angles, n_levels)
        real_glcm_features.append(feats)

    synth_glcm_features = []
    for i in range(synth_imgs.shape[0]):
        feats = compute_glcm_features(synth_imgs[i], distances, angles, n_levels)
        synth_glcm_features.append(feats)

    # Aggregate GLCM results with statistical tests
    glcm_properties = GLCM_PROPERTIES + ["entropy"]
    glcm_results: dict[str, Any] = {}

    for prop in glcm_properties:
        real_vals = np.array([f[prop] for f in real_glcm_features])
        synth_vals = np.array([f[prop] for f in synth_glcm_features])

        ks_stat, ks_pval = ks_2samp(real_vals, synth_vals)
        effect_size = _cohens_d(real_vals, synth_vals)

        glcm_results[prop] = {
            "real_mean": float(np.mean(real_vals)),
            "real_std": float(np.std(real_vals)),
            "synth_mean": float(np.mean(synth_vals)),
            "synth_std": float(np.std(synth_vals)),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "cohens_d": float(effect_size),
            "real_values": real_vals.tolist(),
            "synth_values": synth_vals.tolist(),
        }
        logger.info(
            f"    {prop}: real={np.mean(real_vals):.4f}, synth={np.mean(synth_vals):.4f}, "
            f"KS={ks_stat:.4f} (p={ks_pval:.2e}), d={effect_size:.3f}"
        )

    # --- LBP Histograms ---
    logger.info("  Computing LBP histograms...")
    lbp_results: dict[str, Any] = {}

    for radius, n_pts in zip(lbp_radii, lbp_points):
        key = f"r{radius}_p{n_pts}"

        real_hists = np.array([
            compute_lbp_histogram(real_imgs[i], radius, n_pts)
            for i in range(real_imgs.shape[0])
        ])
        synth_hists = np.array([
            compute_lbp_histogram(synth_imgs[i], radius, n_pts)
            for i in range(synth_imgs.shape[0])
        ])

        # Mean histograms
        real_mean_hist = np.mean(real_hists, axis=0)
        synth_mean_hist = np.mean(synth_hists, axis=0)

        # Per-bin KS tests
        n_bins = n_pts + 2
        bin_ks_stats = []
        for b in range(n_bins):
            ks_s, _ = ks_2samp(real_hists[:, b], synth_hists[:, b])
            bin_ks_stats.append(float(ks_s))

        lbp_results[key] = {
            "radius": radius,
            "n_points": n_pts,
            "real_mean_histogram": real_mean_hist.tolist(),
            "synth_mean_histogram": synth_mean_hist.tolist(),
            "per_bin_ks_statistic": bin_ks_stats,
            "max_bin_ks": float(np.max(bin_ks_stats)),
            "mean_bin_ks": float(np.mean(bin_ks_stats)),
        }
        logger.info(
            f"    LBP {key}: max_bin_KS={np.max(bin_ks_stats):.4f}, "
            f"mean_bin_KS={np.mean(bin_ks_stats):.4f}"
        )

    # --- Gradient Magnitude ---
    logger.info("  Computing gradient magnitude statistics...")
    real_grad = compute_gradient_magnitude_stats(real_imgs)
    synth_grad = compute_gradient_magnitude_stats(synth_imgs)

    grad_ks_stat, grad_ks_pval = ks_2samp(
        real_grad["per_sample_means"], synth_grad["per_sample_means"]
    )
    grad_effect_size = _cohens_d(
        real_grad["per_sample_means"], synth_grad["per_sample_means"]
    )

    gradient_results = {
        "real": {k: v for k, v in real_grad.items() if k != "per_sample_means"},
        "synth": {k: v for k, v in synth_grad.items() if k != "per_sample_means"},
        "ks_statistic": float(grad_ks_stat),
        "ks_pvalue": float(grad_ks_pval),
        "cohens_d": float(grad_effect_size),
        "real_per_sample_means": real_grad["per_sample_means"].tolist(),
        "synth_per_sample_means": synth_grad["per_sample_means"].tolist(),
    }
    logger.info(
        f"    Gradient: real_mean={real_grad['mean']:.4f}, "
        f"synth_mean={synth_grad['mean']:.4f}, "
        f"KS={grad_ks_stat:.4f} (p={grad_ks_pval:.2e})"
    )

    return {
        "channel": channel_name,
        "n_real": int(real_patches.shape[0]),
        "n_synth": int(synth_patches.shape[0]),
        "glcm": glcm_results,
        "lbp": lbp_results,
        "gradient": gradient_results,
    }


def _plot_glcm_violins(
    result: dict,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Plot violin plots of GLCM features for real vs synthetic."""
    glcm = result["glcm"]
    properties = [p for p in GLCM_PROPERTIES + ["entropy"] if p in glcm]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for idx, prop in enumerate(properties):
        if idx >= len(axes):
            break
        ax = axes[idx]
        data = glcm[prop]

        real_vals = np.array(data["real_values"])
        synth_vals = np.array(data["synth_values"])

        plot_data = {
            "value": np.concatenate([real_vals, synth_vals]),
            "source": ["Real"] * len(real_vals) + ["Synthetic"] * len(synth_vals),
        }

        sns.violinplot(
            x="source", y="value", data=plot_data, ax=ax,
            palette={"Real": "steelblue", "Synthetic": "coral"},
            inner="quartile", cut=0,
        )
        ax.set_title(
            f"{prop}\nKS={data['ks_statistic']:.3f}, d={data['cohens_d']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("")
        ax.set_ylabel(prop.capitalize())

    # Remove unused axes
    for idx in range(len(properties), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"GLCM Features: {result['channel']} channel", fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, f"glcm_violins_{result['channel']}", formats=formats, dpi=dpi)
    plt.close(fig)


def _plot_lbp_histograms(
    result: dict,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Plot LBP histogram comparisons."""
    lbp = result["lbp"]
    n_configs = len(lbp)

    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 4))
    if n_configs == 1:
        axes = [axes]

    for idx, (key, data) in enumerate(sorted(lbp.items())):
        ax = axes[idx]
        real_hist = np.array(data["real_mean_histogram"])
        synth_hist = np.array(data["synth_mean_histogram"])
        n_bins = len(real_hist)
        x = np.arange(n_bins)

        width = 0.4
        ax.bar(x - width / 2, real_hist, width, label="Real", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, synth_hist, width, label="Synthetic", color="coral", alpha=0.8)

        ax.set_xlabel("LBP Code")
        ax.set_ylabel("Density")
        ax.set_title(
            f"LBP (R={data['radius']}, P={data['n_points']})\n"
            f"max KS={data['max_bin_ks']:.3f}",
            fontsize=9,
        )
        ax.legend(frameon=True, fontsize=8)

    fig.suptitle(f"LBP Histograms: {result['channel']} channel", fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, f"lbp_histograms_{result['channel']}", formats=formats, dpi=dpi)
    plt.close(fig)


def _plot_gradient_distributions(
    result: dict,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Plot gradient magnitude distribution comparison."""
    gradient = result["gradient"]

    real_means = np.array(gradient["real_per_sample_means"])
    synth_means = np.array(gradient["synth_per_sample_means"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: histogram of per-sample mean gradient magnitudes
    axes[0].hist(
        real_means, bins=50, alpha=0.6, color="steelblue",
        label="Real", density=True,
    )
    axes[0].hist(
        synth_means, bins=50, alpha=0.6, color="coral",
        label="Synthetic", density=True,
    )
    axes[0].set_xlabel("Mean Gradient Magnitude")
    axes[0].set_ylabel("Density")
    axes[0].set_title(
        f"Gradient Magnitude Distribution\n"
        f"KS={gradient['ks_statistic']:.4f} (p={gradient['ks_pvalue']:.2e})"
    )
    axes[0].legend(frameon=True)
    axes[0].grid(True, alpha=0.3)

    # Right: box plot comparison
    data = {
        "value": np.concatenate([real_means, synth_means]),
        "source": ["Real"] * len(real_means) + ["Synthetic"] * len(synth_means),
    }
    sns.boxplot(
        x="source", y="value", data=data, ax=axes[1],
        palette={"Real": "steelblue", "Synthetic": "coral"},
    )
    axes[1].set_ylabel("Mean Gradient Magnitude")
    axes[1].set_title(f"Cohen's d = {gradient['cohens_d']:.3f}")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Gradient Analysis: {result['channel']} channel", fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(
        fig, output_dir, f"gradient_dist_{result['channel']}",
        formats=formats, dpi=dpi,
    )
    plt.close(fig)


def run_texture_analysis(cfg: DictConfig, experiment_name: str) -> dict:
    """Entry point for texture analysis of an experiment.

    Loads patches, runs texture comparison on the image channel (0),
    generates diagnostic plots, and saves JSON results.

    Args:
        cfg: Full diagnostics configuration (OmegaConf DictConfig).
        experiment_name: Name of the experiment to analyze.

    Returns:
        Dictionary with texture analysis results.
    """
    logger.info(f"=== Texture Analysis: {experiment_name} ===")

    # Load data
    patches_dir = Path(cfg.data.patches_base_dir)
    real_patches, synth_patches, _, _ = load_patches(patches_dir, experiment_name)

    # Get config
    texture_cfg = cfg.feature_probes.texture

    # Output directory
    output_dir = ensure_output_dir(
        Path(cfg.output.base_dir), experiment_name, "texture"
    )
    plot_formats = list(cfg.output.plot_format)
    plot_dpi = int(cfg.output.plot_dpi)

    # Run analysis on image channel (texture is most meaningful for FLAIR)
    result = texture_comparison(
        real_patches, synth_patches,
        channel_idx=0,
        cfg=texture_cfg,
    )

    # Generate plots
    _plot_glcm_violins(result, output_dir, dpi=plot_dpi, formats=plot_formats)
    _plot_lbp_histograms(result, output_dir, dpi=plot_dpi, formats=plot_formats)
    _plot_gradient_distributions(result, output_dir, dpi=plot_dpi, formats=plot_formats)

    # Save results (exclude large arrays for JSON)
    json_result = {
        "experiment": experiment_name,
        "analysis": "texture",
        "channel": result["channel"],
        "n_real": result["n_real"],
        "n_synth": result["n_synth"],
        "glcm": {
            prop: {k: v for k, v in data.items() if k not in ("real_values", "synth_values")}
            for prop, data in result["glcm"].items()
        },
        "lbp": {
            key: {k: v for k, v in data.items()}
            for key, data in result["lbp"].items()
        },
        "gradient": {
            k: v for k, v in result["gradient"].items()
            if k not in ("real_per_sample_means", "synth_per_sample_means")
        },
    }
    save_result_json(json_result, output_dir / "texture_results.json")

    # Save CSV: GLCM and gradient summary for inter-experiment analysis
    csv_rows = []
    for prop, data in result["glcm"].items():
        csv_rows.append({
            "experiment": experiment_name,
            "channel": result["channel"],
            "feature_type": "glcm",
            "feature": prop,
            "real_mean": data["real_mean"],
            "real_std": data["real_std"],
            "synth_mean": data["synth_mean"],
            "synth_std": data["synth_std"],
            "ks_statistic": data["ks_statistic"],
            "ks_pvalue": data["ks_pvalue"],
            "cohens_d": data["cohens_d"],
        })
    # Add gradient magnitude
    grad = result["gradient"]
    csv_rows.append({
        "experiment": experiment_name,
        "channel": result["channel"],
        "feature_type": "gradient",
        "feature": "magnitude_mean",
        "real_mean": grad["real"]["mean"],
        "real_std": grad["real"]["std"],
        "synth_mean": grad["synth"]["mean"],
        "synth_std": grad["synth"]["std"],
        "ks_statistic": grad["ks_statistic"],
        "ks_pvalue": grad["ks_pvalue"],
        "cohens_d": grad["cohens_d"],
    })
    # Add LBP summary
    for key, lbp_data in result["lbp"].items():
        csv_rows.append({
            "experiment": experiment_name,
            "channel": result["channel"],
            "feature_type": "lbp",
            "feature": key,
            "real_mean": float(np.mean(lbp_data["real_mean_histogram"])),
            "real_std": 0.0,
            "synth_mean": float(np.mean(lbp_data["synth_mean_histogram"])),
            "synth_std": 0.0,
            "ks_statistic": lbp_data["max_bin_ks"],
            "ks_pvalue": float("nan"),
            "cohens_d": float("nan"),
        })
    save_csv(pd.DataFrame(csv_rows), output_dir / "texture_summary.csv")

    logger.info(f"Texture analysis complete for '{experiment_name}'")
    return result
