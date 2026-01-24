"""Distribution comparison tests for real vs. synthetic MRI patches.

Compares intensity distributions between real and synthetic data using
Kolmogorov-Smirnov tests, Wasserstein (Earth Mover's) distance, and
per-tissue-type stratified analysis. Generates histograms, Q-Q plots,
and summary statistics.

Typical usage:
    results = run_distribution_tests(cfg, "velocity_lp_1.5")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import DictConfig
from scipy.stats import ks_2samp, wasserstein_distance

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    load_patches,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)


def compare_intensity_distributions(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    channel_idx: int = 0,
    n_bins: int = 200,
    region_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare the intensity distributions of real and synthetic patches.

    Extracts all pixel values (optionally within a spatial mask) and computes
    the two-sample Kolmogorov-Smirnov statistic and Wasserstein distance.

    Args:
        real_patches: Real patches, shape (N_real, 2, H, W), float32.
        synth_patches: Synthetic patches, shape (N_synth, 2, H, W), float32.
        channel_idx: Channel to analyze (0=image, 1=mask).
        n_bins: Number of histogram bins for visualization.
        region_mask: Optional boolean mask (H, W) to restrict analysis to
            specific spatial regions. If None, all pixels are used.

    Returns:
        Dictionary with:
            - ks_statistic: KS test statistic
            - ks_pvalue: KS test p-value
            - wasserstein: Earth Mover's distance
            - real_mean, real_std: moments of real distribution
            - synth_mean, synth_std: moments of synthetic distribution
            - histogram_edges: bin edges for plotting
            - real_hist: normalized histogram counts for real
            - synth_hist: normalized histogram counts for synth
            - n_pixels_real: total number of pixels analyzed (real)
            - n_pixels_synth: total number of pixels analyzed (synth)
    """
    real_channel = real_patches[:, channel_idx, :, :]  # (N, H, W)
    synth_channel = synth_patches[:, channel_idx, :, :]

    if region_mask is not None:
        real_values = real_channel[:, region_mask].ravel()
        synth_values = synth_channel[:, region_mask].ravel()
    else:
        real_values = real_channel.ravel()
        synth_values = synth_channel.ravel()

    # Remove NaN/inf values
    real_values = real_values[np.isfinite(real_values)]
    synth_values = synth_values[np.isfinite(synth_values)]

    # Statistical tests
    ks_stat, ks_pvalue = ks_2samp(real_values, synth_values)

    # Subsample for Wasserstein if arrays are very large (memory-efficient)
    max_wasserstein_samples = 500_000
    rng = np.random.default_rng(42)
    if len(real_values) > max_wasserstein_samples:
        real_sub = rng.choice(real_values, max_wasserstein_samples, replace=False)
    else:
        real_sub = real_values
    if len(synth_values) > max_wasserstein_samples:
        synth_sub = rng.choice(synth_values, max_wasserstein_samples, replace=False)
    else:
        synth_sub = synth_values

    w_dist = wasserstein_distance(real_sub, synth_sub)

    # Compute histograms
    combined_range = (
        min(real_values.min(), synth_values.min()),
        max(real_values.max(), synth_values.max()),
    )
    real_hist, edges = np.histogram(
        real_values, bins=n_bins, range=combined_range, density=True
    )
    synth_hist, _ = np.histogram(
        synth_values, bins=n_bins, range=combined_range, density=True
    )

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "wasserstein": float(w_dist),
        "real_mean": float(real_values.mean()),
        "real_std": float(real_values.std()),
        "synth_mean": float(synth_values.mean()),
        "synth_std": float(synth_values.std()),
        "histogram_edges": edges,
        "real_hist": real_hist,
        "synth_hist": synth_hist,
        "n_pixels_real": int(len(real_values)),
        "n_pixels_synth": int(len(synth_values)),
    }


def per_tissue_distributions(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    lesion_thresh: float = 0.3,
    brain_thresh: float = -0.8,
    bg_thresh: float = -0.95,
) -> dict[str, dict[str, Any]]:
    """Compare image intensity distributions stratified by tissue type.

    Tissue segmentation uses the mask channel (channel 1) to define regions:
      - Lesion: mask > lesion_thresh
      - Brain (non-lesion): bg_thresh < mask <= lesion_thresh AND image > brain_thresh
      - Background: mask <= bg_thresh OR image <= brain_thresh

    For each tissue type, the image channel (channel 0) intensity distribution
    is compared between real and synthetic.

    Args:
        real_patches: Real patches, shape (N_real, 2, H, W).
        synth_patches: Synthetic patches, shape (N_synth, 2, H, W).
        lesion_thresh: Mask value threshold above which a pixel is lesion.
        brain_thresh: Image intensity threshold for brain tissue.
        bg_thresh: Mask/image threshold below which a pixel is background.

    Returns:
        Dictionary mapping tissue name to comparison results dict.
    """
    tissue_defs = {
        "lesion": {
            "description": "Lesion tissue (mask > threshold)",
        },
        "brain": {
            "description": "Non-lesion brain parenchyma",
        },
        "background": {
            "description": "Background (outside brain or very low signal)",
        },
    }

    results: dict[str, dict[str, Any]] = {}

    for tissue_name in tissue_defs:
        # Build per-sample tissue masks
        real_images = real_patches[:, 0, :, :]
        real_masks = real_patches[:, 1, :, :]
        synth_images = synth_patches[:, 0, :, :]
        synth_masks = synth_patches[:, 1, :, :]

        if tissue_name == "lesion":
            real_region = real_masks > lesion_thresh
            synth_region = synth_masks > lesion_thresh
        elif tissue_name == "brain":
            real_region = (
                (real_masks <= lesion_thresh)
                & (real_masks > bg_thresh)
                & (real_images > brain_thresh)
            )
            synth_region = (
                (synth_masks <= lesion_thresh)
                & (synth_masks > bg_thresh)
                & (synth_images > brain_thresh)
            )
        else:  # background
            real_region = (real_masks <= bg_thresh) | (real_images <= brain_thresh)
            synth_region = (synth_masks <= bg_thresh) | (synth_images <= brain_thresh)

        # Extract pixel values within region
        real_values = real_images[real_region]
        synth_values = synth_images[synth_region]

        if len(real_values) < 10 or len(synth_values) < 10:
            logger.warning(
                f"Tissue '{tissue_name}': too few pixels "
                f"(real={len(real_values)}, synth={len(synth_values)}). Skipping."
            )
            results[tissue_name] = {
                "skipped": True,
                "reason": "insufficient pixels",
                "n_pixels_real": int(len(real_values)),
                "n_pixels_synth": int(len(synth_values)),
            }
            continue

        # Statistical tests
        ks_stat, ks_pvalue = ks_2samp(real_values, synth_values)

        # Subsample for Wasserstein
        max_samples = 500_000
        rng = np.random.default_rng(42)
        real_sub = (
            rng.choice(real_values, max_samples, replace=False)
            if len(real_values) > max_samples
            else real_values
        )
        synth_sub = (
            rng.choice(synth_values, max_samples, replace=False)
            if len(synth_values) > max_samples
            else synth_values
        )
        w_dist = wasserstein_distance(real_sub, synth_sub)

        results[tissue_name] = {
            "skipped": False,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "wasserstein": float(w_dist),
            "real_mean": float(real_values.mean()),
            "real_std": float(real_values.std()),
            "synth_mean": float(synth_values.mean()),
            "synth_std": float(synth_values.std()),
            "n_pixels_real": int(len(real_values)),
            "n_pixels_synth": int(len(synth_values)),
        }

        logger.info(
            f"Tissue '{tissue_name}': KS={ks_stat:.4f} (p={ks_pvalue:.2e}), "
            f"Wasserstein={w_dist:.5f}"
        )

    return results


def _plot_intensity_histograms(
    dist_result: dict[str, Any],
    output_dir: Path,
    channel_label: str,
    title_suffix: str = "",
) -> None:
    """Plot overlaid intensity histograms for real and synthetic."""
    fig, ax = plt.subplots(figsize=(8, 5))
    edges = dist_result["histogram_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.plot(centers, dist_result["real_hist"], label="Real", color="steelblue", lw=1.5)
    ax.plot(
        centers, dist_result["synth_hist"],
        label="Synthetic", color="coral", lw=1.5, linestyle="--",
    )
    ax.fill_between(centers, dist_result["real_hist"], alpha=0.2, color="steelblue")
    ax.fill_between(centers, dist_result["synth_hist"], alpha=0.2, color="coral")

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    title = f"Intensity distribution ({channel_label})"
    if title_suffix:
        title += f" - {title_suffix}"
    ax.set_title(title)
    ax.legend(loc="upper right")

    # Add test statistics as text
    textstr = (
        f"KS = {dist_result['ks_statistic']:.4f} "
        f"(p = {dist_result['ks_pvalue']:.2e})\n"
        f"Wasserstein = {dist_result['wasserstein']:.5f}"
    )
    ax.text(
        0.02, 0.95, textstr, transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    suffix = f"_{title_suffix.lower().replace(' ', '_')}" if title_suffix else ""
    save_figure(fig, output_dir, f"histogram_{channel_label}{suffix}")
    plt.close(fig)


def _plot_qq(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    channel_idx: int,
    output_dir: Path,
    channel_label: str,
    n_quantiles: int = 200,
) -> None:
    """Generate a Q-Q plot comparing real and synthetic quantiles."""
    real_values = real_patches[:, channel_idx, :, :].ravel()
    synth_values = synth_patches[:, channel_idx, :, :].ravel()

    # Remove non-finite values
    real_values = real_values[np.isfinite(real_values)]
    synth_values = synth_values[np.isfinite(synth_values)]

    quantile_probs = np.linspace(0.001, 0.999, n_quantiles)
    real_quantiles = np.quantile(real_values, quantile_probs)
    synth_quantiles = np.quantile(synth_values, quantile_probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(real_quantiles, synth_quantiles, s=8, alpha=0.7, color="teal")

    # Reference line (y = x)
    lims = [
        min(real_quantiles.min(), synth_quantiles.min()),
        max(real_quantiles.max(), synth_quantiles.max()),
    ]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="y = x")

    ax.set_xlabel("Real quantiles")
    ax.set_ylabel("Synthetic quantiles")
    ax.set_title(f"Q-Q plot ({channel_label})")
    ax.legend()
    ax.set_aspect("equal")
    save_figure(fig, output_dir, f"qq_plot_{channel_label}")
    plt.close(fig)


def _plot_per_tissue_summary(
    tissue_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Plot a bar chart summarizing per-tissue test statistics."""
    tissues = [t for t in tissue_results if not tissue_results[t].get("skipped", False)]
    if not tissues:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # KS statistics
    ks_vals = [tissue_results[t]["ks_statistic"] for t in tissues]
    axes[0].barh(tissues, ks_vals, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("KS statistic")
    axes[0].set_title("KS test statistic by tissue")

    # Wasserstein distances
    w_vals = [tissue_results[t]["wasserstein"] for t in tissues]
    axes[1].barh(tissues, w_vals, color="coral", alpha=0.8)
    axes[1].set_xlabel("Wasserstein distance")
    axes[1].set_title("Wasserstein distance by tissue")

    fig.tight_layout()
    save_figure(fig, output_dir, "per_tissue_summary")
    plt.close(fig)


def run_distribution_tests(cfg: DictConfig, experiment_name: str) -> dict[str, Any]:
    """Run the full distribution comparison analysis pipeline.

    Loads patches, compares intensity distributions (overall and per-tissue),
    generates histograms/Q-Q plots, and saves results.

    Args:
        cfg: Full diagnostics configuration (OmegaConf DictConfig).
        experiment_name: Name of the experiment to analyze.

    Returns:
        Dictionary of all computed results.
    """
    # Load data
    real_patches, synth_patches, real_zbins, synth_zbins = load_patches(
        cfg.data.patches_base_dir, experiment_name
    )

    # Configuration
    dist_cfg = cfg.statistical.get("distributions", {})
    n_bins = dist_cfg.get("n_bins", 200)
    do_tissue = dist_cfg.get("tissue_segmentation", True)
    lesion_thresh = dist_cfg.get("lesion_threshold", 0.3)
    brain_thresh = dist_cfg.get("brain_threshold", -0.8)
    bg_thresh = dist_cfg.get("background_threshold", -0.95)
    channels = cfg.statistical.get("channels", [0])

    output_dir = ensure_output_dir(
        cfg.output.base_dir, experiment_name, "distribution_tests"
    )

    channel_names = {0: "image", 1: "mask"}
    all_results: dict[str, Any] = {"experiment": experiment_name}

    for ch in channels:
        ch_label = channel_names.get(ch, f"ch{ch}")
        logger.info(f"Running distribution tests for channel '{ch_label}'...")

        # Overall intensity distribution comparison
        dist_result = compare_intensity_distributions(
            real_patches, synth_patches,
            channel_idx=ch, n_bins=n_bins,
        )
        all_results[ch_label] = {
            "ks_statistic": dist_result["ks_statistic"],
            "ks_pvalue": dist_result["ks_pvalue"],
            "wasserstein": dist_result["wasserstein"],
            "real_mean": dist_result["real_mean"],
            "real_std": dist_result["real_std"],
            "synth_mean": dist_result["synth_mean"],
            "synth_std": dist_result["synth_std"],
            "n_pixels_real": dist_result["n_pixels_real"],
            "n_pixels_synth": dist_result["n_pixels_synth"],
        }

        # Plots
        _plot_intensity_histograms(dist_result, output_dir, ch_label)
        _plot_qq(real_patches, synth_patches, ch, output_dir, ch_label)

    # Per-tissue analysis (image channel only, uses mask for segmentation)
    if do_tissue:
        logger.info("Running per-tissue distribution analysis...")
        tissue_results = per_tissue_distributions(
            real_patches, synth_patches,
            lesion_thresh=lesion_thresh,
            brain_thresh=brain_thresh,
            bg_thresh=bg_thresh,
        )
        all_results["per_tissue"] = tissue_results
        _plot_per_tissue_summary(tissue_results, output_dir)

        # Per-tissue histograms
        for tissue_name, tissue_res in tissue_results.items():
            if tissue_res.get("skipped", False):
                continue
            # Recompute with histogram for plotting
            real_images = real_patches[:, 0, :, :]
            real_masks = real_patches[:, 1, :, :]
            synth_images = synth_patches[:, 0, :, :]
            synth_masks = synth_patches[:, 1, :, :]

            if tissue_name == "lesion":
                real_region = real_masks > lesion_thresh
                synth_region = synth_masks > lesion_thresh
            elif tissue_name == "brain":
                real_region = (
                    (real_masks <= lesion_thresh)
                    & (real_masks > bg_thresh)
                    & (real_images > brain_thresh)
                )
                synth_region = (
                    (synth_masks <= lesion_thresh)
                    & (synth_masks > bg_thresh)
                    & (synth_images > brain_thresh)
                )
            else:
                real_region = (real_masks <= bg_thresh) | (real_images <= brain_thresh)
                synth_region = (synth_masks <= bg_thresh) | (synth_images <= brain_thresh)

            # Extract per-sample values and build histogram data
            real_vals = real_images[real_region]
            synth_vals = synth_images[synth_region]
            combined_range = (
                min(real_vals.min(), synth_vals.min()),
                max(real_vals.max(), synth_vals.max()),
            )
            real_hist, edges = np.histogram(
                real_vals, bins=n_bins, range=combined_range, density=True
            )
            synth_hist, _ = np.histogram(
                synth_vals, bins=n_bins, range=combined_range, density=True
            )
            hist_data = {
                "histogram_edges": edges,
                "real_hist": real_hist,
                "synth_hist": synth_hist,
                "ks_statistic": tissue_res["ks_statistic"],
                "ks_pvalue": tissue_res["ks_pvalue"],
                "wasserstein": tissue_res["wasserstein"],
            }
            _plot_intensity_histograms(
                hist_data, output_dir, "image", title_suffix=tissue_name
            )

    # Save JSON summary
    save_result_json(all_results, output_dir / "distribution_tests_results.json")

    return all_results
