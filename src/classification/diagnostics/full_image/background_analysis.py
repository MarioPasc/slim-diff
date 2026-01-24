"""Background region consistency analysis for full 160x160 MRI images.

Compares background pixel behavior between real slices (where background is
exactly -1.0) and synthetic replicas (where float16 quantization and diffusion
model imperfections may introduce deviations).

Key diagnostics:
- Background pixel intensity distribution (delta at -1 for real, possibly spread for synth)
- Spatial maps of background deviation from the expected -1.0
- Noise level in background regions (std above threshold)
- Background fraction comparison between real and synthetic
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import DictConfig

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    load_full_replicas,
    load_real_slices,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)


def identify_background(image: np.ndarray, threshold: float = -0.95) -> np.ndarray:
    """Identify background pixels in a single MRI image.

    Background pixels are those with intensity below the threshold. For real
    data these are exactly -1.0; for synthetic data they may deviate slightly.

    Args:
        image: 2D array of shape (H, W) with pixel intensities in [-1, 1].
        threshold: Intensity threshold below which pixels are considered
            background. Default -0.95 captures the background region while
            excluding dark but in-brain tissue.

    Returns:
        Boolean mask of shape (H, W) where True indicates background pixels.
    """
    return image <= threshold


def analyze_background_consistency(
    real_images: np.ndarray,
    synth_images: np.ndarray,
    threshold: float = -0.95,
    noise_threshold: float = 0.01,
) -> dict[str, Any]:
    """Analyze background region consistency between real and synthetic images.

    Computes comprehensive statistics on background pixel behavior for both
    real and synthetic image sets, focusing on deviations from the expected
    -1.0 background value.

    Args:
        real_images: Real MRI slices, shape (N_real, H, W), float32.
        synth_images: Synthetic replicas, shape (N_synth, H, W), float32.
        threshold: Intensity threshold for background identification.
        noise_threshold: Standard deviation threshold above which background
            is considered to have noise contamination.

    Returns:
        Dictionary containing:
        - real/synth sub-dicts with: mean, std, min, max, n_unique,
          fraction_background, has_noise, deviation_from_minus1
        - comparison metrics: std_ratio, mean_difference
    """
    results: dict[str, Any] = {"threshold": threshold, "noise_threshold": noise_threshold}

    for label, images in [("real", real_images), ("synth", synth_images)]:
        bg_pixels_all = []
        fractions = []

        for img in images:
            bg_mask = identify_background(img, threshold=threshold)
            fraction = bg_mask.sum() / bg_mask.size
            fractions.append(fraction)

            if bg_mask.any():
                bg_pixels_all.append(img[bg_mask])

        if not bg_pixels_all:
            logger.warning(f"No background pixels found for {label} images.")
            results[label] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "n_unique": 0,
                "fraction_background_mean": 0.0,
                "fraction_background_std": 0.0,
                "has_noise": False,
                "deviation_from_minus1": None,
                "n_images": len(images),
            }
            continue

        bg_all = np.concatenate(bg_pixels_all)
        bg_std = float(np.std(bg_all))

        results[label] = {
            "mean": float(np.mean(bg_all)),
            "std": bg_std,
            "min": float(np.min(bg_all)),
            "max": float(np.max(bg_all)),
            "n_unique": int(len(np.unique(bg_all))),
            "fraction_background_mean": float(np.mean(fractions)),
            "fraction_background_std": float(np.std(fractions)),
            "has_noise": bg_std > noise_threshold,
            "deviation_from_minus1": float(np.mean(np.abs(bg_all - (-1.0)))),
            "n_images": len(images),
            "n_background_pixels": len(bg_all),
        }

    # Comparison metrics
    if results.get("real", {}).get("std") is not None and results.get("synth", {}).get("std") is not None:
        real_std = results["real"]["std"]
        synth_std = results["synth"]["std"]
        results["comparison"] = {
            "std_ratio": synth_std / max(real_std, 1e-12),
            "mean_difference": abs(
                results["synth"]["mean"] - results["real"]["mean"]
            ),
            "deviation_ratio": (
                results["synth"]["deviation_from_minus1"]
                / max(results["real"]["deviation_from_minus1"], 1e-12)
            ),
            "unique_values_ratio": (
                results["synth"]["n_unique"] / max(results["real"]["n_unique"], 1)
            ),
        }

    return results


def _plot_background_histogram(
    real_images: np.ndarray,
    synth_images: np.ndarray,
    threshold: float,
    output_dir: Path,
    plot_formats: list[str],
    dpi: int,
) -> None:
    """Plot background intensity histograms for real vs synthetic images.

    For real data, the histogram should be a delta function at -1.0.
    For synthetic data, it may show a spread around -1.0.
    """
    real_bg = []
    for img in real_images:
        bg_mask = identify_background(img, threshold=threshold)
        if bg_mask.any():
            real_bg.append(img[bg_mask])

    synth_bg = []
    for img in synth_images:
        bg_mask = identify_background(img, threshold=threshold)
        if bg_mask.any():
            synth_bg.append(img[bg_mask])

    if not real_bg or not synth_bg:
        logger.warning("Insufficient background pixels for histogram plot.")
        return

    real_bg_arr = np.concatenate(real_bg)
    synth_bg_arr = np.concatenate(synth_bg)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Real background histogram
    ax = axes[0]
    ax.hist(real_bg_arr, bins=200, density=True, alpha=0.7, color="#2196F3", edgecolor="none")
    ax.axvline(-1.0, color="red", linestyle="--", linewidth=1.5, label="Expected (-1.0)")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.set_title(f"Real Background (n={len(real_bg_arr):,})")
    ax.legend()
    ax.set_xlim(-1.05, threshold)

    # Synthetic background histogram
    ax = axes[1]
    ax.hist(synth_bg_arr, bins=200, density=True, alpha=0.7, color="#FF9800", edgecolor="none")
    ax.axvline(-1.0, color="red", linestyle="--", linewidth=1.5, label="Expected (-1.0)")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.set_title(f"Synthetic Background (n={len(synth_bg_arr):,})")
    ax.legend()
    ax.set_xlim(-1.05, threshold)

    plt.suptitle("Background Pixel Intensity Distribution", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "background_intensity_histogram", plot_formats, dpi)
    plt.close(fig)


def _plot_spatial_deviation_map(
    real_images: np.ndarray,
    synth_images: np.ndarray,
    threshold: float,
    output_dir: Path,
    plot_formats: list[str],
    dpi: int,
) -> None:
    """Plot spatial maps of mean background deviation from -1.0.

    Computes the per-pixel mean absolute deviation from -1.0 across all
    images, restricted to background pixels. Highlights spatial patterns
    in background noise (e.g., edge effects from diffusion model).
    """
    h, w = real_images.shape[1], real_images.shape[2]

    # Accumulate deviation and count per pixel for both sets
    maps = {}
    for label, images in [("Real", real_images), ("Synthetic", synth_images)]:
        deviation_sum = np.zeros((h, w), dtype=np.float64)
        count = np.zeros((h, w), dtype=np.float64)

        for img in images:
            bg_mask = identify_background(img, threshold=threshold)
            deviation_sum[bg_mask] += np.abs(img[bg_mask] - (-1.0))
            count[bg_mask] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_deviation = np.where(count > 0, deviation_sum / count, np.nan)
        maps[label] = mean_deviation

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Use the same color scale for both
    valid_values = []
    for m in maps.values():
        valid = m[~np.isnan(m)]
        if len(valid) > 0:
            valid_values.append(valid)

    if not valid_values:
        logger.warning("No valid deviation values for spatial map.")
        plt.close(fig)
        return

    all_valid = np.concatenate(valid_values)
    vmax = float(np.percentile(all_valid, 99))

    for ax, (label, dev_map) in zip(axes, maps.items()):
        im = ax.imshow(
            dev_map, cmap="hot", vmin=0, vmax=max(vmax, 1e-6),
            interpolation="nearest", origin="upper",
        )
        ax.set_title(f"{label}: Mean |pixel - (-1)|")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Spatial Map of Background Deviation from -1.0", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "background_spatial_deviation", plot_formats, dpi)
    plt.close(fig)


def run_background_analysis(cfg: DictConfig, experiment_name: str) -> dict[str, Any]:
    """Run full background consistency analysis for an experiment.

    Entry point that loads data, computes statistics, and generates diagnostic
    plots comparing background pixel behavior in real vs synthetic images.

    Args:
        cfg: Diagnostics configuration (OmegaConf DictConfig). Expected keys:
            - data.replicas_base_dir: Path to replica NPZ files.
            - data.real_cache_dir: Path to real slice cache.
            - data.full_image_replica_ids: List of replica indices.
            - full_image.background.threshold: Background identification threshold.
            - full_image.background.noise_threshold: Noise detection threshold.
            - output.base_dir: Base output directory.
            - output.plot_format: List of figure formats.
            - output.plot_dpi: Figure DPI.
        experiment_name: Name of the experiment to analyze.

    Returns:
        Dictionary with all background analysis results.

    Raises:
        FileNotFoundError: If replica or cache directories do not exist.
        ValueError: If no images are loaded.
    """
    output_dir = ensure_output_dir(
        Path(cfg.output.base_dir), experiment_name, "full_image/background"
    )

    threshold = cfg.full_image.background.threshold
    noise_threshold = cfg.full_image.background.noise_threshold
    replica_ids = list(cfg.data.full_image_replica_ids)
    plot_formats = list(cfg.output.plot_format)
    dpi = cfg.output.plot_dpi

    logger.info(
        f"Running background analysis for '{experiment_name}' "
        f"(threshold={threshold}, noise_threshold={noise_threshold})"
    )

    # Load synthetic replicas -- include both lesion and control for background analysis
    synth_images, _, synth_zbins, _ = load_full_replicas(
        replicas_base_dir=Path(cfg.data.replicas_base_dir),
        experiment_name=experiment_name,
        replica_ids=replica_ids,
        lesion_only=False,
    )
    logger.info(f"Loaded {len(synth_images)} synthetic images (all classes)")

    # Load real slices -- include both lesion and non-lesion for background analysis
    real_images, _, real_zbins = load_real_slices(
        cache_dir=Path(cfg.data.real_cache_dir),
        lesion_only=False,
    )
    logger.info(f"Loaded {len(real_images)} real images (all classes)")

    if len(real_images) == 0:
        raise ValueError("No real images loaded from cache.")
    if len(synth_images) == 0:
        raise ValueError("No synthetic images loaded from replicas.")

    # Run analysis
    results = analyze_background_consistency(
        real_images=real_images,
        synth_images=synth_images,
        threshold=threshold,
        noise_threshold=noise_threshold,
    )
    results["experiment"] = experiment_name
    results["n_real_images"] = len(real_images)
    results["n_synth_images"] = len(synth_images)

    # Generate plots
    _plot_background_histogram(
        real_images, synth_images, threshold, output_dir, plot_formats, dpi
    )
    _plot_spatial_deviation_map(
        real_images, synth_images, threshold, output_dir, plot_formats, dpi
    )

    # Save results JSON
    save_result_json(results, output_dir / "background_analysis_results.json")

    logger.info(
        f"Background analysis complete. "
        f"Real bg std={results.get('real', {}).get('std', 'N/A')}, "
        f"Synth bg std={results.get('synth', {}).get('std', 'N/A')}"
    )

    return results
