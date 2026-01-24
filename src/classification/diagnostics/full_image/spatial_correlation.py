"""Spatial autocorrelation structure analysis for full 160x160 MRI images.

Compares the spatial correlation structure between real and synthetic images
using 2D autocorrelation computed via FFT. Differences in correlation length
or structure indicate that the diffusion model produces images with different
spatial smoothness characteristics (e.g., over-smoothing or checkerboard
artifacts).

Key diagnostics:
- 2D autocorrelation maps (real vs synthetic)
- Radial (azimuthally averaged) autocorrelation profiles
- Correlation length (exponential decay fit)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy.optimize import curve_fit

import pandas as pd

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    load_full_replicas,
    load_real_slices,
    save_csv,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)


def compute_autocorrelation_2d(
    images: np.ndarray,
    max_lag: int = 20,
    brain_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the average 2D spatial autocorrelation over a set of images.

    Uses the Wiener-Khinchin theorem: the autocorrelation is the inverse FFT
    of the power spectral density. The result is normalized so that the zero-lag
    value equals 1.0.

    Args:
        images: Array of shape (N, H, W), float32 pixel intensities.
        max_lag: Maximum lag in pixels for the output. The returned array
            has shape (2*max_lag+1, 2*max_lag+1) centered at zero lag.
        brain_mask: Optional boolean mask of shape (H, W). If provided,
            only brain pixels are used (non-brain set to zero before FFT).
            This avoids the background dominating the correlation structure.

    Returns:
        Normalized 2D autocorrelation of shape (2*max_lag+1, 2*max_lag+1).
        The center pixel (max_lag, max_lag) is the zero-lag autocorrelation (=1.0).

    Raises:
        ValueError: If images array is empty or has wrong dimensions.
    """
    if images.ndim != 3:
        raise ValueError(f"Expected 3D array (N, H, W), got shape {images.shape}")
    if len(images) == 0:
        raise ValueError("Empty images array.")

    n_images, h, w = images.shape
    autocorr_sum = np.zeros((h, w), dtype=np.float64)

    for img in images:
        if brain_mask is not None:
            masked = img.copy()
            masked[~brain_mask] = 0.0
            # Subtract mean of brain pixels only
            brain_mean = masked[brain_mask].mean() if brain_mask.any() else 0.0
            masked[brain_mask] -= brain_mean
        else:
            masked = img - img.mean()

        # Wiener-Khinchin: autocorrelation via FFT
        f_transform = np.fft.fft2(masked)
        power_spectrum = np.abs(f_transform) ** 2
        autocorr = np.real(np.fft.ifft2(power_spectrum))

        # Normalize by zero-lag value
        zero_lag = autocorr[0, 0]
        if zero_lag > 0:
            autocorr /= zero_lag

        autocorr_sum += autocorr

    # Average over images
    autocorr_avg = autocorr_sum / n_images

    # Shift so zero-lag is at center and crop to max_lag
    autocorr_shifted = np.fft.fftshift(autocorr_avg)
    center_y, center_x = h // 2, w // 2
    y_start = max(0, center_y - max_lag)
    y_end = min(h, center_y + max_lag + 1)
    x_start = max(0, center_x - max_lag)
    x_end = min(w, center_x + max_lag + 1)

    cropped = autocorr_shifted[y_start:y_end, x_start:x_end]

    # Ensure output is exactly (2*max_lag+1, 2*max_lag+1)
    out_size = 2 * max_lag + 1
    result = np.zeros((out_size, out_size), dtype=np.float64)
    # Place cropped into result (handles edge cases near image borders)
    ry = (out_size - cropped.shape[0]) // 2
    rx = (out_size - cropped.shape[1]) // 2
    result[ry:ry + cropped.shape[0], rx:rx + cropped.shape[1]] = cropped

    return result


def radial_autocorrelation(autocorr_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the azimuthally averaged (radial) autocorrelation profile.

    Averages the 2D autocorrelation over all angles at each radial distance
    from the center (zero-lag point).

    Args:
        autocorr_2d: 2D autocorrelation array of shape (2*max_lag+1, 2*max_lag+1),
            centered at (max_lag, max_lag).

    Returns:
        Tuple of (lags, correlation):
        - lags: 1D array of radial distances in pixels, shape (max_lag+1,).
        - correlation: 1D array of mean autocorrelation at each lag.
    """
    size = autocorr_2d.shape[0]
    max_lag = size // 2
    center = max_lag

    # Compute distance from center for each pixel
    y_coords, x_coords = np.ogrid[:size, :size]
    distances = np.sqrt((y_coords - center) ** 2 + (x_coords - center) ** 2)

    # Bin by integer distance
    lags = np.arange(max_lag + 1)
    correlation = np.zeros(max_lag + 1, dtype=np.float64)

    for lag in lags:
        if lag == 0:
            # Zero-lag is just the center pixel
            correlation[lag] = autocorr_2d[center, center]
        else:
            # Annular ring at distance [lag - 0.5, lag + 0.5)
            ring_mask = (distances >= lag - 0.5) & (distances < lag + 0.5)
            if ring_mask.any():
                correlation[lag] = autocorr_2d[ring_mask].mean()

    return lags.astype(np.float64), correlation


def fit_correlation_length(lags: np.ndarray, correlation: np.ndarray) -> float:
    """Fit an exponential decay to the radial autocorrelation profile.

    Fits the model C(r) = exp(-r / xi) to the autocorrelation data, where
    xi is the correlation length (the distance at which correlation drops
    to 1/e of its zero-lag value).

    Args:
        lags: 1D array of radial distances (pixels).
        correlation: 1D array of normalized autocorrelation values.

    Returns:
        Correlation length xi (in pixels). Returns np.inf if the fit fails
        (e.g., if the autocorrelation does not decay).
    """
    # Use lags > 0 for fitting (skip zero-lag which is always 1.0)
    fit_mask = lags > 0
    if fit_mask.sum() < 3:
        logger.warning("Too few data points for correlation length fit.")
        return np.inf

    x_fit = lags[fit_mask]
    y_fit = correlation[fit_mask]

    # Only fit positive correlation values (log requires > 0)
    positive_mask = y_fit > 0
    if positive_mask.sum() < 3:
        logger.warning("Too few positive correlation values for fit.")
        return np.inf

    x_fit = x_fit[positive_mask]
    y_fit = y_fit[positive_mask]

    def exp_decay(r: np.ndarray, xi: float) -> np.ndarray:
        return np.exp(-r / xi)

    try:
        popt, _ = curve_fit(
            exp_decay, x_fit, y_fit,
            p0=[5.0],
            bounds=(0.1, np.inf),
            maxfev=5000,
        )
        xi = float(popt[0])
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Correlation length fit failed: {e}")
        xi = np.inf

    return xi


def spatial_correlation_analysis(
    real_images: np.ndarray,
    synth_images: np.ndarray,
    max_lag: int = 20,
) -> dict[str, Any]:
    """Compare spatial autocorrelation structure between real and synthetic images.

    Computes 2D and radial autocorrelation profiles for both image sets,
    fits exponential decay models, and quantifies structural differences.

    Args:
        real_images: Real MRI slices, shape (N_real, H, W), float32.
        synth_images: Synthetic replicas, shape (N_synth, H, W), float32.
        max_lag: Maximum lag in pixels for autocorrelation computation.

    Returns:
        Dictionary containing:
        - autocorr_2d_real/synth: 2D autocorrelation maps.
        - radial_lags: Lag values in pixels.
        - radial_real/synth: Radial autocorrelation profiles.
        - correlation_length_real/synth: Fitted xi values.
        - correlation_length_ratio: synth_xi / real_xi.
        - autocorr_2d_difference: Absolute difference between 2D maps.
    """
    logger.info(
        f"Computing spatial autocorrelation (max_lag={max_lag}): "
        f"{len(real_images)} real, {len(synth_images)} synth images"
    )

    # Derive a common brain mask from real images (background = -1.0)
    brain_mask = np.mean(real_images, axis=0) > -0.95

    autocorr_real = compute_autocorrelation_2d(real_images, max_lag, brain_mask)
    autocorr_synth = compute_autocorrelation_2d(synth_images, max_lag, brain_mask)

    lags_real, radial_real = radial_autocorrelation(autocorr_real)
    lags_synth, radial_synth = radial_autocorrelation(autocorr_synth)

    xi_real = fit_correlation_length(lags_real, radial_real)
    xi_synth = fit_correlation_length(lags_synth, radial_synth)

    xi_ratio = xi_synth / xi_real if xi_real > 0 and np.isfinite(xi_real) else np.nan

    results: dict[str, Any] = {
        "max_lag": max_lag,
        "autocorr_2d_real": autocorr_real.tolist(),
        "autocorr_2d_synth": autocorr_synth.tolist(),
        "radial_lags": lags_real.tolist(),
        "radial_real": radial_real.tolist(),
        "radial_synth": radial_synth.tolist(),
        "correlation_length_real": xi_real,
        "correlation_length_synth": xi_synth,
        "correlation_length_ratio": xi_ratio,
        "autocorr_2d_difference": np.abs(autocorr_real - autocorr_synth).tolist(),
    }

    logger.info(
        f"Correlation lengths: real={xi_real:.2f}px, synth={xi_synth:.2f}px, "
        f"ratio={xi_ratio:.3f}"
    )

    return results


def _plot_autocorrelation_2d(
    results: dict[str, Any],
    output_dir: Path,
    plot_formats: list[str],
    dpi: int,
) -> None:
    """Plot 2D autocorrelation maps side by side (real, synthetic, difference)."""
    autocorr_real = np.array(results["autocorr_2d_real"])
    autocorr_synth = np.array(results["autocorr_2d_synth"])
    difference = np.abs(autocorr_real - autocorr_synth)

    max_lag = results["max_lag"]
    extent = [-max_lag, max_lag, -max_lag, max_lag]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Shared color limits for real/synth
    vmin = min(autocorr_real.min(), autocorr_synth.min())
    vmax = max(autocorr_real.max(), autocorr_synth.max())

    im0 = axes[0].imshow(
        autocorr_real, extent=extent, cmap="viridis",
        vmin=vmin, vmax=vmax, interpolation="nearest", origin="lower",
    )
    axes[0].set_title("Real")
    axes[0].set_xlabel("Lag x (pixels)")
    axes[0].set_ylabel("Lag y (pixels)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        autocorr_synth, extent=extent, cmap="viridis",
        vmin=vmin, vmax=vmax, interpolation="nearest", origin="lower",
    )
    axes[1].set_title("Synthetic")
    axes[1].set_xlabel("Lag x (pixels)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        difference, extent=extent, cmap="hot",
        vmin=0, interpolation="nearest", origin="lower",
    )
    axes[2].set_title("|Real - Synth|")
    axes[2].set_xlabel("Lag x (pixels)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("2D Spatial Autocorrelation", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "autocorrelation_2d_maps", plot_formats, dpi)
    plt.close(fig)


def _plot_radial_profiles(
    results: dict[str, Any],
    output_dir: Path,
    plot_formats: list[str],
    dpi: int,
) -> None:
    """Plot radial autocorrelation profiles with exponential fits."""
    lags = np.array(results["radial_lags"])
    radial_real = np.array(results["radial_real"])
    radial_synth = np.array(results["radial_synth"])
    xi_real = results["correlation_length_real"]
    xi_synth = results["correlation_length_synth"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(lags, radial_real, "o-", color="#2196F3", markersize=4,
            label=f"Real (xi={xi_real:.2f} px)")
    ax.plot(lags, radial_synth, "s-", color="#FF9800", markersize=4,
            label=f"Synthetic (xi={xi_synth:.2f} px)")

    # Overlay exponential fits
    lags_fit = np.linspace(0.5, lags.max(), 100)
    if np.isfinite(xi_real) and xi_real > 0:
        ax.plot(lags_fit, np.exp(-lags_fit / xi_real), "--", color="#2196F3",
                alpha=0.6, linewidth=1.5)
    if np.isfinite(xi_synth) and xi_synth > 0:
        ax.plot(lags_fit, np.exp(-lags_fit / xi_synth), "--", color="#FF9800",
                alpha=0.6, linewidth=1.5)

    ax.axhline(1.0 / np.e, color="gray", linestyle=":", alpha=0.5, label="1/e threshold")
    ax.set_xlabel("Radial Lag (pixels)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Radial Autocorrelation Profile")
    ax.legend()
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, "radial_autocorrelation_profiles", plot_formats, dpi)
    plt.close(fig)


def run_spatial_correlation(cfg: DictConfig, experiment_name: str) -> dict[str, Any]:
    """Run full spatial autocorrelation analysis for an experiment.

    Entry point that loads data, computes correlation structure, and
    generates diagnostic plots.

    Args:
        cfg: Diagnostics configuration (OmegaConf DictConfig). Expected keys:
            - data.replicas_base_dir: Path to replica NPZ files.
            - data.real_cache_dir: Path to real slice cache.
            - data.full_image_replica_ids: List of replica indices.
            - full_image.spatial_correlation.max_lag: Maximum lag for autocorrelation.
            - output.base_dir: Base output directory.
            - output.plot_format: List of figure formats.
            - output.plot_dpi: Figure DPI.
        experiment_name: Name of the experiment to analyze.

    Returns:
        Dictionary with spatial correlation analysis results.

    Raises:
        FileNotFoundError: If replica or cache directories do not exist.
        ValueError: If no images are loaded.
    """
    output_dir = ensure_output_dir(
        Path(cfg.output.base_dir), experiment_name, "full_image/spatial_correlation"
    )

    max_lag = cfg.full_image.spatial_correlation.max_lag
    replica_ids = list(cfg.data.full_image_replica_ids)
    plot_formats = list(cfg.output.plot_format)
    dpi = cfg.output.plot_dpi

    logger.info(
        f"Running spatial correlation analysis for '{experiment_name}' "
        f"(max_lag={max_lag})"
    )

    # Load synthetic replicas (lesion only to match training distribution)
    synth_images, _, _, _ = load_full_replicas(
        replicas_base_dir=Path(cfg.data.replicas_base_dir),
        experiment_name=experiment_name,
        replica_ids=replica_ids,
        lesion_only=True,
    )
    logger.info(f"Loaded {len(synth_images)} synthetic images (lesion only)")

    # Load real slices (lesion only to match)
    real_images, _, _ = load_real_slices(
        cache_dir=Path(cfg.data.real_cache_dir),
        lesion_only=True,
    )
    logger.info(f"Loaded {len(real_images)} real images (lesion only)")

    if len(real_images) == 0:
        raise ValueError("No real images loaded from cache.")
    if len(synth_images) == 0:
        raise ValueError("No synthetic images loaded from replicas.")

    # Run analysis
    results = spatial_correlation_analysis(
        real_images=real_images,
        synth_images=synth_images,
        max_lag=max_lag,
    )
    results["experiment"] = experiment_name
    results["n_real_images"] = len(real_images)
    results["n_synth_images"] = len(synth_images)

    # Generate plots
    _plot_autocorrelation_2d(results, output_dir, plot_formats, dpi)
    _plot_radial_profiles(results, output_dir, plot_formats, dpi)

    # Save results JSON (exclude large arrays for JSON readability)
    results_for_json = {
        k: v for k, v in results.items()
        if k not in ("autocorr_2d_real", "autocorr_2d_synth", "autocorr_2d_difference")
    }
    results_for_json["autocorr_2d_shape"] = [2 * max_lag + 1, 2 * max_lag + 1]
    save_result_json(results_for_json, output_dir / "spatial_correlation_results.json")

    # Also save the full numpy arrays for downstream use
    np.savez_compressed(
        output_dir / "autocorrelation_data.npz",
        autocorr_2d_real=np.array(results["autocorr_2d_real"]),
        autocorr_2d_synth=np.array(results["autocorr_2d_synth"]),
        radial_lags=np.array(results["radial_lags"]),
        radial_real=np.array(results["radial_real"]),
        radial_synth=np.array(results["radial_synth"]),
    )

    # Save CSV: radial autocorrelation profiles for inter-experiment analysis
    csv_rows = []
    radial_lags = results.get("radial_lags", [])
    radial_real = results.get("radial_real", [])
    radial_synth = results.get("radial_synth", [])
    for lag, r_real, r_synth in zip(radial_lags, radial_real, radial_synth):
        csv_rows.append({
            "experiment": experiment_name,
            "lag": float(lag),
            "autocorr_real": float(r_real),
            "autocorr_synth": float(r_synth),
        })
    if csv_rows:
        save_csv(pd.DataFrame(csv_rows), output_dir / "spatial_correlation_radial.csv")

    # Summary CSV
    summary_row = {
        "experiment": experiment_name,
        "correlation_length_real": results["correlation_length_real"],
        "correlation_length_synth": results["correlation_length_synth"],
        "correlation_length_ratio": (
            results["correlation_length_synth"]
            / max(results["correlation_length_real"], 1e-12)
        ),
        "n_real_images": len(real_images),
        "n_synth_images": len(synth_images),
    }
    save_csv(pd.DataFrame([summary_row]), output_dir / "spatial_correlation_summary.csv")

    logger.info(
        f"Spatial correlation analysis complete. "
        f"xi_real={results['correlation_length_real']:.2f}, "
        f"xi_synth={results['correlation_length_synth']:.2f}"
    )

    return results
