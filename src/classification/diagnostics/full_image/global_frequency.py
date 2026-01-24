"""Full-image spectral content analysis for 160x160 MRI images.

Compares the power spectral density (PSD) between real and synthetic images
at the full image resolution. Diffusion models often exhibit characteristic
spectral signatures: high-frequency rolloff (over-smoothing) or spectral
slope differences that indicate failure to reproduce fine texture detail.

Key diagnostics:
- 2D power spectral density comparison
- Azimuthally averaged 1D PSD profiles
- Spectral slope estimation via log-log linear regression
- High-frequency to low-frequency power ratio
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

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


def _compute_2d_psd(images: np.ndarray, brain_mask: np.ndarray | None = None) -> np.ndarray:
    """Compute the average 2D power spectral density over a set of images.

    Uses the squared magnitude of the 2D DFT, averaged over all images.
    Optionally masks out background pixels before computing the FFT.

    Args:
        images: Array of shape (N, H, W), float32.
        brain_mask: Optional boolean mask (H, W). If provided, non-brain
            pixels are zeroed before FFT computation.

    Returns:
        2D PSD array of shape (H, W), shifted so DC is at center.
    """
    n_images, h, w = images.shape
    psd_sum = np.zeros((h, w), dtype=np.float64)

    for img in images:
        if brain_mask is not None:
            masked = img.copy()
            masked[~brain_mask] = 0.0
        else:
            masked = img.copy()

        # Subtract mean to remove DC spike dominance
        if brain_mask is not None and brain_mask.any():
            masked[brain_mask] -= masked[brain_mask].mean()
        else:
            masked -= masked.mean()

        # Apply Hann window to reduce spectral leakage
        window_y = np.hanning(h)
        window_x = np.hanning(w)
        window_2d = np.outer(window_y, window_x)
        windowed = masked * window_2d

        f_transform = np.fft.fft2(windowed)
        psd = np.abs(f_transform) ** 2
        psd_sum += psd

    psd_avg = psd_sum / n_images

    # Shift DC to center
    psd_shifted = np.fft.fftshift(psd_avg)

    return psd_shifted


def _azimuthal_average(psd_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the azimuthal (radial) average of a 2D PSD.

    Averages the PSD over all angles at each radial frequency, producing
    a 1D power spectrum as a function of spatial frequency.

    Args:
        psd_2d: 2D PSD of shape (H, W), DC at center.

    Returns:
        Tuple of (frequencies, power):
        - frequencies: Radial frequency values in cycles/pixel, shape (n_bins,).
        - power: Mean PSD at each radial frequency.
    """
    h, w = psd_2d.shape
    center_y, center_x = h // 2, w // 2

    # Distance from center for each pixel (in frequency units)
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

    # Maximum frequency is half the image size (Nyquist)
    max_freq = min(center_y, center_x)
    n_bins = max_freq

    freq_bins = np.arange(n_bins)
    power = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    # Bin by radial distance
    bin_indices = np.floor(distances).astype(int)

    for r in range(n_bins):
        ring_mask = bin_indices == r
        if ring_mask.any():
            power[r] = psd_2d[ring_mask].mean()
            counts[r] = ring_mask.sum()

    # Convert bin indices to normalized frequency (cycles per pixel)
    # freq = bin_index / image_size
    frequencies = freq_bins / float(min(h, w))

    # Remove zero-frequency bin for slope fitting (DC component)
    valid = frequencies > 0
    return frequencies[valid], power[valid]


def _fit_spectral_slope(
    frequencies: np.ndarray,
    power: np.ndarray,
    freq_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Fit a power-law slope to the PSD in log-log space.

    Fits log(P) = slope * log(f) + intercept using least squares.

    Args:
        frequencies: 1D array of frequency values (> 0).
        power: 1D array of PSD values at each frequency.
        freq_range: Optional (min_freq, max_freq) to restrict the fit.
            If None, uses all available frequencies.

    Returns:
        Tuple of (slope, intercept) from the log-log linear fit.
        Returns (np.nan, np.nan) if the fit fails.
    """
    mask = power > 0
    if freq_range is not None:
        mask &= (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])

    if mask.sum() < 3:
        logger.warning("Too few valid points for spectral slope fit.")
        return np.nan, np.nan

    log_freq = np.log10(frequencies[mask])
    log_power = np.log10(power[mask])

    try:
        coeffs = np.polyfit(log_freq, log_power, deg=1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
    except (np.linalg.LinAlgError, ValueError) as e:
        logger.warning(f"Spectral slope fit failed: {e}")
        return np.nan, np.nan

    return slope, intercept


def _compute_high_freq_ratio(
    frequencies: np.ndarray,
    power: np.ndarray,
    cutoff_fraction: float = 0.5,
) -> float:
    """Compute the ratio of high-frequency to low-frequency power.

    Splits the spectrum at the cutoff fraction of the Nyquist frequency
    and computes sum(high) / sum(low).

    Args:
        frequencies: 1D frequency array.
        power: 1D PSD values.
        cutoff_fraction: Fraction of max frequency for the split point.

    Returns:
        Ratio of high-frequency to low-frequency power.
    """
    max_freq = frequencies.max()
    cutoff = max_freq * cutoff_fraction

    low_mask = frequencies <= cutoff
    high_mask = frequencies > cutoff

    low_power = power[low_mask].sum()
    high_power = power[high_mask].sum()

    if low_power <= 0:
        return np.inf
    return float(high_power / low_power)


def full_image_spectral_analysis(
    real_images: np.ndarray,
    synth_images: np.ndarray,
    brain_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare spectral content between real and synthetic full images.

    Computes the power spectral density for both sets, fits spectral slopes,
    and analyzes high-frequency rolloff characteristics.

    Args:
        real_images: Real MRI slices, shape (N_real, H, W), float32.
        synth_images: Synthetic replicas, shape (N_synth, H, W), float32.
        brain_mask: Optional boolean mask (H, W) to restrict analysis
            to brain region. If None, a mask is derived from the mean
            real image (threshold > -0.95).

    Returns:
        Dictionary containing:
        - frequencies: 1D frequency array (cycles/pixel).
        - psd_real / psd_synth: Azimuthally averaged 1D PSDs.
        - slope_real / slope_synth: Fitted spectral slopes.
        - slope_difference: synth_slope - real_slope.
        - high_freq_ratio_real / synth: High-to-low frequency power ratios.
        - high_freq_ratio_difference: Relative difference.
        - psd_2d_real / psd_2d_synth: Full 2D PSDs (for plotting).
    """
    logger.info(
        f"Computing spectral analysis: {len(real_images)} real, "
        f"{len(synth_images)} synth images"
    )

    if brain_mask is None:
        # Derive brain mask from mean real image
        mean_real = np.mean(real_images, axis=0)
        brain_mask = mean_real > -0.95

    # Compute 2D PSDs
    psd_2d_real = _compute_2d_psd(real_images, brain_mask)
    psd_2d_synth = _compute_2d_psd(synth_images, brain_mask)

    # Azimuthal averages
    freq_real, psd_1d_real = _azimuthal_average(psd_2d_real)
    freq_synth, psd_1d_synth = _azimuthal_average(psd_2d_synth)

    # Ensure same frequency axis (should be identical for same image size)
    if len(freq_real) != len(freq_synth):
        n_common = min(len(freq_real), len(freq_synth))
        freq_real = freq_real[:n_common]
        freq_synth = freq_synth[:n_common]
        psd_1d_real = psd_1d_real[:n_common]
        psd_1d_synth = psd_1d_synth[:n_common]

    # Fit spectral slopes
    slope_real, intercept_real = _fit_spectral_slope(freq_real, psd_1d_real)
    slope_synth, intercept_synth = _fit_spectral_slope(freq_synth, psd_1d_synth)

    slope_diff = slope_synth - slope_real if np.isfinite(slope_real) and np.isfinite(slope_synth) else np.nan

    # High-frequency power ratio
    hf_ratio_real = _compute_high_freq_ratio(freq_real, psd_1d_real)
    hf_ratio_synth = _compute_high_freq_ratio(freq_synth, psd_1d_synth)

    hf_ratio_diff = (
        (hf_ratio_synth - hf_ratio_real) / max(hf_ratio_real, 1e-12)
        if np.isfinite(hf_ratio_real) and np.isfinite(hf_ratio_synth)
        else np.nan
    )

    results: dict[str, Any] = {
        "frequencies": freq_real.tolist(),
        "psd_real": psd_1d_real.tolist(),
        "psd_synth": psd_1d_synth.tolist(),
        "slope_real": slope_real,
        "slope_synth": slope_synth,
        "intercept_real": intercept_real,
        "intercept_synth": intercept_synth,
        "slope_difference": slope_diff,
        "high_freq_ratio_real": hf_ratio_real,
        "high_freq_ratio_synth": hf_ratio_synth,
        "high_freq_ratio_difference": hf_ratio_diff,
        "psd_2d_real": psd_2d_real,
        "psd_2d_synth": psd_2d_synth,
    }

    logger.info(
        f"Spectral slopes: real={slope_real:.3f}, synth={slope_synth:.3f}, "
        f"diff={slope_diff:.3f}"
    )
    logger.info(
        f"HF ratios: real={hf_ratio_real:.4f}, synth={hf_ratio_synth:.4f}, "
        f"rel_diff={hf_ratio_diff:.4f}"
    )

    return results


def _plot_psd_comparison(
    results: dict[str, Any],
    output_dir: Path,
    plot_formats: list[str],
    dpi: int,
) -> None:
    """Plot PSD comparison with spectral slope annotations."""
    frequencies = np.array(results["frequencies"])
    psd_real = np.array(results["psd_real"])
    psd_synth = np.array(results["psd_synth"])
    slope_real = results["slope_real"]
    slope_synth = results["slope_synth"]
    intercept_real = results["intercept_real"]
    intercept_synth = results["intercept_synth"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: log-log PSD comparison
    ax = axes[0]
    valid_real = psd_real > 0
    valid_synth = psd_synth > 0

    ax.loglog(frequencies[valid_real], psd_real[valid_real], "-", color="#2196F3",
              linewidth=1.5, alpha=0.8, label=f"Real (slope={slope_real:.2f})")
    ax.loglog(frequencies[valid_synth], psd_synth[valid_synth], "-", color="#FF9800",
              linewidth=1.5, alpha=0.8, label=f"Synthetic (slope={slope_synth:.2f})")

    # Overlay fitted power laws
    if np.isfinite(slope_real) and np.isfinite(intercept_real):
        fit_power_real = 10 ** (slope_real * np.log10(frequencies[valid_real]) + intercept_real)
        ax.loglog(frequencies[valid_real], fit_power_real, "--", color="#2196F3",
                  alpha=0.5, linewidth=1.0)
    if np.isfinite(slope_synth) and np.isfinite(intercept_synth):
        fit_power_synth = 10 ** (slope_synth * np.log10(frequencies[valid_synth]) + intercept_synth)
        ax.loglog(frequencies[valid_synth], fit_power_synth, "--", color="#FF9800",
                  alpha=0.5, linewidth=1.0)

    ax.set_xlabel("Spatial Frequency (cycles/pixel)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("Azimuthally Averaged PSD")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Right panel: PSD ratio (synth / real)
    ax = axes[1]
    common_valid = valid_real & valid_synth
    if common_valid.any():
        ratio = psd_synth[common_valid] / np.maximum(psd_real[common_valid], 1e-30)
        ax.semilogx(frequencies[common_valid], ratio, "-", color="#4CAF50",
                    linewidth=1.5, alpha=0.8)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Unity (perfect match)")

        # Mark the high-frequency cutoff
        max_freq = frequencies.max()
        cutoff = max_freq * 0.5
        ax.axvline(cutoff, color="red", linestyle=":", alpha=0.5, label="HF cutoff (0.5 Nyquist)")

    ax.set_xlabel("Spatial Frequency (cycles/pixel)")
    ax.set_ylabel("PSD Ratio (Synth / Real)")
    ax.set_title("Spectral Power Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0, max(3.0, ratio.max() * 1.1) if common_valid.any() else 3.0)

    plt.suptitle("Full-Image Spectral Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir, "psd_comparison", plot_formats, dpi)
    plt.close(fig)


def _plot_high_freq_analysis(
    results: dict[str, Any],
    output_dir: Path,
    plot_formats: list[str],
    dpi: int,
) -> None:
    """Plot high vs low frequency power comparison bar chart."""
    hf_real = results["high_freq_ratio_real"]
    hf_synth = results["high_freq_ratio_synth"]

    if not (np.isfinite(hf_real) and np.isfinite(hf_synth)):
        logger.warning("Non-finite HF ratios, skipping bar chart.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    categories = ["Real", "Synthetic"]
    values = [hf_real, hf_synth]
    colors = ["#2196F3", "#FF9800"]

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Annotate bars with values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("High-Freq / Low-Freq Power Ratio")
    ax.set_title("High-Frequency Content Comparison\n(cutoff at 0.5 Nyquist)")
    ax.grid(axis="y", alpha=0.3)

    # Add relative difference annotation
    rel_diff = results["high_freq_ratio_difference"]
    if np.isfinite(rel_diff):
        ax.text(
            0.5, 0.95,
            f"Relative difference: {rel_diff:+.2%}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    plt.tight_layout()
    save_figure(fig, output_dir, "high_frequency_ratio", plot_formats, dpi)
    plt.close(fig)


def run_global_frequency(cfg: DictConfig, experiment_name: str) -> dict[str, Any]:
    """Run full-image spectral analysis for an experiment.

    Entry point that loads data, computes PSDs, fits spectral slopes,
    and generates diagnostic plots.

    Args:
        cfg: Diagnostics configuration (OmegaConf DictConfig). Expected keys:
            - data.replicas_base_dir: Path to replica NPZ files.
            - data.real_cache_dir: Path to real slice cache.
            - data.full_image_replica_ids: List of replica indices.
            - full_image.global_frequency.channels: List of channels to analyze.
            - output.base_dir: Base output directory.
            - output.plot_format: List of figure formats.
            - output.plot_dpi: Figure DPI.
        experiment_name: Name of the experiment to analyze.

    Returns:
        Dictionary with spectral analysis results.

    Raises:
        FileNotFoundError: If replica or cache directories do not exist.
        ValueError: If no images are loaded.
    """
    output_dir = ensure_output_dir(
        Path(cfg.output.base_dir), experiment_name, "full_image/global_frequency"
    )

    replica_ids = list(cfg.data.full_image_replica_ids)
    plot_formats = list(cfg.output.plot_format)
    dpi = cfg.output.plot_dpi

    logger.info(f"Running global frequency analysis for '{experiment_name}'")

    # Load synthetic replicas (lesion only for meaningful spectral comparison)
    synth_images, synth_masks, _, _ = load_full_replicas(
        replicas_base_dir=Path(cfg.data.replicas_base_dir),
        experiment_name=experiment_name,
        replica_ids=replica_ids,
        lesion_only=True,
    )
    logger.info(f"Loaded {len(synth_images)} synthetic images (lesion only)")

    # Load real slices (lesion only to match)
    real_images, real_masks, _ = load_real_slices(
        cache_dir=Path(cfg.data.real_cache_dir),
        lesion_only=True,
    )
    logger.info(f"Loaded {len(real_images)} real images (lesion only)")

    if len(real_images) == 0:
        raise ValueError("No real images loaded from cache.")
    if len(synth_images) == 0:
        raise ValueError("No synthetic images loaded from replicas.")

    # Derive brain mask from real images
    mean_real = np.mean(real_images, axis=0)
    brain_mask = mean_real > -0.95

    # Run spectral analysis on image channel (channel 0 = FLAIR intensities)
    results = full_image_spectral_analysis(
        real_images=real_images,
        synth_images=synth_images,
        brain_mask=brain_mask,
    )
    results["experiment"] = experiment_name
    results["n_real_images"] = len(real_images)
    results["n_synth_images"] = len(synth_images)
    results["image_size"] = list(real_images.shape[1:])

    # Generate plots
    _plot_psd_comparison(results, output_dir, plot_formats, dpi)
    _plot_high_freq_analysis(results, output_dir, plot_formats, dpi)

    # Save results JSON (exclude 2D PSD arrays for readability)
    results_for_json = {
        k: v for k, v in results.items()
        if k not in ("psd_2d_real", "psd_2d_synth")
    }
    save_result_json(results_for_json, output_dir / "global_frequency_results.json")

    # Save 2D PSDs as numpy for downstream use
    np.savez_compressed(
        output_dir / "psd_2d_data.npz",
        psd_2d_real=results["psd_2d_real"],
        psd_2d_synth=results["psd_2d_synth"],
        frequencies=np.array(results["frequencies"]),
        psd_1d_real=np.array(results["psd_real"]),
        psd_1d_synth=np.array(results["psd_synth"]),
    )

    # Save CSV: PSD curve data for inter-experiment analysis
    csv_rows = []
    for freq, p_real, p_synth in zip(
        results["frequencies"], results["psd_real"], results["psd_synth"]
    ):
        csv_rows.append({
            "experiment": experiment_name,
            "frequency": freq,
            "psd_real": p_real,
            "psd_synth": p_synth,
        })
    if csv_rows:
        save_csv(pd.DataFrame(csv_rows), output_dir / "global_frequency_psd.csv")

    # Summary CSV
    summary_row = {
        "experiment": experiment_name,
        "slope_real": results["slope_real"],
        "slope_synth": results["slope_synth"],
        "slope_difference": results["slope_difference"],
        "high_freq_ratio_real": results["high_freq_ratio_real"],
        "high_freq_ratio_synth": results["high_freq_ratio_synth"],
        "high_freq_ratio_difference": results["high_freq_ratio_difference"],
        "n_real_images": len(real_images),
        "n_synth_images": len(synth_images),
    }
    save_csv(pd.DataFrame([summary_row]), output_dir / "global_frequency_summary.csv")

    logger.info(
        f"Global frequency analysis complete. "
        f"Slope diff={results['slope_difference']:.3f}, "
        f"HF ratio diff={results['high_freq_ratio_difference']:.4f}"
    )

    return results
