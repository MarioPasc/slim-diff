"""Power spectral density analysis for real vs synthetic MRI comparison.

Compares the frequency-domain characteristics of real and synthetic FLAIR
images and lesion masks. GAN-generated (and diffusion-generated) images
often exhibit spectral artifacts, particularly in high-frequency components.

References:
    Durall, R., Keuper, M., & Keuper, J. (2020). "Watch your Up-Convolution:
        CNN Based Generative Deep Neural Networks are Failing to Reproduce
        Spectral Distributions." CVPR.
    Chandrasegaran, K., et al. (2021). "A Closer Look at Fourier Spectrum
        Discrepancies for CNN-generated Images Detection." ICML.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from scipy.spatial.distance import jensenshannon
from scipy.stats import linregress

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    load_patches,
    save_csv,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)

CHANNEL_NAMES = {0: "image", 1: "mask"}


def compute_2d_psd(images: np.ndarray) -> np.ndarray:
    """Compute the mean 2D power spectral density across a set of images.

    Applies a 2D FFT to each image, computes the squared magnitude of the
    frequency components, and averages across all samples.

    Args:
        images: Array of shape (N, H, W), single-channel images.

    Returns:
        Mean 2D PSD of shape (H, W), with DC component at the center.

    Raises:
        ValueError: If images has fewer than 2 dimensions or is empty.
    """
    if images.ndim != 3 or images.shape[0] == 0:
        raise ValueError(
            f"Expected images with shape (N, H, W), got {images.shape}"
        )

    psd_sum = np.zeros((images.shape[1], images.shape[2]), dtype=np.float64)

    for i in range(images.shape[0]):
        fft_2d = np.fft.fft2(images[i])
        fft_shifted = np.fft.fftshift(fft_2d)
        psd_sum += np.abs(fft_shifted) ** 2

    mean_psd = psd_sum / images.shape[0]
    return mean_psd.astype(np.float64)


def azimuthal_average(psd_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the azimuthal (radial) average of a 2D PSD.

    Converts the 2D power spectrum into a 1D radial profile by averaging
    over concentric annuli centered at the DC component.

    Args:
        psd_2d: 2D PSD array of shape (H, W), DC-centered (via fftshift).

    Returns:
        Tuple of (frequencies, power):
            - frequencies: Normalized spatial frequencies in [0, 0.5], shape (n_bins,).
            - power: Mean power at each radial frequency, shape (n_bins,).
    """
    h, w = psd_2d.shape
    cy, cx = h // 2, w // 2

    y_coords, x_coords = np.ogrid[:h, :w]
    radii = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

    max_radius = min(cx, cy)
    n_bins = max_radius

    # Integer bin assignment
    radii_int = radii.astype(np.int64)

    power = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for r in range(n_bins):
        ring_mask = radii_int == r
        if np.any(ring_mask):
            power[r] = np.mean(psd_2d[ring_mask])
            counts[r] = np.sum(ring_mask)

    # Normalized frequencies: 0 to Nyquist (0.5 cycles/pixel)
    frequencies = np.arange(n_bins) / (2.0 * n_bins)

    # Exclude DC component (index 0) from output
    valid = counts > 0
    valid[0] = False  # Skip DC
    frequencies = frequencies[valid]
    power = power[valid]

    return frequencies, power


def spectral_slope(frequencies: np.ndarray, power: np.ndarray) -> float:
    """Compute the spectral slope via log-log linear regression.

    Natural images exhibit a 1/f^beta power spectrum. The slope beta
    characterizes the rate of power falloff with frequency.

    Args:
        frequencies: Spatial frequencies (must be > 0).
        power: Power values corresponding to each frequency.

    Returns:
        Negative slope beta from log(power) = -beta * log(freq) + c.
        Typical values: ~2.0 for natural images, different for synthetic.

    Raises:
        ValueError: If inputs are empty or contain non-positive values.
    """
    if len(frequencies) == 0 or len(power) == 0:
        raise ValueError("Frequency and power arrays must not be empty.")

    # Filter out zero/negative values for log computation
    valid = (frequencies > 0) & (power > 0)
    if np.sum(valid) < 2:
        raise ValueError("Insufficient valid data points for regression.")

    log_freq = np.log10(frequencies[valid])
    log_power = np.log10(power[valid])

    result = linregress(log_freq, log_power)
    # Return negative slope so that steeper falloff = larger beta
    return -float(result.slope)


def spectral_divergence(psd1: np.ndarray, psd2: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two normalized PSDs.

    Both PSDs are normalized to form probability distributions before
    computing the symmetric divergence metric.

    Args:
        psd1: First radial PSD profile (1D array).
        psd2: Second radial PSD profile (1D array).

    Returns:
        Jensen-Shannon divergence (base-2 log, range [0, 1]).

    Raises:
        ValueError: If arrays have different lengths or are all zeros.
    """
    if len(psd1) != len(psd2):
        raise ValueError(
            f"PSD arrays must have the same length, got {len(psd1)} and {len(psd2)}."
        )

    # Normalize to probability distributions
    p = psd1.astype(np.float64)
    q = psd2.astype(np.float64)

    p_sum = np.sum(p)
    q_sum = np.sum(q)

    if p_sum <= 0 or q_sum <= 0:
        raise ValueError("PSD arrays must have positive total power.")

    p = p / p_sum
    q = q / q_sum

    # Add small epsilon to avoid log(0)
    eps = 1e-12
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)

    return float(jensenshannon(p, q, base=2) ** 2)


def spectral_analysis(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    channel_idx: int = 0,
    n_frequency_bands: int = 5,
) -> dict:
    """Run full spectral analysis comparing real and synthetic patches.

    Computes 2D PSD, radial averages, spectral slopes, divergence, and
    per-band power metrics for the specified channel.

    Args:
        real_patches: Real patches of shape (N, 2, H, W).
        synth_patches: Synthetic patches of shape (M, 2, H, W).
        channel_idx: Which channel to analyze (0=image, 1=mask).
        n_frequency_bands: Number of frequency bands for per-band analysis.

    Returns:
        Dictionary containing:
            - channel: Channel name.
            - n_real, n_synth: Sample counts.
            - real_slope, synth_slope: Spectral slopes (beta).
            - slope_difference: Absolute difference in slopes.
            - js_divergence: Jensen-Shannon divergence of radial PSDs.
            - per_band_power: Dict with band-wise power for real and synth.
            - frequencies_real, power_real: Radial PSD for real data.
            - frequencies_synth, power_synth: Radial PSD for synthetic data.
    """
    channel_name = CHANNEL_NAMES.get(channel_idx, f"channel_{channel_idx}")
    logger.info(
        f"Running spectral analysis on {channel_name} channel: "
        f"real={real_patches.shape[0]}, synth={synth_patches.shape[0]}"
    )

    real_imgs = real_patches[:, channel_idx, :, :]
    synth_imgs = synth_patches[:, channel_idx, :, :]

    # Compute 2D PSDs
    real_psd_2d = compute_2d_psd(real_imgs)
    synth_psd_2d = compute_2d_psd(synth_imgs)

    # Radial averages
    freq_real, power_real = azimuthal_average(real_psd_2d)
    freq_synth, power_synth = azimuthal_average(synth_psd_2d)

    # Align lengths (use the shorter one)
    min_len = min(len(freq_real), len(freq_synth))
    freq_real, power_real = freq_real[:min_len], power_real[:min_len]
    freq_synth, power_synth = freq_synth[:min_len], power_synth[:min_len]

    # Spectral slopes
    real_beta = spectral_slope(freq_real, power_real)
    synth_beta = spectral_slope(freq_synth, power_synth)

    # Jensen-Shannon divergence
    js_div = spectral_divergence(power_real, power_synth)

    # Per-band power analysis
    band_edges = np.logspace(
        np.log10(freq_real[0]),
        np.log10(freq_real[-1]),
        n_frequency_bands + 1,
    )
    per_band = {
        "band_edges": band_edges.tolist(),
        "real_power": [],
        "synth_power": [],
        "power_ratio": [],
    }

    for b in range(n_frequency_bands):
        low, high = band_edges[b], band_edges[b + 1]
        band_mask = (freq_real >= low) & (freq_real < high)
        if np.any(band_mask):
            real_band_power = float(np.mean(power_real[band_mask]))
            synth_band_power = float(np.mean(power_synth[band_mask]))
        else:
            real_band_power = 0.0
            synth_band_power = 0.0

        per_band["real_power"].append(real_band_power)
        per_band["synth_power"].append(synth_band_power)
        ratio = synth_band_power / max(real_band_power, 1e-12)
        per_band["power_ratio"].append(float(ratio))

    result = {
        "channel": channel_name,
        "n_real": int(real_patches.shape[0]),
        "n_synth": int(synth_patches.shape[0]),
        "real_slope": float(real_beta),
        "synth_slope": float(synth_beta),
        "slope_difference": float(abs(real_beta - synth_beta)),
        "js_divergence": float(js_div),
        "per_band_power": per_band,
        "frequencies_real": freq_real.tolist(),
        "power_real": power_real.tolist(),
        "frequencies_synth": freq_synth.tolist(),
        "power_synth": power_synth.tolist(),
    }

    logger.info(
        f"  {channel_name}: beta_real={real_beta:.3f}, beta_synth={synth_beta:.3f}, "
        f"JS-div={js_div:.6f}"
    )
    return result


def _plot_psd_overlay(
    result: dict,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Plot overlaid radial PSD curves for real and synthetic data."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    freq_real = np.array(result["frequencies_real"])
    freq_synth = np.array(result["frequencies_synth"])
    power_real = np.array(result["power_real"])
    power_synth = np.array(result["power_synth"])

    ax.loglog(freq_real, power_real, label="Real", color="steelblue", linewidth=1.5)
    ax.loglog(freq_synth, power_synth, label="Synthetic", color="coral", linewidth=1.5)

    ax.set_xlabel("Spatial Frequency (cycles/pixel)")
    ax.set_ylabel("Power")
    ax.set_title(
        f"Radial PSD: {result['channel']} channel\n"
        f"Slope: real={result['real_slope']:.2f}, synth={result['synth_slope']:.2f} | "
        f"JS-div={result['js_divergence']:.4f}"
    )
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    save_figure(fig, output_dir, f"psd_overlay_{result['channel']}", formats=formats, dpi=dpi)
    plt.close(fig)


def _plot_spectral_slopes(
    results: list[dict],
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Plot spectral slope comparison across channels."""
    channels = [r["channel"] for r in results]
    real_slopes = [r["real_slope"] for r in results]
    synth_slopes = [r["synth_slope"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    x = np.arange(len(channels))
    width = 0.35

    ax.bar(x - width / 2, real_slopes, width, label="Real", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, synth_slopes, width, label="Synthetic", color="coral", alpha=0.8)

    ax.set_xlabel("Channel")
    ax.set_ylabel("Spectral Slope (beta)")
    ax.set_title("Spectral Slope Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, output_dir, "spectral_slopes", formats=formats, dpi=dpi)
    plt.close(fig)


def _plot_band_power(
    result: dict,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Plot per-band power comparison bar chart."""
    per_band = result["per_band_power"]
    n_bands = len(per_band["real_power"])
    band_edges = per_band["band_edges"]

    # Create band labels
    labels = []
    for b in range(n_bands):
        labels.append(f"{band_edges[b]:.3f}-{band_edges[b+1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: absolute power comparison
    x = np.arange(n_bands)
    width = 0.35
    axes[0].bar(
        x - width / 2, per_band["real_power"], width,
        label="Real", color="steelblue", alpha=0.8,
    )
    axes[0].bar(
        x + width / 2, per_band["synth_power"], width,
        label="Synthetic", color="coral", alpha=0.8,
    )
    axes[0].set_xlabel("Frequency Band")
    axes[0].set_ylabel("Mean Power")
    axes[0].set_title(f"Per-Band Power: {result['channel']}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].legend(frameon=True)
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right: power ratio (synth/real)
    axes[1].bar(x, per_band["power_ratio"], color="mediumpurple", alpha=0.8)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_xlabel("Frequency Band")
    axes[1].set_ylabel("Power Ratio (Synth / Real)")
    axes[1].set_title(f"Power Ratio: {result['channel']}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, output_dir, f"band_power_{result['channel']}", formats=formats, dpi=dpi)
    plt.close(fig)


def run_spectral_analysis(cfg: DictConfig, experiment_name: str) -> dict:
    """Entry point for spectral analysis of an experiment.

    Loads patches, runs spectral analysis on configured channels, generates
    plots, and saves JSON results.

    Args:
        cfg: Full diagnostics configuration (OmegaConf DictConfig).
        experiment_name: Name of the experiment to analyze.

    Returns:
        Dictionary with per-channel spectral analysis results.
    """
    logger.info(f"=== Spectral Analysis: {experiment_name} ===")

    # Load data
    patches_dir = Path(cfg.data.patches_base_dir)
    real_patches, synth_patches, _, _ = load_patches(patches_dir, experiment_name)

    # Get config
    spectral_cfg = cfg.feature_probes.spectral
    channels = list(spectral_cfg.channels)
    n_bands = int(spectral_cfg.n_frequency_bands)

    # Output directory
    output_dir = ensure_output_dir(
        Path(cfg.output.base_dir), experiment_name, "spectral"
    )
    plot_formats = list(cfg.output.plot_format)
    plot_dpi = int(cfg.output.plot_dpi)

    # Run analysis per channel
    all_results = []
    for ch in channels:
        result = spectral_analysis(
            real_patches, synth_patches,
            channel_idx=ch,
            n_frequency_bands=n_bands,
        )
        all_results.append(result)

        # Generate per-channel plots
        _plot_psd_overlay(result, output_dir, dpi=plot_dpi, formats=plot_formats)
        _plot_band_power(result, output_dir, dpi=plot_dpi, formats=plot_formats)

    # Cross-channel slope comparison plot
    if len(all_results) > 1:
        _plot_spectral_slopes(all_results, output_dir, dpi=plot_dpi, formats=plot_formats)

    # Save results
    combined_result = {
        "experiment": experiment_name,
        "analysis": "spectral",
        "channels": {r["channel"]: r for r in all_results},
    }
    save_result_json(combined_result, output_dir / "spectral_results.json")

    # Save CSV: radial PSD curves for inter-experiment analysis
    csv_rows = []
    for r in all_results:
        for freq, p_real, p_synth in zip(
            r["frequencies_real"], r["power_real"], r["power_synth"]
        ):
            csv_rows.append({
                "experiment": experiment_name,
                "channel": r["channel"],
                "frequency": freq,
                "power_real": p_real,
                "power_synth": p_synth,
            })
    save_csv(pd.DataFrame(csv_rows), output_dir / "spectral_psd.csv")

    # Save CSV: summary metrics per channel
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "experiment": experiment_name,
            "channel": r["channel"],
            "real_slope": r["real_slope"],
            "synth_slope": r["synth_slope"],
            "slope_difference": r["slope_difference"],
            "js_divergence": r["js_divergence"],
            "n_real": r["n_real"],
            "n_synth": r["n_synth"],
        })
    save_csv(pd.DataFrame(summary_rows), output_dir / "spectral_summary.csv")

    logger.info(f"Spectral analysis complete for '{experiment_name}'")
    return combined_result
