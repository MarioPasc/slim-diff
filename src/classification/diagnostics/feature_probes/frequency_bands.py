"""Band-pass frequency analysis for real vs synthetic MRI comparison.

Decomposes images into log-spaced frequency bands and compares the power
distribution in each band between real and synthetic data. This reveals
which spatial frequency ranges exhibit the largest discrepancies.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import DictConfig
from scipy.stats import ks_2samp

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    load_patches,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)

CHANNEL_NAMES = {0: "image", 1: "mask"}


def bandpass_filter_2d(
    image: np.ndarray,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """Apply an ideal bandpass filter in the frequency domain.

    Passes frequencies in the annular region [low_freq, high_freq) and
    zeros out all others. Frequencies are normalized to [0, 0.5] where
    0.5 is the Nyquist frequency.

    Args:
        image: 2D float array of shape (H, W).
        low_freq: Lower cutoff frequency (normalized, 0 to 0.5).
        high_freq: Upper cutoff frequency (normalized, 0 to 0.5).

    Returns:
        Bandpass-filtered image of shape (H, W).

    Raises:
        ValueError: If frequency bounds are invalid.
    """
    if low_freq < 0 or high_freq > 0.5 or low_freq >= high_freq:
        raise ValueError(
            f"Invalid frequency bounds: [{low_freq}, {high_freq}]. "
            f"Must satisfy 0 <= low_freq < high_freq <= 0.5."
        )

    h, w = image.shape
    cy, cx = h // 2, w // 2

    # Compute FFT and shift DC to center
    fft_2d = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_2d)

    # Build radial frequency grid (normalized to [0, 0.5])
    y_coords = np.arange(h) - cy
    x_coords = np.arange(w) - cx
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    # Normalize radius: max possible radius corresponds to 0.5
    max_radius = np.sqrt(cy ** 2 + cx ** 2)
    radii = np.sqrt(xx ** 2 + yy ** 2) / (2.0 * max_radius)

    # Create bandpass mask
    mask = ((radii >= low_freq) & (radii < high_freq)).astype(np.float64)

    # Apply filter and inverse FFT
    filtered_fft = fft_shifted * mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))

    return np.real(filtered_image).astype(np.float64)


def compute_band_power(
    images: np.ndarray,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """Compute the mean power per sample in a given frequency band.

    For each image, applies the bandpass filter and computes the mean
    squared amplitude (power) of the filtered signal.

    Args:
        images: Array of shape (N, H, W), single-channel images.
        low_freq: Lower cutoff frequency (normalized, 0 to 0.5).
        high_freq: Upper cutoff frequency (normalized, 0 to 0.5).

    Returns:
        Array of shape (N,) with mean power per sample.
    """
    n_samples = images.shape[0]
    powers = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        filtered = bandpass_filter_2d(images[i], low_freq, high_freq)
        powers[i] = np.mean(filtered ** 2)

    return powers


def band_analysis(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    channel_idx: int = 0,
    n_bands: int = 5,
) -> dict:
    """Perform band-pass frequency analysis comparing real and synthetic patches.

    Divides the frequency range into log-spaced bands and compares the
    power distribution in each band using KS tests and summary statistics.

    Args:
        real_patches: Real patches of shape (N, 2, H, W).
        synth_patches: Synthetic patches of shape (M, 2, H, W).
        channel_idx: Which channel to analyze (0=image, 1=mask).
        n_bands: Number of frequency bands (log-spaced).

    Returns:
        Dictionary containing per-band power distributions, KS test results,
        and summary statistics.
    """
    channel_name = CHANNEL_NAMES.get(channel_idx, f"channel_{channel_idx}")
    logger.info(
        f"Running band analysis on {channel_name} channel: "
        f"real={real_patches.shape[0]}, synth={synth_patches.shape[0]}, "
        f"n_bands={n_bands}"
    )

    real_imgs = real_patches[:, channel_idx, :, :]
    synth_imgs = synth_patches[:, channel_idx, :, :]

    # Define log-spaced frequency band edges from near-DC to Nyquist
    # Start at a small positive frequency to avoid DC
    min_freq = 1.0 / max(real_imgs.shape[1], real_imgs.shape[2])
    max_freq = 0.5
    band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bands + 1)

    bands = []
    for b in range(n_bands):
        low = float(band_edges[b])
        high = float(band_edges[b + 1])

        logger.info(f"  Band {b}: [{low:.4f}, {high:.4f}]")

        real_power = compute_band_power(real_imgs, low, high)
        synth_power = compute_band_power(synth_imgs, low, high)

        # KS test
        ks_stat, ks_pval = ks_2samp(real_power, synth_power)

        # Effect size (mean difference relative to pooled std)
        pooled_std = np.sqrt(
            (np.var(real_power, ddof=1) + np.var(synth_power, ddof=1)) / 2.0
        )
        if pooled_std > 1e-12:
            cohens_d = float((np.mean(real_power) - np.mean(synth_power)) / pooled_std)
        else:
            cohens_d = 0.0

        band_result = {
            "band_idx": b,
            "low_freq": low,
            "high_freq": high,
            "real_power_mean": float(np.mean(real_power)),
            "real_power_std": float(np.std(real_power)),
            "real_power_median": float(np.median(real_power)),
            "synth_power_mean": float(np.mean(synth_power)),
            "synth_power_std": float(np.std(synth_power)),
            "synth_power_median": float(np.median(synth_power)),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "cohens_d": cohens_d,
            "power_ratio": float(
                np.mean(synth_power) / max(np.mean(real_power), 1e-12)
            ),
            "real_power_samples": real_power.tolist(),
            "synth_power_samples": synth_power.tolist(),
        }
        bands.append(band_result)

        logger.info(
            f"    real_mean={np.mean(real_power):.6f}, synth_mean={np.mean(synth_power):.6f}, "
            f"KS={ks_stat:.4f} (p={ks_pval:.2e}), d={cohens_d:.3f}"
        )

    result = {
        "channel": channel_name,
        "n_real": int(real_patches.shape[0]),
        "n_synth": int(synth_patches.shape[0]),
        "n_bands": n_bands,
        "band_edges": band_edges.tolist(),
        "bands": bands,
    }
    return result


def _plot_band_comparison(
    result: dict,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Plot per-band power comparison with KS statistics."""
    bands = result["bands"]
    n_bands = len(bands)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Top: power comparison bar chart
    x = np.arange(n_bands)
    width = 0.35
    band_labels = [f"{b['low_freq']:.3f}-{b['high_freq']:.3f}" for b in bands]

    real_means = [b["real_power_mean"] for b in bands]
    synth_means = [b["synth_power_mean"] for b in bands]
    real_stds = [b["real_power_std"] for b in bands]
    synth_stds = [b["synth_power_std"] for b in bands]

    axes[0].bar(
        x - width / 2, real_means, width,
        yerr=real_stds, capsize=3,
        label="Real", color="steelblue", alpha=0.8,
    )
    axes[0].bar(
        x + width / 2, synth_means, width,
        yerr=synth_stds, capsize=3,
        label="Synthetic", color="coral", alpha=0.8,
    )
    axes[0].set_xlabel("Frequency Band")
    axes[0].set_ylabel("Mean Power")
    axes[0].set_title(f"Per-Band Power: {result['channel']} channel")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(band_labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_yscale("log")
    axes[0].legend(frameon=True)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Bottom: KS statistic and significance markers
    ks_stats = [b["ks_statistic"] for b in bands]
    ks_pvals = [b["ks_pvalue"] for b in bands]

    colors = ["crimson" if p < 0.05 else "gray" for p in ks_pvals]
    axes[1].bar(x, ks_stats, color=colors, alpha=0.8)
    axes[1].axhline(0.1, color="black", linestyle="--", linewidth=0.7, alpha=0.5, label="KS=0.1")
    axes[1].set_xlabel("Frequency Band")
    axes[1].set_ylabel("KS Statistic")
    axes[1].set_title("KS Test Results (red = p < 0.05)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(band_labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add p-value annotations
    for i, (ks, pval) in enumerate(zip(ks_stats, ks_pvals)):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        axes[1].text(
            i, ks + 0.02, sig, ha="center", va="bottom", fontsize=8, fontweight="bold"
        )

    plt.tight_layout()
    save_figure(
        fig, output_dir, f"band_comparison_{result['channel']}",
        formats=formats, dpi=dpi,
    )
    plt.close(fig)


def _plot_band_distributions(
    result: dict,
    output_dir: Path,
    dpi: int = 300,
    formats: list[str] | None = None,
) -> None:
    """Plot violin/box plots of per-sample power in each band."""
    bands = result["bands"]
    n_bands = len(bands)

    # Limit to at most 8 panels per row
    n_cols = min(n_bands, 4)
    n_rows = (n_bands + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_bands == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for b, band_data in enumerate(bands):
        row, col = divmod(b, n_cols)
        ax = axes[row, col]

        real_power = np.array(band_data["real_power_samples"])
        synth_power = np.array(band_data["synth_power_samples"])

        plot_data = {
            "power": np.concatenate([real_power, synth_power]),
            "source": ["Real"] * len(real_power) + ["Synthetic"] * len(synth_power),
        }

        sns.violinplot(
            x="source", y="power", data=plot_data, ax=ax,
            palette={"Real": "steelblue", "Synthetic": "coral"},
            inner="quartile", cut=0,
        )
        ax.set_title(
            f"Band {b}: [{band_data['low_freq']:.3f}, {band_data['high_freq']:.3f}]\n"
            f"KS={band_data['ks_statistic']:.3f}, d={band_data['cohens_d']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Power")

    # Remove unused axes
    for b in range(n_bands, n_rows * n_cols):
        row, col = divmod(b, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Per-Band Power Distributions: {result['channel']} channel",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    save_figure(
        fig, output_dir, f"band_distributions_{result['channel']}",
        formats=formats, dpi=dpi,
    )
    plt.close(fig)


def run_band_analysis(cfg: DictConfig, experiment_name: str) -> dict:
    """Entry point for band-pass frequency analysis of an experiment.

    Loads patches, runs per-band power analysis on configured channels,
    generates comparison plots, and saves JSON results.

    Args:
        cfg: Full diagnostics configuration (OmegaConf DictConfig).
        experiment_name: Name of the experiment to analyze.

    Returns:
        Dictionary with per-channel band analysis results.
    """
    logger.info(f"=== Band-Pass Frequency Analysis: {experiment_name} ===")

    # Load data
    patches_dir = Path(cfg.data.patches_base_dir)
    real_patches, synth_patches, _, _ = load_patches(patches_dir, experiment_name)

    # Get config
    band_cfg = cfg.feature_probes.frequency_bands
    channels = list(band_cfg.channels)
    n_bands = int(band_cfg.n_bands)

    # Output directory
    output_dir = ensure_output_dir(
        Path(cfg.output.base_dir), experiment_name, "frequency_bands"
    )
    plot_formats = list(cfg.output.plot_format)
    plot_dpi = int(cfg.output.plot_dpi)

    # Run analysis per channel
    all_results = {}
    for ch in channels:
        channel_name = CHANNEL_NAMES.get(ch, f"channel_{ch}")
        result = band_analysis(
            real_patches, synth_patches,
            channel_idx=ch,
            n_bands=n_bands,
        )
        all_results[channel_name] = result

        # Generate plots
        _plot_band_comparison(result, output_dir, dpi=plot_dpi, formats=plot_formats)
        _plot_band_distributions(result, output_dir, dpi=plot_dpi, formats=plot_formats)

    # Save results (exclude large per-sample arrays for JSON)
    json_results = {
        "experiment": experiment_name,
        "analysis": "frequency_bands",
        "channels": {},
    }
    for ch_name, result in all_results.items():
        ch_data = {
            "channel": result["channel"],
            "n_real": result["n_real"],
            "n_synth": result["n_synth"],
            "n_bands": result["n_bands"],
            "band_edges": result["band_edges"],
            "bands": [
                {k: v for k, v in b.items() if k not in ("real_power_samples", "synth_power_samples")}
                for b in result["bands"]
            ],
        }
        json_results["channels"][ch_name] = ch_data

    save_result_json(json_results, output_dir / "frequency_bands_results.json")

    logger.info(f"Band-pass frequency analysis complete for '{experiment_name}'")
    return all_results
