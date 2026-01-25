"""Spectral attribution analysis for real vs. synthetic classifier.

Identifies which frequency bands the classifier focuses on by computing the
FFT of attribution maps (GradCAM heatmaps or input gradients). This directly
links classifier XAI to the Focal Frequency Loss (FFL) design: if the
classifier focuses on specific frequency bands, those are where synthetic
images fail and FFL weighting should concentrate.

Scientific basis:
    Durall et al. (2020) showed GAN-generated images exhibit spectral
    artifacts detectable in frequency domain. Jiang et al. (2021, FFL)
    proposed adaptively weighting frequencies. This module connects the
    two: by measuring WHERE in frequency space the classifier discriminates,
    we identify which bands FFL should target.

    References:
        Durall, R., et al. (2020). "Watch your Up-Convolution: CNN Based
        Generative Deep Neural Networks are Failing to Reproduce Spectral
        Distributions." CVPR 2020.

        Jiang, L., et al. (2021). "Focal Frequency Loss for Image
        Reconstruction and Synthesis." ICCV 2021.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy import stats

from src.classification.diagnostics.utils import (
    discover_checkpoint,
    ensure_output_dir,
    load_model_from_checkpoint,
    load_patches,
    save_csv,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)


def _compute_radial_psd(image_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute radially-averaged power spectral density of a 2D image.

    Args:
        image_2d: 2D array (H, W).

    Returns:
        Tuple of (frequencies, power) arrays.
    """
    h, w = image_2d.shape
    fft = np.fft.fft2(image_2d)
    psd = np.abs(np.fft.fftshift(fft)) ** 2

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_radius = min(cx, cy)
    radial_power = np.zeros(max_radius)
    for radius in range(max_radius):
        mask = r == radius
        if mask.any():
            radial_power[radius] = psd[mask].mean()

    frequencies = np.arange(max_radius) / max_radius
    return frequencies, radial_power


def _compute_band_edges(n_bands: int, max_freq: float = 1.0) -> list[tuple[float, float]]:
    """Compute log-spaced frequency band edges.

    Args:
        n_bands: Number of frequency bands.
        max_freq: Maximum normalized frequency.

    Returns:
        List of (low, high) frequency tuples.
    """
    edges = np.logspace(np.log10(1.0 / 64), np.log10(max_freq), n_bands + 1)
    bands = [(edges[i], edges[i + 1]) for i in range(n_bands)]
    return bands


def _compute_band_power(
    frequencies: np.ndarray,
    power: np.ndarray,
    bands: list[tuple[float, float]],
) -> np.ndarray:
    """Compute power within each frequency band.

    Args:
        frequencies: Normalized frequency array.
        power: Power spectral density array.
        bands: List of (low, high) band edges.

    Returns:
        Array of per-band integrated power.
    """
    band_powers = np.zeros(len(bands))
    for i, (low, high) in enumerate(bands):
        mask = (frequencies >= low) & (frequencies < high)
        if mask.any():
            band_powers[i] = power[mask].sum()
    return band_powers


def _get_attribution_maps(
    cfg: DictConfig,
    experiment_name: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get attribution maps from GradCAM results or compute input gradients.

    Returns:
        Tuple of (attribution_maps, labels, zbins).
        attribution_maps shape: (N, H, W).
    """
    output_base_dir = Path(cfg.output.base_dir)
    source = cfg.get("xai", {}).get("spectral_attribution", {}).get("source", "gradcam")

    if source == "gradcam":
        # Load pre-computed GradCAM heatmaps
        heatmap_path = output_base_dir / experiment_name / "gradcam" / "joint" / "heatmaps.npz"
        if heatmap_path.exists():
            data = np.load(heatmap_path)
            real_maps = data["real_heatmaps"]
            synth_maps = data["synth_heatmaps"]
            real_zbins = data["real_zbins"]
            synth_zbins = data["synth_zbins"]

            all_maps = np.concatenate([real_maps, synth_maps], axis=0)
            labels = np.concatenate([
                np.zeros(len(real_maps), dtype=np.int32),
                np.ones(len(synth_maps), dtype=np.int32),
            ])
            zbins = np.concatenate([real_zbins, synth_zbins])
            logger.info(f"Loaded {len(all_maps)} GradCAM heatmaps from {heatmap_path}")
            return all_maps, labels, zbins
        else:
            logger.warning(f"GradCAM heatmaps not found at {heatmap_path}, falling back to input gradients")
            source = "input_gradient"

    # Compute input gradients as attribution maps
    patches_base_dir = Path(cfg.data.patches_base_dir)
    checkpoints_base_dir = Path(cfg.data.checkpoints_base_dir)
    n_folds = cfg.dithering.reclassification.n_folds

    real_patches, synth_patches, real_zbins, synth_zbins = load_patches(
        patches_base_dir, experiment_name
    )
    all_patches = np.concatenate([real_patches, synth_patches], axis=0)
    labels = np.concatenate([
        np.zeros(len(real_patches), dtype=np.int32),
        np.ones(len(synth_patches), dtype=np.int32),
    ])
    zbins = np.concatenate([real_zbins, synth_zbins])

    # Use first available fold
    all_grads = None
    for fold_idx in range(n_folds):
        ckpt_path = discover_checkpoint(checkpoints_base_dir, experiment_name, fold_idx, input_mode="joint")
        if ckpt_path is None:
            continue

        model = load_model_from_checkpoint(ckpt_path, in_channels=2, device=device)
        grads = []
        for i in range(len(all_patches)):
            x = torch.from_numpy(all_patches[i:i+1]).float().to(device).requires_grad_(True)
            logit = model(x).squeeze()
            model.zero_grad()
            logit.backward()
            # Sum gradient magnitude across channels for spatial attribution
            grad_mag = x.grad.detach().cpu().numpy()[0].sum(axis=0)  # (H, W)
            grads.append(np.abs(grad_mag))
        all_grads = np.stack(grads, axis=0)
        del model
        torch.cuda.empty_cache() if "cuda" in device else None
        break

    if all_grads is None:
        raise RuntimeError("No checkpoints available for gradient computation")

    return all_grads, labels, zbins


def _plot_spectral_attribution(
    results: dict,
    output_dir: Path,
) -> None:
    """Generate spectral attribution visualizations."""
    bands = results["bands"]
    n_bands = len(bands)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Attribution PSD overlay with frequency bands
    if "attribution_psd" in results:
        psd_data = results["attribution_psd"]
        freqs = np.array(psd_data["frequencies"])
        power_real = np.array(psd_data["power_real"])
        power_synth = np.array(psd_data["power_synth"])

        axes[0].semilogy(freqs, power_real, "b-", linewidth=1.5, label="Real attributions")
        axes[0].semilogy(freqs, power_synth, "r-", linewidth=1.5, label="Synthetic attributions")

        # Shade frequency bands
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_bands))
        for i, (low, high) in enumerate(bands):
            axes[0].axvspan(low, high, alpha=0.1, color=colors[i])

        axes[0].set_xlabel("Normalized frequency")
        axes[0].set_ylabel("Attribution power (log)")
        axes[0].set_title("Attribution Power Spectrum")
        axes[0].legend()

    # Panel 2: Per-band attribution fraction with concordance overlay
    band_attr = np.array(results["per_band_attribution_fraction"])
    band_labels = [f"B{i}" for i in range(n_bands)]

    bars = axes[1].bar(band_labels, band_attr, color="#2196F3", alpha=0.7,
                       label="Attribution fraction")

    # Overlay power ratio from frequency bands analysis if available
    if results.get("power_ratio_comparison"):
        power_ratios = np.array(results["power_ratio_comparison"])
        # Normalize to same scale for overlay
        pr_norm = power_ratios / (power_ratios.max() + 1e-12)
        ax2 = axes[1].twinx()
        ax2.plot(band_labels, pr_norm, "ro-", linewidth=2, markersize=8,
                 label="Power ratio (norm)")
        ax2.set_ylabel("Normalized power ratio", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.legend(loc="upper left")

    axes[1].set_xlabel("Frequency band")
    axes[1].set_ylabel("Attribution fraction")
    axes[1].set_title(f"Per-band Attribution (concordance={results.get('concordance', 0):.2f})")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    save_figure(fig, output_dir, "attribution_spectrum")
    plt.close(fig)

    # Concordance overlay plot
    if results.get("power_ratio_comparison") is not None:
        fig2, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(n_bands)
        width = 0.35

        ax.bar(x - width/2, band_attr, width, label="Classifier attention", color="#2196F3")
        pr = np.array(results["power_ratio_comparison"])
        # Show deviation from 1.0 (where 1.0 = no difference)
        pr_deviation = np.abs(pr - 1.0)
        pr_deviation_norm = pr_deviation / (pr_deviation.sum() + 1e-12)
        ax.bar(x + width/2, pr_deviation_norm, width,
               label="Real-synth difference", color="#FF9800")

        ax.set_xticks(x)
        ax.set_xticklabels([f"Band {i}" for i in range(n_bands)])
        ax.set_ylabel("Normalized weight")
        ax.set_title(
            f"Concordance: Classifier Focus vs. Actual Differences\n"
            f"(Spearman ρ = {results.get('concordance', 0):.3f})"
        )
        ax.legend()
        plt.tight_layout()
        save_figure(fig2, output_dir, "concordance_overlay")
        plt.close(fig2)


def run_spectral_attribution(
    cfg: DictConfig,
    experiment_name: str,
    device: str = "cuda",
) -> dict:
    """Run spectral attribution analysis.

    Computes the frequency-domain decomposition of classifier attribution
    maps to identify which frequency bands the classifier uses for
    discrimination.

    Args:
        cfg: Diagnostics configuration.
        experiment_name: Experiment to analyze.
        device: Torch device string.

    Returns:
        Dictionary with spectral attribution metrics.
    """
    n_bands = cfg.get("xai", {}).get("spectral_attribution", {}).get("n_bands", 5)
    output_base_dir = Path(cfg.output.base_dir)

    # Get attribution maps
    attribution_maps, labels, zbins = _get_attribution_maps(cfg, experiment_name, device)
    logger.info(f"Computing spectral attribution for {len(attribution_maps)} samples")

    # Compute radial PSD of attribution maps per class
    real_mask = labels == 0
    synth_mask = labels == 1

    # Average PSD across samples
    psd_real_list = []
    psd_synth_list = []

    for i in range(len(attribution_maps)):
        freqs, power = _compute_radial_psd(attribution_maps[i])
        if labels[i] == 0:
            psd_real_list.append(power)
        else:
            psd_synth_list.append(power)

    psd_real_mean = np.mean(psd_real_list, axis=0)
    psd_synth_mean = np.mean(psd_synth_list, axis=0)

    # Define frequency bands (same structure as frequency_bands.py)
    bands = _compute_band_edges(n_bands, max_freq=1.0)

    # Compute per-band attribution fraction (using all samples)
    all_band_powers = []
    for i in range(len(attribution_maps)):
        freqs, power = _compute_radial_psd(attribution_maps[i])
        bp = _compute_band_power(freqs, power, bands)
        all_band_powers.append(bp)

    mean_band_power = np.mean(all_band_powers, axis=0)
    total_power = mean_band_power.sum() + 1e-12
    band_attribution_fraction = mean_band_power / total_power

    # Identify peak attribution band
    peak_band = int(np.argmax(band_attribution_fraction))

    # High-frequency fraction (last 2 bands)
    hf_fraction = float(band_attribution_fraction[-2:].sum()) if n_bands >= 3 else float(
        band_attribution_fraction[-1]
    )

    # Cross-reference with existing frequency band power ratios
    freq_bands_path = (
        output_base_dir / experiment_name / "frequency_bands" /
        "frequency_bands_results.json"
    )
    power_ratio_comparison = None
    concordance = None

    if freq_bands_path.exists():
        with open(freq_bands_path) as f:
            freq_data = json.load(f)

        # Extract power ratios from image channel
        channels = freq_data.get("channels", {})
        image_channel = channels.get("image", channels.get("ch0", {}))
        if image_channel:
            band_info = image_channel.get("bands", [])
            power_ratios = [b.get("power_ratio", 1.0) for b in band_info]
            if len(power_ratios) == n_bands:
                power_ratio_comparison = power_ratios
                # Concordance: do bands with high attribution also have high deviation?
                deviation = np.abs(np.array(power_ratios) - 1.0)
                if deviation.std() > 1e-12 and band_attribution_fraction.std() > 1e-12:
                    rho, p_value = stats.spearmanr(band_attribution_fraction, deviation)
                    concordance = float(rho)
                else:
                    concordance = 0.0

    results = {
        "experiment": experiment_name,
        "n_samples": len(attribution_maps),
        "n_bands": n_bands,
        "bands": [(float(lo), float(hi)) for lo, hi in bands],
        "per_band_attribution_fraction": band_attribution_fraction.tolist(),
        "peak_attribution_band": peak_band,
        "attribution_hf_fraction": hf_fraction,
        "concordance": concordance,
        "power_ratio_comparison": power_ratio_comparison,
        "attribution_psd": {
            "frequencies": freqs.tolist(),
            "power_real": psd_real_mean.tolist(),
            "power_synth": psd_synth_mean.tolist(),
        },
    }

    # Save results
    output_dir = ensure_output_dir(output_base_dir, experiment_name, "spectral_attribution")
    save_result_json(results, output_dir / "spectral_attribution_results.json")

    # CSV for cross-experiment
    csv_row = {
        "experiment": experiment_name,
        "peak_attribution_band": peak_band,
        "attribution_hf_fraction": hf_fraction,
        "concordance": concordance,
    }
    for i, frac in enumerate(band_attribution_fraction):
        csv_row[f"band_{i}_attribution"] = float(frac)
    save_csv(pd.DataFrame([csv_row]), output_dir / "spectral_attribution_summary.csv")

    # Visualizations
    _plot_spectral_attribution(results, output_dir)

    logger.info(
        f"Spectral attribution: peak_band={peak_band}, "
        f"hf_fraction={hf_fraction:.3f}, concordance={concordance}"
    )
    return results
