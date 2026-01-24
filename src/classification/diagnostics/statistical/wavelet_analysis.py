"""Multi-scale wavelet decomposition comparison for real vs. synthetic patches.

Performs 2D discrete wavelet transform (DWT) on image patches and compares
the energy and coefficient distributions at each decomposition level and
subband (LH, HL, HH). Detects scale-specific differences such as missing
high-frequency texture or over-smoothing in synthetic data.

Requires PyWavelets (pywt) as an optional dependency.

Typical usage:
    results = run_wavelet_analysis(cfg, "velocity_lp_1.5")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy.stats import ks_2samp

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    load_patches,
    save_figure,
    save_result_json,
)

try:
    import pywt

    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Subband names for detail coefficients at each level
SUBBAND_NAMES = ("LH", "HL", "HH")


def _check_pywt() -> None:
    """Raise an informative error if PyWavelets is not installed."""
    if not PYWT_AVAILABLE:
        raise ImportError(
            "PyWavelets (pywt) is required for wavelet analysis but is not installed. "
            "Install it with: pip install PyWavelets"
        )


def wavelet_decompose(
    images: np.ndarray,
    wavelet: str = "db4",
    levels: int = 4,
) -> dict[int, dict[str, np.ndarray]]:
    """Compute 2D DWT decomposition for a batch of images.

    Decomposes each image using the specified wavelet and number of levels.
    Returns detail coefficients (LH, HL, HH) at each level, averaged across
    the batch dimension.

    Args:
        images: Batch of 2D images, shape (N, H, W).
        wavelet: Wavelet name (e.g., 'db4', 'haar', 'sym4').
            Must be a valid PyWavelets wavelet name.
        levels: Number of decomposition levels.

    Returns:
        Dictionary mapping level (1-indexed, 1=finest) to a dict with keys
        'LH', 'HL', 'HH', each containing an array of shape (N, h_l, w_l)
        where h_l, w_l are the spatial dimensions at that level.

    Raises:
        ImportError: If PyWavelets is not installed.
    """
    _check_pywt()

    n_samples = images.shape[0]
    result: dict[int, dict[str, list[np.ndarray]]] = {
        lvl: {"LH": [], "HL": [], "HH": []}
        for lvl in range(1, levels + 1)
    }

    for i in range(n_samples):
        coeffs = pywt.wavedec2(images[i], wavelet=wavelet, level=levels)
        # coeffs[0] = approximation at coarsest level
        # coeffs[j] = (LH, HL, HH) detail tuple at level (levels - j + 1)
        for j in range(1, len(coeffs)):
            level = levels - j + 1  # 1-indexed, 1 = finest
            lh, hl, hh = coeffs[j]
            result[level]["LH"].append(lh)
            result[level]["HL"].append(hl)
            result[level]["HH"].append(hh)

    # Stack into arrays
    output: dict[int, dict[str, np.ndarray]] = {}
    for lvl in range(1, levels + 1):
        output[lvl] = {
            sb: np.stack(result[lvl][sb], axis=0)
            for sb in SUBBAND_NAMES
        }

    return output


def compute_subband_energy(coeffs: np.ndarray) -> float:
    """Compute the energy (mean squared value) of wavelet coefficients.

    Args:
        coeffs: Wavelet coefficient array, shape (N, h, w) or (h, w).

    Returns:
        Mean squared coefficient value (scalar).
    """
    return float(np.mean(coeffs ** 2))


def wavelet_analysis(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    channel_idx: int = 0,
    wavelet: str = "db4",
    levels: int = 4,
) -> dict[str, Any]:
    """Compare wavelet decompositions of real and synthetic patches.

    For each decomposition level and subband, computes:
      - Energy (mean squared coefficient)
      - Energy ratio (synth/real)
      - KS test on coefficient value distributions

    Args:
        real_patches: Real patches, shape (N_real, 2, H, W).
        synth_patches: Synthetic patches, shape (N_synth, 2, H, W).
        channel_idx: Channel to analyze (0=image, 1=mask).
        wavelet: Wavelet name for PyWavelets.
        levels: Number of DWT decomposition levels.

    Returns:
        Dictionary with per-level, per-subband results including energy
        values, ratios, and KS test statistics.

    Raises:
        ImportError: If PyWavelets is not installed.
    """
    _check_pywt()

    real_images = real_patches[:, channel_idx, :, :]
    synth_images = synth_patches[:, channel_idx, :, :]

    logger.info(
        f"Wavelet decomposition: wavelet={wavelet}, levels={levels}, "
        f"real={real_images.shape[0]} samples, synth={synth_images.shape[0]} samples"
    )

    real_coeffs = wavelet_decompose(real_images, wavelet=wavelet, levels=levels)
    synth_coeffs = wavelet_decompose(synth_images, wavelet=wavelet, levels=levels)

    results: dict[str, Any] = {
        "wavelet": wavelet,
        "levels": levels,
        "channel_idx": channel_idx,
        "n_real": int(real_images.shape[0]),
        "n_synth": int(synth_images.shape[0]),
        "per_level": {},
    }

    for lvl in range(1, levels + 1):
        level_results: dict[str, Any] = {}

        for sb in SUBBAND_NAMES:
            real_sb = real_coeffs[lvl][sb]
            synth_sb = synth_coeffs[lvl][sb]

            # Energy comparison
            real_energy = compute_subband_energy(real_sb)
            synth_energy = compute_subband_energy(synth_sb)
            energy_ratio = synth_energy / max(real_energy, 1e-10)

            # KS test on flattened coefficient distributions
            # Subsample for memory efficiency
            max_ks_samples = 200_000
            rng = np.random.default_rng(42 + lvl)
            real_flat = real_sb.ravel()
            synth_flat = synth_sb.ravel()

            if len(real_flat) > max_ks_samples:
                real_flat = rng.choice(real_flat, max_ks_samples, replace=False)
            if len(synth_flat) > max_ks_samples:
                synth_flat = rng.choice(synth_flat, max_ks_samples, replace=False)

            ks_stat, ks_pvalue = ks_2samp(real_flat, synth_flat)

            level_results[sb] = {
                "real_energy": real_energy,
                "synth_energy": synth_energy,
                "energy_ratio": energy_ratio,
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "real_coeff_mean": float(real_sb.mean()),
                "synth_coeff_mean": float(synth_sb.mean()),
                "real_coeff_std": float(real_sb.std()),
                "synth_coeff_std": float(synth_sb.std()),
            }

        results["per_level"][lvl] = level_results

        logger.info(
            f"  Level {lvl}: "
            + ", ".join(
                f"{sb} energy_ratio={level_results[sb]['energy_ratio']:.3f}"
                for sb in SUBBAND_NAMES
            )
        )

    return results


def _plot_energy_bar_chart(
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot energy per level and subband as grouped bar chart."""
    levels = sorted(results["per_level"].keys())
    n_levels = len(levels)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_levels)
    width = 0.12
    offsets = {"LH": -width, "HL": 0, "HH": width}
    colors_real = {"LH": "#4878A8", "HL": "#48A878", "HH": "#A84878"}
    colors_synth = {"LH": "#7AAAD4", "HL": "#7AD4AA", "HH": "#D47AAA"}

    for sb in SUBBAND_NAMES:
        real_energies = [results["per_level"][lvl][sb]["real_energy"] for lvl in levels]
        synth_energies = [results["per_level"][lvl][sb]["synth_energy"] for lvl in levels]

        ax.bar(
            x + offsets[sb] - width / 2, real_energies, width,
            label=f"Real {sb}", color=colors_real[sb], alpha=0.85,
        )
        ax.bar(
            x + offsets[sb] + width / 2, synth_energies, width,
            label=f"Synth {sb}", color=colors_synth[sb], alpha=0.85,
        )

    ax.set_xlabel("Decomposition level (1=finest)")
    ax.set_ylabel("Energy (mean squared coefficient)")
    ax.set_title(f"Wavelet subband energy ({results['wavelet']}, {n_levels} levels)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {lvl}" for lvl in levels])
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_figure(fig, output_dir, "wavelet_energy_per_level")
    plt.close(fig)


def _plot_ks_statistics_table(
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot a heatmap of KS statistics per level and subband."""
    levels = sorted(results["per_level"].keys())
    n_levels = len(levels)
    n_subbands = len(SUBBAND_NAMES)

    ks_matrix = np.zeros((n_levels, n_subbands))
    pval_matrix = np.zeros((n_levels, n_subbands))

    for i, lvl in enumerate(levels):
        for j, sb in enumerate(SUBBAND_NAMES):
            ks_matrix[i, j] = results["per_level"][lvl][sb]["ks_statistic"]
            pval_matrix[i, j] = results["per_level"][lvl][sb]["ks_pvalue"]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(ks_matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    # Annotate cells with KS statistic and significance marker
    for i in range(n_levels):
        for j in range(n_subbands):
            sig = "*" if pval_matrix[i, j] < 0.05 else ""
            ax.text(
                j, i, f"{ks_matrix[i, j]:.3f}{sig}",
                ha="center", va="center", fontsize=9,
                color="white" if ks_matrix[i, j] > 0.3 else "black",
            )

    ax.set_xticks(range(n_subbands))
    ax.set_xticklabels(SUBBAND_NAMES)
    ax.set_yticks(range(n_levels))
    ax.set_yticklabels([f"Level {lvl}" for lvl in levels])
    ax.set_xlabel("Subband")
    ax.set_ylabel("Decomposition level")
    ax.set_title("KS statistics per subband (* = p < 0.05)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="KS statistic")

    fig.tight_layout()
    save_figure(fig, output_dir, "wavelet_ks_heatmap")
    plt.close(fig)


def _plot_coefficient_violins(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    channel_idx: int,
    wavelet: str,
    levels: int,
    output_dir: Path,
    max_samples_per_violin: int = 50_000,
) -> None:
    """Plot violin plots of coefficient distributions per subband."""
    _check_pywt()

    real_images = real_patches[:, channel_idx, :, :]
    synth_images = synth_patches[:, channel_idx, :, :]

    real_coeffs = wavelet_decompose(real_images, wavelet=wavelet, levels=levels)
    synth_coeffs = wavelet_decompose(synth_images, wavelet=wavelet, levels=levels)

    n_plots = levels * len(SUBBAND_NAMES)
    ncols = len(SUBBAND_NAMES)
    nrows = levels

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    rng = np.random.default_rng(42)

    for i, lvl in enumerate(range(1, levels + 1)):
        for j, sb in enumerate(SUBBAND_NAMES):
            ax = axes[i, j]

            real_vals = real_coeffs[lvl][sb].ravel()
            synth_vals = synth_coeffs[lvl][sb].ravel()

            # Subsample for visualization
            if len(real_vals) > max_samples_per_violin:
                real_vals = rng.choice(real_vals, max_samples_per_violin, replace=False)
            if len(synth_vals) > max_samples_per_violin:
                synth_vals = rng.choice(synth_vals, max_samples_per_violin, replace=False)

            parts = ax.violinplot(
                [real_vals, synth_vals],
                positions=[0, 1],
                showmeans=True,
                showextrema=False,
            )
            # Color the violins
            for idx, body in enumerate(parts["bodies"]):
                body.set_facecolor("steelblue" if idx == 0 else "coral")
                body.set_alpha(0.6)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Real", "Synth"], fontsize=8)
            ax.set_title(f"L{lvl} {sb}", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

            if j == 0:
                ax.set_ylabel("Coefficient value")

    fig.suptitle(
        f"Wavelet coefficient distributions ({wavelet})", fontsize=12, y=1.01
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "wavelet_coefficient_violins")
    plt.close(fig)


def run_wavelet_analysis(cfg: DictConfig, experiment_name: str) -> dict[str, Any]:
    """Run the full wavelet decomposition analysis pipeline.

    Loads patches, performs DWT decomposition, compares energy and coefficient
    distributions per level/subband, and generates diagnostic plots.

    Args:
        cfg: Full diagnostics configuration (OmegaConf DictConfig).
        experiment_name: Name of the experiment to analyze.

    Returns:
        Dictionary of wavelet analysis results.

    Raises:
        ImportError: If PyWavelets is not installed.
    """
    _check_pywt()

    # Load data
    real_patches, synth_patches, real_zbins, synth_zbins = load_patches(
        cfg.data.patches_base_dir, experiment_name
    )

    # Configuration
    wavelet_cfg = cfg.statistical.get("wavelet", {})
    wavelet = wavelet_cfg.get("wavelet", "db4")
    levels = wavelet_cfg.get("levels", 4)
    channels = cfg.statistical.get("channels", [0])

    output_dir = ensure_output_dir(
        cfg.output.base_dir, experiment_name, "wavelet_analysis"
    )

    channel_names = {0: "image", 1: "mask"}
    all_results: dict[str, Any] = {"experiment": experiment_name}

    for ch in channels:
        ch_label = channel_names.get(ch, f"ch{ch}")
        logger.info(f"Running wavelet analysis for channel '{ch_label}'...")

        results = wavelet_analysis(
            real_patches, synth_patches,
            channel_idx=ch, wavelet=wavelet, levels=levels,
        )

        # Store results
        all_results[ch_label] = {
            "wavelet": wavelet,
            "levels": levels,
            "n_real": results["n_real"],
            "n_synth": results["n_synth"],
            "per_level": results["per_level"],
        }

        # Generate plots
        _plot_energy_bar_chart(results, output_dir)
        _plot_ks_statistics_table(results, output_dir)
        _plot_coefficient_violins(
            real_patches, synth_patches,
            channel_idx=ch, wavelet=wavelet, levels=levels,
            output_dir=output_dir,
        )

    # Save JSON summary
    save_result_json(all_results, output_dir / "wavelet_analysis_results.json")

    return all_results
