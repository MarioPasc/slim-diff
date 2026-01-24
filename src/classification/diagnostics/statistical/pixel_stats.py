"""Per-pixel statistical analysis of real vs. synthetic MRI patches.

Performs Welch's t-test at each pixel location to identify spatially-localized
differences between real and synthetic FLAIR images or lesion masks. Multiple
comparison correction is applied via FDR (Benjamini-Hochberg) or other methods
supported by statsmodels.

Typical usage:
    results = run_pixel_stats(cfg, "velocity_lp_1.5")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import DictConfig
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    load_patches,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)


def compute_pixel_statistics(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    channel_idx: int = 0,
    alpha: float = 0.05,
    correction: str = "fdr_bh",
) -> dict[str, Any]:
    """Compute per-pixel Welch's t-test between real and synthetic patches.

    For each spatial location (i, j), an independent two-sample Welch's t-test
    is performed comparing the distribution of pixel values across samples.
    P-values are then corrected for multiple comparisons.

    Args:
        real_patches: Real patches, shape (N_real, 2, H, W), float32.
        synth_patches: Synthetic patches, shape (N_synth, 2, H, W), float32.
        channel_idx: Channel to analyze (0=image, 1=mask).
        alpha: Significance level for the corrected tests.
        correction: Multiple testing correction method passed to
            statsmodels.stats.multitest.multipletests. Common options:
            'fdr_bh' (Benjamini-Hochberg), 'bonferroni', 'holm'.

    Returns:
        Dictionary with:
            - mean_diff_map: (H, W) array of mean(real) - mean(synth)
            - t_stat_map: (H, W) array of t-statistics
            - p_value_map: (H, W) array of uncorrected p-values
            - p_corrected_map: (H, W) array of corrected p-values
            - significant_mask: (H, W) boolean array of significant pixels
            - fraction_significant: scalar, proportion of significant pixels
            - alpha: significance level used
            - correction: correction method used
            - n_real: number of real samples
            - n_synth: number of synthetic samples
    """
    real_channel = real_patches[:, channel_idx, :, :]  # (N, H, W)
    synth_channel = synth_patches[:, channel_idx, :, :]

    h, w = real_channel.shape[1], real_channel.shape[2]

    # Compute means for the difference map
    mean_real = real_channel.mean(axis=0)
    mean_synth = synth_channel.mean(axis=0)
    mean_diff_map = mean_real - mean_synth

    # Flatten spatial dimensions for vectorized t-test
    # scipy ttest_ind with axis=0 handles (N, pixels) efficiently
    real_flat = real_channel.reshape(real_channel.shape[0], -1)  # (N_real, H*W)
    synth_flat = synth_channel.reshape(synth_channel.shape[0], -1)  # (N_synth, H*W)

    t_stats, p_values = ttest_ind(real_flat, synth_flat, axis=0, equal_var=False)

    # Handle NaN p-values (e.g., constant pixels)
    nan_mask = np.isnan(p_values)
    p_values_clean = np.where(nan_mask, 1.0, p_values)

    # Multiple comparison correction
    reject, p_corrected, _, _ = multipletests(
        p_values_clean, alpha=alpha, method=correction
    )
    # Restore NaN positions as non-significant
    reject[nan_mask] = False
    p_corrected[nan_mask] = 1.0

    # Reshape back to spatial maps
    t_stat_map = t_stats.reshape(h, w)
    p_value_map = p_values.reshape(h, w)
    p_corrected_map = p_corrected.reshape(h, w)
    significant_mask = reject.reshape(h, w)

    fraction_significant = float(significant_mask.sum()) / significant_mask.size

    logger.info(
        f"Pixel stats (channel={channel_idx}): "
        f"{fraction_significant:.2%} significant pixels "
        f"(alpha={alpha}, correction={correction})"
    )

    return {
        "mean_diff_map": mean_diff_map,
        "t_stat_map": t_stat_map,
        "p_value_map": p_value_map,
        "p_corrected_map": p_corrected_map,
        "significant_mask": significant_mask,
        "fraction_significant": fraction_significant,
        "alpha": alpha,
        "correction": correction,
        "n_real": int(real_patches.shape[0]),
        "n_synth": int(synth_patches.shape[0]),
    }


def per_zbin_pixel_stats(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    real_zbins: np.ndarray,
    synth_zbins: np.ndarray,
    channel_idx: int = 0,
    alpha: float = 0.05,
    correction: str = "fdr_bh",
) -> dict[int, dict[str, Any]]:
    """Run per-pixel statistical tests stratified by z-bin.

    Args:
        real_patches: Real patches, shape (N_real, 2, H, W).
        synth_patches: Synthetic patches, shape (N_synth, 2, H, W).
        real_zbins: Z-bin labels for real patches, shape (N_real,).
        synth_zbins: Z-bin labels for synthetic patches, shape (N_synth,).
        channel_idx: Channel to analyze (0=image, 1=mask).
        alpha: Significance level.
        correction: Multiple testing correction method.

    Returns:
        Dictionary mapping z-bin index to pixel statistics dict.
        Bins with insufficient samples (< 5 in either group) are skipped.
    """
    all_zbins = sorted(set(np.unique(real_zbins)) & set(np.unique(synth_zbins)))
    results: dict[int, dict[str, Any]] = {}

    for zbin in all_zbins:
        real_mask = real_zbins == zbin
        synth_mask = synth_zbins == zbin

        n_real = int(real_mask.sum())
        n_synth = int(synth_mask.sum())

        if n_real < 5 or n_synth < 5:
            logger.warning(
                f"Skipping z-bin {zbin}: insufficient samples "
                f"(real={n_real}, synth={n_synth})"
            )
            continue

        stats = compute_pixel_statistics(
            real_patches[real_mask],
            synth_patches[synth_mask],
            channel_idx=channel_idx,
            alpha=alpha,
            correction=correction,
        )
        results[int(zbin)] = stats

    logger.info(
        f"Per-zbin pixel stats: computed for {len(results)}/{len(all_zbins)} z-bins"
    )
    return results


def _plot_mean_diff_heatmap(
    mean_diff_map: np.ndarray,
    output_dir: Path,
    channel_label: str,
) -> None:
    """Plot the mean difference map with a diverging colormap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.abs(mean_diff_map).max()
    im = ax.imshow(
        mean_diff_map,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean difference (real - synth)")
    ax.set_title(f"Per-pixel mean difference ({channel_label})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save_figure(fig, output_dir, f"mean_diff_{channel_label}")
    plt.close(fig)


def _plot_tstat_heatmap(
    t_stat_map: np.ndarray,
    output_dir: Path,
    channel_label: str,
) -> None:
    """Plot the t-statistic map."""
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.percentile(np.abs(t_stat_map[np.isfinite(t_stat_map)]), 99)
    im = ax.imshow(
        t_stat_map,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("t-statistic")
    ax.set_title(f"Welch's t-statistic ({channel_label})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save_figure(fig, output_dir, f"t_stat_{channel_label}")
    plt.close(fig)


def _plot_significance_mask(
    significant_mask: np.ndarray,
    mean_diff_map: np.ndarray,
    output_dir: Path,
    channel_label: str,
    alpha: float,
    correction: str,
) -> None:
    """Plot the significance mask overlaid on the mean difference."""
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.abs(mean_diff_map).max()
    ax.imshow(
        mean_diff_map,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.5,
        interpolation="nearest",
    )
    # Overlay significant pixels in semi-transparent red/blue
    overlay = np.zeros((*significant_mask.shape, 4))
    overlay[significant_mask, 3] = 0.4  # alpha channel
    overlay[significant_mask & (mean_diff_map > 0), 0] = 1.0  # red for real > synth
    overlay[significant_mask & (mean_diff_map < 0), 2] = 1.0  # blue for synth > real
    ax.imshow(overlay, interpolation="nearest")
    frac = significant_mask.sum() / significant_mask.size
    ax.set_title(
        f"Significant pixels ({channel_label})\n"
        f"{correction}, alpha={alpha}, {frac:.1%} significant"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save_figure(fig, output_dir, f"significance_mask_{channel_label}")
    plt.close(fig)


def _plot_zbin_mean_diff_grid(
    zbin_results: dict[int, dict[str, Any]],
    output_dir: Path,
    channel_label: str,
) -> None:
    """Plot a grid of per-z-bin mean difference maps."""
    zbins = sorted(zbin_results.keys())
    n_zbins = len(zbins)
    if n_zbins == 0:
        return

    ncols = min(5, n_zbins)
    nrows = (n_zbins + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    # Determine global color range
    all_diffs = [zbin_results[z]["mean_diff_map"] for z in zbins]
    vmax = max(np.abs(d).max() for d in all_diffs)

    for idx, zbin in enumerate(zbins):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.imshow(
            zbin_results[zbin]["mean_diff_map"],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        frac = zbin_results[zbin]["fraction_significant"]
        ax.set_title(f"z={zbin} ({frac:.0%} sig.)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(n_zbins, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Per-z-bin mean difference ({channel_label})", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, f"zbin_mean_diff_grid_{channel_label}")
    plt.close(fig)


def run_pixel_stats(cfg: DictConfig, experiment_name: str) -> dict[str, Any]:
    """Run the full per-pixel statistical analysis pipeline.

    Loads patches, computes pixel-level statistics (overall and per-z-bin),
    generates diagnostic plots, and saves results to disk.

    Args:
        cfg: Full diagnostics configuration (OmegaConf DictConfig).
        experiment_name: Name of the experiment to analyze.

    Returns:
        Dictionary of all computed results (per-channel and per-z-bin).
    """
    # Load data
    real_patches, synth_patches, real_zbins, synth_zbins = load_patches(
        cfg.data.patches_base_dir, experiment_name
    )

    # Configuration
    alpha = cfg.statistical.get("alpha", 0.05)
    correction = cfg.statistical.get("correction", "fdr_bh")
    channels = cfg.statistical.get("channels", [0])
    do_per_zbin = cfg.statistical.get("per_zbin", True)

    output_dir = ensure_output_dir(
        cfg.output.base_dir, experiment_name, "pixel_stats"
    )

    channel_names = {0: "image", 1: "mask"}
    all_results: dict[str, Any] = {"experiment": experiment_name}

    for ch in channels:
        ch_label = channel_names.get(ch, f"ch{ch}")
        logger.info(f"Computing pixel statistics for channel '{ch_label}'...")

        # Overall pixel stats
        stats = compute_pixel_statistics(
            real_patches, synth_patches,
            channel_idx=ch, alpha=alpha, correction=correction,
        )
        all_results[ch_label] = {
            "fraction_significant": stats["fraction_significant"],
            "n_real": stats["n_real"],
            "n_synth": stats["n_synth"],
            "alpha": alpha,
            "correction": correction,
            "mean_diff_range": [
                float(stats["mean_diff_map"].min()),
                float(stats["mean_diff_map"].max()),
            ],
            "t_stat_range": [
                float(np.nanmin(stats["t_stat_map"])),
                float(np.nanmax(stats["t_stat_map"])),
            ],
        }

        # Generate plots
        _plot_mean_diff_heatmap(stats["mean_diff_map"], output_dir, ch_label)
        _plot_tstat_heatmap(stats["t_stat_map"], output_dir, ch_label)
        _plot_significance_mask(
            stats["significant_mask"], stats["mean_diff_map"],
            output_dir, ch_label, alpha, correction,
        )

        # Per z-bin analysis
        if do_per_zbin:
            zbin_results = per_zbin_pixel_stats(
                real_patches, synth_patches, real_zbins, synth_zbins,
                channel_idx=ch, alpha=alpha, correction=correction,
            )
            _plot_zbin_mean_diff_grid(zbin_results, output_dir, ch_label)

            all_results[ch_label]["per_zbin"] = {
                int(z): {
                    "fraction_significant": r["fraction_significant"],
                    "n_real": r["n_real"],
                    "n_synth": r["n_synth"],
                }
                for z, r in zbin_results.items()
            }

    # Save JSON summary
    save_result_json(all_results, output_dir / "pixel_stats_results.json")

    return all_results
