"""Lesion boundary gradient analysis for real vs. synthetic MRI patches.

Analyzes the intensity transition profile at lesion boundaries by computing
radial intensity profiles at varying distances from the mask edge. Compares
sharpness and transition width between real and synthetic data to detect
blurring or unnatural boundary characteristics.

Typical usage:
    results = run_boundary_analysis(cfg, "velocity_lp_1.5")
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.ndimage import distance_transform_edt
from scipy.stats import ks_2samp

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    load_patches,
    save_csv,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)


def extract_boundary_profiles(
    image: np.ndarray,
    mask: np.ndarray,
    n_radii: int = 15,
    max_distance: float = 10.0,
) -> np.ndarray | None:
    """Extract radial intensity profile around the lesion boundary.

    Computes the signed distance transform from the lesion boundary, then
    averages image intensity at discrete distance bins. Negative distances
    are inside the lesion; positive distances are outside.

    Args:
        image: 2D image array, shape (H, W), values in [-1, 1].
        mask: 2D binary mask, shape (H, W), values in {-1, +1}.
        n_radii: Number of distance bins for the profile.
        max_distance: Maximum absolute distance (pixels) from boundary.

    Returns:
        1D intensity profile of shape (n_radii,) where index 0 corresponds
        to the deepest interior point and index n_radii-1 to the furthest
        exterior point. Returns None if the mask has no boundary pixels.
    """
    # Binarize mask: lesion = True
    binary_mask = mask > 0

    # Check for valid lesion region
    if binary_mask.sum() < 4 or (~binary_mask).sum() < 4:
        return None

    # Compute signed distance: positive outside, negative inside
    dist_outside = distance_transform_edt(~binary_mask)
    dist_inside = distance_transform_edt(binary_mask)
    signed_distance = dist_outside - dist_inside

    # Define distance bins from -max_distance to +max_distance
    bin_edges = np.linspace(-max_distance, max_distance, n_radii + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    profile = np.full(n_radii, np.nan)
    for i in range(n_radii):
        in_bin = (signed_distance >= bin_edges[i]) & (signed_distance < bin_edges[i + 1])
        if in_bin.sum() > 0:
            profile[i] = image[in_bin].mean()

    # If too many NaN bins, return None
    if np.isnan(profile).sum() > n_radii // 2:
        return None

    return profile


def compute_sharpness_index(profile: np.ndarray) -> float:
    """Compute the sharpness index of a boundary intensity profile.

    Defined as the peak absolute gradient magnitude along the profile.
    Higher values indicate sharper transitions.

    Args:
        profile: 1D intensity profile, shape (n_radii,).

    Returns:
        Peak absolute gradient value. Returns 0.0 if profile is invalid.
    """
    valid = ~np.isnan(profile)
    if valid.sum() < 3:
        return 0.0

    # Interpolate NaN values for gradient computation
    interp_profile = np.interp(
        np.arange(len(profile)),
        np.where(valid)[0],
        profile[valid],
    )
    gradient = np.gradient(interp_profile)
    return float(np.abs(gradient).max())


def compute_transition_width(profile: np.ndarray) -> float:
    """Compute the transition width (FWHM of gradient peak).

    Measures the full width at half maximum of the absolute gradient,
    representing how spatially extended the intensity transition is.
    Wider transitions indicate more blurring.

    Args:
        profile: 1D intensity profile, shape (n_radii,).

    Returns:
        Transition width in profile-bin units. Returns 0.0 if invalid.
    """
    valid = ~np.isnan(profile)
    if valid.sum() < 3:
        return 0.0

    # Interpolate NaN values
    interp_profile = np.interp(
        np.arange(len(profile)),
        np.where(valid)[0],
        profile[valid],
    )
    abs_gradient = np.abs(np.gradient(interp_profile))
    peak_val = abs_gradient.max()

    if peak_val < 1e-8:
        return float(len(profile))  # Flat profile, maximum width

    half_max = peak_val / 2.0
    above_half = abs_gradient >= half_max

    # Find the width of the region above half-max
    indices = np.where(above_half)[0]
    if len(indices) < 1:
        return 0.0

    width = float(indices[-1] - indices[0] + 1)
    return width


def boundary_analysis(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    n_radii: int = 15,
    max_distance: float = 10.0,
) -> dict[str, Any]:
    """Compare lesion boundary profiles between real and synthetic patches.

    Extracts boundary intensity profiles for all lesion-containing patches,
    computes sharpness and transition width metrics, and performs KS tests
    comparing these distributions.

    Args:
        real_patches: Real patches, shape (N_real, 2, H, W).
        synth_patches: Synthetic patches, shape (N_synth, 2, H, W).
        n_radii: Number of distance bins for boundary profiles.
        max_distance: Maximum distance from boundary to sample.

    Returns:
        Dictionary with:
            - real_profiles: (M_real, n_radii) array of profiles
            - synth_profiles: (M_synth, n_radii) array of profiles
            - real_mean_profile: mean profile for real
            - synth_mean_profile: mean profile for synth
            - real_sharpness: array of sharpness indices
            - synth_sharpness: array of sharpness indices
            - real_transition_width: array of transition widths
            - synth_transition_width: array of transition widths
            - sharpness_ks: KS test result for sharpness distributions
            - width_ks: KS test result for width distributions
            - n_real_valid: number of valid real profiles
            - n_synth_valid: number of valid synthetic profiles
    """
    def _extract_all_profiles(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract profiles, sharpness, and width for all patches."""
        profiles = []
        sharpness_vals = []
        width_vals = []

        for i in range(patches.shape[0]):
            image = patches[i, 0, :, :]
            mask = patches[i, 1, :, :]

            profile = extract_boundary_profiles(
                image, mask, n_radii=n_radii, max_distance=max_distance
            )
            if profile is not None:
                profiles.append(profile)
                sharpness_vals.append(compute_sharpness_index(profile))
                width_vals.append(compute_transition_width(profile))

        if not profiles:
            return np.empty((0, n_radii)), np.array([]), np.array([])

        return (
            np.stack(profiles, axis=0),
            np.array(sharpness_vals),
            np.array(width_vals),
        )

    logger.info("Extracting boundary profiles for real patches...")
    real_profiles, real_sharpness, real_width = _extract_all_profiles(real_patches)

    logger.info("Extracting boundary profiles for synthetic patches...")
    synth_profiles, synth_sharpness, synth_width = _extract_all_profiles(synth_patches)

    n_real = len(real_sharpness)
    n_synth = len(synth_sharpness)
    logger.info(
        f"Valid boundary profiles: real={n_real}, synth={n_synth}"
    )

    result: dict[str, Any] = {
        "n_real_valid": n_real,
        "n_synth_valid": n_synth,
        "n_radii": n_radii,
        "max_distance": max_distance,
    }

    if n_real < 5 or n_synth < 5:
        logger.warning("Insufficient valid profiles for statistical comparison.")
        result["insufficient_data"] = True
        return result

    # Mean profiles (suppress warnings for all-NaN bins at extreme distances
    # where no pixels fall within the distance range)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        real_mean_profile = np.nanmean(real_profiles, axis=0)
        synth_mean_profile = np.nanmean(synth_profiles, axis=0)
        real_std_profile = np.nanstd(real_profiles, axis=0)
        synth_std_profile = np.nanstd(synth_profiles, axis=0)

    # KS tests
    sharp_ks, sharp_p = ks_2samp(real_sharpness, synth_sharpness)
    width_ks, width_p = ks_2samp(real_width, synth_width)

    result.update({
        "real_profiles": real_profiles,
        "synth_profiles": synth_profiles,
        "real_mean_profile": real_mean_profile,
        "synth_mean_profile": synth_mean_profile,
        "real_std_profile": real_std_profile,
        "synth_std_profile": synth_std_profile,
        "real_sharpness": real_sharpness,
        "synth_sharpness": synth_sharpness,
        "real_transition_width": real_width,
        "synth_transition_width": synth_width,
        "sharpness_ks": {"statistic": float(sharp_ks), "pvalue": float(sharp_p)},
        "width_ks": {"statistic": float(width_ks), "pvalue": float(width_p)},
        "real_sharpness_mean": float(real_sharpness.mean()),
        "synth_sharpness_mean": float(synth_sharpness.mean()),
        "real_width_mean": float(real_width.mean()),
        "synth_width_mean": float(synth_width.mean()),
        "insufficient_data": False,
    })

    logger.info(
        f"Sharpness: real_mean={real_sharpness.mean():.4f}, "
        f"synth_mean={synth_sharpness.mean():.4f}, "
        f"KS={sharp_ks:.4f} (p={sharp_p:.2e})"
    )
    logger.info(
        f"Transition width: real_mean={real_width.mean():.2f}, "
        f"synth_mean={synth_width.mean():.2f}, "
        f"KS={width_ks:.4f} (p={width_p:.2e})"
    )

    return result


def _plot_mean_profiles(
    result: dict[str, Any],
    output_dir: Path,
    max_distance: float,
) -> None:
    """Plot mean boundary profiles with confidence bands."""
    if result.get("insufficient_data", True):
        return

    n_radii = result["n_radii"]
    bin_centers = np.linspace(-max_distance, max_distance, n_radii)

    real_mean = result["real_mean_profile"]
    synth_mean = result["synth_mean_profile"]
    real_std = result["real_std_profile"]
    synth_std = result["synth_std_profile"]

    # 95% CI using SEM
    n_real = result["n_real_valid"]
    n_synth = result["n_synth_valid"]
    real_sem = real_std / np.sqrt(n_real)
    synth_sem = synth_std / np.sqrt(n_synth)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(bin_centers, real_mean, color="steelblue", lw=2, label="Real")
    ax.fill_between(
        bin_centers,
        real_mean - 1.96 * real_sem,
        real_mean + 1.96 * real_sem,
        alpha=0.2, color="steelblue",
    )

    ax.plot(bin_centers, synth_mean, color="coral", lw=2, linestyle="--", label="Synthetic")
    ax.fill_between(
        bin_centers,
        synth_mean - 1.96 * synth_sem,
        synth_mean + 1.96 * synth_sem,
        alpha=0.2, color="coral",
    )

    ax.axvline(0, color="gray", linestyle=":", lw=1, alpha=0.7, label="Boundary")
    ax.set_xlabel("Signed distance from boundary (pixels)")
    ax.set_ylabel("Mean intensity")
    ax.set_title("Boundary intensity profile (mean +/- 95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, "boundary_profiles")
    plt.close(fig)


def _plot_sharpness_comparison(
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot box/violin plots comparing sharpness distributions."""
    if result.get("insufficient_data", True):
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Sharpness box plot
    data_sharpness = [result["real_sharpness"], result["synth_sharpness"]]
    bp = axes[0].boxplot(
        data_sharpness, labels=["Real", "Synthetic"],
        patch_artist=True, widths=0.5,
    )
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("coral")
    bp["boxes"][1].set_alpha(0.6)
    axes[0].set_ylabel("Sharpness index")
    axes[0].set_title(
        f"Boundary sharpness\n"
        f"KS={result['sharpness_ks']['statistic']:.4f} "
        f"(p={result['sharpness_ks']['pvalue']:.2e})"
    )
    axes[0].grid(True, alpha=0.3, axis="y")

    # Transition width box plot
    data_width = [result["real_transition_width"], result["synth_transition_width"]]
    bp = axes[1].boxplot(
        data_width, labels=["Real", "Synthetic"],
        patch_artist=True, widths=0.5,
    )
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("coral")
    bp["boxes"][1].set_alpha(0.6)
    axes[1].set_ylabel("Transition width (bins)")
    axes[1].set_title(
        f"Transition width (FWHM)\n"
        f"KS={result['width_ks']['statistic']:.4f} "
        f"(p={result['width_ks']['pvalue']:.2e})"
    )
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_figure(fig, output_dir, "boundary_sharpness_comparison")
    plt.close(fig)


def _plot_gradient_profiles(
    result: dict[str, Any],
    output_dir: Path,
    max_distance: float,
) -> None:
    """Plot mean gradient magnitude profiles."""
    if result.get("insufficient_data", True):
        return

    n_radii = result["n_radii"]
    bin_centers = np.linspace(-max_distance, max_distance, n_radii)

    # Compute gradient of mean profiles
    real_grad = np.abs(np.gradient(result["real_mean_profile"]))
    synth_grad = np.abs(np.gradient(result["synth_mean_profile"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bin_centers, real_grad, color="steelblue", lw=2, label="Real")
    ax.plot(bin_centers, synth_grad, color="coral", lw=2, linestyle="--", label="Synthetic")
    ax.axvline(0, color="gray", linestyle=":", lw=1, alpha=0.7)
    ax.set_xlabel("Signed distance from boundary (pixels)")
    ax.set_ylabel("|Gradient|")
    ax.set_title("Gradient magnitude profile at lesion boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, "gradient_profiles")
    plt.close(fig)


def run_boundary_analysis(cfg: DictConfig, experiment_name: str) -> dict[str, Any]:
    """Run the full lesion boundary analysis pipeline.

    Loads patches, extracts boundary profiles, compares sharpness and
    transition width between real and synthetic, and generates plots.

    Args:
        cfg: Full diagnostics configuration (OmegaConf DictConfig).
        experiment_name: Name of the experiment to analyze.

    Returns:
        Dictionary of boundary analysis results (JSON-serializable subset).
    """
    # Load data
    real_patches, synth_patches, real_zbins, synth_zbins = load_patches(
        cfg.data.patches_base_dir, experiment_name
    )

    # Configuration
    boundary_cfg = cfg.statistical.get("boundary", {})
    n_radii = boundary_cfg.get("n_radii", 15)
    max_distance = boundary_cfg.get("max_distance", 10.0)

    output_dir = ensure_output_dir(
        cfg.output.base_dir, experiment_name, "boundary_analysis"
    )

    logger.info(
        f"Running boundary analysis: n_radii={n_radii}, max_distance={max_distance}"
    )

    # Run analysis
    result = boundary_analysis(
        real_patches, synth_patches,
        n_radii=n_radii, max_distance=max_distance,
    )

    # Generate plots
    _plot_mean_profiles(result, output_dir, max_distance)
    _plot_sharpness_comparison(result, output_dir)
    _plot_gradient_profiles(result, output_dir, max_distance)

    # Build JSON-serializable summary (exclude large arrays)
    summary: dict[str, Any] = {
        "experiment": experiment_name,
        "n_radii": n_radii,
        "max_distance": max_distance,
        "n_real_valid": result["n_real_valid"],
        "n_synth_valid": result["n_synth_valid"],
        "insufficient_data": result.get("insufficient_data", False),
    }

    if not result.get("insufficient_data", True):
        summary.update({
            "real_sharpness_mean": result["real_sharpness_mean"],
            "synth_sharpness_mean": result["synth_sharpness_mean"],
            "real_width_mean": result["real_width_mean"],
            "synth_width_mean": result["synth_width_mean"],
            "sharpness_ks": result["sharpness_ks"],
            "width_ks": result["width_ks"],
            "real_mean_profile": result["real_mean_profile"].tolist(),
            "synth_mean_profile": result["synth_mean_profile"].tolist(),
        })

    save_result_json(summary, output_dir / "boundary_analysis_results.json")

    # Save CSV: boundary profile data and summary metrics
    csv_rows = []
    if not result.get("insufficient_data", True):
        # Profile curve data
        profile_rows = []
        bin_centers = np.linspace(-max_distance, max_distance, n_radii)
        for i, dist in enumerate(bin_centers):
            profile_rows.append({
                "experiment": experiment_name,
                "distance": float(dist),
                "real_mean_intensity": float(result["real_mean_profile"][i]),
                "synth_mean_intensity": float(result["synth_mean_profile"][i]),
                "real_std": float(result["real_std_profile"][i]),
                "synth_std": float(result["synth_std_profile"][i]),
            })
        save_csv(pd.DataFrame(profile_rows), output_dir / "boundary_profiles.csv")

        # Summary metrics
        csv_rows.append({
            "experiment": experiment_name,
            "real_sharpness_mean": result["real_sharpness_mean"],
            "synth_sharpness_mean": result["synth_sharpness_mean"],
            "real_width_mean": result["real_width_mean"],
            "synth_width_mean": result["synth_width_mean"],
            "sharpness_ks_statistic": result["sharpness_ks"]["statistic"],
            "sharpness_ks_pvalue": result["sharpness_ks"]["pvalue"],
            "width_ks_statistic": result["width_ks"]["statistic"],
            "width_ks_pvalue": result["width_ks"]["pvalue"],
            "n_real_valid": result["n_real_valid"],
            "n_synth_valid": result["n_synth_valid"],
        })
        save_csv(pd.DataFrame(csv_rows), output_dir / "boundary_summary.csv")

    return summary
