"""Cross-experiment aggregation of diagnostic results.

Collects per-experiment CSV files from the diagnostics output directory and
produces combined DataFrames and diagnostic reports for inter-experiment
comparison. Synthesizes results into actionable categories:

1. Spectral fidelity: frequency content accuracy (PSD slopes, band power)
2. Spatial texture: microstructure accuracy (GLCM, LBP, wavelet subbands)
3. Boundary quality: lesion edge sharpness and transition accuracy
4. Spatial coherence: correlation structure (autocorrelation length)
5. Background behavior: signal leakage outside brain (deviation from -1.0)
6. Distribution accuracy: per-tissue intensity distribution matching

Also decomposes experiment names into prediction_type and lp_norm
to identify which model design choices matter most.

Usage:
    python -m src.classification.diagnostics aggregate \
        --config src/classification/diagnostics/config/diagnostics.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.classification.diagnostics.utils import save_csv

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# JSON result file locations
# ──────────────────────────────────────────────────────────────────────────────

ANALYSIS_JSON_FILES = {
    "spectral": "spectral/spectral_results.json",
    "texture": "texture/texture_results.json",
    "frequency_bands": "frequency_bands/frequency_bands_results.json",
    "pixel_stats": "pixel_stats/pixel_stats_results.json",
    "distribution_tests": "distribution_tests/distribution_tests_results.json",
    "boundary": "boundary_analysis/boundary_analysis_results.json",
    "wavelet": "wavelet_analysis/wavelet_analysis_results.json",
    "background": "full_image/background/background_analysis_results.json",
    "spatial_correlation": "full_image/spatial_correlation/spatial_correlation_results.json",
    "global_frequency": "full_image/global_frequency/global_frequency_results.json",
}


# ──────────────────────────────────────────────────────────────────────────────
# Experiment name parsing
# ──────────────────────────────────────────────────────────────────────────────


def _parse_experiment_name(name: str) -> dict[str, str]:
    """Parse experiment name into prediction_type and lp_norm.

    Expected format: '{prediction_type}_lp_{norm_value}'
    E.g., 'epsilon_lp_1.5' -> {'prediction_type': 'epsilon', 'lp_norm': '1.5'}
    """
    parts = name.split("_lp_")
    if len(parts) == 2:
        return {"prediction_type": parts[0], "lp_norm": parts[1]}
    return {"prediction_type": name, "lp_norm": "unknown"}


# ──────────────────────────────────────────────────────────────────────────────
# JSON loading helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict | None:
    """Load a JSON file, return None on failure."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Per-analysis extraction functions
# ──────────────────────────────────────────────────────────────────────────────


def _extract_spectral(data: dict, experiment: str) -> list[dict]:
    """Extract spectral metrics from JSON result."""
    rows = []
    channels = data.get("channels", {})
    for ch_name, ch_data in channels.items():
        row = {
            "experiment": experiment,
            "channel": ch_name,
            "real_slope": ch_data.get("real_slope"),
            "synth_slope": ch_data.get("synth_slope"),
            "slope_difference": ch_data.get("slope_difference"),
            "js_divergence": ch_data.get("js_divergence"),
            "n_real": ch_data.get("n_real"),
            "n_synth": ch_data.get("n_synth"),
        }
        # Per-band power ratios
        per_band = ch_data.get("per_band_power", {})
        power_ratios = per_band.get("power_ratio", [])
        for i, ratio in enumerate(power_ratios):
            row[f"band_{i}_power_ratio"] = ratio
        rows.append(row)
    return rows


def _extract_spectral_psd(data: dict, experiment: str) -> list[dict]:
    """Extract PSD curve data from spectral JSON."""
    rows = []
    channels = data.get("channels", {})
    for ch_name, ch_data in channels.items():
        freqs = ch_data.get("frequencies_real", [])
        power_real = ch_data.get("power_real", [])
        power_synth = ch_data.get("power_synth", [])
        for f, pr, ps in zip(freqs, power_real, power_synth):
            rows.append({
                "experiment": experiment,
                "channel": ch_name,
                "frequency": f,
                "power_real": pr,
                "power_synth": ps,
                "power_ratio": ps / max(pr, 1e-12),
            })
    return rows


def _extract_texture(data: dict, experiment: str) -> list[dict]:
    """Extract texture feature metrics from JSON result."""
    rows = []
    channel = data.get("channel", "image")
    glcm = data.get("glcm", {})
    for prop, prop_data in glcm.items():
        rows.append({
            "experiment": experiment,
            "channel": channel,
            "feature_type": "glcm",
            "feature": prop,
            "real_mean": prop_data.get("real_mean"),
            "real_std": prop_data.get("real_std"),
            "synth_mean": prop_data.get("synth_mean"),
            "synth_std": prop_data.get("synth_std"),
            "ks_statistic": prop_data.get("ks_statistic"),
            "ks_pvalue": prop_data.get("ks_pvalue"),
            "cohens_d": prop_data.get("cohens_d"),
        })

    gradient = data.get("gradient", {})
    if "real" in gradient and "synth" in gradient:
        rows.append({
            "experiment": experiment,
            "channel": channel,
            "feature_type": "gradient",
            "feature": "magnitude_mean",
            "real_mean": gradient["real"].get("mean"),
            "real_std": gradient["real"].get("std"),
            "synth_mean": gradient["synth"].get("mean"),
            "synth_std": gradient["synth"].get("std"),
            "ks_statistic": gradient.get("ks_statistic"),
            "ks_pvalue": gradient.get("ks_pvalue"),
            "cohens_d": gradient.get("cohens_d"),
        })

    lbp = data.get("lbp", {})
    for key, lbp_data in lbp.items():
        rows.append({
            "experiment": experiment,
            "channel": channel,
            "feature_type": "lbp",
            "feature": key,
            "real_mean": float(np.mean(lbp_data.get("real_mean_histogram", [0]))),
            "real_std": 0.0,
            "synth_mean": float(np.mean(lbp_data.get("synth_mean_histogram", [0]))),
            "synth_std": 0.0,
            "ks_statistic": lbp_data.get("max_bin_ks"),
            "ks_pvalue": None,
            "cohens_d": None,
        })
    return rows


def _extract_frequency_bands(data: dict, experiment: str) -> list[dict]:
    """Extract frequency band metrics from JSON result."""
    rows = []
    channels = data.get("channels", {})
    for ch_name, ch_data in channels.items():
        for band in ch_data.get("bands", []):
            rows.append({
                "experiment": experiment,
                "channel": ch_name,
                "band_idx": band.get("band_idx"),
                "low_freq": band.get("low_freq"),
                "high_freq": band.get("high_freq"),
                "real_power_mean": band.get("real_power_mean"),
                "synth_power_mean": band.get("synth_power_mean"),
                "power_ratio": band.get("power_ratio"),
                "ks_statistic": band.get("ks_statistic"),
                "ks_pvalue": band.get("ks_pvalue"),
                "cohens_d": band.get("cohens_d"),
            })
    return rows


def _extract_pixel_stats(data: dict, experiment: str) -> list[dict]:
    """Extract pixel statistics from JSON result."""
    rows = []
    for ch_label in ("image", "mask"):
        ch_data = data.get(ch_label, {})
        if not ch_data:
            continue
        rows.append({
            "experiment": experiment,
            "channel": ch_label,
            "z_bin": "all",
            "fraction_significant": ch_data.get("fraction_significant"),
            "n_real": ch_data.get("n_real"),
            "n_synth": ch_data.get("n_synth"),
        })
        for zbin, zdata in ch_data.get("per_zbin", {}).items():
            rows.append({
                "experiment": experiment,
                "channel": ch_label,
                "z_bin": str(zbin),
                "fraction_significant": zdata.get("fraction_significant"),
                "n_real": zdata.get("n_real"),
                "n_synth": zdata.get("n_synth"),
            })
    return rows


def _extract_distributions(data: dict, experiment: str) -> list[dict]:
    """Extract distribution test metrics from JSON result."""
    rows = []
    for ch_label in ("image", "mask"):
        ch_data = data.get(ch_label, {})
        if not ch_data or "ks_statistic" not in ch_data:
            continue
        rows.append({
            "experiment": experiment,
            "channel": ch_label,
            "tissue": "all",
            "ks_statistic": ch_data.get("ks_statistic"),
            "ks_pvalue": ch_data.get("ks_pvalue"),
            "wasserstein": ch_data.get("wasserstein"),
            "real_mean": ch_data.get("real_mean"),
            "real_std": ch_data.get("real_std"),
            "synth_mean": ch_data.get("synth_mean"),
            "synth_std": ch_data.get("synth_std"),
        })
    for tissue_name, tissue_data in data.get("per_tissue", {}).items():
        if tissue_data.get("skipped"):
            continue
        rows.append({
            "experiment": experiment,
            "channel": "image",
            "tissue": tissue_name,
            "ks_statistic": tissue_data.get("ks_statistic"),
            "ks_pvalue": tissue_data.get("ks_pvalue"),
            "wasserstein": tissue_data.get("wasserstein"),
            "real_mean": tissue_data.get("real_mean"),
            "real_std": tissue_data.get("real_std"),
            "synth_mean": tissue_data.get("synth_mean"),
            "synth_std": tissue_data.get("synth_std"),
        })
    return rows


def _extract_boundary(data: dict, experiment: str) -> list[dict]:
    """Extract boundary analysis metrics from JSON result."""
    if data.get("insufficient_data"):
        return []
    return [{
        "experiment": experiment,
        "real_sharpness_mean": data.get("real_sharpness_mean"),
        "synth_sharpness_mean": data.get("synth_sharpness_mean"),
        "sharpness_ratio": (
            data.get("synth_sharpness_mean", 0)
            / max(data.get("real_sharpness_mean", 1e-12), 1e-12)
        ),
        "real_width_mean": data.get("real_width_mean"),
        "synth_width_mean": data.get("synth_width_mean"),
        "width_ratio": (
            data.get("synth_width_mean", 0)
            / max(data.get("real_width_mean", 1e-12), 1e-12)
        ),
        "sharpness_ks_statistic": data.get("sharpness_ks", {}).get("statistic"),
        "sharpness_ks_pvalue": data.get("sharpness_ks", {}).get("pvalue"),
        "width_ks_statistic": data.get("width_ks", {}).get("statistic"),
        "width_ks_pvalue": data.get("width_ks", {}).get("pvalue"),
        "n_real_valid": data.get("n_real_valid"),
        "n_synth_valid": data.get("n_synth_valid"),
    }]


def _extract_boundary_profiles(data: dict, experiment: str) -> list[dict]:
    """Extract boundary profile data from JSON result."""
    if data.get("insufficient_data"):
        return []
    real_profile = data.get("real_mean_profile", [])
    synth_profile = data.get("synth_mean_profile", [])
    n_radii = data.get("n_radii", len(real_profile))
    max_dist = data.get("max_distance", 10.0)
    if not real_profile:
        return []
    distances = np.linspace(-max_dist, max_dist, n_radii).tolist()
    rows = []
    for i, (d, rp, sp) in enumerate(zip(distances, real_profile, synth_profile)):
        rows.append({
            "experiment": experiment,
            "distance": d,
            "real_mean_intensity": rp,
            "synth_mean_intensity": sp,
        })
    return rows


def _extract_wavelet(data: dict, experiment: str) -> list[dict]:
    """Extract wavelet analysis metrics from JSON result."""
    rows = []
    for ch_label in ("image", "mask"):
        ch_data = data.get(ch_label, {})
        if not ch_data:
            continue
        for lvl_str, level_data in ch_data.get("per_level", {}).items():
            for sb, sb_data in level_data.items():
                rows.append({
                    "experiment": experiment,
                    "channel": ch_label,
                    "level": int(lvl_str),
                    "subband": sb,
                    "real_energy": sb_data.get("real_energy"),
                    "synth_energy": sb_data.get("synth_energy"),
                    "energy_ratio": sb_data.get("energy_ratio"),
                    "ks_statistic": sb_data.get("ks_statistic"),
                    "ks_pvalue": sb_data.get("ks_pvalue"),
                    "real_coeff_std": sb_data.get("real_coeff_std"),
                    "synth_coeff_std": sb_data.get("synth_coeff_std"),
                })
    return rows


def _extract_background(data: dict, experiment: str) -> list[dict]:
    """Extract background analysis metrics from JSON result."""
    rows = []
    for source in ("real", "synth"):
        src_data = data.get(source, {})
        if src_data.get("mean") is None:
            continue
        rows.append({
            "experiment": experiment,
            "source": source,
            "mean": src_data.get("mean"),
            "std": src_data.get("std"),
            "n_unique": src_data.get("n_unique"),
            "fraction_background_mean": src_data.get("fraction_background_mean"),
            "has_noise": src_data.get("has_noise"),
            "deviation_from_minus1": src_data.get("deviation_from_minus1"),
        })
    return rows


def _extract_spatial_correlation(data: dict, experiment: str) -> list[dict]:
    """Extract spatial correlation metrics from JSON result."""
    xi_real = data.get("correlation_length_real")
    xi_synth = data.get("correlation_length_synth")
    if xi_real is None:
        return []
    return [{
        "experiment": experiment,
        "correlation_length_real": xi_real,
        "correlation_length_synth": xi_synth,
        "correlation_length_ratio": xi_synth / max(xi_real, 1e-12),
    }]


def _extract_global_frequency(data: dict, experiment: str) -> list[dict]:
    """Extract global frequency analysis metrics from JSON result."""
    return [{
        "experiment": experiment,
        "slope_real": data.get("slope_real"),
        "slope_synth": data.get("slope_synth"),
        "slope_difference": data.get("slope_difference"),
        "high_freq_ratio_real": data.get("high_freq_ratio_real"),
        "high_freq_ratio_synth": data.get("high_freq_ratio_synth"),
        "high_freq_ratio_difference": data.get("high_freq_ratio_difference"),
    }]


def _extract_global_frequency_psd(data: dict, experiment: str) -> list[dict]:
    """Extract PSD curve data from global frequency JSON."""
    rows = []
    freqs = data.get("frequencies", [])
    psd_real = data.get("psd_real", [])
    psd_synth = data.get("psd_synth", [])
    for f, pr, ps in zip(freqs, psd_real, psd_synth):
        rows.append({
            "experiment": experiment,
            "frequency": f,
            "psd_real": pr,
            "psd_synth": ps,
            "power_ratio": ps / max(pr, 1e-12),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main aggregation logic
# ──────────────────────────────────────────────────────────────────────────────


# Map analysis name to (json_key, extraction_function, output_csv_name)
EXTRACTORS = {
    "spectral_summary": ("spectral", _extract_spectral, "spectral_summary.csv"),
    "spectral_psd": ("spectral", _extract_spectral_psd, "spectral_psd.csv"),
    "texture": ("texture", _extract_texture, "texture_summary.csv"),
    "frequency_bands": ("frequency_bands", _extract_frequency_bands, "frequency_bands.csv"),
    "pixel_stats": ("pixel_stats", _extract_pixel_stats, "pixel_stats.csv"),
    "distributions": ("distribution_tests", _extract_distributions, "distribution_tests.csv"),
    "boundary_summary": ("boundary", _extract_boundary, "boundary_summary.csv"),
    "boundary_profiles": ("boundary", _extract_boundary_profiles, "boundary_profiles.csv"),
    "wavelet": ("wavelet", _extract_wavelet, "wavelet.csv"),
    "background": ("background", _extract_background, "background.csv"),
    "spatial_correlation": ("spatial_correlation", _extract_spatial_correlation, "spatial_correlation.csv"),
    "global_frequency_summary": ("global_frequency", _extract_global_frequency, "global_frequency_summary.csv"),
    "global_frequency_psd": ("global_frequency", _extract_global_frequency_psd, "global_frequency_psd.csv"),
}


def aggregate_experiments(cfg: DictConfig) -> dict[str, pd.DataFrame]:
    """Aggregate all diagnostic results across experiments.

    Reads JSON result files from each experiment directory, extracts key
    metrics, and produces combined DataFrames and a diagnostic report.

    Args:
        cfg: Diagnostics configuration with output.base_dir.

    Returns:
        Dictionary mapping analysis name to combined DataFrame.
    """
    base_dir = Path(cfg.output.base_dir)
    if not base_dir.exists():
        logger.error(f"Output directory not found: {base_dir}")
        return {}

    # Discover experiments from output directory
    experiments = sorted([
        d.name for d in base_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and d.name != "cross_experiment"
    ])

    if not experiments:
        logger.error("No experiment directories found in output.")
        return {}

    logger.info(f"Aggregating {len(experiments)} experiments: {experiments}")

    # Output directory for aggregated files
    agg_dir = base_dir / "cross_experiment"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Load all JSON results
    all_json: dict[str, dict[str, dict]] = {}  # {analysis: {experiment: data}}
    for analysis_key, json_relpath in ANALYSIS_JSON_FILES.items():
        all_json[analysis_key] = {}
        for exp in experiments:
            json_path = base_dir / exp / json_relpath
            data = _load_json(json_path)
            if data is not None:
                all_json[analysis_key][exp] = data

    # Extract and combine
    combined: dict[str, pd.DataFrame] = {}

    for output_name, (json_key, extractor, csv_name) in EXTRACTORS.items():
        all_rows = []
        for exp, data in all_json.get(json_key, {}).items():
            try:
                rows = extractor(data, exp)
                all_rows.extend(rows)
            except Exception as e:
                logger.warning(f"Failed extracting {output_name} for {exp}: {e}")

        if all_rows:
            df = pd.DataFrame(all_rows)
            # Add experiment metadata columns
            meta = df["experiment"].apply(_parse_experiment_name).apply(pd.Series)
            df = pd.concat([df, meta], axis=1)
            combined[output_name] = df
            save_csv(df, agg_dir / csv_name)
            n_exps = df["experiment"].nunique()
            logger.info(f"  {output_name}: {len(df)} rows from {n_exps} experiments")

    # Generate diagnostic reports
    if combined:
        _generate_experiment_ranking(combined, agg_dir)
        _generate_diagnostic_report(combined, agg_dir)
        _generate_grouped_analysis(combined, agg_dir)

    logger.info(f"Cross-experiment aggregation complete. Output: {agg_dir}")
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Experiment ranking
# ──────────────────────────────────────────────────────────────────────────────


def _generate_experiment_ranking(
    combined: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Generate a ranking of experiments by key diagnostic metrics.

    One row per experiment, showing the most important metrics from each
    analysis. Lower values generally indicate more realistic synthetic images.
    """
    experiments = set()
    for df in combined.values():
        if "experiment" in df.columns:
            experiments.update(df["experiment"].unique())
    experiments = sorted(experiments)

    rows = []
    for exp in experiments:
        meta = _parse_experiment_name(exp)
        row: dict = {
            "experiment": exp,
            "prediction_type": meta["prediction_type"],
            "lp_norm": meta["lp_norm"],
        }

        # Spectral: slope difference and JS divergence (image channel)
        if "spectral_summary" in combined:
            spec = combined["spectral_summary"]
            exp_spec = spec[(spec["experiment"] == exp) & (spec["channel"] == "image")]
            if not exp_spec.empty:
                row["spectral_slope_diff"] = exp_spec["slope_difference"].values[0]
                row["spectral_js_divergence"] = exp_spec["js_divergence"].values[0]
                row["spectral_synth_slope"] = exp_spec["synth_slope"].values[0]

        # Texture: max and mean effect sizes
        if "texture" in combined:
            tex = combined["texture"]
            exp_tex = tex[(tex["experiment"] == exp) & (tex["feature_type"] == "glcm")]
            if not exp_tex.empty:
                row["texture_max_cohens_d"] = exp_tex["cohens_d"].abs().max()
                row["texture_mean_ks"] = exp_tex["ks_statistic"].mean()

        # Frequency bands: max KS and which band is worst (image channel)
        if "frequency_bands" in combined:
            fb = combined["frequency_bands"]
            exp_fb = fb[(fb["experiment"] == exp) & (fb["channel"] == "image")]
            if not exp_fb.empty:
                row["freq_bands_max_ks"] = exp_fb["ks_statistic"].max()
                worst_band = exp_fb.loc[exp_fb["ks_statistic"].idxmax()]
                row["freq_bands_worst_band"] = int(worst_band["band_idx"])
                row["freq_bands_worst_ratio"] = worst_band["power_ratio"]

        # Pixel stats: fraction significant (image, all zbins)
        if "pixel_stats" in combined:
            ps = combined["pixel_stats"]
            exp_ps = ps[
                (ps["experiment"] == exp) & (ps["channel"] == "image")
                & (ps["z_bin"] == "all")
            ]
            if not exp_ps.empty:
                row["pixel_frac_significant"] = exp_ps["fraction_significant"].values[0]

        # Distribution tests: per-tissue Wasserstein
        if "distributions" in combined:
            dt = combined["distributions"]
            for tissue in ("all", "lesion", "brain", "background"):
                exp_dt = dt[
                    (dt["experiment"] == exp) & (dt["channel"] == "image")
                    & (dt["tissue"] == tissue)
                ]
                if not exp_dt.empty:
                    row[f"wasserstein_{tissue}"] = exp_dt["wasserstein"].values[0]
                    row[f"ks_{tissue}"] = exp_dt["ks_statistic"].values[0]

        # Boundary: sharpness and width ratios
        if "boundary_summary" in combined:
            bd = combined["boundary_summary"]
            exp_bd = bd[bd["experiment"] == exp]
            if not exp_bd.empty:
                row["boundary_sharpness_ratio"] = exp_bd["sharpness_ratio"].values[0]
                row["boundary_width_ratio"] = exp_bd["width_ratio"].values[0]

        # Wavelet: per-level summary (image channel)
        if "wavelet" in combined:
            wv = combined["wavelet"]
            exp_wv = wv[(wv["experiment"] == exp) & (wv["channel"] == "image")]
            if not exp_wv.empty:
                row["wavelet_max_energy_ratio"] = exp_wv["energy_ratio"].max()
                row["wavelet_min_energy_ratio"] = exp_wv["energy_ratio"].min()
                # Level 1 (finest) HH subband is most diagnostic
                l1_hh = exp_wv[(exp_wv["level"] == 1) & (exp_wv["subband"] == "HH")]
                if not l1_hh.empty:
                    row["wavelet_L1_HH_energy_ratio"] = l1_hh["energy_ratio"].values[0]

        # Global frequency: slope difference and HF ratio
        if "global_frequency_summary" in combined:
            gf = combined["global_frequency_summary"]
            exp_gf = gf[gf["experiment"] == exp]
            if not exp_gf.empty:
                row["global_slope_diff"] = exp_gf["slope_difference"].values[0]
                row["global_hf_ratio_diff"] = exp_gf["high_freq_ratio_difference"].values[0]

        # Spatial correlation: correlation length ratio
        if "spatial_correlation" in combined:
            sc = combined["spatial_correlation"]
            exp_sc = sc[sc["experiment"] == exp]
            if not exp_sc.empty:
                row["spatial_corr_ratio"] = exp_sc["correlation_length_ratio"].values[0]

        # Background: synth std and deviation
        if "background" in combined:
            bg = combined["background"]
            exp_bg_synth = bg[(bg["experiment"] == exp) & (bg["source"] == "synth")]
            if not exp_bg_synth.empty:
                row["background_synth_std"] = exp_bg_synth["std"].values[0]
                row["background_synth_deviation"] = exp_bg_synth["deviation_from_minus1"].values[0]

        rows.append(row)

    if rows:
        ranking_df = pd.DataFrame(rows)
        save_csv(ranking_df, output_dir / "experiment_ranking.csv")
        logger.info(f"Generated experiment ranking: {len(rows)} experiments")


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic report: categorized artifact analysis
# ──────────────────────────────────────────────────────────────────────────────


def _generate_diagnostic_report(
    combined: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Generate a diagnostic report categorizing artifacts by type.

    Produces a CSV with one row per (experiment, artifact_category) with
    severity scores normalized to [0, 1] range across experiments.
    Categories:
      - spectral_rolloff: Is the frequency content too flat/steep?
      - texture_deviation: Is the microstructure different?
      - boundary_blur: Are lesion edges blurred?
      - spatial_incoherence: Is the spatial correlation structure wrong?
      - background_noise: Is there signal leakage in background?
      - distribution_shift: Are intensity distributions shifted?
      - high_freq_excess: Is there excess high-frequency content?
    """
    experiments = set()
    for df in combined.values():
        if "experiment" in df.columns:
            experiments.update(df["experiment"].unique())
    experiments = sorted(experiments)

    # Collect raw scores per category per experiment
    raw_scores: dict[str, dict[str, float]] = {exp: {} for exp in experiments}

    for exp in experiments:
        # Spectral rolloff: absolute slope difference
        if "spectral_summary" in combined:
            spec = combined["spectral_summary"]
            exp_s = spec[(spec["experiment"] == exp) & (spec["channel"] == "image")]
            if not exp_s.empty:
                raw_scores[exp]["spectral_slope_diff"] = abs(exp_s["slope_difference"].values[0])
                raw_scores[exp]["spectral_js_div"] = exp_s["js_divergence"].values[0]

        # Texture deviation: mean KS across GLCM features
        if "texture" in combined:
            tex = combined["texture"]
            exp_t = tex[(tex["experiment"] == exp) & (tex["feature_type"] == "glcm")]
            if not exp_t.empty:
                raw_scores[exp]["texture_mean_ks"] = exp_t["ks_statistic"].mean()
                raw_scores[exp]["texture_max_d"] = exp_t["cohens_d"].abs().max()

        # Boundary blur: sharpness ratio (< 1 means synth is blurrier)
        if "boundary_summary" in combined:
            bd = combined["boundary_summary"]
            exp_b = bd[bd["experiment"] == exp]
            if not exp_b.empty:
                # Distance from ideal ratio of 1.0
                raw_scores[exp]["boundary_sharpness_deficit"] = abs(
                    1.0 - exp_b["sharpness_ratio"].values[0]
                )
                raw_scores[exp]["boundary_width_deficit"] = abs(
                    1.0 - exp_b["width_ratio"].values[0]
                )

        # Spatial incoherence: deviation of correlation ratio from 1.0
        if "spatial_correlation" in combined:
            sc = combined["spatial_correlation"]
            exp_sc = sc[sc["experiment"] == exp]
            if not exp_sc.empty:
                raw_scores[exp]["spatial_corr_deficit"] = abs(
                    1.0 - exp_sc["correlation_length_ratio"].values[0]
                )

        # Background noise: synth background std
        if "background" in combined:
            bg = combined["background"]
            exp_bg = bg[(bg["experiment"] == exp) & (bg["source"] == "synth")]
            if not exp_bg.empty:
                raw_scores[exp]["background_noise"] = exp_bg["std"].values[0]

        # Distribution shift: Wasserstein distance (all tissue, image)
        if "distributions" in combined:
            dt = combined["distributions"]
            exp_dt = dt[
                (dt["experiment"] == exp) & (dt["tissue"] == "all")
                & (dt["channel"] == "image")
            ]
            if not exp_dt.empty:
                raw_scores[exp]["distribution_wasserstein"] = exp_dt["wasserstein"].values[0]
            # Lesion-specific
            exp_les = dt[
                (dt["experiment"] == exp) & (dt["tissue"] == "lesion")
                & (dt["channel"] == "image")
            ]
            if not exp_les.empty:
                raw_scores[exp]["lesion_wasserstein"] = exp_les["wasserstein"].values[0]

        # High-frequency excess: global HF ratio difference
        if "global_frequency_summary" in combined:
            gf = combined["global_frequency_summary"]
            exp_gf = gf[gf["experiment"] == exp]
            if not exp_gf.empty:
                raw_scores[exp]["high_freq_excess"] = abs(
                    exp_gf["high_freq_ratio_difference"].values[0]
                )

        # Wavelet: finest-level energy ratio deviation
        if "wavelet" in combined:
            wv = combined["wavelet"]
            exp_wv = wv[(wv["experiment"] == exp) & (wv["channel"] == "image")]
            if not exp_wv.empty:
                # Average deviation from 1.0 across all subbands
                raw_scores[exp]["wavelet_energy_deviation"] = (
                    (exp_wv["energy_ratio"] - 1.0).abs().mean()
                )

    # Build the report DataFrame
    all_metrics = set()
    for scores in raw_scores.values():
        all_metrics.update(scores.keys())
    all_metrics = sorted(all_metrics)

    report_rows = []
    for exp in experiments:
        meta = _parse_experiment_name(exp)
        row = {
            "experiment": exp,
            "prediction_type": meta["prediction_type"],
            "lp_norm": meta["lp_norm"],
        }
        for metric in all_metrics:
            row[metric] = raw_scores[exp].get(metric)
        report_rows.append(row)

    if report_rows:
        report_df = pd.DataFrame(report_rows)

        # Normalize each metric column to [0, 1] across experiments
        # (higher = worse artifact)
        norm_cols = [c for c in report_df.columns if c not in
                     ("experiment", "prediction_type", "lp_norm")]
        for col in norm_cols:
            vals = report_df[col].dropna()
            if len(vals) > 0:
                vmin, vmax = vals.min(), vals.max()
                if vmax - vmin > 1e-12:
                    report_df[f"{col}_norm"] = (report_df[col] - vmin) / (vmax - vmin)
                else:
                    report_df[f"{col}_norm"] = 0.0

        # Compute overall artifact severity (mean of normalized scores)
        norm_col_names = [c for c in report_df.columns
                          if c.endswith("_norm") and c != "lp_norm"]
        if norm_col_names:
            report_df["overall_artifact_severity"] = (
                report_df[norm_col_names].mean(axis=1)
            )
            # Sort by severity (best = lowest)
            report_df = report_df.sort_values("overall_artifact_severity")

        save_csv(report_df, output_dir / "diagnostic_report.csv")
        logger.info(
            f"Generated diagnostic report with {len(all_metrics)} metrics "
            f"across {len(experiments)} experiments"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Grouped analysis: by prediction_type and lp_norm
# ──────────────────────────────────────────────────────────────────────────────


def _generate_grouped_analysis(
    combined: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Analyze metrics grouped by prediction_type and lp_norm.

    Produces two CSVs:
      - by_prediction_type.csv: mean metrics per prediction type
      - by_lp_norm.csv: mean metrics per Lp norm value

    This reveals whether the prediction type (epsilon/velocity/x0) or the
    Lp norm (1.5/2.0/2.5) has more impact on synthetic image quality.
    """
    # Build a summary row per experiment first
    experiments = set()
    for df in combined.values():
        if "experiment" in df.columns:
            experiments.update(df["experiment"].unique())
    experiments = sorted(experiments)

    summary_rows = []
    for exp in experiments:
        meta = _parse_experiment_name(exp)
        row: dict[str, Any] = {
            "experiment": exp,
            "prediction_type": meta["prediction_type"],
            "lp_norm": float(meta["lp_norm"]) if meta["lp_norm"] != "unknown" else None,
        }

        # Key metrics for grouping
        if "spectral_summary" in combined:
            spec = combined["spectral_summary"]
            exp_s = spec[(spec["experiment"] == exp) & (spec["channel"] == "image")]
            if not exp_s.empty:
                row["spectral_slope_diff"] = exp_s["slope_difference"].values[0]
                row["spectral_js_div"] = exp_s["js_divergence"].values[0]

        if "texture" in combined:
            tex = combined["texture"]
            exp_t = tex[(tex["experiment"] == exp) & (tex["feature_type"] == "glcm")]
            if not exp_t.empty:
                row["texture_mean_ks"] = exp_t["ks_statistic"].mean()

        if "boundary_summary" in combined:
            bd = combined["boundary_summary"]
            exp_b = bd[bd["experiment"] == exp]
            if not exp_b.empty:
                row["boundary_sharpness_ratio"] = exp_b["sharpness_ratio"].values[0]

        if "spatial_correlation" in combined:
            sc = combined["spatial_correlation"]
            exp_sc = sc[sc["experiment"] == exp]
            if not exp_sc.empty:
                row["spatial_corr_ratio"] = exp_sc["correlation_length_ratio"].values[0]

        if "distributions" in combined:
            dt = combined["distributions"]
            exp_dt = dt[
                (dt["experiment"] == exp) & (dt["tissue"] == "all")
                & (dt["channel"] == "image")
            ]
            if not exp_dt.empty:
                row["distribution_wasserstein"] = exp_dt["wasserstein"].values[0]

        if "global_frequency_summary" in combined:
            gf = combined["global_frequency_summary"]
            exp_gf = gf[gf["experiment"] == exp]
            if not exp_gf.empty:
                row["global_slope_diff"] = exp_gf["slope_difference"].values[0]
                row["global_hf_ratio_diff"] = exp_gf["high_freq_ratio_difference"].values[0]

        if "wavelet" in combined:
            wv = combined["wavelet"]
            exp_wv = wv[(wv["experiment"] == exp) & (wv["channel"] == "image")]
            if not exp_wv.empty:
                row["wavelet_mean_energy_ratio"] = exp_wv["energy_ratio"].mean()

        summary_rows.append(row)

    if not summary_rows:
        return

    summary_df = pd.DataFrame(summary_rows)
    metric_cols = [c for c in summary_df.columns
                   if c not in ("experiment", "prediction_type", "lp_norm")]

    # Group by prediction type
    if "prediction_type" in summary_df.columns:
        by_pred = summary_df.groupby("prediction_type")[metric_cols].agg(["mean", "std"])
        # Flatten multi-level columns
        by_pred.columns = [f"{col}_{stat}" for col, stat in by_pred.columns]
        by_pred = by_pred.reset_index()
        save_csv(by_pred, output_dir / "by_prediction_type.csv")
        logger.info(f"Generated grouped analysis by prediction_type: {len(by_pred)} groups")

    # Group by Lp norm
    if "lp_norm" in summary_df.columns:
        by_norm = summary_df.groupby("lp_norm")[metric_cols].agg(["mean", "std"])
        by_norm.columns = [f"{col}_{stat}" for col, stat in by_norm.columns]
        by_norm = by_norm.reset_index()
        save_csv(by_norm, output_dir / "by_lp_norm.csv")
        logger.info(f"Generated grouped analysis by lp_norm: {len(by_norm)} groups")
