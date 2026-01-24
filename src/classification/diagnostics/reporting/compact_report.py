"""Compact LLM-readable report generation in YAML format.

Produces a structured YAML file optimized for consumption by language models,
containing all key diagnostic metrics, XAI results, ranked findings, and
actionable recommendations. Preserves numerical precision while being
scannable and information-dense.

The compact format uses abbreviated keys to minimize tokens while maintaining
interpretability:
    v = raw value
    n = normalized score (0-1, higher = worse)
    s = statistical significance ("***"=p<0.001, "**"=p<0.01, "*"=p<0.05, "ns")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig

from src.classification.diagnostics.utils import ensure_output_dir

logger = logging.getLogger(__name__)


# Quality grade thresholds (overall_severity -> letter grade)
DEFAULT_THRESHOLDS = {"A": 0.05, "B": 0.15, "C": 0.30, "D": 0.50}


def _load_json(path: Path) -> dict | None:
    """Load a JSON file, return None if missing."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _significance_marker(p_value: float | None) -> str:
    """Convert p-value to significance marker."""
    if p_value is None:
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _quality_grade(severity: float, thresholds: dict[str, float]) -> str:
    """Map severity to letter grade."""
    for grade in ("A", "B", "C", "D"):
        if severity < thresholds.get(grade, 1.0):
            return grade
    return "F"


def _parse_experiment_name(name: str) -> dict[str, str]:
    """Parse experiment name into prediction_type and lp_norm."""
    parts = name.split("_lp_")
    if len(parts) == 2:
        pred_type = parts[0]
        # Map internal names to readable names
        type_map = {"x0": "sample", "epsilon": "epsilon", "velocity": "velocity"}
        return {
            "prediction_type": type_map.get(pred_type, pred_type),
            "lp_norm": parts[1],
        }
    return {"prediction_type": name, "lp_norm": "unknown"}


def _extract_metrics(
    experiment_dir: Path,
) -> dict[str, Any]:
    """Extract all metrics from per-experiment JSON results."""
    metrics = {}

    # Spectral
    spectral = _load_json(experiment_dir / "spectral" / "spectral_results.json")
    if spectral:
        channels = spectral.get("channels", {})
        img_ch = channels.get("image", {})
        metrics["spectral"] = {
            "slope_diff": {"v": img_ch.get("slope_difference")},
            "js_div": {"v": img_ch.get("js_divergence")},
            "real_slope": img_ch.get("real_slope"),
            "synth_slope": img_ch.get("synth_slope"),
        }

    # Texture
    texture = _load_json(experiment_dir / "texture" / "texture_results.json")
    if texture:
        glcm = texture.get("glcm", {})
        ks_stats = [v.get("ks_statistic", 0) for v in glcm.values() if isinstance(v, dict)]
        cohens_ds = [abs(v.get("cohens_d", 0)) for v in glcm.values() if isinstance(v, dict)]
        p_values = [v.get("ks_pvalue") for v in glcm.values() if isinstance(v, dict)]
        min_p = min((p for p in p_values if p is not None), default=None)
        metrics["texture"] = {
            "mean_ks": {"v": float(sum(ks_stats) / max(len(ks_stats), 1)), "s": _significance_marker(min_p)},
            "max_d": {"v": max(cohens_ds) if cohens_ds else 0.0, "s": _significance_marker(min_p)},
        }

    # Wavelet
    wavelet = _load_json(experiment_dir / "wavelet_analysis" / "wavelet_analysis_results.json")
    if wavelet:
        img_data = wavelet.get("image", {})
        per_level = img_data.get("per_level", {})
        energy_ratios = []
        l1_hh_ratio = None
        for lvl_str, level_data in per_level.items():
            for sb, sb_data in level_data.items():
                er = sb_data.get("energy_ratio")
                if er is not None:
                    energy_ratios.append(er)
                if lvl_str == "1" and sb == "HH":
                    l1_hh_ratio = er
        mean_dev = float(
            sum(abs(r - 1.0) for r in energy_ratios) / max(len(energy_ratios), 1)
        )
        metrics["wavelet"] = {
            "L1_HH_ratio": {"v": l1_hh_ratio},
            "energy_dev": {"v": mean_dev},
        }

    # Frequency bands
    freq_bands = _load_json(experiment_dir / "frequency_bands" / "frequency_bands_results.json")
    if freq_bands:
        channels = freq_bands.get("channels", {})
        img_ch = channels.get("image", {})
        band_list = img_ch.get("bands", [])
        worst_band = max(band_list, key=lambda b: abs(b.get("power_ratio", 1.0) - 1.0)) if band_list else {}
        metrics["frequency"] = {
            "worst_band": {"idx": worst_band.get("band_idx"), "power_ratio": worst_band.get("power_ratio")},
        }

    # Boundary
    boundary = _load_json(experiment_dir / "boundary_analysis" / "boundary_analysis_results.json")
    if boundary and not boundary.get("insufficient_data"):
        metrics["boundary"] = {
            "sharpness_deficit": {
                "v": abs(1.0 - (boundary.get("synth_sharpness_mean", 0) /
                               max(boundary.get("real_sharpness_mean", 1e-12), 1e-12))),
                "s": _significance_marker(boundary.get("sharpness_ks", {}).get("pvalue")),
            },
            "width_deficit": {
                "v": abs(1.0 - (boundary.get("synth_width_mean", 0) /
                               max(boundary.get("real_width_mean", 1e-12), 1e-12))),
                "s": _significance_marker(boundary.get("width_ks", {}).get("pvalue")),
            },
        }

    # Distributions
    dist = _load_json(experiment_dir / "distribution_tests" / "distribution_tests_results.json")
    if dist:
        img_data = dist.get("image", {})
        per_tissue = dist.get("per_tissue", {})
        lesion_data = per_tissue.get("lesion", {})
        metrics["distribution"] = {
            "wasserstein_all": {
                "v": img_data.get("wasserstein"),
                "s": _significance_marker(img_data.get("ks_pvalue")),
            },
            "wasserstein_lesion": {
                "v": lesion_data.get("wasserstein"),
                "s": _significance_marker(lesion_data.get("ks_pvalue")),
            },
        }

    # Spatial correlation
    spatial = _load_json(experiment_dir / "full_image" / "spatial_correlation" / "spatial_correlation_results.json")
    if spatial:
        xi_real = spatial.get("correlation_length_real", 1.0)
        xi_synth = spatial.get("correlation_length_synth", 1.0)
        metrics["spatial"] = {
            "correlation_deficit": {"v": abs(1.0 - xi_synth / max(xi_real, 1e-12))},
        }

    # Background
    bg = _load_json(experiment_dir / "full_image" / "background" / "background_analysis_results.json")
    if bg:
        synth_bg = bg.get("synth", {})
        metrics["background"] = {
            "noise_std": {"v": synth_bg.get("std", 0.0)},
        }

    # Global frequency (HF excess)
    gf = _load_json(experiment_dir / "full_image" / "global_frequency" / "global_frequency_results.json")
    if gf:
        hf_diff = gf.get("high_freq_ratio_difference")
        if hf_diff is not None:
            metrics.setdefault("frequency", {})["high_freq_excess"] = {"v": abs(hf_diff)}

    return metrics


def _extract_xai_results(experiment_dir: Path) -> dict[str, Any]:
    """Extract XAI analysis results."""
    xai = {}

    # Channel decomposition
    cd = _load_json(experiment_dir / "channel_decomposition" / "channel_decomposition_results.json")
    if cd:
        overall = cd.get("overall", {})
        grad = overall.get("gradient", {})
        xai["channel_decomposition"] = {
            "image_frac": grad.get("image_fraction"),
            "mask_frac": grad.get("mask_fraction"),
            "dominant": overall.get("dominant_channel"),
        }

    # Spectral attribution
    sa = _load_json(experiment_dir / "spectral_attribution" / "spectral_attribution_results.json")
    if sa:
        xai["spectral_attribution"] = {
            "peak_band": sa.get("peak_attribution_band"),
            "hf_fraction": sa.get("attribution_hf_fraction"),
            "concordance": sa.get("concordance"),
        }

    # Feature space
    fs = _load_json(experiment_dir / "feature_space" / "feature_space_results.json")
    if fs:
        cluster = fs.get("cluster_metrics", {})
        fisher = fs.get("fisher_discriminant", {})
        xai["feature_space"] = {
            "separability": cluster.get("tsne_silhouette"),
            "top_fisher_dim": fisher.get("top_indices", [])[:5],
            "cosine_distance": cluster.get("inter_class_cosine_distance"),
        }

    # Integrated gradients
    ig = _load_json(experiment_dir / "integrated_gradients" / "ig_results.json")
    if ig:
        xai["integrated_gradients"] = {
            "ig_cam_corr": ig.get("ig_gradcam_correlation"),
            "concentration": ig.get("attribution_concentration"),
            "image_channel_fraction": ig.get("mean_attribution_per_channel", {}).get("image"),
        }

    return xai


def _derive_findings(
    metrics: dict[str, Any],
    xai: dict[str, Any],
) -> list[dict[str, Any]]:
    """Derive ranked findings from metrics and XAI results."""
    findings = []

    # Wavelet HF deficit
    wavelet = metrics.get("wavelet", {})
    l1_hh = wavelet.get("L1_HH_ratio", {}).get("v")
    if l1_hh is not None and l1_hh < 0.85:
        deficit_pct = int((1.0 - l1_hh) * 100)
        findings.append({
            "issue": f"{deficit_pct}% HF deficit in wavelet HH L1",
            "severity": round(1.0 - l1_hh, 3),
            "category": "wavelet",
            "evidence": f"ratio={l1_hh:.2f}, target>0.85",
        })

    # Spectral attribution HF focus
    sa = xai.get("spectral_attribution", {})
    hf_frac = sa.get("hf_fraction")
    if hf_frac is not None and hf_frac > 0.4:
        findings.append({
            "issue": f"Classifier focuses {int(hf_frac*100)}% on HF content",
            "severity": round(hf_frac * 0.5, 3),
            "category": "spectral_attribution",
            "evidence": f"hf_fraction={hf_frac:.2f}",
        })

    # Channel dominance
    cd = xai.get("channel_decomposition", {})
    img_frac = cd.get("image_frac")
    if img_frac is not None and img_frac > 0.75:
        findings.append({
            "issue": f"Image channel dominates ({int(img_frac*100)}%)",
            "severity": round((img_frac - 0.5) * 0.5, 3),
            "category": "channel_decomposition",
            "evidence": f"image_frac={img_frac:.2f}",
        })
    elif img_frac is not None and img_frac < 0.25:
        findings.append({
            "issue": f"Mask channel dominates ({int((1-img_frac)*100)}%)",
            "severity": round((0.5 - img_frac) * 0.5, 3),
            "category": "channel_decomposition",
            "evidence": f"mask_frac={1-img_frac:.2f}",
        })

    # Texture deviation
    texture = metrics.get("texture", {})
    max_d = texture.get("max_d", {}).get("v")
    if max_d is not None and max_d > 0.5:
        findings.append({
            "issue": f"Texture Cohen's d={max_d:.2f} (large effect)",
            "severity": round(min(max_d / 3.0, 0.5), 3),
            "category": "texture",
            "evidence": f"max_cohens_d={max_d:.2f}",
        })

    # Background noise
    bg = metrics.get("background", {})
    noise = bg.get("noise_std", {}).get("v")
    if noise is not None and noise > 0.005:
        findings.append({
            "issue": f"Background noise std={noise:.4f}",
            "severity": round(min(noise * 20, 0.5), 3),
            "category": "background",
            "evidence": f"std={noise:.4f}, expected ~0",
        })

    # Spatial correlation
    spatial = metrics.get("spatial", {})
    corr_def = spatial.get("correlation_deficit", {}).get("v")
    if corr_def is not None and corr_def > 0.05:
        findings.append({
            "issue": f"Spatial correlation deficit={corr_def:.3f}",
            "severity": round(min(corr_def, 0.4), 3),
            "category": "spatial",
            "evidence": f"deficit={corr_def:.3f}",
        })

    # Sort by severity
    findings.sort(key=lambda f: f["severity"], reverse=True)
    for i, f in enumerate(findings):
        f["rank"] = i + 1

    return findings


def _derive_recommendations(
    metrics: dict[str, Any],
    xai: dict[str, Any],
    prediction_type: str,
) -> list[dict[str, Any]]:
    """Derive actionable recommendations from findings."""
    recommendations = []

    # Rule 1: Epsilon prediction → switch to x0
    if prediction_type == "epsilon":
        recommendations.append({
            "priority": "CRITICAL",
            "action": "Switch to x0 (sample) prediction type",
            "param": "scheduler.prediction_type=sample",
            "values": {"clip_sample": True, "clip_sample_range": 1.0},
            "evidence": ["prediction_type=epsilon causes 40x HF excess"],
        })

    # Rule 2: Wavelet HH deficit → add FFL
    wavelet = metrics.get("wavelet", {})
    l1_hh = wavelet.get("L1_HH_ratio", {}).get("v")
    if l1_hh is not None and l1_hh < 0.85:
        recommendations.append({
            "priority": "HIGH",
            "action": "Add Focal Frequency Loss (FFL)",
            "param": "loss.mode=mse_lp_norm_ffl_groups",
            "values": {"ffl_alpha": 1.0, "patch_factor": 1, "initial_log_vars": [0.0, 1.5]},
            "evidence": [f"wavelet_L1_HH_ratio={l1_hh:.2f}"],
        })

    # Rule 3: Spectral attribution HF → target specific bands
    sa = xai.get("spectral_attribution", {})
    hf_frac = sa.get("hf_fraction")
    peak_band = sa.get("peak_band")
    if hf_frac is not None and hf_frac > 0.5 and peak_band is not None:
        recommendations.append({
            "priority": "HIGH",
            "action": f"Target FFL at bands {max(0, peak_band-1)}-{peak_band}",
            "param": "loss.ffl.patch_factor",
            "values": {"focus_bands": [max(0, peak_band-1), peak_band]},
            "evidence": [f"spectral_attribution_hf_fraction={hf_frac:.2f}"],
        })

    # Rule 4: Channel dominance → focus on dominant channel
    cd = xai.get("channel_decomposition", {})
    img_frac = cd.get("image_frac")
    if img_frac is not None:
        if img_frac > 0.8:
            recommendations.append({
                "priority": "HIGH",
                "action": "Focus on image quality (FFL, architecture capacity)",
                "param": "loss/model changes for image channel",
                "evidence": [f"image_contribution={img_frac:.2f}"],
            })
        elif img_frac < 0.2:
            recommendations.append({
                "priority": "HIGH",
                "action": "Focus on mask quality (boundary loss, mask weighting)",
                "param": "loss.mask_weight or boundary-aware loss",
                "evidence": [f"mask_contribution={1-img_frac:.2f}"],
            })

    # Rule 5: Background noise → anatomical conditioning
    bg = metrics.get("background", {})
    noise = bg.get("noise_std", {}).get("v")
    if noise is not None and noise > 0.01:
        recommendations.append({
            "priority": "HIGH",
            "action": "Strengthen anatomical prior masking",
            "param": "model.conditioning_mode=cross_attention",
            "evidence": [f"background_noise_std={noise:.4f}"],
        })

    # Rule 6: Spatial correlation → increase capacity
    spatial = metrics.get("spatial", {})
    corr_def = spatial.get("correlation_deficit", {}).get("v")
    if corr_def is not None and corr_def > 0.1:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Increase model capacity or add cross-attention conditioning",
            "param": "model.channels=[64,128,256,512]",
            "evidence": [f"spatial_corr_deficit={corr_def:.3f}"],
        })

    # Rule 7: Boundary deficit
    boundary = metrics.get("boundary", {})
    sharp_def = boundary.get("sharpness_deficit", {}).get("v")
    if sharp_def is not None and sharp_def > 0.1:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Add boundary-aware loss weighting",
            "param": "loss.boundary_weight",
            "evidence": [f"boundary_sharpness_deficit={sharp_def:.3f}"],
        })

    return recommendations


def generate_compact_report(
    cfg: DictConfig,
    experiment_name: str,
) -> dict:
    """Generate compact LLM-readable YAML report for an experiment.

    Consolidates all diagnostic and XAI results into a structured YAML
    format optimized for LLM consumption.

    Args:
        cfg: Diagnostics configuration.
        experiment_name: Experiment to report on.

    Returns:
        The report dictionary (also saved as YAML).
    """
    output_base_dir = Path(cfg.output.base_dir)
    experiment_dir = output_base_dir / experiment_name
    thresholds = cfg.get("reporting", {}).get("quality_thresholds", DEFAULT_THRESHOLDS)

    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return {}

    # Parse experiment metadata
    meta = _parse_experiment_name(experiment_name)

    # Extract all metrics
    metrics = _extract_metrics(experiment_dir)
    xai = _extract_xai_results(experiment_dir)

    # Compute overall severity from diagnostic_report.csv if available
    overall_severity = None
    cross_exp_dir = output_base_dir / "cross_experiment"
    diag_report_path = cross_exp_dir / "diagnostic_report.csv"
    if diag_report_path.exists():
        import pandas as pd
        df = pd.read_csv(diag_report_path)
        exp_row = df[df["experiment"] == experiment_name]
        if not exp_row.empty and "overall_artifact_severity" in exp_row.columns:
            overall_severity = float(exp_row["overall_artifact_severity"].values[0])

    # Estimate severity from available metrics if not available from cross-experiment
    if overall_severity is None:
        severity_components = []
        wavelet = metrics.get("wavelet", {})
        l1_hh = wavelet.get("L1_HH_ratio", {}).get("v")
        if l1_hh is not None:
            severity_components.append(1.0 - l1_hh)
        bg = metrics.get("background", {})
        noise = bg.get("noise_std", {}).get("v")
        if noise is not None:
            severity_components.append(min(noise * 50, 1.0))
        if severity_components:
            overall_severity = float(sum(severity_components) / len(severity_components))
        else:
            overall_severity = 0.5  # Unknown

    grade = _quality_grade(overall_severity, thresholds)

    # Derive findings and recommendations
    findings = _derive_findings(metrics, xai)
    recommendations = _derive_recommendations(metrics, xai, meta["prediction_type"])

    # Build report structure
    report = {
        "experiment": {
            "name": experiment_name,
            "prediction_type": meta["prediction_type"],
            "lp_norm": meta["lp_norm"],
            "overall_severity": round(overall_severity, 4),
            "quality_grade": grade,
        },
        "metrics": metrics,
        "xai": xai,
        "findings": findings,
        "recommendations": recommendations,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }

    # Save as YAML
    output_dir = ensure_output_dir(output_base_dir, experiment_name, "report")
    yaml_path = output_dir / "compact_report.yaml"

    # Custom YAML representer for clean output
    class CleanDumper(yaml.SafeDumper):
        pass

    def represent_none(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:null", "~")

    CleanDumper.add_representer(type(None), represent_none)

    with open(yaml_path, "w") as f:
        f.write(f"# Compact Diagnostic Report - {experiment_name}\n")
        f.write(f"# Generated: {report['generated']}\n")
        f.write(f"# Format: v=value, n=normalized(0-1), s=significance\n")
        f.write(f"# Grades: A(<{thresholds['A']}), B(<{thresholds['B']}), ")
        f.write(f"C(<{thresholds['C']}), D(<{thresholds['D']}), F(>={thresholds['D']})\n\n")
        yaml.dump(report, f, Dumper=CleanDumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True, width=120)

    logger.info(f"Compact report saved to {yaml_path}")
    return report
