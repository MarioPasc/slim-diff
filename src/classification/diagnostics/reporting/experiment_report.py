"""Structured per-experiment diagnostic report generation.

Consolidates all analysis results (existing diagnostics + new XAI) into a
comprehensive structured JSON report with per-category severity scores,
ranked artifacts, and actionable recommendations via a rule-based engine
encoding domain knowledge from the ICIP 2026 ablation study.

This replaces the thinner summary_report.py with a more comprehensive
and actionable report structure.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from src.classification.diagnostics.utils import (
    NumpyEncoder,
    ensure_output_dir,
    save_result_json,
)

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict | None:
    """Load JSON file, return None if missing."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def _parse_experiment_name(name: str) -> dict[str, str]:
    """Parse experiment name into components."""
    parts = name.split("_lp_")
    if len(parts) == 2:
        return {"prediction_type": parts[0], "lp_norm": parts[1]}
    return {"prediction_type": name, "lp_norm": "unknown"}


def _collect_all_results(experiment_dir: Path) -> dict[str, Any]:
    """Collect all analysis JSON results from experiment directory."""
    results = {}

    # Standard analyses
    json_paths = {
        "spectral": "spectral/spectral_results.json",
        "texture": "texture/texture_results.json",
        "frequency_bands": "frequency_bands/frequency_bands_results.json",
        "pixel_stats": "pixel_stats/pixel_stats_results.json",
        "distributions": "distribution_tests/distribution_tests_results.json",
        "boundary": "boundary_analysis/boundary_analysis_results.json",
        "wavelet": "wavelet_analysis/wavelet_analysis_results.json",
        "background": "full_image/background/background_analysis_results.json",
        "spatial_correlation": "full_image/spatial_correlation/spatial_correlation_results.json",
        "global_frequency": "full_image/global_frequency/global_frequency_results.json",
        # XAI analyses
        "gradcam": "gradcam/joint/gradcam_summary.json",
        "channel_decomposition": "channel_decomposition/channel_decomposition_results.json",
        "spectral_attribution": "spectral_attribution/spectral_attribution_results.json",
        "feature_space": "feature_space/feature_space_results.json",
        "integrated_gradients": "integrated_gradients/ig_results.json",
    }

    for name, relpath in json_paths.items():
        data = _load_json(experiment_dir / relpath)
        if data is not None:
            results[name] = data

    return results


def _compute_category_scores(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compute severity scores per diagnostic category.

    Returns dict mapping category -> {score, raw_metrics, interpretation}.
    Scores are in [0, 1] where 0 is best.
    """
    categories = {}

    # Spectral fidelity
    spectral = results.get("spectral", {})
    if spectral:
        channels = spectral.get("channels", {})
        img = channels.get("image", {})
        slope_diff = abs(img.get("slope_difference", 0))
        js_div = img.get("js_divergence", 0)
        # Normalize: slope_diff typically 0-2, js_div typically 0-0.15
        score = min(1.0, slope_diff / 1.5 * 0.5 + js_div / 0.1 * 0.5)
        categories["spectral_fidelity"] = {
            "score": round(score, 4),
            "raw": {"slope_difference": slope_diff, "js_divergence": js_div},
            "interpretation": (
                "Over-smoothed" if img.get("slope_difference", 0) > 0
                else "Excess high-frequency"
            ) if slope_diff > 0.05 else "Good spectral match",
        }

    # Texture quality
    texture = results.get("texture", {})
    if texture:
        glcm = texture.get("glcm", {})
        ks_stats = [v.get("ks_statistic", 0) for v in glcm.values() if isinstance(v, dict)]
        cohens_ds = [abs(v.get("cohens_d", 0)) for v in glcm.values() if isinstance(v, dict)]
        mean_ks = sum(ks_stats) / max(len(ks_stats), 1)
        max_d = max(cohens_ds) if cohens_ds else 0
        score = min(1.0, mean_ks * 2 + max_d / 5.0)
        categories["texture_quality"] = {
            "score": round(score, 4),
            "raw": {"mean_ks": mean_ks, "max_cohens_d": max_d},
            "interpretation": f"Cohen's d={max_d:.2f}" if max_d > 0.3 else "Good texture match",
        }

    # High-frequency content
    gf = results.get("global_frequency", {})
    wavelet = results.get("wavelet", {})
    if gf or wavelet:
        hf_ratio_diff = abs(gf.get("high_freq_ratio_difference", 0)) if gf else 0
        # Wavelet L1 HH ratio
        l1_hh = None
        if wavelet:
            img_data = wavelet.get("image", {})
            per_level = img_data.get("per_level", {})
            l1_data = per_level.get("1", {})
            hh_data = l1_data.get("HH", {})
            l1_hh = hh_data.get("energy_ratio")

        hf_deficit = 1.0 - l1_hh if l1_hh is not None else hf_ratio_diff
        score = min(1.0, abs(hf_deficit))
        categories["high_frequency"] = {
            "score": round(score, 4),
            "raw": {"hf_ratio_difference": hf_ratio_diff, "wavelet_L1_HH_ratio": l1_hh},
            "interpretation": (
                f"HF deficit: wavelet ratio={l1_hh:.2f}" if l1_hh is not None and l1_hh < 0.85
                else f"HF excess: ratio_diff={hf_ratio_diff:.3f}" if hf_ratio_diff > 0.3
                else "Good HF content"
            ),
        }

    # Boundary quality
    boundary = results.get("boundary", {})
    if boundary and not boundary.get("insufficient_data"):
        sharp_ratio = boundary.get("synth_sharpness_mean", 0) / max(
            boundary.get("real_sharpness_mean", 1e-12), 1e-12
        )
        width_ratio = boundary.get("synth_width_mean", 0) / max(
            boundary.get("real_width_mean", 1e-12), 1e-12
        )
        score = min(1.0, abs(1.0 - sharp_ratio) + abs(1.0 - width_ratio))
        categories["boundary_quality"] = {
            "score": round(score, 4),
            "raw": {"sharpness_ratio": sharp_ratio, "width_ratio": width_ratio},
            "interpretation": (
                "Blurred boundaries" if sharp_ratio < 0.9
                else "Over-sharp boundaries" if sharp_ratio > 1.1
                else "Good boundary quality"
            ),
        }

    # Spatial coherence
    spatial = results.get("spatial_correlation", {})
    if spatial:
        xi_real = spatial.get("correlation_length_real", 1.0)
        xi_synth = spatial.get("correlation_length_synth", 1.0)
        ratio = xi_synth / max(xi_real, 1e-12)
        deficit = abs(1.0 - ratio)
        score = min(1.0, deficit * 2)
        categories["spatial_coherence"] = {
            "score": round(score, 4),
            "raw": {"correlation_ratio": ratio, "deficit": deficit},
            "interpretation": (
                "Over-smooth (long correlation)" if ratio > 1.1
                else "Too granular (short correlation)" if ratio < 0.9
                else "Good spatial structure"
            ),
        }

    # Background integrity
    bg = results.get("background", {})
    if bg:
        synth_std = bg.get("synth", {}).get("std", 0)
        score = min(1.0, synth_std * 50)
        categories["background_integrity"] = {
            "score": round(score, 4),
            "raw": {"synth_bg_std": synth_std},
            "interpretation": (
                f"Noise leakage (std={synth_std:.4f})" if synth_std > 0.005
                else "Clean background"
            ),
        }

    # Distribution accuracy
    dist = results.get("distributions", {})
    if dist:
        img_data = dist.get("image", {})
        wasserstein = img_data.get("wasserstein", 0)
        per_tissue = dist.get("per_tissue", {})
        lesion_w = per_tissue.get("lesion", {}).get("wasserstein", 0)
        score = min(1.0, wasserstein * 5 + lesion_w * 3)
        categories["distribution_accuracy"] = {
            "score": round(score, 4),
            "raw": {"wasserstein_all": wasserstein, "wasserstein_lesion": lesion_w},
            "interpretation": f"W-distance: all={wasserstein:.4f}, lesion={lesion_w:.4f}",
        }

    return categories


def _rank_artifacts(
    categories: dict[str, dict],
    results: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rank artifacts by severity × statistical significance."""
    artifacts = []

    for cat_name, cat_data in categories.items():
        score = cat_data["score"]
        if score < 0.01:
            continue
        artifacts.append({
            "category": cat_name,
            "severity": score,
            "raw_metrics": cat_data["raw"],
            "interpretation": cat_data["interpretation"],
        })

    # Sort by severity
    artifacts.sort(key=lambda a: a["severity"], reverse=True)
    for i, a in enumerate(artifacts):
        a["rank"] = i + 1

    return artifacts


def _generate_recommendations(
    categories: dict[str, dict],
    results: dict[str, Any],
    prediction_type: str,
) -> list[dict[str, Any]]:
    """Generate rule-based recommendations from diagnostic findings.

    Rules encode domain knowledge from the ICIP 2026 ablation study and
    the known relationships between model parameters and artifact types.
    """
    recommendations = []

    # Rule: Epsilon prediction
    if prediction_type == "epsilon":
        recommendations.append({
            "priority": "CRITICAL",
            "action": "Switch to x0 (sample) prediction type",
            "target_parameter": "scheduler.prediction_type",
            "target_value": "sample",
            "additional_params": {
                "scheduler.clip_sample": True,
                "scheduler.clip_sample_range": 1.0,
            },
            "expected_improvement": "40x reduction in HF excess",
            "evidence": "Epsilon prediction amplifies noise at high timesteps via x0=(x_t-sqrt(1-a)*eps)/sqrt(a)",
        })

    # Rule: HF deficit → FFL
    hf = categories.get("high_frequency", {})
    if hf.get("score", 0) > 0.15:
        raw = hf.get("raw", {})
        l1_hh = raw.get("wavelet_L1_HH_ratio")
        recommendations.append({
            "priority": "HIGH",
            "action": "Add Focal Frequency Loss (FFL) with group uncertainty weighting",
            "target_parameter": "loss.mode",
            "target_value": "mse_lp_norm_ffl_groups",
            "additional_params": {
                "loss.ffl.alpha": 1.0,
                "loss.ffl.patch_factor": 1,
                "loss.group_uncertainty_weighting.initial_log_vars": [0.0, 1.5],
            },
            "expected_improvement": f"wavelet HH ratio {l1_hh:.2f} -> 0.85+" if l1_hh else "Reduce HF deficit",
            "evidence": f"wavelet_L1_HH_ratio={l1_hh}" if l1_hh else "high_freq_score",
        })

    # Rule: Spectral attribution → targeted FFL
    sa = results.get("spectral_attribution", {})
    hf_frac = sa.get("attribution_hf_fraction")
    peak_band = sa.get("peak_attribution_band")
    if hf_frac is not None and hf_frac > 0.5 and peak_band is not None:
        recommendations.append({
            "priority": "HIGH",
            "action": f"Target FFL at frequency bands {max(0,peak_band-1)}-{peak_band}",
            "target_parameter": "loss.ffl.focus_bands",
            "target_value": [max(0, peak_band-1), peak_band],
            "expected_improvement": "Focus frequency correction where classifier detects artifacts",
            "evidence": f"spectral_attribution: {int(hf_frac*100)}% attention on HF, peak at band {peak_band}",
        })

    # Rule: Channel dominance
    cd = results.get("channel_decomposition", {})
    overall = cd.get("overall", {})
    img_frac = overall.get("gradient", {}).get("image_fraction")
    if img_frac is not None:
        if img_frac > 0.8:
            recommendations.append({
                "priority": "HIGH",
                "action": "Focus on image channel quality (FFL, model capacity)",
                "target_parameter": "loss/model architecture",
                "expected_improvement": "Reduce image-channel artifacts that dominate classifier attention",
                "evidence": f"channel_decomposition: image={img_frac:.2f}, mask={1-img_frac:.2f}",
            })
        elif img_frac < 0.2:
            recommendations.append({
                "priority": "HIGH",
                "action": "Focus on mask quality (boundary loss, mask-specific weighting)",
                "target_parameter": "loss.mask_weight or boundary loss",
                "expected_improvement": "Reduce mask-channel artifacts",
                "evidence": f"channel_decomposition: mask={1-img_frac:.2f} dominates",
            })

    # Rule: Background noise → conditioning
    bg = categories.get("background_integrity", {})
    if bg.get("score", 0) > 0.3:
        recommendations.append({
            "priority": "HIGH",
            "action": "Strengthen anatomical prior masking or switch to cross-attention conditioning",
            "target_parameter": "model.conditioning_mode",
            "target_value": "cross_attention",
            "expected_improvement": "Reduce background noise leakage",
            "evidence": f"background_noise_std={bg.get('raw', {}).get('synth_bg_std', 0):.4f}",
        })

    # Rule: Spatial coherence → capacity
    sc = categories.get("spatial_coherence", {})
    if sc.get("score", 0) > 0.2:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Increase model capacity (channels, attention levels)",
            "target_parameter": "model.channels",
            "target_value": [64, 128, 256, 512],
            "additional_params": {"model.attention_levels": [False, False, True, True]},
            "expected_improvement": "Better spatial correlation structure",
            "evidence": f"spatial_corr_deficit={sc.get('raw', {}).get('deficit', 0):.3f}",
        })

    # Rule: Boundary issues
    bq = categories.get("boundary_quality", {})
    if bq.get("score", 0) > 0.2:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Add boundary-aware loss weighting",
            "target_parameter": "loss.boundary_weight",
            "expected_improvement": "Sharper, more accurate lesion boundaries",
            "evidence": f"boundary ratio: sharp={bq.get('raw', {}).get('sharpness_ratio', 0):.2f}",
        })

    return recommendations


def _generate_narrative(
    experiment_name: str,
    meta: dict,
    categories: dict,
    artifacts: list,
    recommendations: list,
    xai_summary: dict,
) -> str:
    """Generate human-readable markdown narrative."""
    lines = []
    lines.append(f"# Diagnostic Report: {experiment_name}")
    lines.append(f"**Prediction type**: {meta['prediction_type']} | **Lp norm**: {meta['lp_norm']}")
    lines.append("")

    # Overall assessment
    scores = [c["score"] for c in categories.values()]
    avg_score = sum(scores) / max(len(scores), 1)
    lines.append(f"## Overall Assessment")
    lines.append(f"Mean category severity: **{avg_score:.3f}** (0=perfect, 1=worst)")
    lines.append("")

    # Category scores
    lines.append("## Category Scores")
    lines.append("| Category | Score | Interpretation |")
    lines.append("|----------|-------|----------------|")
    for name, data in sorted(categories.items(), key=lambda x: x[1]["score"], reverse=True):
        lines.append(f"| {name} | {data['score']:.4f} | {data['interpretation']} |")
    lines.append("")

    # Top artifacts
    if artifacts:
        lines.append("## Top Artifacts")
        for a in artifacts[:5]:
            lines.append(f"{a['rank']}. **{a['category']}** (severity={a['severity']:.3f}): {a['interpretation']}")
        lines.append("")

    # XAI insights
    if xai_summary:
        lines.append("## XAI Insights")
        if "channel_decomposition" in xai_summary:
            cd = xai_summary["channel_decomposition"]
            lines.append(f"- **Channel focus**: Image={cd.get('image_frac', 0):.0%}, Mask={cd.get('mask_frac', 0):.0%}")
        if "spectral_attribution" in xai_summary:
            sa = xai_summary["spectral_attribution"]
            lines.append(f"- **Spectral focus**: {sa.get('hf_fraction', 0):.0%} on high-frequency, peak band={sa.get('peak_band')}")
        if "feature_space" in xai_summary:
            fs = xai_summary["feature_space"]
            lines.append(f"- **Separability**: silhouette={fs.get('silhouette', 0):.3f}")
        lines.append("")

    # Recommendations
    if recommendations:
        lines.append("## Recommendations")
        for r in recommendations:
            lines.append(f"- [{r['priority']}] {r['action']}")
            if "target_parameter" in r and "target_value" in r:
                lines.append(f"  - Set `{r['target_parameter']}` = `{r.get('target_value')}`")
            if "expected_improvement" in r:
                lines.append(f"  - Expected: {r['expected_improvement']}")
        lines.append("")

    return "\n".join(lines)


def generate_experiment_report(
    cfg: DictConfig,
    experiment_name: str,
) -> dict:
    """Generate comprehensive structured diagnostic report for an experiment.

    Args:
        cfg: Diagnostics configuration.
        experiment_name: Experiment to report on.

    Returns:
        The complete report dictionary.
    """
    output_base_dir = Path(cfg.output.base_dir)
    experiment_dir = output_base_dir / experiment_name

    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return {}

    meta = _parse_experiment_name(experiment_name)

    # Collect all results
    all_results = _collect_all_results(experiment_dir)
    n_analyses = len(all_results)
    logger.info(f"Collected {n_analyses} analysis results for '{experiment_name}'")

    # Compute category scores
    categories = _compute_category_scores(all_results)

    # Rank artifacts
    artifacts = _rank_artifacts(categories, all_results)

    # XAI summary
    xai_summary = {}
    cd = all_results.get("channel_decomposition", {})
    if cd:
        overall = cd.get("overall", {})
        xai_summary["channel_decomposition"] = {
            "image_frac": overall.get("gradient", {}).get("image_fraction"),
            "mask_frac": overall.get("gradient", {}).get("mask_fraction"),
            "dominant": overall.get("dominant_channel"),
        }
    sa = all_results.get("spectral_attribution", {})
    if sa:
        xai_summary["spectral_attribution"] = {
            "peak_band": sa.get("peak_attribution_band"),
            "hf_fraction": sa.get("attribution_hf_fraction"),
            "concordance": sa.get("concordance"),
        }
    fs = all_results.get("feature_space", {})
    if fs:
        xai_summary["feature_space"] = {
            "silhouette": fs.get("cluster_metrics", {}).get("tsne_silhouette"),
            "n_significant": fs.get("statistical_tests", {}).get("n_significant_fdr"),
        }

    # Generate recommendations
    recommendations = _generate_recommendations(
        categories, all_results, meta["prediction_type"]
    )

    # Generate narrative
    narrative = _generate_narrative(
        experiment_name, meta, categories, artifacts, recommendations, xai_summary
    )

    # Compile report
    report = {
        "experiment": experiment_name,
        "prediction_type": meta["prediction_type"],
        "lp_norm": meta["lp_norm"],
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_analyses_available": n_analyses,
        "category_scores": categories,
        "ranked_artifacts": artifacts,
        "xai_summary": xai_summary,
        "recommendations": recommendations,
    }

    # Save outputs
    output_dir = ensure_output_dir(output_base_dir, experiment_name, "report")

    # JSON report
    save_result_json(report, output_dir / "experiment_report.json")

    # Markdown narrative
    md_path = output_dir / "experiment_report.md"
    with open(md_path, "w") as f:
        f.write(narrative)
    logger.info(f"Narrative report saved to {md_path}")

    return report
