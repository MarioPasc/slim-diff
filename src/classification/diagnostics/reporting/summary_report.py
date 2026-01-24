"""Summary report generation for diagnostic analyses.

Aggregates findings from all analysis components and produces a comprehensive
report identifying the key differences between real and synthetic images.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.classification.diagnostics.utils import (
    NumpyEncoder,
    ensure_output_dir,
    save_figure,
)

logger = logging.getLogger(__name__)


def _load_json_result(path: Path) -> dict | None:
    """Load a JSON result file, returning None if not found."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _collect_findings(results_dir: Path) -> dict[str, Any]:
    """Collect all analysis results from the output directory.

    Args:
        results_dir: Per-experiment results directory.

    Returns:
        Dict mapping analysis name to loaded results.
    """
    findings = {}

    # Dithering
    dither_path = results_dir / "dithering" / "reclassification_results.json"
    findings["dithering"] = _load_json_result(dither_path)

    # GradCAM
    gradcam_path = results_dir / "gradcam" / "gradcam_results.json"
    findings["gradcam"] = _load_json_result(gradcam_path)

    # Spectral
    for ch in [0, 1]:
        spectral_path = results_dir / "spectral" / f"spectral_results_ch{ch}.json"
        findings[f"spectral_ch{ch}"] = _load_json_result(spectral_path)

    # Texture
    for ch in [0, 1]:
        texture_path = results_dir / "texture" / f"texture_results_ch{ch}.json"
        findings[f"texture_ch{ch}"] = _load_json_result(texture_path)

    # Frequency bands
    for ch in [0, 1]:
        band_path = results_dir / "frequency_bands" / f"band_results_ch{ch}.json"
        findings[f"bands_ch{ch}"] = _load_json_result(band_path)

    # Pixel stats
    for ch in [0, 1]:
        pixel_path = results_dir / "pixel_stats" / f"pixel_stats_ch{ch}.json"
        findings[f"pixel_stats_ch{ch}"] = _load_json_result(pixel_path)

    # Distribution tests
    dist_path = results_dir / "distributions" / "distribution_results.json"
    findings["distributions"] = _load_json_result(dist_path)

    # Boundary
    boundary_path = results_dir / "boundary" / "boundary_results.json"
    findings["boundary"] = _load_json_result(boundary_path)

    # Wavelet
    for ch in [0, 1]:
        wavelet_path = results_dir / "wavelet" / f"wavelet_results_ch{ch}.json"
        findings[f"wavelet_ch{ch}"] = _load_json_result(wavelet_path)

    # Full image analyses
    bg_path = results_dir / "background" / "background_results.json"
    findings["background"] = _load_json_result(bg_path)

    corr_path = results_dir / "spatial_correlation" / "spatial_correlation_results.json"
    findings["spatial_correlation"] = _load_json_result(corr_path)

    freq_path = results_dir / "global_frequency" / "global_frequency_results.json"
    findings["global_frequency"] = _load_json_result(freq_path)

    return findings


def _extract_key_metrics(findings: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the most informative metrics from all analyses.

    Returns a list of ranked findings sorted by significance/effect size.
    """
    metrics = []

    # Dithering: delta-AUC
    if findings.get("dithering"):
        for mode, data in findings["dithering"].get("per_mode", {}).items():
            metrics.append({
                "analysis": "dithering",
                "metric": f"AUC after dithering ({mode})",
                "value": data.get("auc_after_dithering"),
                "delta": data.get("delta_auc"),
                "interpretation": (
                    "Low residual AUC (near 0.5) = float16 was the only issue. "
                    "High residual AUC = genuine model artifacts remain."
                ),
                "significance": abs(data.get("auc_after_dithering", 0.5) - 0.5),
            })

    # Spectral: slope difference and divergence
    for ch in [0, 1]:
        ch_name = "image" if ch == 0 else "mask"
        data = findings.get(f"spectral_ch{ch}")
        if data:
            metrics.append({
                "analysis": "spectral",
                "metric": f"Spectral slope difference ({ch_name})",
                "value": data.get("slope_difference"),
                "real_slope": data.get("real_slope"),
                "synth_slope": data.get("synth_slope"),
                "js_divergence": data.get("spectral_divergence"),
                "interpretation": (
                    "Steeper synthetic slope = less high-frequency content (over-smoothed). "
                    "Shallower = excess high-frequency noise."
                ),
                "significance": abs(data.get("slope_difference", 0)),
            })

    # Texture: features with largest effect sizes
    for ch in [0, 1]:
        ch_name = "image" if ch == 0 else "mask"
        data = findings.get(f"texture_ch{ch}")
        if data and "effect_sizes" in data:
            for feature, effect_size in data["effect_sizes"].items():
                if abs(effect_size) > 0.3:  # Medium or larger effect
                    metrics.append({
                        "analysis": "texture",
                        "metric": f"GLCM {feature} ({ch_name})",
                        "effect_size": effect_size,
                        "ks_stat": data.get("ks_statistics", {}).get(feature),
                        "p_value": data.get("ks_pvalues", {}).get(feature),
                        "interpretation": f"Cohen's d = {effect_size:.2f}",
                        "significance": abs(effect_size),
                    })

    # Pixel stats: fraction significant
    for ch in [0, 1]:
        ch_name = "image" if ch == 0 else "mask"
        data = findings.get(f"pixel_stats_ch{ch}")
        if data:
            metrics.append({
                "analysis": "pixel_stats",
                "metric": f"Fraction significant pixels ({ch_name})",
                "value": data.get("fraction_significant"),
                "interpretation": (
                    "Fraction of pixels with significantly different mean values "
                    "after FDR correction."
                ),
                "significance": data.get("fraction_significant", 0),
            })

    # Boundary: sharpness difference
    if findings.get("boundary"):
        data = findings["boundary"]
        metrics.append({
            "analysis": "boundary",
            "metric": "Boundary sharpness difference",
            "real_sharpness": data.get("mean_sharpness_real"),
            "synth_sharpness": data.get("mean_sharpness_synth"),
            "ks_stat": data.get("sharpness_ks_stat"),
            "p_value": data.get("sharpness_ks_pvalue"),
            "interpretation": (
                "Higher synthetic sharpness = over-crisp boundaries. "
                "Lower = blurred boundaries."
            ),
            "significance": abs(
                (data.get("mean_sharpness_real") or 0) -
                (data.get("mean_sharpness_synth") or 0)
            ),
        })

    # Background analysis
    if findings.get("background"):
        data = findings["background"]
        synth_std = data.get("synth_bg_std", 0)
        metrics.append({
            "analysis": "background",
            "metric": "Background noise (synthetic std)",
            "value": synth_std,
            "real_std": data.get("real_bg_std"),
            "interpretation": (
                "Non-zero synthetic background std indicates noise leakage "
                "past brain boundary masking."
            ),
            "significance": synth_std,
        })

    # Spatial correlation
    if findings.get("spatial_correlation"):
        data = findings["spatial_correlation"]
        real_xi = data.get("real_correlation_length", 0)
        synth_xi = data.get("synth_correlation_length", 0)
        if real_xi and synth_xi:
            metrics.append({
                "analysis": "spatial_correlation",
                "metric": "Correlation length difference",
                "real_xi": real_xi,
                "synth_xi": synth_xi,
                "ratio": synth_xi / real_xi if real_xi > 0 else None,
                "interpretation": (
                    "Longer synthetic correlation = over-smooth textures. "
                    "Shorter = too granular."
                ),
                "significance": abs(real_xi - synth_xi) / max(real_xi, 1e-6),
            })

    # Sort by significance
    metrics.sort(key=lambda x: x.get("significance", 0), reverse=True)
    return metrics


def _generate_summary_table(metrics: list[dict], output_dir: Path) -> None:
    """Generate a summary table of top findings."""
    import matplotlib.pyplot as plt

    if not metrics:
        logger.warning("No metrics to summarize")
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(metrics[:15]) * 0.4)))
    ax.axis("off")

    headers = ["Rank", "Analysis", "Metric", "Significance", "Key Value"]
    rows = []
    for i, m in enumerate(metrics[:15], 1):
        value_str = ""
        if "value" in m and m["value"] is not None:
            value_str = f"{m['value']:.4f}"
        elif "effect_size" in m:
            value_str = f"d={m['effect_size']:.2f}"
        elif "delta" in m and m["delta"] is not None:
            value_str = f"delta={m['delta']:.4f}"

        rows.append([
            str(i),
            m["analysis"],
            m["metric"],
            f"{m.get('significance', 0):.4f}",
            value_str,
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        loc="center",
        colWidths=[0.05, 0.15, 0.35, 0.12, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)

    ax.set_title("Diagnostic Findings Summary (Ranked by Significance)", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    save_figure(fig, output_dir, "findings_summary", ["png", "pdf"], dpi=300)
    plt.close(fig)


def generate_report(cfg, experiment_name: str) -> None:
    """Generate comprehensive diagnostic report.

    Collects all analysis results, ranks findings by significance,
    and produces summary visualizations and a structured JSON report.

    Args:
        cfg: Diagnostics configuration.
        experiment_name: Experiment to report on.
    """
    output_base = Path(cfg.output.base_dir)
    results_dir = output_base / experiment_name
    report_dir = ensure_output_dir(output_base, experiment_name, "report")

    if not results_dir.exists():
        logger.error(f"No results found at {results_dir}")
        return

    logger.info(f"Collecting findings from {results_dir}")
    findings = _collect_findings(results_dir)

    # Count successful analyses
    n_available = sum(1 for v in findings.values() if v is not None)
    n_total = len(findings)
    logger.info(f"Found {n_available}/{n_total} analysis results")

    # Extract and rank key metrics
    metrics = _extract_key_metrics(findings)
    logger.info(f"Extracted {len(metrics)} key metrics")

    # Generate summary table
    _generate_summary_table(metrics, report_dir)

    # Save structured report
    report = {
        "experiment": experiment_name,
        "n_analyses_available": n_available,
        "n_analyses_total": n_total,
        "top_findings": metrics[:10],
        "all_findings": metrics,
        "interpretation": _generate_interpretation(metrics),
    }

    report_path = report_dir / "diagnostics_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2)
    logger.info(f"Report saved to {report_path}")


def _generate_interpretation(metrics: list[dict]) -> dict[str, str]:
    """Generate human-readable interpretation of the findings.

    Returns:
        Dict with 'summary', 'primary_issues', 'recommendations'.
    """
    if not metrics:
        return {
            "summary": "No significant findings.",
            "primary_issues": [],
            "recommendations": [],
        }

    # Check dithering result
    dither_metrics = [m for m in metrics if m["analysis"] == "dithering"]
    residual_aucs = [m.get("value") for m in dither_metrics if m.get("value") is not None]

    issues = []
    recommendations = []

    if residual_aucs:
        max_residual = max(residual_aucs)
        if max_residual > 0.8:
            issues.append(
                "High residual AUC after dithering indicates genuine model artifacts "
                "beyond the float16 quantization issue."
            )
        elif max_residual > 0.6:
            issues.append(
                "Moderate residual AUC after dithering suggests some detectable "
                "differences remain beyond float16 quantization."
            )
        else:
            issues.append(
                "Low residual AUC after dithering suggests float16 quantization was "
                "the primary distinguishing factor. Fix storage format to float32."
            )
            recommendations.append(
                "Store replicas in float32 instead of float16 to eliminate the "
                "trivial quantization artifact."
            )

    # Check spectral findings
    spectral_metrics = [m for m in metrics if m["analysis"] == "spectral"]
    for sm in spectral_metrics:
        slope_diff = sm.get("value")
        if slope_diff is not None and abs(slope_diff) > 0.1:
            if slope_diff > 0:
                issues.append("Synthetic images lack high-frequency content (over-smoothed).")
                recommendations.append(
                    "Consider increasing Focal Frequency Loss weight or reducing "
                    "DDIM eta to preserve high-frequency detail."
                )
            else:
                issues.append("Synthetic images have excess high-frequency content.")
                recommendations.append(
                    "Check for noise injection artifacts in the sampling process."
                )

    # Check texture findings
    texture_metrics = [m for m in metrics if m["analysis"] == "texture" and m.get("significance", 0) > 0.5]
    if texture_metrics:
        features = [m["metric"] for m in texture_metrics[:3]]
        issues.append(f"Significant texture differences in: {', '.join(features)}.")
        recommendations.append(
            "Texture artifacts suggest the model may benefit from perceptual "
            "or style-based loss terms during training."
        )

    # Check boundary findings
    boundary_metrics = [m for m in metrics if m["analysis"] == "boundary"]
    for bm in boundary_metrics:
        if bm.get("p_value") is not None and bm["p_value"] < 0.01:
            issues.append("Lesion boundary characteristics significantly differ.")
            recommendations.append(
                "Consider lesion boundary-aware loss weighting or dedicated "
                "boundary refinement during sampling."
            )

    # Check background
    bg_metrics = [m for m in metrics if m["analysis"] == "background"]
    for bgm in bg_metrics:
        if bgm.get("value") is not None and bgm["value"] > 0.01:
            issues.append("Background noise leakage detected in synthetic images.")
            recommendations.append(
                "Ensure z-bin prior masking is applied correctly during generation, "
                "or tighten the background suppression post-processing."
            )

    summary = (
        f"Identified {len(issues)} potential issues across {len(metrics)} analyzed metrics. "
        f"Top concern: {issues[0] if issues else 'None identified.'}"
    )

    return {
        "summary": summary,
        "primary_issues": issues,
        "recommendations": recommendations,
    }
