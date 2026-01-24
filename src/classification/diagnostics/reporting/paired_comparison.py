"""Cross-experiment paired comparison and next-experiment recommendation.

Computes metric deltas between experiment pairs that differ by a single
parameter (prediction type or Lp norm), identifies monotonic trends, and
recommends the next experiment configuration based on remaining artifacts.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    save_csv,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)


def _parse_experiment_name(name: str) -> dict[str, str]:
    """Parse experiment into components."""
    parts = name.split("_lp_")
    if len(parts) == 2:
        return {"prediction_type": parts[0], "lp_norm": parts[1]}
    return {"prediction_type": name, "lp_norm": "unknown"}


def _load_diagnostic_report(output_dir: Path) -> pd.DataFrame | None:
    """Load the cross-experiment diagnostic report."""
    report_path = output_dir / "cross_experiment" / "diagnostic_report.csv"
    if not report_path.exists():
        logger.warning(f"Diagnostic report not found: {report_path}")
        return None
    return pd.read_csv(report_path)


def _compute_paired_deltas(
    df: pd.DataFrame,
    metric_cols: list[str],
) -> list[dict[str, Any]]:
    """Compute metric deltas for experiment pairs differing by one parameter.

    Returns list of dicts describing each paired comparison.
    """
    pairs = []

    experiments = df["experiment"].tolist()
    for exp_a, exp_b in combinations(experiments, 2):
        meta_a = _parse_experiment_name(exp_a)
        meta_b = _parse_experiment_name(exp_b)

        # Only compare pairs that differ by exactly one parameter
        same_pred = meta_a["prediction_type"] == meta_b["prediction_type"]
        same_norm = meta_a["lp_norm"] == meta_b["lp_norm"]

        if not (same_pred ^ same_norm):  # XOR: exactly one must differ
            continue

        varying_param = "lp_norm" if same_pred else "prediction_type"
        row_a = df[df["experiment"] == exp_a].iloc[0]
        row_b = df[df["experiment"] == exp_b].iloc[0]

        deltas = {}
        for col in metric_cols:
            val_a = row_a.get(col)
            val_b = row_b.get(col)
            if pd.notna(val_a) and pd.notna(val_b):
                deltas[col] = float(val_b - val_a)

        pairs.append({
            "exp_a": exp_a,
            "exp_b": exp_b,
            "varying_param": varying_param,
            "value_a": meta_a[varying_param],
            "value_b": meta_b[varying_param],
            "fixed_param": "prediction_type" if varying_param == "lp_norm" else "lp_norm",
            "fixed_value": meta_a["prediction_type"] if varying_param == "lp_norm" else meta_a["lp_norm"],
            "severity_a": float(row_a.get("overall_artifact_severity", 0)),
            "severity_b": float(row_b.get("overall_artifact_severity", 0)),
            "deltas": deltas,
        })

    return pairs


def _detect_trends(
    df: pd.DataFrame,
    metric_cols: list[str],
) -> dict[str, list[dict]]:
    """Detect monotonic trends within each prediction type as Lp norm varies.

    Returns dict mapping prediction_type -> list of trend dicts.
    """
    from scipy.stats import spearmanr

    trends = {}

    for pred_type in df["prediction_type"].unique():
        subset = df[df["prediction_type"] == pred_type].copy()
        if len(subset) < 3:
            continue

        # Sort by Lp norm
        subset = subset.sort_values("lp_norm")
        lp_values = subset["lp_norm"].astype(float).values

        type_trends = []
        for col in metric_cols:
            values = subset[col].values
            if np.all(np.isnan(values)):
                continue
            valid = ~np.isnan(values)
            if valid.sum() < 3:
                continue

            rho, p_value = spearmanr(lp_values[valid], values[valid])
            if abs(rho) > 0.8:  # Strong monotonic trend
                type_trends.append({
                    "metric": col,
                    "direction": "increasing" if rho > 0 else "decreasing",
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                    "values": {f"lp_{lp:.1f}": float(v) for lp, v in zip(lp_values[valid], values[valid])},
                })

        if type_trends:
            trends[pred_type] = type_trends

    return trends


def _compute_effect_sizes(
    df: pd.DataFrame,
    metric_cols: list[str],
) -> dict[str, dict[str, float]]:
    """Compute effect size (Cohen's d) for each parameter choice on each metric.

    Returns dict mapping "prediction_type" or "lp_norm" -> {metric: effect_size}.
    """
    effects = {}

    # Effect of prediction type
    pred_types = df["prediction_type"].unique()
    if len(pred_types) >= 2:
        pred_effects = {}
        for col in metric_cols:
            group_means = df.groupby("prediction_type")[col].mean()
            group_stds = df.groupby("prediction_type")[col].std()
            if len(group_means) >= 2:
                overall_std = df[col].std()
                if overall_std > 1e-12:
                    # Range of group means / overall std ≈ effect size
                    effect = float((group_means.max() - group_means.min()) / overall_std)
                    pred_effects[col] = effect
        effects["prediction_type"] = pred_effects

    # Effect of lp_norm
    lp_norms = df["lp_norm"].unique()
    if len(lp_norms) >= 2:
        norm_effects = {}
        for col in metric_cols:
            group_means = df.groupby("lp_norm")[col].mean()
            overall_std = df[col].std()
            if overall_std > 1e-12:
                effect = float((group_means.max() - group_means.min()) / overall_std)
                norm_effects[col] = effect
        effects["lp_norm"] = norm_effects

    return effects


def _recommend_next_experiment(
    df: pd.DataFrame,
    metric_cols: list[str],
) -> dict[str, Any]:
    """Recommend next experiment configuration based on remaining artifacts.

    Logic:
    1. Identify best experiment
    2. Find its top-3 remaining artifacts (highest normalized scores)
    3. For each artifact, check if another experiment does better on that metric
    4. If yes, identify what parameter difference caused the improvement
    5. Propose hybrid configuration
    """
    if "overall_artifact_severity" not in df.columns:
        return {"error": "no_severity_scores"}

    best_idx = df["overall_artifact_severity"].idxmin()
    best_exp = df.loc[best_idx]
    best_name = best_exp["experiment"]
    best_meta = _parse_experiment_name(best_name)

    # Find normalized metric columns
    norm_cols = [c for c in df.columns if c.endswith("_norm") and c != "lp_norm"]

    # Top-3 remaining artifacts for best experiment
    best_norm_scores = best_exp[norm_cols].sort_values(ascending=False)
    top_artifacts = []
    for col in best_norm_scores.index[:5]:
        if best_norm_scores[col] > 0.01:  # Non-trivial
            base_metric = col.replace("_norm", "")
            top_artifacts.append({
                "metric": base_metric,
                "normalized_score": float(best_norm_scores[col]),
                "raw_value": float(best_exp.get(base_metric, 0)),
            })

    # For each artifact, find if any experiment does better
    suggestions = []
    for artifact in top_artifacts[:3]:
        metric = artifact["metric"]
        if metric not in df.columns:
            continue
        best_val = df.loc[best_idx, metric]
        # Find experiment with lowest value for this metric (lower = better)
        better_mask = df[metric] < best_val * 0.8  # At least 20% better
        if better_mask.any():
            better_exp = df.loc[df[metric].idxmin()]
            better_meta = _parse_experiment_name(better_exp["experiment"])
            suggestions.append({
                "artifact": metric,
                "current_value": float(best_val),
                "better_experiment": better_exp["experiment"],
                "better_value": float(better_exp[metric]),
                "parameter_difference": {
                    k: {"current": best_meta[k], "better": better_meta[k]}
                    for k in ("prediction_type", "lp_norm")
                    if best_meta[k] != better_meta[k]
                },
            })

    recommendation = {
        "current_best": best_name,
        "current_severity": float(best_exp["overall_artifact_severity"]),
        "top_remaining_artifacts": top_artifacts,
        "improvement_suggestions": suggestions,
    }

    # Propose hybrid configuration
    if suggestions:
        hybrid = dict(best_meta)
        for s in suggestions:
            for param, values in s.get("parameter_difference", {}).items():
                # Only adopt if it doesn't conflict with a previous suggestion
                if param not in hybrid or hybrid[param] == best_meta[param]:
                    hybrid[param] = values["better"]
        recommendation["proposed_hybrid"] = hybrid

    return recommendation


def _plot_effect_sizes(
    effects: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot effect size heatmap."""
    import matplotlib.pyplot as plt

    if not effects:
        return

    # Build matrix
    params = list(effects.keys())
    all_metrics = set()
    for param_effects in effects.values():
        all_metrics.update(param_effects.keys())
    metrics = sorted(all_metrics)

    matrix = np.zeros((len(params), len(metrics)))
    for i, param in enumerate(params):
        for j, metric in enumerate(metrics):
            matrix[i, j] = effects[param].get(metric, 0)

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 0.6), max(3, len(params) * 1.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params)
    ax.set_title("Effect Size: Parameter Impact on Metrics")
    plt.colorbar(im, ax=ax, label="Effect size (range/std)")
    plt.tight_layout()
    save_figure(fig, output_dir, "effect_sizes_heatmap")
    plt.close(fig)


def run_paired_comparison(
    cfg: DictConfig,
) -> dict:
    """Run paired comparison analysis across all experiments.

    Args:
        cfg: Diagnostics configuration.

    Returns:
        Dictionary with paired comparison results and recommendations.
    """
    output_base_dir = Path(cfg.output.base_dir)

    # Load diagnostic report
    df = _load_diagnostic_report(output_base_dir)
    if df is None or df.empty:
        logger.error("Cannot run paired comparison: no diagnostic report available")
        return {"error": "no_diagnostic_report"}

    # Identify metric columns (raw, non-metadata)
    meta_cols = {"experiment", "prediction_type", "lp_norm", "overall_artifact_severity"}
    norm_cols = {c for c in df.columns if c.endswith("_norm")}
    metric_cols = [c for c in df.columns if c not in meta_cols and c not in norm_cols]

    logger.info(f"Paired comparison across {len(df)} experiments, {len(metric_cols)} metrics")

    # Compute paired deltas
    pairs = _compute_paired_deltas(df, metric_cols)
    logger.info(f"Found {len(pairs)} valid experiment pairs")

    # Detect trends
    trends = _detect_trends(df, metric_cols)

    # Effect sizes
    effects = _compute_effect_sizes(df, metric_cols)

    # Next experiment recommendation
    recommendation = _recommend_next_experiment(df, metric_cols)

    results = {
        "n_experiments": len(df),
        "n_metrics": len(metric_cols),
        "n_pairs": len(pairs),
        "paired_deltas": pairs,
        "trends": trends,
        "effect_sizes": effects,
        "next_experiment": recommendation,
    }

    # Save results
    output_dir = ensure_output_dir(output_base_dir, "cross_experiment", "paired_comparison")
    save_result_json(results, output_dir / "paired_comparison_results.json")

    # Paired deltas CSV
    if pairs:
        delta_rows = []
        for pair in pairs:
            row = {
                "exp_a": pair["exp_a"],
                "exp_b": pair["exp_b"],
                "varying_param": pair["varying_param"],
                "value_a": pair["value_a"],
                "value_b": pair["value_b"],
                "severity_a": pair["severity_a"],
                "severity_b": pair["severity_b"],
                "severity_delta": pair["severity_b"] - pair["severity_a"],
            }
            for metric, delta in pair["deltas"].items():
                row[f"delta_{metric}"] = delta
            delta_rows.append(row)
        save_csv(pd.DataFrame(delta_rows), output_dir / "paired_deltas.csv")

    # Effect sizes CSV
    if effects:
        effect_rows = []
        for param, param_effects in effects.items():
            for metric, effect in param_effects.items():
                effect_rows.append({
                    "parameter": param,
                    "metric": metric,
                    "effect_size": effect,
                })
        save_csv(pd.DataFrame(effect_rows), output_dir / "effect_sizes.csv")

    # Visualizations
    _plot_effect_sizes(effects, output_dir)

    # Log key findings
    if recommendation.get("proposed_hybrid"):
        logger.info(f"Recommended next experiment: {recommendation['proposed_hybrid']}")
    if effects.get("prediction_type"):
        top_effect = max(effects["prediction_type"].items(), key=lambda x: x[1])
        logger.info(f"Strongest prediction_type effect: {top_effect[0]} (d={top_effect[1]:.2f})")

    return results
