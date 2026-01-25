"""Inter-experiment comparison of confusion-stratified XAI results.

Compares confusion matrix distributions, classifier behavior, and attention
patterns across experiments to identify which experimental conditions lead
to higher FP rates (synthetic samples indistinguishable from real).

The goal is to guide experiment design toward maximizing FP rate, which
indicates synthetic images that successfully fool the classifier.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from omegaconf import DictConfig
from scipy.stats import spearmanr, pearsonr

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    save_csv,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfusionData:
    """Confusion-stratified data for a single experiment."""

    experiment_name: str
    fp_rate: float
    fn_rate: float
    n_tp: int
    n_tn: int
    n_fp: int
    n_fn: int

    # Channel contributions per category
    tp_image_fraction: float = 0.0
    tp_mask_fraction: float = 0.0
    tn_image_fraction: float = 0.0
    tn_mask_fraction: float = 0.0
    fp_image_fraction: float = 0.0
    fp_mask_fraction: float = 0.0
    fn_image_fraction: float = 0.0
    fn_mask_fraction: float = 0.0

    # Classifier confidence
    tp_mean_prob: float = 0.0
    tn_mean_prob: float = 0.0
    fp_mean_prob: float = 0.0
    fn_mean_prob: float = 0.0

    # Comparison metrics (if available)
    fp_vs_tn_cosine_sim: float = 0.0
    fp_vs_tn_feature_dist: float = 0.0
    fn_vs_tp_cosine_sim: float = 0.0
    fn_vs_tp_feature_dist: float = 0.0

    # Metadata
    prediction_type: str = ""
    lp_norm: str = ""

    # Mean GradCAM heatmaps (optional, loaded separately)
    tp_heatmap: Optional[np.ndarray] = None
    tn_heatmap: Optional[np.ndarray] = None
    fp_heatmap: Optional[np.ndarray] = None
    fn_heatmap: Optional[np.ndarray] = None


def _parse_experiment_name(name: str) -> dict[str, str]:
    """Parse experiment name into components."""
    parts = name.split("_lp_")
    if len(parts) == 2:
        return {"prediction_type": parts[0], "lp_norm": parts[1]}
    return {"prediction_type": name, "lp_norm": "unknown"}


def _load_confusion_summary(summary_path: Path) -> Optional[dict]:
    """Load confusion_stratified_summary.json."""
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def _load_confusion_heatmaps(heatmaps_path: Path) -> dict[str, np.ndarray]:
    """Load confusion_stratified_heatmaps.npz."""
    if not heatmaps_path.exists():
        return {}
    data = np.load(heatmaps_path)
    return {k: data[k] for k in data.files if k.endswith("_mean_gradcam")}


def load_experiment_data(
    diagnostics_dir: Path,
    experiment_name: str,
) -> Optional[ExperimentConfusionData]:
    """Load confusion-stratified data for a single experiment.

    Args:
        diagnostics_dir: Base diagnostics directory.
        experiment_name: Name of the experiment.

    Returns:
        ExperimentConfusionData or None if not available.
    """
    confusion_dir = diagnostics_dir / experiment_name / "confusion_stratified"
    summary_path = confusion_dir / "confusion_stratified_summary.json"

    summary = _load_confusion_summary(summary_path)
    if summary is None:
        logger.warning(f"No confusion summary for {experiment_name}")
        return None

    # Parse experiment metadata
    meta = _parse_experiment_name(experiment_name)

    # Extract category data
    categories = summary.get("categories", {})

    def get_cat_field(cat: str, field: str, default=0.0):
        cat_data = categories.get(cat, {})
        if field == "n_samples":
            return cat_data.get(field, 0)
        elif field.endswith("_fraction"):
            contrib = cat_data.get("channel_contributions", {})
            return contrib.get(field, default)
        elif field == "mean_prob":
            return cat_data.get(field, default)
        return default

    # Extract comparison metrics
    comparisons = summary.get("comparisons", {})
    fp_vs_tn = comparisons.get("FP_vs_TN", {})
    fn_vs_tp = comparisons.get("FN_vs_TP", {})

    data = ExperimentConfusionData(
        experiment_name=experiment_name,
        fp_rate=summary.get("fp_rate", 0.0),
        fn_rate=summary.get("fn_rate", 0.0),
        n_tp=get_cat_field("TP", "n_samples"),
        n_tn=get_cat_field("TN", "n_samples"),
        n_fp=get_cat_field("FP", "n_samples"),
        n_fn=get_cat_field("FN", "n_samples"),
        tp_image_fraction=get_cat_field("TP", "image_fraction"),
        tp_mask_fraction=get_cat_field("TP", "mask_fraction"),
        tn_image_fraction=get_cat_field("TN", "image_fraction"),
        tn_mask_fraction=get_cat_field("TN", "mask_fraction"),
        fp_image_fraction=get_cat_field("FP", "image_fraction"),
        fp_mask_fraction=get_cat_field("FP", "mask_fraction"),
        fn_image_fraction=get_cat_field("FN", "image_fraction"),
        fn_mask_fraction=get_cat_field("FN", "mask_fraction"),
        tp_mean_prob=get_cat_field("TP", "mean_prob"),
        tn_mean_prob=get_cat_field("TN", "mean_prob"),
        fp_mean_prob=get_cat_field("FP", "mean_prob"),
        fn_mean_prob=get_cat_field("FN", "mean_prob"),
        fp_vs_tn_cosine_sim=fp_vs_tn.get("cosine_similarity", 0.0),
        fp_vs_tn_feature_dist=fp_vs_tn.get("feature_distance", 0.0),
        fn_vs_tp_cosine_sim=fn_vs_tp.get("cosine_similarity", 0.0),
        fn_vs_tp_feature_dist=fn_vs_tp.get("feature_distance", 0.0),
        prediction_type=meta["prediction_type"],
        lp_norm=meta["lp_norm"],
    )

    # Load heatmaps if available
    heatmaps_path = confusion_dir / "confusion_stratified_heatmaps.npz"
    heatmaps = _load_confusion_heatmaps(heatmaps_path)
    if "TP_mean_gradcam" in heatmaps:
        data.tp_heatmap = heatmaps["TP_mean_gradcam"]
    if "TN_mean_gradcam" in heatmaps:
        data.tn_heatmap = heatmaps["TN_mean_gradcam"]
    if "FP_mean_gradcam" in heatmaps:
        data.fp_heatmap = heatmaps["FP_mean_gradcam"]
    if "FN_mean_gradcam" in heatmaps:
        data.fn_heatmap = heatmaps["FN_mean_gradcam"]

    return data


def load_all_experiments(
    diagnostics_dir: Path,
    experiment_names: Optional[list[str]] = None,
) -> list[ExperimentConfusionData]:
    """Load confusion data for all (or specified) experiments.

    Args:
        diagnostics_dir: Base diagnostics directory.
        experiment_names: Optional list of experiment names to load.
            If None, auto-discovers all experiments with confusion_stratified results.

    Returns:
        List of ExperimentConfusionData objects.
    """
    diagnostics_dir = Path(diagnostics_dir)

    if experiment_names is None:
        # Auto-discover experiments
        experiment_names = []
        for exp_dir in sorted(diagnostics_dir.iterdir()):
            if exp_dir.is_dir():
                confusion_dir = exp_dir / "confusion_stratified"
                if (confusion_dir / "confusion_stratified_summary.json").exists():
                    experiment_names.append(exp_dir.name)

    experiments = []
    for exp_name in experiment_names:
        data = load_experiment_data(diagnostics_dir, exp_name)
        if data is not None:
            experiments.append(data)

    logger.info(f"Loaded confusion data for {len(experiments)} experiments")
    return experiments


def build_comparison_dataframe(
    experiments: list[ExperimentConfusionData],
) -> pd.DataFrame:
    """Build a DataFrame comparing all experiments.

    Args:
        experiments: List of experiment data.

    Returns:
        DataFrame with one row per experiment.
    """
    rows = []
    for exp in experiments:
        row = {
            "experiment": exp.experiment_name,
            "prediction_type": exp.prediction_type,
            "lp_norm": exp.lp_norm,
            "fp_rate": exp.fp_rate,
            "fn_rate": exp.fn_rate,
            "accuracy": (exp.n_tp + exp.n_tn) / max(1, exp.n_tp + exp.n_tn + exp.n_fp + exp.n_fn),
            "n_tp": exp.n_tp,
            "n_tn": exp.n_tn,
            "n_fp": exp.n_fp,
            "n_fn": exp.n_fn,
            "n_total": exp.n_tp + exp.n_tn + exp.n_fp + exp.n_fn,
            # Channel contributions
            "tp_image_fraction": exp.tp_image_fraction,
            "tp_mask_fraction": exp.tp_mask_fraction,
            "tn_image_fraction": exp.tn_image_fraction,
            "tn_mask_fraction": exp.tn_mask_fraction,
            "fp_image_fraction": exp.fp_image_fraction,
            "fp_mask_fraction": exp.fp_mask_fraction,
            "fn_image_fraction": exp.fn_image_fraction,
            "fn_mask_fraction": exp.fn_mask_fraction,
            # Classifier confidence
            "tp_mean_prob": exp.tp_mean_prob,
            "tn_mean_prob": exp.tn_mean_prob,
            "fp_mean_prob": exp.fp_mean_prob,
            "fn_mean_prob": exp.fn_mean_prob,
            # Comparison metrics
            "fp_vs_tn_cosine_sim": exp.fp_vs_tn_cosine_sim,
            "fp_vs_tn_feature_dist": exp.fp_vs_tn_feature_dist,
            "fn_vs_tp_cosine_sim": exp.fn_vs_tp_cosine_sim,
            "fn_vs_tp_feature_dist": exp.fn_vs_tp_feature_dist,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_fp_rate_correlations(
    df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute correlations between FP rate and other metrics.

    Higher correlations indicate metrics that co-vary with synthetic quality.

    Args:
        df: Comparison DataFrame.

    Returns:
        Dict mapping metric -> {"pearson": r, "spearman": rho, "p_value": p}.
    """
    correlations = {}

    # Metrics to correlate with FP rate
    metrics = [
        "fn_rate", "accuracy",
        "tp_image_fraction", "tn_image_fraction",
        "tp_mean_prob", "tn_mean_prob",
        "fn_vs_tp_feature_dist",
    ]

    fp_rate = df["fp_rate"].values

    # Skip if all FP rates are 0 or constant
    if np.std(fp_rate) < 1e-12:
        logger.warning("FP rate has no variance, cannot compute correlations")
        return correlations

    for metric in metrics:
        if metric not in df.columns:
            continue

        values = df[metric].values
        if np.std(values) < 1e-12:
            continue

        # Filter valid pairs
        valid = ~(np.isnan(fp_rate) | np.isnan(values))
        if valid.sum() < 3:
            continue

        try:
            pearson_r, pearson_p = pearsonr(fp_rate[valid], values[valid])
            spearman_rho, spearman_p = spearmanr(fp_rate[valid], values[valid])

            correlations[metric] = {
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_rho": float(spearman_rho),
                "spearman_p": float(spearman_p),
            }
        except Exception as e:
            logger.warning(f"Failed to compute correlation for {metric}: {e}")

    return correlations


def compute_prediction_type_effects(
    df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute effect of prediction type on confusion metrics.

    Args:
        df: Comparison DataFrame.

    Returns:
        Dict mapping prediction_type -> {metric: mean_value}.
    """
    effects = {}

    metrics = ["fp_rate", "fn_rate", "accuracy", "tn_image_fraction"]

    for pred_type in df["prediction_type"].unique():
        subset = df[df["prediction_type"] == pred_type]
        type_effects = {}
        for metric in metrics:
            if metric in subset.columns:
                type_effects[metric] = float(subset[metric].mean())
        effects[pred_type] = type_effects

    return effects


def compute_lp_norm_trends(
    df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute trends in metrics as Lp norm varies within each prediction type.

    Args:
        df: Comparison DataFrame.

    Returns:
        Dict mapping prediction_type -> {metric: {"trend": "increasing/decreasing", "rho": value}}.
    """
    trends = {}

    metrics = ["fp_rate", "fn_rate", "tn_image_fraction"]

    for pred_type in df["prediction_type"].unique():
        subset = df[df["prediction_type"] == pred_type].copy()
        if len(subset) < 3:
            continue

        # Convert lp_norm to float for correlation
        try:
            lp_values = subset["lp_norm"].astype(float).values
        except (ValueError, TypeError):
            continue

        type_trends = {}
        for metric in metrics:
            if metric not in subset.columns:
                continue

            values = subset[metric].values
            valid = ~np.isnan(values)
            if valid.sum() < 3:
                continue

            try:
                rho, p_value = spearmanr(lp_values[valid], values[valid])
                if abs(rho) > 0.5:  # Moderate to strong trend
                    type_trends[metric] = {
                        "trend": "increasing" if rho > 0 else "decreasing",
                        "spearman_rho": float(rho),
                        "p_value": float(p_value),
                    }
            except Exception:
                continue

        if type_trends:
            trends[pred_type] = type_trends

    return trends


def identify_best_experiments(
    df: pd.DataFrame,
    n_top: int = 3,
) -> list[dict[str, Any]]:
    """Identify experiments with highest FP rate (best synthetic quality).

    Args:
        df: Comparison DataFrame.
        n_top: Number of top experiments to return.

    Returns:
        List of dicts with experiment details.
    """
    sorted_df = df.sort_values("fp_rate", ascending=False).head(n_top)

    top_experiments = []
    for _, row in sorted_df.iterrows():
        top_experiments.append({
            "experiment": row["experiment"],
            "fp_rate": row["fp_rate"],
            "fn_rate": row["fn_rate"],
            "accuracy": row["accuracy"],
            "prediction_type": row["prediction_type"],
            "lp_norm": row["lp_norm"],
            "n_fp": row["n_fp"],
            "n_tn": row["n_tn"],
        })

    return top_experiments


def recommend_next_experiment(
    df: pd.DataFrame,
    experiments: list[ExperimentConfusionData],
) -> dict[str, Any]:
    """Recommend next experiment configuration to maximize FP rate.

    Logic:
    1. Find experiments with any FP samples (FP > 0)
    2. Analyze what differentiates them from FP=0 experiments
    3. If no FP samples anywhere, analyze classifier behavior patterns
    4. Suggest configuration most likely to increase FP

    Args:
        df: Comparison DataFrame.
        experiments: List of experiment data with heatmaps.

    Returns:
        Recommendation dict.
    """
    recommendation = {
        "current_best": None,
        "analysis": [],
        "suggested_changes": [],
        "reasoning": [],
    }

    # Find best current experiment
    best_idx = df["fp_rate"].idxmax()
    best_exp = df.loc[best_idx]
    recommendation["current_best"] = {
        "experiment": best_exp["experiment"],
        "fp_rate": float(best_exp["fp_rate"]),
        "fn_rate": float(best_exp["fn_rate"]),
    }

    # Analyze by prediction type
    pred_type_fp = df.groupby("prediction_type")["fp_rate"].mean()
    best_pred_type = pred_type_fp.idxmax()
    recommendation["analysis"].append({
        "factor": "prediction_type",
        "best_value": best_pred_type,
        "mean_fp_rate": float(pred_type_fp[best_pred_type]),
        "all_values": {k: float(v) for k, v in pred_type_fp.items()},
    })

    # Analyze by lp_norm within best prediction type
    best_pred_df = df[df["prediction_type"] == best_pred_type]
    if len(best_pred_df) > 1:
        try:
            lp_values = best_pred_df["lp_norm"].astype(float)
            fp_values = best_pred_df["fp_rate"]
            rho, _ = spearmanr(lp_values, fp_values)

            if abs(rho) > 0.3:
                trend = "higher" if rho > 0 else "lower"
                recommendation["analysis"].append({
                    "factor": "lp_norm",
                    "within": best_pred_type,
                    "trend": f"{trend} Lp norm correlates with higher FP rate",
                    "spearman_rho": float(rho),
                })
        except (ValueError, TypeError):
            pass

    # Analyze classifier behavior patterns
    # If TN samples have very high confidence, the classifier is too good
    mean_tn_prob = df["tn_mean_prob"].mean()
    if mean_tn_prob > 0.9:
        recommendation["analysis"].append({
            "observation": "classifier_confidence",
            "detail": f"Mean TN probability is {mean_tn_prob:.3f} (very confident on synthetic)",
            "implication": "Classifier easily detects synthetic samples - need to address artifacts",
        })

    # Check channel contributions
    # If classifier relies heavily on one channel, that's where artifacts are
    mean_image_frac = df["tn_image_fraction"].mean()
    if mean_image_frac > 0.6:
        recommendation["analysis"].append({
            "observation": "channel_reliance",
            "detail": f"Classifier relies {mean_image_frac:.1%} on image channel for TN",
            "implication": "Focus on improving FLAIR image quality (spectral, texture)",
        })
    elif mean_image_frac < 0.4:
        recommendation["analysis"].append({
            "observation": "channel_reliance",
            "detail": f"Classifier relies {1-mean_image_frac:.1%} on mask channel for TN",
            "implication": "Focus on improving lesion mask quality (boundaries, shape)",
        })

    # Generate suggestions
    if best_exp["fp_rate"] == 0:
        recommendation["suggested_changes"].append({
            "change": "Try FFL (Focal Frequency Loss)",
            "reasoning": "If HF artifacts are the issue, FFL can help preserve high-frequency details",
        })
        recommendation["suggested_changes"].append({
            "change": "Increase self-conditioning probability",
            "reasoning": "Higher self-conditioning (0.8) allows iterative refinement of details",
        })
        recommendation["suggested_changes"].append({
            "change": "Reduce DDIM eta",
            "reasoning": "Lower eta (0.1) reduces stochastic noise that may corrupt fine details",
        })

    recommendation["suggested_changes"].append({
        "change": f"Use prediction_type={best_pred_type}",
        "reasoning": f"This prediction type has highest mean FP rate ({pred_type_fp[best_pred_type]:.3f})",
    })

    return recommendation


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_fp_rate_comparison(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot FP rate comparison across experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart by experiment
    ax = axes[0]
    sorted_df = df.sort_values("fp_rate", ascending=True)
    colors = []
    for _, row in sorted_df.iterrows():
        if row["prediction_type"] == "x0":
            colors.append("#2ecc71")  # Green
        elif row["prediction_type"] == "velocity":
            colors.append("#3498db")  # Blue
        else:
            colors.append("#e74c3c")  # Red (epsilon)

    ax.barh(range(len(sorted_df)), sorted_df["fp_rate"], color=colors)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df["experiment"], fontsize=9)
    ax.set_xlabel("FP Rate (Synthetic classified as Real)")
    ax.set_title("FP Rate by Experiment\n(Higher = Better Synthetic Quality)")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Random chance")

    # Add legend for prediction types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="x0 (sample)"),
        Patch(facecolor="#3498db", label="velocity"),
        Patch(facecolor="#e74c3c", label="epsilon"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Grouped bar chart by prediction type
    ax = axes[1]
    pred_types = df["prediction_type"].unique()
    x = np.arange(len(pred_types))
    width = 0.25

    for i, lp in enumerate(sorted(df["lp_norm"].unique())):
        lp_data = df[df["lp_norm"] == lp]
        fp_rates = [lp_data[lp_data["prediction_type"] == pt]["fp_rate"].values[0]
                    if len(lp_data[lp_data["prediction_type"] == pt]) > 0 else 0
                    for pt in pred_types]
        ax.bar(x + i * width, fp_rates, width, label=f"Lp={lp}")

    ax.set_xlabel("Prediction Type")
    ax.set_ylabel("FP Rate")
    ax.set_title("FP Rate by Prediction Type and Lp Norm")
    ax.set_xticks(x + width)
    ax.set_xticklabels(pred_types)
    ax.legend(title="Lp Norm")
    ax.set_ylim(0, max(0.1, df["fp_rate"].max() * 1.2))

    plt.tight_layout()
    save_figure(fig, output_dir, "fp_rate_comparison")
    plt.close(fig)


def plot_confusion_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot confusion matrix category distribution across experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked bar chart
    experiments = df["experiment"].tolist()
    x = np.arange(len(experiments))
    width = 0.6

    # Normalize to percentages
    totals = df["n_tp"] + df["n_tn"] + df["n_fp"] + df["n_fn"]
    tp_pct = df["n_tp"] / totals * 100
    tn_pct = df["n_tn"] / totals * 100
    fp_pct = df["n_fp"] / totals * 100
    fn_pct = df["n_fn"] / totals * 100

    ax.bar(x, tp_pct, width, label="TP (Real→Real)", color="#2ecc71")
    ax.bar(x, tn_pct, width, bottom=tp_pct, label="TN (Synth→Synth)", color="#e74c3c")
    ax.bar(x, fp_pct, width, bottom=tp_pct + tn_pct, label="FP (Synth→Real)", color="#3498db")
    ax.bar(x, fn_pct, width, bottom=tp_pct + tn_pct + fp_pct, label="FN (Real→Synth)", color="#f39c12")

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Percentage of Samples")
    ax.set_title("Confusion Matrix Distribution by Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    save_figure(fig, output_dir, "confusion_distribution")
    plt.close(fig)


def plot_channel_contributions(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot classifier channel reliance across experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TN channel contributions (most informative - correctly identified synthetic)
    ax = axes[0]
    sorted_df = df.sort_values("tn_image_fraction", ascending=True)
    x = np.arange(len(sorted_df))

    ax.barh(x, sorted_df["tn_image_fraction"], label="Image Channel", color="#3498db", alpha=0.8)
    ax.barh(x, sorted_df["tn_mask_fraction"], left=sorted_df["tn_image_fraction"],
            label="Mask Channel", color="#e74c3c", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(sorted_df["experiment"], fontsize=8)
    ax.set_xlabel("Channel Contribution Fraction")
    ax.set_title("TN Samples: Classifier Channel Reliance\n(What the classifier uses to detect synthetic)")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)

    # Comparison: TN vs TP channel reliance
    ax = axes[1]
    experiments = df["experiment"].tolist()
    x = np.arange(len(experiments))
    width = 0.35

    ax.bar(x - width/2, df["tn_image_fraction"], width, label="TN (Synthetic)", color="#e74c3c", alpha=0.8)
    ax.bar(x + width/2, df["tp_image_fraction"], width, label="TP (Real)", color="#2ecc71", alpha=0.8)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Image Channel Fraction")
    ax.set_title("Image Channel Reliance: TN vs TP")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=7)
    ax.legend(loc="best", fontsize=8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_figure(fig, output_dir, "channel_contributions")
    plt.close(fig)


def plot_classifier_confidence(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot classifier confidence distribution across experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))

    experiments = df["experiment"].tolist()
    x = np.arange(len(experiments))
    width = 0.2

    # Plot mean probabilities for each category
    ax.bar(x - 1.5*width, df["tp_mean_prob"], width, label="TP (Real→Real)", color="#2ecc71")
    ax.bar(x - 0.5*width, df["tn_mean_prob"], width, label="TN (Synth→Synth)", color="#e74c3c")
    ax.bar(x + 0.5*width, df["fp_mean_prob"], width, label="FP (Synth→Real)", color="#3498db")
    ax.bar(x + 1.5*width, df["fn_mean_prob"], width, label="FN (Real→Synth)", color="#f39c12")

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Mean Classifier Probability (P(synthetic))")
    ax.set_title("Classifier Confidence by Category\n(TP/FP should be low, TN/FN should be high)")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="best", fontsize=8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_figure(fig, output_dir, "classifier_confidence")
    plt.close(fig)


def plot_heatmap_comparison(
    experiments: list[ExperimentConfusionData],
    output_dir: Path,
    category: str = "TN",
) -> None:
    """Plot mean GradCAM heatmaps across experiments for a category."""
    # Filter experiments with heatmaps
    heatmap_attr = f"{category.lower()}_heatmap"
    exp_with_heatmaps = [e for e in experiments if getattr(e, heatmap_attr) is not None]

    if len(exp_with_heatmaps) < 2:
        logger.info(f"Not enough experiments with {category} heatmaps for comparison")
        return

    n_exp = len(exp_with_heatmaps)
    n_cols = min(4, n_exp)
    n_rows = (n_exp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, exp in enumerate(exp_with_heatmaps):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        heatmap = getattr(exp, heatmap_attr)
        im = ax.imshow(heatmap, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"{exp.experiment_name}\n(n={getattr(exp, f'n_{category.lower()}')})", fontsize=9)
        ax.axis("off")

    # Turn off empty axes
    for idx in range(len(exp_with_heatmaps), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)

    category_labels = {"TP": "Real→Real", "TN": "Synth→Synth", "FP": "Synth→Real", "FN": "Real→Synth"}
    plt.suptitle(f"Mean GradCAM Attention: {category} ({category_labels.get(category, category)})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, f"heatmap_comparison_{category.lower()}")
    plt.close(fig)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_confusion_comparison(
    cfg: DictConfig,
    experiment_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run inter-experiment comparison of confusion-stratified results.

    Args:
        cfg: Diagnostics configuration.
        experiment_names: Optional list of experiments to compare.
            If None, auto-discovers all available experiments.

    Returns:
        Dictionary with comparison results and recommendations.
    """
    diagnostics_dir = Path(cfg.output.base_dir)

    # Load all experiment data
    experiments = load_all_experiments(diagnostics_dir, experiment_names)

    if len(experiments) < 2:
        logger.error("Need at least 2 experiments for comparison")
        return {"error": "insufficient_experiments", "n_loaded": len(experiments)}

    # Build comparison DataFrame
    df = build_comparison_dataframe(experiments)

    logger.info(f"Comparing {len(experiments)} experiments")
    logger.info(f"FP rates range: {df['fp_rate'].min():.3f} - {df['fp_rate'].max():.3f}")

    # Compute analysis
    correlations = compute_fp_rate_correlations(df)
    pred_type_effects = compute_prediction_type_effects(df)
    lp_norm_trends = compute_lp_norm_trends(df)
    top_experiments = identify_best_experiments(df, n_top=3)
    recommendation = recommend_next_experiment(df, experiments)

    # Build results
    results = {
        "n_experiments": len(experiments),
        "experiments": [e.experiment_name for e in experiments],
        "fp_rate_range": {
            "min": float(df["fp_rate"].min()),
            "max": float(df["fp_rate"].max()),
            "mean": float(df["fp_rate"].mean()),
        },
        "fn_rate_range": {
            "min": float(df["fn_rate"].min()),
            "max": float(df["fn_rate"].max()),
            "mean": float(df["fn_rate"].mean()),
        },
        "top_experiments": top_experiments,
        "fp_rate_correlations": correlations,
        "prediction_type_effects": pred_type_effects,
        "lp_norm_trends": lp_norm_trends,
        "recommendation": recommendation,
    }

    # Save results
    output_dir = ensure_output_dir(diagnostics_dir, "cross_experiment", "confusion_comparison")

    # Save JSON
    save_result_json(results, output_dir / "confusion_comparison_results.json")

    # Save DataFrame
    save_csv(df, output_dir / "confusion_comparison_table.csv")

    # Generate visualizations
    logger.info("Generating comparison visualizations...")
    plot_fp_rate_comparison(df, output_dir)
    plot_confusion_distribution(df, output_dir)
    plot_channel_contributions(df, output_dir)
    plot_classifier_confidence(df, output_dir)

    # Heatmap comparisons
    for category in ["TN", "TP", "FP", "FN"]:
        plot_heatmap_comparison(experiments, output_dir, category)

    # Log key findings
    logger.info("=" * 60)
    logger.info("CONFUSION COMPARISON SUMMARY")
    logger.info("=" * 60)

    if top_experiments:
        best = top_experiments[0]
        logger.info(f"Best experiment: {best['experiment']} (FP rate: {best['fp_rate']:.3f})")

    if pred_type_effects:
        best_pred = max(pred_type_effects.items(), key=lambda x: x[1].get("fp_rate", 0))
        logger.info(f"Best prediction type: {best_pred[0]} (mean FP rate: {best_pred[1].get('fp_rate', 0):.3f})")

    if recommendation.get("suggested_changes"):
        logger.info("Suggested changes to increase FP rate:")
        for change in recommendation["suggested_changes"][:3]:
            logger.info(f"  - {change['change']}")

    logger.info("=" * 60)

    return results
