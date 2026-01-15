"""Evaluation metrics for audition analysis.

This module provides comprehensive metrics for evaluating the real vs synthetic
classifier, including global AUC, per-zbin analysis, and bootstrap confidence intervals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class GlobalMetrics:
    """Container for global evaluation metrics."""

    auc_roc: float
    auc_roc_ci_lower: float
    auc_roc_ci_upper: float
    pr_auc: float
    pr_auc_ci_lower: float
    pr_auc_ci_upper: float
    accuracy: float
    f1: float
    precision: float
    recall: float
    optimal_threshold: float
    n_samples: int
    n_real: int
    n_synthetic: int


@dataclass
class PerZbinMetrics:
    """Container for per-zbin metrics."""

    z_bin: int
    auc_roc: float
    n_samples: int
    n_real: int
    n_synthetic: int


def compute_global_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    cfg: DictConfig | None = None,
) -> GlobalMetrics:
    """Compute global evaluation metrics with bootstrap confidence intervals.

    Args:
        probs: Predicted probabilities (N,).
        labels: True labels (N,), 0=real, 1=synthetic.
        cfg: Optional configuration for bootstrap settings.

    Returns:
        GlobalMetrics object with all computed metrics.
    """
    n_samples = len(labels)
    n_real = int((labels == 0).sum())
    n_synthetic = int((labels == 1).sum())

    # Basic metrics
    auc_roc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)

    # Optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Metrics at optimal threshold
    preds = (probs >= optimal_threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    # Bootstrap confidence intervals
    if cfg is not None and cfg.evaluation.bootstrap.enabled:
        n_iterations = cfg.evaluation.bootstrap.n_iterations
        confidence_level = cfg.evaluation.bootstrap.confidence_level
    else:
        n_iterations = 1000
        confidence_level = 0.95

    auc_ci_lower, auc_ci_upper = _bootstrap_ci(
        probs, labels, roc_auc_score, n_iterations, confidence_level
    )
    pr_ci_lower, pr_ci_upper = _bootstrap_ci(
        probs, labels, average_precision_score, n_iterations, confidence_level
    )

    return GlobalMetrics(
        auc_roc=auc_roc,
        auc_roc_ci_lower=auc_ci_lower,
        auc_roc_ci_upper=auc_ci_upper,
        pr_auc=pr_auc,
        pr_auc_ci_lower=pr_ci_lower,
        pr_auc_ci_upper=pr_ci_upper,
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
        optimal_threshold=optimal_threshold,
        n_samples=n_samples,
        n_real=n_real,
        n_synthetic=n_synthetic,
    )


def _bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    metric_fn: callable,
    n_iterations: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        probs: Predicted probabilities.
        labels: True labels.
        metric_fn: Function to compute metric (probs, labels) -> float.
        n_iterations: Number of bootstrap iterations.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    rng = np.random.default_rng(42)
    n_samples = len(labels)
    bootstrap_scores = []

    for _ in range(n_iterations):
        # Sample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_probs = probs[indices]
        boot_labels = labels[indices]

        # Skip if only one class present
        if len(np.unique(boot_labels)) < 2:
            continue

        try:
            score = metric_fn(boot_labels, boot_probs)
            bootstrap_scores.append(score)
        except ValueError:
            continue

    if len(bootstrap_scores) < 100:
        logger.warning(f"Only {len(bootstrap_scores)} valid bootstrap samples")
        return 0.0, 1.0

    alpha = (1 - confidence_level) / 2
    lower = np.percentile(bootstrap_scores, alpha * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

    return lower, upper


def compute_per_zbin_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    z_bins: np.ndarray,
    min_samples: int = 10,
) -> list[PerZbinMetrics]:
    """Compute AUC for each z-bin.

    Args:
        probs: Predicted probabilities (N,).
        labels: True labels (N,).
        z_bins: Z-bin indices (N,).
        min_samples: Minimum samples required per z-bin.

    Returns:
        List of PerZbinMetrics objects.
    """
    results = []
    unique_zbins = np.unique(z_bins)

    for zbin in sorted(unique_zbins):
        mask = z_bins == zbin
        zbin_probs = probs[mask]
        zbin_labels = labels[mask]

        n_samples = len(zbin_labels)
        n_real = int((zbin_labels == 0).sum())
        n_synthetic = int((zbin_labels == 1).sum())

        # Check if we have enough samples and both classes
        if n_samples < min_samples:
            logger.warning(f"Z-bin {zbin}: only {n_samples} samples, skipping")
            continue

        if n_real == 0 or n_synthetic == 0:
            logger.warning(f"Z-bin {zbin}: missing class, skipping")
            continue

        try:
            auc_roc = roc_auc_score(zbin_labels, zbin_probs)
        except ValueError:
            logger.warning(f"Z-bin {zbin}: could not compute AUC")
            continue

        results.append(
            PerZbinMetrics(
                z_bin=int(zbin),
                auc_roc=auc_roc,
                n_samples=n_samples,
                n_real=n_real,
                n_synthetic=n_synthetic,
            )
        )

    return results


def generate_evaluation_report(
    probs: np.ndarray,
    labels: np.ndarray,
    z_bins: np.ndarray,
    output_dir: Path,
    cfg: DictConfig | None = None,
) -> dict:
    """Generate comprehensive evaluation report with figures.

    Args:
        probs: Predicted probabilities.
        labels: True labels.
        z_bins: Z-bin indices.
        output_dir: Output directory for report and figures.
        cfg: Optional configuration.

    Returns:
        Dictionary with all computed metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Compute global metrics
    logger.info("Computing global metrics...")
    global_metrics = compute_global_metrics(probs, labels, cfg)

    # Compute per-zbin metrics
    logger.info("Computing per-zbin metrics...")
    min_samples = cfg.evaluation.per_zbin.min_samples if cfg else 10
    per_zbin_metrics = compute_per_zbin_metrics(probs, labels, z_bins, min_samples)

    # Generate figures
    logger.info("Generating figures...")
    _plot_roc_curve(probs, labels, global_metrics, figures_dir / "roc_curve.png")
    _plot_pr_curve(probs, labels, global_metrics, figures_dir / "pr_curve.png")
    _plot_per_zbin_auc(per_zbin_metrics, figures_dir / "per_zbin_auc.png")
    _plot_probability_histogram(probs, labels, figures_dir / "prob_histogram.png")

    # Save global metrics
    global_dict = {
        "auc_roc": global_metrics.auc_roc,
        "auc_roc_ci": [global_metrics.auc_roc_ci_lower, global_metrics.auc_roc_ci_upper],
        "pr_auc": global_metrics.pr_auc,
        "pr_auc_ci": [global_metrics.pr_auc_ci_lower, global_metrics.pr_auc_ci_upper],
        "accuracy": global_metrics.accuracy,
        "f1": global_metrics.f1,
        "precision": global_metrics.precision,
        "recall": global_metrics.recall,
        "optimal_threshold": global_metrics.optimal_threshold,
        "n_samples": global_metrics.n_samples,
        "n_real": global_metrics.n_real,
        "n_synthetic": global_metrics.n_synthetic,
        "interpretation": _interpret_auc(global_metrics.auc_roc),
    }

    with open(output_dir / "global_metrics.json", "w") as f:
        json.dump(global_dict, f, indent=2)

    # Save per-zbin metrics
    per_zbin_df = pd.DataFrame([
        {
            "z_bin": m.z_bin,
            "auc_roc": m.auc_roc,
            "n_samples": m.n_samples,
            "n_real": m.n_real,
            "n_synthetic": m.n_synthetic,
        }
        for m in per_zbin_metrics
    ])
    per_zbin_df.to_csv(output_dir / "per_zbin_metrics.csv", index=False)

    # Log summary
    logger.info("=" * 60)
    logger.info("AUDITION EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"AUC-ROC: {global_metrics.auc_roc:.4f} [{global_metrics.auc_roc_ci_lower:.4f}, {global_metrics.auc_roc_ci_upper:.4f}]")
    logger.info(f"PR-AUC: {global_metrics.pr_auc:.4f} [{global_metrics.pr_auc_ci_lower:.4f}, {global_metrics.pr_auc_ci_upper:.4f}]")
    logger.info(f"Accuracy: {global_metrics.accuracy:.4f}")
    logger.info(f"F1 Score: {global_metrics.f1:.4f}")
    logger.info(f"Interpretation: {_interpret_auc(global_metrics.auc_roc)}")
    logger.info("=" * 60)

    return {
        "global": global_dict,
        "per_zbin": per_zbin_df.to_dict(orient="records"),
    }


def _interpret_auc(auc: float) -> str:
    """Interpret AUC value in terms of synthetic quality.

    Args:
        auc: AUC-ROC value.

    Returns:
        Interpretation string.
    """
    if auc < 0.55:
        return "EXCELLENT - Real and synthetic are nearly indistinguishable"
    elif auc < 0.65:
        return "GOOD - Synthetic quality is high, minor differences detectable"
    elif auc < 0.75:
        return "FAIR - Some distinguishable differences between real and synthetic"
    elif auc < 0.85:
        return "MODERATE - Clear differences exist between real and synthetic"
    else:
        return "POOR - Real and synthetic are easily distinguishable"


def _plot_roc_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    metrics: GlobalMetrics,
    save_path: Path,
) -> None:
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {metrics.auc_roc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.2)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random classifier")

    # Add CI annotation
    ci_text = f"95% CI: [{metrics.auc_roc_ci_lower:.3f}, {metrics.auc_roc_ci_upper:.3f}]"
    ax.annotate(ci_text, xy=(0.6, 0.2), fontsize=10)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve: Real vs Synthetic Classification", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved ROC curve to {save_path}")


def _plot_pr_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    metrics: GlobalMetrics,
    save_path: Path,
) -> None:
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {metrics.pr_auc:.3f})")
    ax.fill_between(recall, precision, alpha=0.2)

    # Baseline (random classifier)
    baseline = metrics.n_synthetic / metrics.n_samples
    ax.axhline(y=baseline, color="gray", linestyle="--", label=f"Random baseline ({baseline:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve: Real vs Synthetic Classification", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved PR curve to {save_path}")


def _plot_per_zbin_auc(
    per_zbin_metrics: list[PerZbinMetrics],
    save_path: Path,
) -> None:
    """Plot per-zbin AUC bar chart."""
    if not per_zbin_metrics:
        logger.warning("No per-zbin metrics to plot")
        return

    z_bins = [m.z_bin for m in per_zbin_metrics]
    aucs = [m.auc_roc for m in per_zbin_metrics]
    n_samples = [m.n_samples for m in per_zbin_metrics]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color bars by AUC value
    colors = plt.cm.RdYlGn_r(np.array(aucs))  # Red=high (bad), Green=low (good)
    bars = ax.bar(z_bins, aucs, color=colors, edgecolor="black", linewidth=0.5)

    # Add sample count annotations
    for bar, n in zip(bars, n_samples):
        ax.annotate(
            f"n={n}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    # Reference lines
    ax.axhline(y=0.5, color="green", linestyle="--", linewidth=2, label="Random (AUC=0.5)")
    ax.axhline(y=0.7, color="orange", linestyle=":", linewidth=1, label="Threshold (AUC=0.7)")

    ax.set_xlabel("Z-bin", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Per-Zbin Discriminability (lower is better)", fontsize=14)
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved per-zbin AUC plot to {save_path}")


def _plot_probability_histogram(
    probs: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
) -> None:
    """Plot histogram of predicted probabilities by class."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate by class
    real_probs = probs[labels == 0]
    synth_probs = probs[labels == 1]

    # Plot histograms
    ax.hist(
        real_probs,
        bins=50,
        alpha=0.6,
        color="blue",
        label=f"Real (n={len(real_probs)})",
        density=True,
    )
    ax.hist(
        synth_probs,
        bins=50,
        alpha=0.6,
        color="orange",
        label=f"Synthetic (n={len(synth_probs)})",
        density=True,
    )

    ax.set_xlabel("Predicted Probability (P(synthetic))", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Predicted Probabilities", fontsize=14)
    ax.legend(loc="upper center", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved probability histogram to {save_path}")
