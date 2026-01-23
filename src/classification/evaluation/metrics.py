"""Metrics computation with bootstrap confidence intervals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class GlobalMetrics:
    """Global classification metrics with confidence intervals."""

    auc_roc: float
    auc_roc_ci_lower: float
    auc_roc_ci_upper: float
    pr_auc: float
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
    """Per-z-bin classification metrics."""

    z_bin: int
    auc_roc: float
    n_samples: int
    n_real: int
    n_synthetic: int


@dataclass
class FoldResult:
    """Results from a single fold."""

    fold_idx: int
    global_metrics: GlobalMetrics
    per_zbin_metrics: list[PerZbinMetrics]
    probs: np.ndarray
    labels: np.ndarray
    z_bins: np.ndarray


@dataclass
class ExperimentResult:
    """Aggregated results across all folds for one experiment."""

    experiment_name: str
    input_mode: str
    fold_results: list[FoldResult]
    mean_auc: float
    std_auc: float
    pooled_auc: float
    pooled_ci_lower: float
    pooled_ci_upper: float
    per_zbin_mean_auc: dict[int, float] = field(default_factory=dict)


def compute_fold_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    z_bins: np.ndarray,
    fold_idx: int,
    bootstrap_n: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
    min_samples_per_zbin: int = 10,
) -> FoldResult:
    """Compute all metrics for a single fold.

    Args:
        probs: Predicted probabilities (N,).
        labels: True labels (N,), 0=real, 1=synthetic.
        z_bins: Z-bin indices (N,).
        fold_idx: Fold index.
        bootstrap_n: Number of bootstrap iterations for CI.
        confidence_level: Confidence level for CI.
        bootstrap_seed: Random seed for bootstrap.
        min_samples_per_zbin: Minimum samples for per-z-bin AUC.

    Returns:
        FoldResult with all computed metrics.
    """
    # Global metrics
    auc = roc_auc_score(labels, probs)
    ci_lower, ci_upper = _bootstrap_ci(
        labels, probs, roc_auc_score, bootstrap_n, confidence_level, bootstrap_seed
    )

    pr_auc = average_precision_score(labels, probs)

    # Optimal threshold via Youden's J
    threshold = _youden_threshold(labels, probs)
    preds = (probs >= threshold).astype(int)

    global_metrics = GlobalMetrics(
        auc_roc=auc,
        auc_roc_ci_lower=ci_lower,
        auc_roc_ci_upper=ci_upper,
        pr_auc=pr_auc,
        accuracy=accuracy_score(labels, preds),
        f1=f1_score(labels, preds, zero_division=0),
        precision=precision_score(labels, preds, zero_division=0),
        recall=recall_score(labels, preds, zero_division=0),
        optimal_threshold=threshold,
        n_samples=len(labels),
        n_real=int((labels == 0).sum()),
        n_synthetic=int((labels == 1).sum()),
    )

    # Per-z-bin metrics
    per_zbin = []
    unique_zbins = np.unique(z_bins)
    for zb in sorted(unique_zbins):
        mask = z_bins == zb
        if mask.sum() < min_samples_per_zbin:
            continue
        zb_labels = labels[mask]
        zb_probs = probs[mask]
        # Need both classes present
        if len(np.unique(zb_labels)) < 2:
            continue
        zb_auc = roc_auc_score(zb_labels, zb_probs)
        per_zbin.append(PerZbinMetrics(
            z_bin=int(zb),
            auc_roc=zb_auc,
            n_samples=int(mask.sum()),
            n_real=int((zb_labels == 0).sum()),
            n_synthetic=int((zb_labels == 1).sum()),
        ))

    return FoldResult(
        fold_idx=fold_idx,
        global_metrics=global_metrics,
        per_zbin_metrics=per_zbin,
        probs=probs,
        labels=labels,
        z_bins=z_bins,
    )


def aggregate_fold_metrics(
    fold_results: list[FoldResult],
    experiment_name: str,
    input_mode: str,
    bootstrap_n: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> ExperimentResult:
    """Aggregate metrics across folds.

    Computes mean/std of per-fold AUCs and bootstrap CI on pooled predictions.

    Args:
        fold_results: List of FoldResult objects.
        experiment_name: Name of the experiment.
        input_mode: Input mode used.
        bootstrap_n: Bootstrap iterations for pooled CI.
        confidence_level: Confidence level.
        bootstrap_seed: Random seed.

    Returns:
        ExperimentResult with aggregated metrics.
    """
    # Per-fold AUCs
    fold_aucs = [fr.global_metrics.auc_roc for fr in fold_results]
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0

    # Pooled predictions
    all_probs = np.concatenate([fr.probs for fr in fold_results])
    all_labels = np.concatenate([fr.labels for fr in fold_results])
    all_zbins = np.concatenate([fr.z_bins for fr in fold_results])

    pooled_auc = roc_auc_score(all_labels, all_probs)
    pooled_ci_lower, pooled_ci_upper = _bootstrap_ci(
        all_labels, all_probs, roc_auc_score, bootstrap_n, confidence_level, bootstrap_seed
    )

    # Per-z-bin mean AUC across folds
    all_zbins_unique = np.unique(all_zbins)
    per_zbin_mean_auc: dict[int, float] = {}
    for zb in sorted(all_zbins_unique):
        zb_aucs = []
        for fr in fold_results:
            for pz in fr.per_zbin_metrics:
                if pz.z_bin == int(zb):
                    zb_aucs.append(pz.auc_roc)
        if zb_aucs:
            per_zbin_mean_auc[int(zb)] = float(np.mean(zb_aucs))

    return ExperimentResult(
        experiment_name=experiment_name,
        input_mode=input_mode,
        fold_results=fold_results,
        mean_auc=mean_auc,
        std_auc=std_auc,
        pooled_auc=pooled_auc,
        pooled_ci_lower=pooled_ci_lower,
        pooled_ci_upper=pooled_ci_upper,
        per_zbin_mean_auc=per_zbin_mean_auc,
    )


def _bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    metric_fn,
    n_iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    scores = []

    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels = labels[idx]
        boot_probs = probs[idx]
        # Need both classes
        if len(np.unique(boot_labels)) < 2:
            continue
        scores.append(metric_fn(boot_labels, boot_probs))

    if not scores:
        return (0.0, 1.0)

    alpha = 1 - confidence_level
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return (lower, upper)


def _youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Find optimal threshold via Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])
