"""Evaluation metrics and statistical testing for classification."""

from src.classification.evaluation.metrics import compute_fold_metrics, aggregate_fold_metrics
from src.classification.evaluation.statistical_tests import permutation_test_auc

__all__ = [
    "compute_fold_metrics",
    "aggregate_fold_metrics",
    "permutation_test_auc",
]
