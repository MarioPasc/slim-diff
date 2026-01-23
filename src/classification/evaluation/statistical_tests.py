"""Statistical tests for classification quality assessment.

Implements permutation testing for the null hypothesis H0: AUC = 0.5,
meaning the classifier cannot distinguish real from synthetic samples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class PermutationTestResult:
    """Result of a permutation test."""

    observed_auc: float
    p_value: float
    n_permutations: int
    null_distribution_mean: float
    null_distribution_std: float
    significant: bool  # p < alpha


def permutation_test_auc(
    probs: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> PermutationTestResult:
    """Test H0: AUC = 0.5 via label permutation.

    Under the null hypothesis, the classifier has no discrimination ability,
    and the AUC should be close to 0.5. We compute the null distribution by
    shuffling labels and computing AUC on shuffled data.

    The p-value is the fraction of permuted AUCs >= observed AUC (one-sided
    test for AUC > 0.5). For AUC < 0.5, we test the other tail.

    Args:
        probs: Predicted probabilities (N,).
        labels: True labels (N,), 0=real, 1=synthetic.
        n_permutations: Number of permutations.
        seed: Random seed.
        alpha: Significance level.

    Returns:
        PermutationTestResult with observed AUC, p-value, and null distribution stats.
    """
    rng = np.random.default_rng(seed)

    # Observed AUC
    observed_auc = roc_auc_score(labels, probs)

    # Generate null distribution
    null_aucs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_labels = rng.permutation(labels)
        if len(np.unique(shuffled_labels)) < 2:
            null_aucs[i] = 0.5
            continue
        null_aucs[i] = roc_auc_score(shuffled_labels, probs)

    # Two-sided p-value: how extreme is observed AUC relative to null?
    # Null distribution should be centered around 0.5
    observed_deviation = abs(observed_auc - 0.5)
    null_deviations = np.abs(null_aucs - 0.5)
    p_value = float(np.mean(null_deviations >= observed_deviation))

    result = PermutationTestResult(
        observed_auc=observed_auc,
        p_value=p_value,
        n_permutations=n_permutations,
        null_distribution_mean=float(np.mean(null_aucs)),
        null_distribution_std=float(np.std(null_aucs)),
        significant=p_value < alpha,
    )

    logger.info(
        f"Permutation test: AUC={observed_auc:.4f}, p={p_value:.4f}, "
        f"null_mean={result.null_distribution_mean:.4f}, "
        f"significant={result.significant}"
    )

    return result
