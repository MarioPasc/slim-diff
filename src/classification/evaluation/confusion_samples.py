"""Confusion matrix sample tracking for per-sample XAI analysis.

This module provides data structures and functions to categorize classification
samples into confusion matrix positions (TP, TN, FP, FN) and track their metadata
for downstream XAI analysis.

The key insight is that FP samples (synthetic incorrectly classified as real)
represent the highest-quality synthetic samples that are indistinguishable from
real data, while TN samples (synthetic correctly classified) contain detectable
artifacts. Comparing these categories enables targeted XAI analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SampleReference:
    """Reference to a specific sample in the dataset.

    Provides full traceability from a confusion matrix category back to the
    original data source (real slice or synthetic replica).

    Attributes:
        original_idx: Index into real_patches or synth_patches array.
        is_real: True if from real_patches, False if from synth_patches.
        z_bin: Anatomical axial level (0-29).
        prob: Classifier probability output (0=real, 1=synthetic).
        subject_id: For real samples, the patient identifier (e.g., "MRIe_072").
        replica_id: For synthetic samples, which replica file (-1 for real).
        sample_idx: For synthetic samples, index within replica (-1 for real).
    """

    original_idx: int
    is_real: bool
    z_bin: int
    prob: float
    subject_id: str = ""
    replica_id: int = -1
    sample_idx: int = -1

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SampleReference:
        """Create from dictionary."""
        return cls(**d)


@dataclass
class ConfusionMatrixSamples:
    """Samples categorized by confusion matrix position.

    For binary classification with real=0 (negative) and synthetic=1 (positive):
    - TP (True Positive): Real correctly classified as real (pred=0, label=0)
    - TN (True Negative): Synthetic correctly classified as synthetic (pred=1, label=1)
    - FP (False Positive): Synthetic incorrectly classified as real (pred=0, label=1)
      ** KEY TARGET: These are the best synthetic samples **
    - FN (False Negative): Real incorrectly classified as synthetic (pred=1, label=0)

    Note: In our convention, label=0 is "real" (negative class) and label=1 is
    "synthetic" (positive class). The classifier predicts probability of synthetic.

    Attributes:
        true_positives: Real samples correctly classified as real.
        true_negatives: Synthetic samples correctly classified as synthetic.
        false_positives: Synthetic samples incorrectly classified as real.
        false_negatives: Real samples incorrectly classified as synthetic.
        threshold: Decision threshold used for classification.
        fold_idx: Cross-validation fold index.
        experiment_name: Name of the experiment.
        input_mode: Input mode used (joint, image_only, mask_only).
    """

    true_positives: list[SampleReference] = field(default_factory=list)
    true_negatives: list[SampleReference] = field(default_factory=list)
    false_positives: list[SampleReference] = field(default_factory=list)
    false_negatives: list[SampleReference] = field(default_factory=list)
    threshold: float = 0.5
    fold_idx: int = 0
    experiment_name: str = ""
    input_mode: str = "joint"

    @property
    def n_tp(self) -> int:
        return len(self.true_positives)

    @property
    def n_tn(self) -> int:
        return len(self.true_negatives)

    @property
    def n_fp(self) -> int:
        return len(self.false_positives)

    @property
    def n_fn(self) -> int:
        return len(self.false_negatives)

    @property
    def n_real(self) -> int:
        return self.n_tp + self.n_fn

    @property
    def n_synthetic(self) -> int:
        return self.n_tn + self.n_fp

    @property
    def fp_rate(self) -> float:
        """False positive rate: fraction of synthetic classified as real."""
        if self.n_synthetic == 0:
            return 0.0
        return self.n_fp / self.n_synthetic

    @property
    def fn_rate(self) -> float:
        """False negative rate: fraction of real classified as synthetic."""
        if self.n_real == 0:
            return 0.0
        return self.n_fn / self.n_real

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        total = self.n_tp + self.n_tn + self.n_fp + self.n_fn
        if total == 0:
            return 0.0
        return (self.n_tp + self.n_tn) / total

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "n_tp": self.n_tp,
            "n_tn": self.n_tn,
            "n_fp": self.n_fp,
            "n_fn": self.n_fn,
            "n_real": self.n_real,
            "n_synthetic": self.n_synthetic,
            "fp_rate": self.fp_rate,
            "fn_rate": self.fn_rate,
            "accuracy": self.accuracy,
            "threshold": self.threshold,
            "fold_idx": self.fold_idx,
            "experiment_name": self.experiment_name,
            "input_mode": self.input_mode,
        }

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        data = {
            "true_positives": [s.to_dict() for s in self.true_positives],
            "true_negatives": [s.to_dict() for s in self.true_negatives],
            "false_positives": [s.to_dict() for s in self.false_positives],
            "false_negatives": [s.to_dict() for s in self.false_negatives],
            "threshold": self.threshold,
            "fold_idx": self.fold_idx,
            "experiment_name": self.experiment_name,
            "input_mode": self.input_mode,
            "summary": self.summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved confusion samples to {path}")

    @classmethod
    def load(cls, path: Path) -> ConfusionMatrixSamples:
        """Load from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls(
            true_positives=[SampleReference.from_dict(d) for d in data["true_positives"]],
            true_negatives=[SampleReference.from_dict(d) for d in data["true_negatives"]],
            false_positives=[SampleReference.from_dict(d) for d in data["false_positives"]],
            false_negatives=[SampleReference.from_dict(d) for d in data["false_negatives"]],
            threshold=data["threshold"],
            fold_idx=data.get("fold_idx", 0),
            experiment_name=data.get("experiment_name", ""),
            input_mode=data.get("input_mode", "joint"),
        )

    def get_category_indices(self, category: str) -> list[int]:
        """Get original indices for a category.

        Args:
            category: One of "TP", "TN", "FP", "FN".

        Returns:
            List of original_idx values for samples in that category.
        """
        category_map = {
            "TP": self.true_positives,
            "TN": self.true_negatives,
            "FP": self.false_positives,
            "FN": self.false_negatives,
        }
        if category not in category_map:
            raise ValueError(f"Unknown category: {category}. Must be one of TP, TN, FP, FN.")
        return [s.original_idx for s in category_map[category]]

    def get_category_zbins(self, category: str) -> list[int]:
        """Get z-bins for a category."""
        category_map = {
            "TP": self.true_positives,
            "TN": self.true_negatives,
            "FP": self.false_positives,
            "FN": self.false_negatives,
        }
        if category not in category_map:
            raise ValueError(f"Unknown category: {category}.")
        return [s.z_bin for s in category_map[category]]


def compute_confusion_samples(
    probs: np.ndarray,
    labels: np.ndarray,
    z_bins: np.ndarray,
    original_indices: np.ndarray,
    is_real: np.ndarray,
    threshold: float,
    real_metadata: Optional[dict] = None,
    synth_metadata: Optional[dict] = None,
    fold_idx: int = 0,
    experiment_name: str = "",
    input_mode: str = "joint",
) -> ConfusionMatrixSamples:
    """Categorize samples into confusion matrix positions.

    Args:
        probs: Classifier probabilities (N,), values in [0, 1].
        labels: Ground truth labels (N,), 0=real, 1=synthetic.
        z_bins: Z-bin indices (N,).
        original_indices: Original indices into real/synth patches (N,).
        is_real: Boolean array indicating real (True) or synthetic (False) (N,).
        threshold: Decision threshold for binary classification.
        real_metadata: Optional dict with 'subject_ids' array for real samples.
        synth_metadata: Optional dict with 'replica_ids', 'sample_indices' arrays.
        fold_idx: Cross-validation fold index.
        experiment_name: Name of the experiment.
        input_mode: Input mode used.

    Returns:
        ConfusionMatrixSamples with categorized samples.
    """
    n_samples = len(probs)
    predictions = (probs >= threshold).astype(int)

    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    for i in range(n_samples):
        label = int(labels[i])
        pred = int(predictions[i])
        is_real_sample = bool(is_real[i])
        orig_idx = int(original_indices[i])

        # Build sample reference
        ref = SampleReference(
            original_idx=orig_idx,
            is_real=is_real_sample,
            z_bin=int(z_bins[i]),
            prob=float(probs[i]),
        )

        # Add metadata if available
        if is_real_sample and real_metadata is not None:
            subject_ids = real_metadata.get("subject_ids")
            if subject_ids is not None and orig_idx < len(subject_ids):
                ref.subject_id = str(subject_ids[orig_idx])
        elif not is_real_sample and synth_metadata is not None:
            replica_ids = synth_metadata.get("replica_ids")
            sample_indices = synth_metadata.get("sample_indices")
            if replica_ids is not None and orig_idx < len(replica_ids):
                ref.replica_id = int(replica_ids[orig_idx])
            if sample_indices is not None and orig_idx < len(sample_indices):
                ref.sample_idx = int(sample_indices[orig_idx])

        # Categorize based on label and prediction
        # label=0 is real (negative), label=1 is synthetic (positive)
        # pred=0 means classified as real, pred=1 means classified as synthetic
        if label == 0 and pred == 0:
            # Real correctly classified as real
            true_positives.append(ref)
        elif label == 1 and pred == 1:
            # Synthetic correctly classified as synthetic
            true_negatives.append(ref)
        elif label == 1 and pred == 0:
            # Synthetic incorrectly classified as real (FP - key target)
            false_positives.append(ref)
        elif label == 0 and pred == 1:
            # Real incorrectly classified as synthetic
            false_negatives.append(ref)

    result = ConfusionMatrixSamples(
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        threshold=threshold,
        fold_idx=fold_idx,
        experiment_name=experiment_name,
        input_mode=input_mode,
    )

    logger.info(
        f"Computed confusion samples: TP={result.n_tp}, TN={result.n_tn}, "
        f"FP={result.n_fp}, FN={result.n_fn} (FP rate={result.fp_rate:.3f})"
    )

    return result


def aggregate_confusion_samples(
    fold_samples: list[ConfusionMatrixSamples],
) -> ConfusionMatrixSamples:
    """Aggregate confusion samples across folds.

    Note: Original indices will refer to fold-specific validation sets,
    so this aggregation is primarily for summary statistics.

    Args:
        fold_samples: List of ConfusionMatrixSamples from each fold.

    Returns:
        Aggregated ConfusionMatrixSamples with combined samples.
    """
    if not fold_samples:
        return ConfusionMatrixSamples()

    aggregated = ConfusionMatrixSamples(
        threshold=fold_samples[0].threshold,
        fold_idx=-1,  # Indicates aggregated
        experiment_name=fold_samples[0].experiment_name,
        input_mode=fold_samples[0].input_mode,
    )

    for fs in fold_samples:
        aggregated.true_positives.extend(fs.true_positives)
        aggregated.true_negatives.extend(fs.true_negatives)
        aggregated.false_positives.extend(fs.false_positives)
        aggregated.false_negatives.extend(fs.false_negatives)

    return aggregated


def load_all_fold_confusion_samples(
    results_dir: Path,
    n_folds: int = 3,
) -> list[ConfusionMatrixSamples]:
    """Load confusion samples from all folds.

    Args:
        results_dir: Directory containing fold{idx}_confusion_samples.json files.
        n_folds: Number of folds.

    Returns:
        List of ConfusionMatrixSamples, one per fold.
    """
    results_dir = Path(results_dir)
    fold_samples = []

    for fold_idx in range(n_folds):
        path = results_dir / f"fold{fold_idx}_confusion_samples.json"
        if path.exists():
            fold_samples.append(ConfusionMatrixSamples.load(path))
        else:
            logger.warning(f"Confusion samples not found: {path}")

    return fold_samples
