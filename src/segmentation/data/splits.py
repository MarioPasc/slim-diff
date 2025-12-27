"""Subject-level k-fold splitting for segmentation."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class SubjectKFoldSplitter:
    """Create subject-level k-fold splits excluding test set."""

    def __init__(
        self,
        cache_dir: Path | str,
        n_folds: int = 5,
        exclude_test: bool = True,
        stratify_by: str = "has_lesion_subject",
        seed: int = 42,
    ):
        """Initialize splitter.

        Args:
            cache_dir: Path to slice_cache
            n_folds: Number of folds
            exclude_test: Whether to exclude test.csv subjects
            stratify_by: Stratification strategy
            seed: Random seed
        """
        self.cache_dir = Path(cache_dir)
        self.n_folds = n_folds
        self.exclude_test = exclude_test
        self.stratify_by = stratify_by
        self.seed = seed

        # Load and process data
        self.subjects_info = self._load_subject_info()
        self.folds = self._create_folds()

        logger.info(
            f"K-fold splitter initialized: {n_folds} folds, "
            f"{len(self.subjects_info)} subjects, "
            f"exclude_test={exclude_test}"
        )

    def _load_subject_info(self) -> dict:
        """Load subject-level information from CSVs.

        Returns:
            dict: {subject_id: {'has_lesion': bool, 'n_slices': int, 'slices': list}}
        """
        subjects = {}

        # Load train and val CSVs (exclude test if configured)
        csv_files = ["train.csv", "val.csv"]

        for csv_name in csv_files:
            csv_path = self.cache_dir / csv_name

            if not csv_path.exists():
                logger.warning(f"CSV not found: {csv_path}")
                continue

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    subject_id = row["subject_id"]
                    has_lesion = row["has_lesion"].lower() == "true"

                    if subject_id not in subjects:
                        subjects[subject_id] = {
                            "has_lesion": False,
                            "n_slices": 0,
                            "slices": [],
                        }

                    # Subject has lesion if ANY slice has lesion
                    if has_lesion:
                        subjects[subject_id]["has_lesion"] = True

                    subjects[subject_id]["n_slices"] += 1
                    subjects[subject_id]["slices"].append(row)

        logger.info(
            f"Loaded {len(subjects)} subjects, "
            f"{sum(1 for s in subjects.values() if s['has_lesion'])} with lesions"
        )

        return subjects

    def _create_folds(self) -> list[tuple[list[str], list[str]]]:
        """Create stratified k-fold splits at subject level.

        Returns:
            List of (train_subjects, val_subjects) tuples
        """
        # Get subject IDs and stratification labels
        subject_ids = sorted(self.subjects_info.keys())

        if self.stratify_by == "has_lesion_subject":
            # Stratify by whether subject has any lesions
            strat_labels = np.array([
                int(self.subjects_info[sid]["has_lesion"])
                for sid in subject_ids
            ])
        else:
            # No stratification
            strat_labels = None

        # Create k-fold splitter
        kfold = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.seed,
        )

        # Generate folds
        folds = []
        for train_idx, val_idx in kfold.split(subject_ids, strat_labels):
            train_subjects = [subject_ids[i] for i in train_idx]
            val_subjects = [subject_ids[i] for i in val_idx]

            folds.append((train_subjects, val_subjects))

        return folds

    def get_fold(self, fold_idx: int) -> tuple[list[str], list[str]]:
        """Get train/val subject IDs for a specific fold.

        Args:
            fold_idx: Fold index (0 to n_folds-1)

        Returns:
            (train_subjects, val_subjects) tuple
        """
        if fold_idx < 0 or fold_idx >= self.n_folds:
            raise ValueError(
                f"Fold index {fold_idx} out of range [0, {self.n_folds})"
            )

        return self.folds[fold_idx]

    def get_fold_sample_lists(
        self, fold_idx: int
    ) -> tuple[list[dict], list[dict]]:
        """Get sample-level lists for a fold.

        Args:
            fold_idx: Fold index

        Returns:
            (train_samples, val_samples) where each is list of slice metadata dicts
        """
        train_subjects, val_subjects = self.get_fold(fold_idx)

        # Collect all slices for subjects in each split
        train_samples = []
        val_samples = []

        for subject_id in train_subjects:
            train_samples.extend(self.subjects_info[subject_id]["slices"])

        for subject_id in val_subjects:
            val_samples.extend(self.subjects_info[subject_id]["slices"])

        logger.info(
            f"Fold {fold_idx}: {len(train_samples)} train slices from "
            f"{len(train_subjects)} subjects, {len(val_samples)} val slices "
            f"from {len(val_subjects)} subjects"
        )

        return train_samples, val_samples

    def print_fold_statistics(self, fold_idx: int) -> None:
        """Print statistics for a fold.

        Args:
            fold_idx: Fold index
        """
        train_subs, val_subs = self.get_fold(fold_idx)

        # Count lesion subjects
        train_lesion = sum(
            self.subjects_info[s]["has_lesion"] for s in train_subs
        )
        val_lesion = sum(
            self.subjects_info[s]["has_lesion"] for s in val_subs
        )

        print(f"\nFold {fold_idx} Statistics:")
        print(f"  Train: {len(train_subs)} subjects ({train_lesion} with lesions)")
        print(f"  Val:   {len(val_subs)} subjects ({val_lesion} with lesions)")
        print(f"  Train lesion rate: {train_lesion/len(train_subs)*100:.1f}%")
        print(f"  Val lesion rate:   {val_lesion/len(val_subs)*100:.1f}%")
