"""K-fold cross-validation data module for classification with held-out test set."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.classification.data.dataset import ClassificationDataset
from src.classification.data.patch_extractor import PatchExtractor
from src.classification.diagnostics.preprocessing.dithering import apply_uniform_dithering

logger = logging.getLogger(__name__)


class KFoldClassificationDataModule(pl.LightningDataModule):
    """Lightning DataModule with k-fold cross-validation and held-out test set.

    Uses subject-level splits for real data (prevents same-patient slices
    in both train and val) and z-bin-stratified splits for synthetic data.

    The data is split as follows:
    1. First, 20% is held out as a test set (same for all folds)
    2. The remaining 80% is used for k-fold cross-validation (train/val)

    Args:
        cfg: Master configuration.
        experiment_name: Experiment name for synthetic data.
        input_mode: Channel selection mode.
        patches_dir: Directory containing pre-extracted patches.
    """

    def __init__(
        self,
        cfg: DictConfig,
        experiment_name: str,
        input_mode: Literal["joint", "image_only", "mask_only"] = "joint",
        patches_dir: Path | None = None,
        dithering: bool = False,
        dithering_seed: int = 42,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.experiment_name = experiment_name
        self.input_mode = input_mode
        self.dithering = dithering
        self.dithering_seed = dithering_seed
        self.n_folds = cfg.data.kfold.n_folds
        self.kfold_seed = cfg.data.kfold.seed

        # Test split configuration (held-out test set)
        test_split_cfg = cfg.data.get("test_split", None)
        self.test_fraction = test_split_cfg.fraction if test_split_cfg else 0.0
        self.test_seed = test_split_cfg.seed if test_split_cfg else 42

        # Determine patches directory
        if patches_dir is not None:
            self.patches_dir = Path(patches_dir)
        else:
            base = Path(cfg.output.base_dir) / cfg.output.patches_subdir
            self.patches_dir = base / experiment_name

        self._current_fold: int = 0
        self._train_dataset: ClassificationDataset | None = None
        self._val_dataset: ClassificationDataset | None = None
        self._test_dataset: ClassificationDataset | None = None

        # Loaded data (full dataset before splits)
        self._real_patches: np.ndarray | None = None
        self._real_zbins: np.ndarray | None = None
        self._real_subjects: np.ndarray | None = None
        self._synth_patches: np.ndarray | None = None
        self._synth_zbins: np.ndarray | None = None
        self._synth_replica_ids: np.ndarray | None = None
        self._synth_sample_indices: np.ndarray | None = None
        self._image_size: int = 0

        # Held-out test set data (created once, same for all folds)
        self._test_real_patches: np.ndarray | None = None
        self._test_real_zbins: np.ndarray | None = None
        self._test_real_subjects: np.ndarray | None = None
        self._test_synth_patches: np.ndarray | None = None
        self._test_synth_zbins: np.ndarray | None = None
        self._test_synth_replica_ids: np.ndarray | None = None
        self._test_synth_sample_indices: np.ndarray | None = None

        # Training pool data (after test split, used for k-fold)
        self._trainpool_real_patches: np.ndarray | None = None
        self._trainpool_real_zbins: np.ndarray | None = None
        self._trainpool_real_subjects: np.ndarray | None = None
        self._trainpool_synth_patches: np.ndarray | None = None
        self._trainpool_synth_zbins: np.ndarray | None = None
        self._trainpool_synth_replica_ids: np.ndarray | None = None
        self._trainpool_synth_sample_indices: np.ndarray | None = None

        # Fold splits (computed once on training pool)
        self._fold_splits: list[dict] | None = None

        # Test set metadata (stored after setup for confusion samples)
        self._test_real_subject_ids: np.ndarray | None = None
        self._test_synth_replica_ids_meta: np.ndarray | None = None
        self._test_synth_sample_indices_meta: np.ndarray | None = None

    @property
    def current_fold(self) -> int:
        return self._current_fold

    @property
    def in_channels(self) -> int:
        if self.input_mode == "joint":
            return 2
        return 1

    @property
    def _is_full_image(self) -> bool:
        """Detect full-image mode from loaded patch dimensions."""
        return self._image_size > 128

    def set_fold(self, fold_idx: int) -> None:
        """Set the active fold index. Must call setup() after."""
        if fold_idx < 0 or fold_idx >= self.n_folds:
            raise ValueError(f"fold_idx must be in [0, {self.n_folds - 1}]")
        self._current_fold = fold_idx
        self._train_dataset = None
        self._val_dataset = None

    def prepare_data(self) -> None:
        """Verify that patches exist."""
        real_path = self.patches_dir / "real_patches.npz"
        if not real_path.exists():
            raise FileNotFoundError(
                f"Real patches not found at {real_path}. "
                f"Run 'python -m src.classification extract' first."
            )

    def setup(self, stage: str | None = None) -> None:
        """Load patches and create fold-specific datasets.

        First splits off a held-out test set (if configured), then creates
        train/val splits for the current fold from the remaining data.
        """
        if self._real_patches is None:
            self._load_patches()

        # Split off held-out test set (done once, before k-fold)
        if self._trainpool_real_patches is None:
            self._split_test_set()

        if self._fold_splits is None:
            self._compute_fold_splits()

        fold = self._fold_splits[self._current_fold]

        # Build train/val datasets for this fold from the training pool
        real_train = self._trainpool_real_patches[fold["real_train_idx"]].copy()
        real_val = self._trainpool_real_patches[fold["real_val_idx"]].copy()
        real_zbins_train = self._trainpool_real_zbins[fold["real_train_idx"]].copy()
        real_zbins_val = self._trainpool_real_zbins[fold["real_val_idx"]].copy()

        synth_train = self._trainpool_synth_patches[fold["synth_train_idx"]].copy()
        synth_val = self._trainpool_synth_patches[fold["synth_val_idx"]].copy()
        synth_zbins_train = self._trainpool_synth_zbins[fold["synth_train_idx"]].copy()
        synth_zbins_val = self._trainpool_synth_zbins[fold["synth_val_idx"]].copy()

        self._train_dataset = ClassificationDataset(
            real_patches=real_train,
            synth_patches=synth_train,
            real_zbins=real_zbins_train,
            synth_zbins=synth_zbins_train,
            input_mode=self.input_mode,
        )
        self._val_dataset = ClassificationDataset(
            real_patches=real_val,
            synth_patches=synth_val,
            real_zbins=real_zbins_val,
            synth_zbins=synth_zbins_val,
            input_mode=self.input_mode,
        )

        # Create test dataset (same for all folds)
        if self._test_dataset is None and self._test_real_patches is not None:
            self._test_dataset = ClassificationDataset(
                real_patches=self._test_real_patches,
                synth_patches=self._test_synth_patches,
                real_zbins=self._test_real_zbins,
                synth_zbins=self._test_synth_zbins,
                input_mode=self.input_mode,
            )
            # Store test metadata for confusion samples
            self._test_real_subject_ids = self._test_real_subjects.copy()
            if self._test_synth_replica_ids is not None and len(self._test_synth_replica_ids) > 0:
                self._test_synth_replica_ids_meta = self._test_synth_replica_ids.copy()
                self._test_synth_sample_indices_meta = self._test_synth_sample_indices.copy()
            else:
                self._test_synth_replica_ids_meta = np.empty((0,), dtype=np.int32)
                self._test_synth_sample_indices_meta = np.empty((0,), dtype=np.int32)

        test_info = ""
        if self._test_dataset is not None:
            test_info = f", test={len(self._test_dataset)} (real={len(self._test_real_patches)}, synth={len(self._test_synth_patches)})"

        logger.info(
            f"Fold {self._current_fold}: "
            f"train={len(self._train_dataset)} "
            f"(real={len(real_train)}, synth={len(synth_train)}), "
            f"val={len(self._val_dataset)} "
            f"(real={len(real_val)}, synth={len(synth_val)})"
            f"{test_info}"
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None, "Call setup() first."
        weights = self._train_dataset.get_class_weights()
        sampler = WeightedRandomSampler(
            weights, num_samples=len(self._train_dataset), replacement=True
        )
        # Use num_workers=0 for full-image mode to avoid forking large arrays
        num_workers = 0 if self._is_full_image else self.cfg.training.num_workers
        return DataLoader(
            self._train_dataset,
            batch_size=self.cfg.training.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.cfg.training.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, "Call setup() first."
        num_workers = 0 if self._is_full_image else self.cfg.training.num_workers
        return DataLoader(
            self._val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.cfg.training.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the held-out test set dataloader.

        This is the same test set for all folds, used for final evaluation.
        """
        assert self._test_dataset is not None, "Call setup() first. No test set available."
        num_workers = 0 if self._is_full_image else self.cfg.training.num_workers
        return DataLoader(
            self._test_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.cfg.training.pin_memory,
        )

    @property
    def has_test_set(self) -> bool:
        """Check if a held-out test set is configured."""
        return self.test_fraction > 0

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _load_patches(self) -> None:
        """Load pre-extracted patches from disk, optionally applying dithering."""
        real_path = self.patches_dir / "real_patches.npz"
        synth_path = self.patches_dir / "synthetic_patches.npz"

        real_data = np.load(real_path, allow_pickle=True)
        self._real_patches = real_data["patches"]
        self._real_zbins = real_data["z_bins"]
        self._real_subjects = real_data["subject_ids"]
        self._image_size = self._real_patches.shape[-1] if self._real_patches.ndim == 4 else 0

        if synth_path.exists():
            synth_data = np.load(synth_path, allow_pickle=True)
            self._synth_patches = synth_data["patches"]
            self._synth_zbins = synth_data["z_bins"]
            # Load metadata for confusion samples tracking
            if "replica_ids" in synth_data:
                self._synth_replica_ids = synth_data["replica_ids"]
            else:
                self._synth_replica_ids = np.full(len(self._synth_patches), -1, dtype=np.int32)
            if "sample_indices" in synth_data:
                self._synth_sample_indices = synth_data["sample_indices"]
            else:
                self._synth_sample_indices = np.full(len(self._synth_patches), -1, dtype=np.int32)
        else:
            # Control experiment: no synthetic data
            self._synth_patches = np.empty((0, 2, 0, 0), dtype=np.float32)
            self._synth_zbins = np.empty((0,), dtype=np.int32)
            self._synth_replica_ids = np.empty((0,), dtype=np.int32)
            self._synth_sample_indices = np.empty((0,), dtype=np.int32)

        # Apply dithering to synthetic patches to remove float16 quantization artifact
        if self.dithering and len(self._synth_patches) > 0:
            logger.info("Applying uniform dithering to synthetic patches...")
            self._synth_patches, dither_stats = apply_uniform_dithering(
                self._synth_patches, seed=self.dithering_seed
            )
            logger.info(
                f"Dithering applied: {dither_stats.unique_values_before} -> "
                f"{dither_stats.unique_values_after} unique values"
            )

        logger.info(
            f"Loaded patches: {len(self._real_patches)} real, "
            f"{len(self._synth_patches)} synthetic"
            f"{' (dithered)' if self.dithering else ''}"
        )

    def _split_test_set(self) -> None:
        """Split off held-out test set before k-fold cross-validation.

        Uses subject-level splitting for real data (to prevent data leakage)
        and stratified splitting by z-bin for synthetic data.
        """
        if self.test_fraction <= 0:
            # No test split: use all data for k-fold
            self._trainpool_real_patches = self._real_patches
            self._trainpool_real_zbins = self._real_zbins
            self._trainpool_real_subjects = self._real_subjects
            self._trainpool_synth_patches = self._synth_patches
            self._trainpool_synth_zbins = self._synth_zbins
            self._trainpool_synth_replica_ids = self._synth_replica_ids
            self._trainpool_synth_sample_indices = self._synth_sample_indices

            # No test set
            self._test_real_patches = np.empty((0, 2, 0, 0), dtype=np.float32)
            self._test_real_zbins = np.empty((0,), dtype=np.int32)
            self._test_real_subjects = np.empty((0,), dtype=self._real_subjects.dtype)
            self._test_synth_patches = np.empty((0, 2, 0, 0), dtype=np.float32)
            self._test_synth_zbins = np.empty((0,), dtype=np.int32)
            self._test_synth_replica_ids = np.empty((0,), dtype=np.int32)
            self._test_synth_sample_indices = np.empty((0,), dtype=np.int32)
            return

        logger.info(f"Splitting {self.test_fraction:.0%} held-out test set...")

        # ---- Real data: subject-level split ----
        unique_subjects = np.unique(self._real_subjects)
        # Compute median z-bin per subject for stratification
        subject_zbins = []
        for subj in unique_subjects:
            mask = self._real_subjects == subj
            zbins_for_subj = self._real_zbins[mask]
            subject_zbins.append(int(np.median(zbins_for_subj)))
        subject_zbins = np.array(subject_zbins)

        # Split subjects into train-pool and test
        # Try stratified split first, fall back to random if z-bins are too sparse
        try:
            trainpool_subj_idx, test_subj_idx = train_test_split(
                np.arange(len(unique_subjects)),
                test_size=self.test_fraction,
                stratify=subject_zbins,
                random_state=self.test_seed,
            )
        except ValueError as e:
            logger.warning(
                f"Stratified test split failed ({e}). "
                "Falling back to random (non-stratified) split."
            )
            trainpool_subj_idx, test_subj_idx = train_test_split(
                np.arange(len(unique_subjects)),
                test_size=self.test_fraction,
                stratify=None,
                random_state=self.test_seed,
            )

        trainpool_subjects = unique_subjects[trainpool_subj_idx]
        test_subjects = unique_subjects[test_subj_idx]

        # Expand to sample indices
        trainpool_real_mask = np.isin(self._real_subjects, trainpool_subjects)
        test_real_mask = np.isin(self._real_subjects, test_subjects)

        self._trainpool_real_patches = self._real_patches[trainpool_real_mask]
        self._trainpool_real_zbins = self._real_zbins[trainpool_real_mask]
        self._trainpool_real_subjects = self._real_subjects[trainpool_real_mask]

        self._test_real_patches = self._real_patches[test_real_mask]
        self._test_real_zbins = self._real_zbins[test_real_mask]
        self._test_real_subjects = self._real_subjects[test_real_mask]

        # ---- Synthetic data: stratified split by z-bin ----
        if len(self._synth_patches) > 0:
            try:
                trainpool_synth_idx, test_synth_idx = train_test_split(
                    np.arange(len(self._synth_patches)),
                    test_size=self.test_fraction,
                    stratify=self._synth_zbins,
                    random_state=self.test_seed,
                )
            except ValueError:
                logger.warning(
                    "Stratified synthetic test split failed. "
                    "Falling back to random split."
                )
                trainpool_synth_idx, test_synth_idx = train_test_split(
                    np.arange(len(self._synth_patches)),
                    test_size=self.test_fraction,
                    stratify=None,
                    random_state=self.test_seed,
                )

            self._trainpool_synth_patches = self._synth_patches[trainpool_synth_idx]
            self._trainpool_synth_zbins = self._synth_zbins[trainpool_synth_idx]
            self._trainpool_synth_replica_ids = self._synth_replica_ids[trainpool_synth_idx]
            self._trainpool_synth_sample_indices = self._synth_sample_indices[trainpool_synth_idx]

            self._test_synth_patches = self._synth_patches[test_synth_idx]
            self._test_synth_zbins = self._synth_zbins[test_synth_idx]
            self._test_synth_replica_ids = self._synth_replica_ids[test_synth_idx]
            self._test_synth_sample_indices = self._synth_sample_indices[test_synth_idx]
        else:
            self._trainpool_synth_patches = np.empty((0, 2, 0, 0), dtype=np.float32)
            self._trainpool_synth_zbins = np.empty((0,), dtype=np.int32)
            self._trainpool_synth_replica_ids = np.empty((0,), dtype=np.int32)
            self._trainpool_synth_sample_indices = np.empty((0,), dtype=np.int32)

            self._test_synth_patches = np.empty((0, 2, 0, 0), dtype=np.float32)
            self._test_synth_zbins = np.empty((0,), dtype=np.int32)
            self._test_synth_replica_ids = np.empty((0,), dtype=np.int32)
            self._test_synth_sample_indices = np.empty((0,), dtype=np.int32)

        # Clear original data to save memory
        self._real_patches = None
        self._real_zbins = None
        self._real_subjects = None
        self._synth_patches = None
        self._synth_zbins = None
        self._synth_replica_ids = None
        self._synth_sample_indices = None

        logger.info(
            f"Test split: {len(self._test_real_patches)} real samples "
            f"({len(test_subjects)} subjects), "
            f"{len(self._test_synth_patches)} synthetic samples"
        )
        logger.info(
            f"Training pool: {len(self._trainpool_real_patches)} real samples "
            f"({len(trainpool_subjects)} subjects), "
            f"{len(self._trainpool_synth_patches)} synthetic samples"
        )

    def _compute_fold_splits(self) -> None:
        """Compute k-fold splits on the training pool (after test split)."""
        self._fold_splits = []

        # Real data: subject-level grouped stratified k-fold
        # Group by subject_id so slices from same patient stay in same fold
        unique_subjects = np.unique(self._trainpool_real_subjects)
        subject_zbins = []
        for subj in unique_subjects:
            mask = self._trainpool_real_subjects == subj
            # Use most common z-bin as the stratification label
            zbins_for_subj = self._trainpool_real_zbins[mask]
            subject_zbins.append(int(np.median(zbins_for_subj)))
        subject_zbins = np.array(subject_zbins)

        # Build subject-to-indices mapping
        subject_to_indices: dict[str, np.ndarray] = {}
        for subj in unique_subjects:
            subject_to_indices[str(subj)] = np.where(self._trainpool_real_subjects == subj)[0]

        # Use StratifiedKFold on subject level
        sgkf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.kfold_seed
        )
        real_fold_splits = list(sgkf.split(unique_subjects, subject_zbins))

        # Synthetic data: z-bin stratified k-fold (no grouping needed)
        if len(self._trainpool_synth_patches) > 0:
            skf_synth = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.kfold_seed
            )
            synth_fold_splits = list(skf_synth.split(
                np.arange(len(self._trainpool_synth_patches)), self._trainpool_synth_zbins
            ))
        else:
            synth_fold_splits = [(np.array([]), np.array([]))] * self.n_folds

        for fold_idx in range(self.n_folds):
            # Real: expand subject indices to sample indices
            train_subj_idx, val_subj_idx = real_fold_splits[fold_idx]
            train_subjects = unique_subjects[train_subj_idx]
            val_subjects = unique_subjects[val_subj_idx]

            real_train_idx = np.concatenate([
                subject_to_indices[str(s)] for s in train_subjects
            ])
            real_val_idx = np.concatenate([
                subject_to_indices[str(s)] for s in val_subjects
            ])

            # Synthetic
            synth_train_idx, synth_val_idx = synth_fold_splits[fold_idx]

            self._fold_splits.append({
                "real_train_idx": real_train_idx,
                "real_val_idx": real_val_idx,
                "synth_train_idx": synth_train_idx,
                "synth_val_idx": synth_val_idx,
            })

    def get_real_metadata(self) -> dict:
        """Get metadata for real test samples.

        Returns:
            Dict with 'subject_ids' array for confusion sample tracking.
        """
        return {
            "subject_ids": self._test_real_subject_ids,
        }

    def get_synth_metadata(self) -> dict:
        """Get metadata for synthetic test samples.

        Returns:
            Dict with 'replica_ids' and 'sample_indices' arrays for confusion
            sample tracking.
        """
        return {
            "replica_ids": self._test_synth_replica_ids_meta,
            "sample_indices": self._test_synth_sample_indices_meta,
        }

    def get_test_set_size(self) -> tuple[int, int]:
        """Get the number of samples in the test set.

        Returns:
            Tuple of (n_real, n_synth) samples in the test set.
        """
        if self._test_real_patches is None:
            return 0, 0
        return len(self._test_real_patches), len(self._test_synth_patches)


class ControlDataModule(pl.LightningDataModule):
    """Data module for real-vs-real control experiment.

    Splits real patches into two halves and treats one as "synthetic"
    to verify that the methodology produces AUC ~ 0.5 when there is
    no true difference.

    Args:
        cfg: Master configuration.
        patches_dir: Directory containing real_patches.npz.
        input_mode: Channel selection mode.
        repeat_idx: Repeat index for different random splits.
    """

    def __init__(
        self,
        cfg: DictConfig,
        patches_dir: Path,
        input_mode: Literal["joint", "image_only", "mask_only"] = "joint",
        repeat_idx: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.patches_dir = Path(patches_dir)
        self.input_mode = input_mode
        self.repeat_idx = repeat_idx
        self.n_folds = cfg.data.kfold.n_folds
        self.kfold_seed = cfg.data.kfold.seed + repeat_idx

        self._current_fold: int = 0
        self._train_dataset: ClassificationDataset | None = None
        self._val_dataset: ClassificationDataset | None = None
        self._patches: np.ndarray | None = None
        self._zbins: np.ndarray | None = None
        self._split_labels: np.ndarray | None = None
        self._fold_splits: list[dict] | None = None

    def set_fold(self, fold_idx: int) -> None:
        self._current_fold = fold_idx
        self._train_dataset = None
        self._val_dataset = None

    @property
    def in_channels(self) -> int:
        return 2 if self.input_mode == "joint" else 1

    @property
    def has_test_set(self) -> bool:
        """Control experiment doesn't use held-out test set."""
        return False

    def prepare_data(self) -> None:
        real_path = self.patches_dir / "real_patches.npz"
        if not real_path.exists():
            raise FileNotFoundError(f"Real patches not found at {real_path}")

    def setup(self, stage: str | None = None) -> None:
        if self._patches is None:
            self._load_and_split()
        if self._fold_splits is None:
            self._compute_fold_splits()

        fold = self._fold_splits[self._current_fold]

        # "Group A" = real (label=0), "Group B" = fake real (label=1)
        group_a_mask = self._split_labels == 0
        group_b_mask = self._split_labels == 1

        group_a_patches = self._patches[group_a_mask]
        group_b_patches = self._patches[group_b_mask]
        group_a_zbins = self._zbins[group_a_mask]
        group_b_zbins = self._zbins[group_b_mask]

        # Apply fold split
        a_train = group_a_patches[fold["a_train_idx"]]
        a_val = group_a_patches[fold["a_val_idx"]]
        b_train = group_b_patches[fold["b_train_idx"]]
        b_val = group_b_patches[fold["b_val_idx"]]

        a_zbins_train = group_a_zbins[fold["a_train_idx"]]
        a_zbins_val = group_a_zbins[fold["a_val_idx"]]
        b_zbins_train = group_b_zbins[fold["b_train_idx"]]
        b_zbins_val = group_b_zbins[fold["b_val_idx"]]

        self._train_dataset = ClassificationDataset(
            real_patches=a_train, synth_patches=b_train,
            real_zbins=a_zbins_train, synth_zbins=b_zbins_train,
            input_mode=self.input_mode,
        )
        self._val_dataset = ClassificationDataset(
            real_patches=a_val, synth_patches=b_val,
            real_zbins=a_zbins_val, synth_zbins=b_zbins_val,
            input_mode=self.input_mode,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None
        weights = self._train_dataset.get_class_weights()
        sampler = WeightedRandomSampler(weights, len(self._train_dataset), replacement=True)
        return DataLoader(
            self._train_dataset,
            batch_size=self.cfg.training.batch_size,
            sampler=sampler,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
        )

    def _load_and_split(self) -> None:
        """Load real patches and split into two random halves."""
        data = np.load(self.patches_dir / "real_patches.npz", allow_pickle=True)
        self._patches = data["patches"]
        self._zbins = data["z_bins"]

        rng = np.random.default_rng(self.kfold_seed)
        n = len(self._patches)
        indices = rng.permutation(n)
        half = n // 2

        self._split_labels = np.zeros(n, dtype=np.int32)
        self._split_labels[indices[half:]] = 1

    def _compute_fold_splits(self) -> None:
        """Compute k-fold splits for both groups."""
        group_a_mask = self._split_labels == 0
        group_b_mask = self._split_labels == 1

        group_a_zbins = self._zbins[group_a_mask]
        group_b_zbins = self._zbins[group_b_mask]

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.kfold_seed)

        a_splits = list(skf.split(np.arange(group_a_mask.sum()), group_a_zbins))
        b_splits = list(skf.split(np.arange(group_b_mask.sum()), group_b_zbins))

        self._fold_splits = []
        for fold_idx in range(self.n_folds):
            self._fold_splits.append({
                "a_train_idx": a_splits[fold_idx][0],
                "a_val_idx": a_splits[fold_idx][1],
                "b_train_idx": b_splits[fold_idx][0],
                "b_val_idx": b_splits[fold_idx][1],
            })
