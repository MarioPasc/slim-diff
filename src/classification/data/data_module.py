"""K-fold cross-validation data module for classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.classification.data.dataset import ClassificationDataset
from src.classification.data.patch_extractor import PatchExtractor

logger = logging.getLogger(__name__)


class KFoldClassificationDataModule(pl.LightningDataModule):
    """Lightning DataModule with k-fold cross-validation support.

    Uses subject-level splits for real data (prevents same-patient slices
    in both train and val) and z-bin-stratified splits for synthetic data.

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
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.experiment_name = experiment_name
        self.input_mode = input_mode
        self.n_folds = cfg.data.kfold.n_folds
        self.kfold_seed = cfg.data.kfold.seed

        # Determine patches directory
        if patches_dir is not None:
            self.patches_dir = Path(patches_dir)
        else:
            base = Path(cfg.output.base_dir) / cfg.output.patches_subdir
            self.patches_dir = base / experiment_name

        self._current_fold: int = 0
        self._train_dataset: ClassificationDataset | None = None
        self._val_dataset: ClassificationDataset | None = None

        # Loaded data
        self._real_patches: np.ndarray | None = None
        self._real_zbins: np.ndarray | None = None
        self._real_subjects: np.ndarray | None = None
        self._synth_patches: np.ndarray | None = None
        self._synth_zbins: np.ndarray | None = None

        # Fold splits (computed once)
        self._fold_splits: list[dict] | None = None

    @property
    def current_fold(self) -> int:
        return self._current_fold

    @property
    def in_channels(self) -> int:
        if self.input_mode == "joint":
            return 2
        return 1

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
        """Load patches and create fold-specific datasets."""
        if self._real_patches is None:
            self._load_patches()

        if self._fold_splits is None:
            self._compute_fold_splits()

        fold = self._fold_splits[self._current_fold]

        # Build train/val datasets for this fold
        real_train = self._real_patches[fold["real_train_idx"]]
        real_val = self._real_patches[fold["real_val_idx"]]
        real_zbins_train = self._real_zbins[fold["real_train_idx"]]
        real_zbins_val = self._real_zbins[fold["real_val_idx"]]

        synth_train = self._synth_patches[fold["synth_train_idx"]]
        synth_val = self._synth_patches[fold["synth_val_idx"]]
        synth_zbins_train = self._synth_zbins[fold["synth_train_idx"]]
        synth_zbins_val = self._synth_zbins[fold["synth_val_idx"]]

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

        logger.info(
            f"Fold {self._current_fold}: "
            f"train={len(self._train_dataset)} "
            f"(real={len(real_train)}, synth={len(synth_train)}), "
            f"val={len(self._val_dataset)} "
            f"(real={len(real_val)}, synth={len(synth_val)})"
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None, "Call setup() first."
        weights = self._train_dataset.get_class_weights()
        sampler = WeightedRandomSampler(
            weights, num_samples=len(self._train_dataset), replacement=True
        )
        return DataLoader(
            self._train_dataset,
            batch_size=self.cfg.training.batch_size,
            sampler=sampler,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, "Call setup() first."
        return DataLoader(
            self._val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
        )

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _load_patches(self) -> None:
        """Load pre-extracted patches from disk."""
        real_path = self.patches_dir / "real_patches.npz"
        synth_path = self.patches_dir / "synthetic_patches.npz"

        real_data = np.load(real_path, allow_pickle=True)
        self._real_patches = real_data["patches"]
        self._real_zbins = real_data["z_bins"]
        self._real_subjects = real_data["subject_ids"]

        if synth_path.exists():
            synth_data = np.load(synth_path, allow_pickle=True)
            self._synth_patches = synth_data["patches"]
            self._synth_zbins = synth_data["z_bins"]
        else:
            # Control experiment: no synthetic data
            self._synth_patches = np.empty((0, 2, 0, 0), dtype=np.float32)
            self._synth_zbins = np.empty((0,), dtype=np.int32)

        logger.info(
            f"Loaded patches: {len(self._real_patches)} real, "
            f"{len(self._synth_patches)} synthetic"
        )

    def _compute_fold_splits(self) -> None:
        """Compute k-fold splits with subject-level grouping for real data."""
        self._fold_splits = []

        # Real data: subject-level grouped stratified k-fold
        # Group by subject_id so slices from same patient stay in same fold
        unique_subjects = np.unique(self._real_subjects)
        subject_zbins = []
        for subj in unique_subjects:
            mask = self._real_subjects == subj
            # Use most common z-bin as the stratification label
            zbins_for_subj = self._real_zbins[mask]
            subject_zbins.append(int(np.median(zbins_for_subj)))
        subject_zbins = np.array(subject_zbins)

        # Build subject-to-indices mapping
        subject_to_indices: dict[str, np.ndarray] = {}
        for subj in unique_subjects:
            subject_to_indices[str(subj)] = np.where(self._real_subjects == subj)[0]

        # Use StratifiedGroupKFold on subject level
        sgkf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.kfold_seed
        )
        real_fold_splits = list(sgkf.split(unique_subjects, subject_zbins))

        # Synthetic data: z-bin stratified k-fold (no grouping needed)
        if len(self._synth_patches) > 0:
            skf_synth = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.kfold_seed
            )
            synth_fold_splits = list(skf_synth.split(
                np.arange(len(self._synth_patches)), self._synth_zbins
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
