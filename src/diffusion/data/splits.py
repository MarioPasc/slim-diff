"""Patient-level split utilities for JS-DDPM.

Handles train/val/test splitting at the subject level to prevent
data leakage between splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from src.diffusion.utils.io import (
    discover_subjects,
    get_image_path,
    get_label_path,
    parse_subject_prefix,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Container for dataset split information."""

    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]

    @property
    def n_train(self) -> int:
        return len(self.train_subjects)

    @property
    def n_val(self) -> int:
        return len(self.val_subjects)

    @property
    def n_test(self) -> int:
        return len(self.test_subjects)

    def __repr__(self) -> str:
        return (
            f"DatasetSplit(train={self.n_train}, "
            f"val={self.n_val}, test={self.n_test})"
        )


def shuffle_subjects(subjects: list[str], seed: int) -> list[str]:
    """Shuffle a list of subjects deterministically.

    Args:
        subjects: List of subject IDs.
        seed: Random seed.

    Returns:
        Shuffled list.
    """
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    return shuffled


def split_subjects(
    subjects: list[str],
    val_fraction: float,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split subjects into train/val/test sets.

    Args:
        subjects: List of subject IDs.
        val_fraction: Fraction for validation set.
        test_fraction: Fraction for test set (if not using predefined).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects).
    """
    n = len(subjects)
    if n == 0:
        return [], [], []

    shuffled = shuffle_subjects(subjects, seed)

    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)

    test_subjects = shuffled[:n_test]
    val_subjects = shuffled[n_test : n_test + n_val]
    train_subjects = shuffled[n_test + n_val :]

    return train_subjects, val_subjects, test_subjects


def create_epilepsy_splits(
    cfg: DictConfig,
) -> DatasetSplit:
    """Create splits for the epilepsy dataset.

    Uses predefined test set (imagesTs) if configured, otherwise
    splits from training pool.

    Args:
        cfg: Configuration with data paths and split parameters.

    Returns:
        DatasetSplit for epilepsy subjects.
    """
    data_cfg = cfg.data
    dataset_path = Path(data_cfg.root_dir) / data_cfg.epilepsy.name
    prefix = parse_subject_prefix(data_cfg.epilepsy.name)
    modality_index = data_cfg.epilepsy.modality_index

    # Discover training subjects
    train_pool = discover_subjects(
        dataset_path,
        image_dir="imagesTr",
        prefix=prefix,
        modality_index=modality_index,
    )

    # Handle test set
    if data_cfg.splits.use_predefined_test:
        # Use predefined test subjects from imagesTs
        test_subjects = discover_subjects(
            dataset_path,
            image_dir="imagesTs",
            prefix=prefix,
            modality_index=modality_index,
        )
        # Split remaining into train/val
        train_subjects, val_subjects, _ = split_subjects(
            train_pool,
            val_fraction=data_cfg.splits.val_fraction,
            test_fraction=0.0,
            seed=data_cfg.splits.seed,
        )
    else:
        # Split everything from train pool
        train_subjects, val_subjects, test_subjects = split_subjects(
            train_pool,
            val_fraction=data_cfg.splits.val_fraction,
            test_fraction=0.15,  # Default test fraction
            seed=data_cfg.splits.seed,
        )

    split = DatasetSplit(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )

    logger.info(f"Epilepsy splits: {split}")
    return split


def create_control_splits(
    cfg: DictConfig,
    target_test_size: int | None = None,
) -> DatasetSplit:
    """Create splits for the control dataset.

    Control dataset doesn't have predefined test set, so we create
    one from the training pool to match epilepsy test size.

    Args:
        cfg: Configuration with data paths and split parameters.
        target_test_size: Number of test subjects to match (from epilepsy).

    Returns:
        DatasetSplit for control subjects.
    """
    data_cfg = cfg.data
    dataset_path = Path(data_cfg.root_dir) / data_cfg.control.name
    prefix = parse_subject_prefix(data_cfg.control.name)
    modality_index = data_cfg.control.modality_index

    # Discover all control subjects
    all_subjects = discover_subjects(
        dataset_path,
        image_dir="imagesTr",
        prefix=prefix,
        modality_index=modality_index,
    )

    n = len(all_subjects)
    if n == 0:
        logger.warning("No control subjects found")
        return DatasetSplit([], [], [])

    # Calculate test fraction
    if target_test_size is not None:
        test_fraction = min(target_test_size / n, data_cfg.splits.control_test_fraction)
    else:
        test_fraction = data_cfg.splits.control_test_fraction

    # Split subjects
    train_subjects, val_subjects, test_subjects = split_subjects(
        all_subjects,
        val_fraction=data_cfg.splits.val_fraction,
        test_fraction=test_fraction,
        seed=data_cfg.splits.seed,
    )

    split = DatasetSplit(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )

    logger.info(f"Control splits: {split}")
    return split


def create_all_splits(cfg: DictConfig) -> dict[str, DatasetSplit]:
    """Create splits for both epilepsy and control datasets.

    Args:
        cfg: Configuration.

    Returns:
        Dictionary with 'epilepsy' and 'control' splits.
    """
    epilepsy_split = create_epilepsy_splits(cfg)
    control_split = create_control_splits(
        cfg,
        target_test_size=epilepsy_split.n_test,
    )

    return {
        "epilepsy": epilepsy_split,
        "control": control_split,
    }


@dataclass
class SubjectInfo:
    """Information about a single subject."""

    subject_id: str
    image_path: Path
    label_path: Path | None  # None for control subjects
    source: str  # "epilepsy" or "control"
    split: str  # "train", "val", or "test"


def get_subject_paths(
    subject_id: str,
    cfg: DictConfig,
    source: str,
    split: str,
) -> SubjectInfo:
    """Get file paths for a subject.

    Args:
        subject_id: Subject ID.
        cfg: Configuration.
        source: "epilepsy" or "control".
        split: "train", "val", or "test".

    Returns:
        SubjectInfo with paths.
    """
    data_cfg = cfg.data

    if source == "epilepsy":
        dataset_path = Path(data_cfg.root_dir) / data_cfg.epilepsy.name
        modality_index = data_cfg.epilepsy.modality_index

        # Determine image directory based on split
        if split == "test" and data_cfg.splits.use_predefined_test:
            image_dir = "imagesTs"
            label_dir = "labelsTs"
        else:
            image_dir = "imagesTr"
            label_dir = "labelsTr"

        image_path = get_image_path(
            dataset_path, subject_id, modality_index, image_dir
        )
        label_path = get_label_path(dataset_path, subject_id, label_dir)

    else:  # control
        dataset_path = Path(data_cfg.root_dir) / data_cfg.control.name
        modality_index = data_cfg.control.modality_index

        image_path = get_image_path(
            dataset_path, subject_id, modality_index, "imagesTr"
        )
        label_path = None  # Control subjects have no labels

    return SubjectInfo(
        subject_id=subject_id,
        image_path=image_path,
        label_path=label_path,
        source=source,
        split=split,
    )


def get_all_subject_infos(
    cfg: DictConfig,
    splits: dict[str, DatasetSplit] | None = None,
) -> dict[str, list[SubjectInfo]]:
    """Get SubjectInfo for all subjects organized by split.

    Args:
        cfg: Configuration.
        splits: Pre-computed splits or None to compute.

    Returns:
        Dictionary mapping split name to list of SubjectInfo.
    """
    if splits is None:
        splits = create_all_splits(cfg)

    result: dict[str, list[SubjectInfo]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Process epilepsy subjects
    epi_split = splits["epilepsy"]
    for split_name, subjects in [
        ("train", epi_split.train_subjects),
        ("val", epi_split.val_subjects),
        ("test", epi_split.test_subjects),
    ]:
        for subject_id in subjects:
            info = get_subject_paths(subject_id, cfg, "epilepsy", split_name)
            result[split_name].append(info)

    # Process control subjects
    ctl_split = splits["control"]
    for split_name, subjects in [
        ("train", ctl_split.train_subjects),
        ("val", ctl_split.val_subjects),
        ("test", ctl_split.test_subjects),
    ]:
        for subject_id in subjects:
            info = get_subject_paths(subject_id, cfg, "control", split_name)
            result[split_name].append(info)

    for split_name, infos in result.items():
        logger.info(f"Split '{split_name}': {len(infos)} subjects")

    return result
