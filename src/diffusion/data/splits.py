"""Patient-level split utilities for JS-DDPM.

Handles train/val/test splitting at the subject level to prevent
data leakage between splits. Supports stratified splitting based on
subject characteristics.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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


@dataclass
class SubjectCharacteristics:
    """Subject-level characteristics for stratified splitting."""

    subject_id: str
    n_slices: int = 0
    n_lesion_slices: int = 0
    lesion_slice_percentage: float = 0.0
    mean_lesion_area: float = 0.0
    z_bin_coverage: set = field(default_factory=set)
    lesion_z_bins: set = field(default_factory=set)


def compute_subject_characteristics_from_csv(
    csv_path: Path,
    subjects: Optional[list[str]] = None,
) -> dict[str, SubjectCharacteristics]:
    """Compute per-subject characteristics from a cache CSV file.

    Args:
        csv_path: Path to the CSV file (e.g., train.csv).
        subjects: Optional list of subjects to filter. If None, all subjects are used.

    Returns:
        Dictionary mapping subject_id to SubjectCharacteristics.
    """
    characteristics: dict[str, SubjectCharacteristics] = {}

    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return characteristics

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            subject_id = row["subject_id"]

            # Filter by subjects list if provided
            if subjects is not None and subject_id not in subjects:
                continue

            # Initialize if new subject
            if subject_id not in characteristics:
                characteristics[subject_id] = SubjectCharacteristics(subject_id=subject_id)

            char = characteristics[subject_id]
            char.n_slices += 1

            # Parse z_bin
            z_bin = int(row["z_bin"])
            char.z_bin_coverage.add(z_bin)

            # Parse lesion info
            has_lesion = row["has_lesion"].lower() == "true"
            if has_lesion:
                char.n_lesion_slices += 1
                char.lesion_z_bins.add(z_bin)

                # Accumulate lesion area for mean calculation
                if "lesion_area_px" in row and row["lesion_area_px"]:
                    area = float(row["lesion_area_px"])
                    # Running average: new_mean = old_mean + (value - old_mean) / n
                    char.mean_lesion_area += (area - char.mean_lesion_area) / char.n_lesion_slices

    # Compute lesion percentage for each subject
    for char in characteristics.values():
        if char.n_slices > 0:
            char.lesion_slice_percentage = (char.n_lesion_slices / char.n_slices) * 100

    logger.info(f"Computed characteristics for {len(characteristics)} subjects from {csv_path}")
    return characteristics


def create_stratification_bins(
    characteristics: dict[str, SubjectCharacteristics],
    n_bins: int = 4,
    stratify_by: list[str] = None,
    min_subjects_per_bin: int = 2,
) -> dict[str, int]:
    """Create discretized stratification labels for sklearn.

    Args:
        characteristics: Dictionary mapping subject_id to SubjectCharacteristics.
        n_bins: Number of bins for discretizing continuous features.
        stratify_by: Features to stratify on. Options: "lesion_percentage", "lesion_area".
        min_subjects_per_bin: Minimum subjects per bin before merging.

    Returns:
        Dictionary mapping subject_id to composite stratification label.
    """
    if stratify_by is None:
        stratify_by = ["lesion_percentage"]

    if not characteristics:
        return {}

    subjects = list(characteristics.keys())
    n_subjects = len(subjects)

    # Start with all subjects in bin 0
    labels = {s: 0 for s in subjects}

    for feature in stratify_by:
        # Extract values for this feature
        if feature == "lesion_percentage":
            values = [characteristics[s].lesion_slice_percentage for s in subjects]
        elif feature == "lesion_area":
            values = [characteristics[s].mean_lesion_area for s in subjects]
        else:
            logger.warning(f"Unknown stratification feature: {feature}")
            continue

        # Handle edge case: all values are the same
        if len(set(values)) <= 1:
            logger.info(f"All subjects have same {feature}, skipping stratification")
            continue

        # Create bins using percentiles
        values_array = np.array(values)

        # Compute percentile thresholds
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        thresholds = np.percentile(values_array, percentiles)

        # Assign bin indices
        bin_indices = np.digitize(values_array, thresholds)

        # Update composite labels
        current_max_label = max(labels.values()) + 1
        for i, s in enumerate(subjects):
            labels[s] = labels[s] * n_bins + bin_indices[i]

    # Check for bins with too few subjects and merge if needed
    label_counts = {}
    for label in labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1

    # If any bin has too few subjects, simplify to binary (has_lesion vs no_lesion)
    min_count = min(label_counts.values()) if label_counts else 0
    if min_count < min_subjects_per_bin:
        logger.warning(
            f"Some stratification bins have <{min_subjects_per_bin} subjects. "
            "Falling back to binary lesion stratification."
        )
        labels = {
            s: 1 if characteristics[s].n_lesion_slices > 0 else 0
            for s in subjects
        }

    return labels


def stratified_split_subjects(
    subjects: list[str],
    characteristics: dict[str, SubjectCharacteristics],
    val_fraction: float,
    test_fraction: float = 0.0,
    seed: int = 42,
    stratify_by: list[str] = None,
    n_stratification_bins: int = 4,
    min_subjects_per_bin: int = 2,
) -> tuple[list[str], list[str], list[str]]:
    """Split subjects using sklearn's train_test_split with stratification.

    Args:
        subjects: List of subject IDs to split.
        characteristics: Dictionary of subject characteristics.
        val_fraction: Fraction for validation set.
        test_fraction: Fraction for test set (if not using predefined).
        seed: Random seed for reproducibility.
        stratify_by: Features to stratify on.
        n_stratification_bins: Number of bins for discretization.
        min_subjects_per_bin: Minimum subjects per bin.

    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects).
    """
    if stratify_by is None:
        stratify_by = ["lesion_percentage"]

    n = len(subjects)
    if n == 0:
        return [], [], []

    # Filter characteristics to only include subjects in our list
    filtered_chars = {s: characteristics[s] for s in subjects if s in characteristics}

    # If no characteristics available, fall back to random split
    if not filtered_chars:
        logger.warning("No subject characteristics available, falling back to random split")
        return split_subjects(subjects, val_fraction, test_fraction, seed)

    # Create stratification bins
    strat_labels = create_stratification_bins(
        filtered_chars,
        n_bins=n_stratification_bins,
        stratify_by=stratify_by,
        min_subjects_per_bin=min_subjects_per_bin,
    )

    # Get labels in same order as subjects
    labels = [strat_labels.get(s, 0) for s in subjects]

    try:
        from sklearn.model_selection import train_test_split

        if test_fraction > 0:
            # First split: separate test set
            train_val, test_subjects, train_val_labels, _ = train_test_split(
                subjects,
                labels,
                test_size=test_fraction,
                stratify=labels,
                random_state=seed,
            )
            # Recompute labels for train_val
            labels_tv = [strat_labels.get(s, 0) for s in train_val]
        else:
            train_val = subjects
            test_subjects = []
            labels_tv = labels

        # Second split: separate validation set
        if val_fraction > 0 and len(train_val) > 1:
            # Adjust val_fraction since we already removed test
            val_adjusted = val_fraction / (1 - test_fraction) if test_fraction < 1 else val_fraction

            train_subjects, val_subjects, _, _ = train_test_split(
                train_val,
                labels_tv,
                test_size=val_adjusted,
                stratify=labels_tv,
                random_state=seed,
            )
        else:
            train_subjects = train_val
            val_subjects = []

        logger.info(
            f"Stratified split: train={len(train_subjects)}, "
            f"val={len(val_subjects)}, test={len(test_subjects)}"
        )
        return train_subjects, val_subjects, test_subjects

    except (ValueError, ImportError) as e:
        logger.warning(f"Stratified split failed: {e}. Falling back to random split.")
        return split_subjects(subjects, val_fraction, test_fraction, seed)


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
