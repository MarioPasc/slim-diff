"""K-fold planner for segmentation with real + synthetic data support.

Handles five scenarios:
1. Real data only (synthetic.enabled: false)
2. Real + Synthetic with concat strategy
3. Real + Synthetic with balance strategy
4. Synthetic data only (real.enabled: false)
5. Synthetic for training, real for val/test (synthetic.training_only: true)

The planner generates a unified CSV with columns:
- fold, split, subject_id, image_path, mask_path, has_lesion_slice, has_lesion_subject,
  source (real/synthetic), zbin
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class SampleRecord:
    """A single sample record for the fold CSV."""

    subject_id: str
    filepath: str  # Relative path to NPZ file
    z_index: int
    z_bin: int
    has_lesion: bool
    source: Literal["real", "synthetic"]
    has_lesion_subject: bool = False
    replica_name: str | None = None  # For synthetic samples

    def to_dict(self, fold: int, split: str) -> dict:
        """Convert to dict for CSV writing."""
        return {
            "fold": fold,
            "split": split,
            "subject_id": self.subject_id,
            "filepath": self.filepath,
            "z_index": self.z_index,
            "z_bin": self.z_bin,
            "has_lesion_slice": self.has_lesion,
            "has_lesion_subject": self.has_lesion_subject,
            "source": self.source,
            "replica": self.replica_name or "",
        }


@dataclass
class SubjectInfo:
    """Subject-level information aggregating all slices."""

    subject_id: str
    has_lesion: bool = False
    samples: list[SampleRecord] = field(default_factory=list)

    def add_sample(self, sample: SampleRecord):
        """Add a sample and update has_lesion status."""
        self.samples.append(sample)
        if sample.has_lesion:
            self.has_lesion = True
        sample.has_lesion_subject = self.has_lesion

    def update_has_lesion_subject(self):
        """Update has_lesion_subject for all samples after loading."""
        for sample in self.samples:
            sample.has_lesion_subject = self.has_lesion


class KFoldPlanner:
    """Plan k-fold splits with support for real and synthetic data.

    Usage:
        planner = KFoldPlanner(cfg)
        planner.plan()  # Generates fold CSVs

        # Access fold data:
        train_samples, val_samples = planner.get_fold(0)
    """

    def __init__(self, cfg: DictConfig):
        """Initialize planner from config.

        Args:
            cfg: Configuration with data.real, data.synthetic, k_fold sections
        """
        self.cfg = cfg

        # Data configuration
        self.real_enabled = cfg.data.real.enabled
        self.synthetic_enabled = cfg.data.synthetic.enabled

        # Check for training_only mode: synthetic for training, real for val/test
        self.synthetic_training_only = cfg.data.synthetic.get("training_only", False) if self.synthetic_enabled else False

        # Paths
        self.real_cache_dir = Path(cfg.data.real.cache_dir) if self.real_enabled or self.synthetic_training_only else None
        self.synthetic_dir = Path(cfg.data.synthetic.samples_dir) if self.synthetic_enabled else None

        # Synthetic configuration
        self.replicas = cfg.data.synthetic.get("replicas", []) if self.synthetic_enabled else []
        self.merging_strategy = cfg.data.synthetic.get("merging_strategy", "concat")

        # K-fold configuration
        self.n_folds = cfg.k_fold.n_folds
        self.exclude_test = cfg.k_fold.exclude_test
        self.stratify_by = cfg.k_fold.stratify_by
        self.seed = cfg.k_fold.seed

        # Negative case filtering
        self.use_negative_cases = cfg.data.get("use_negative_cases", True)

        # Augmentation multiplier: adds N augmented copies of each training sample
        # multiplier=0: no duplication (default, augmentation applied probabilistically)
        # multiplier=2: adds 2 augmented copies per sample (total = base * 3)
        aug_cfg = cfg.get("augmentation", {})
        self.augmentation_multiplier = aug_cfg.get("multiplier", 0) if aug_cfg else 0

        # Output directory
        self.output_dir = Path(cfg.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.real_subjects: dict[str, SubjectInfo] = {}
        self.test_samples: list[SampleRecord] = []  # Test set samples
        self.synthetic_samples: list[SampleRecord] = []
        self.folds: list[tuple[list[str], list[str]]] = []

        # Load data
        self._load_data()

        # Create folds if real data is available (for validation)
        if (self.real_enabled or self.synthetic_training_only) and self.real_subjects:
            self._create_folds()

        self._log_summary()

    def _load_data(self):
        """Load both real and synthetic data."""
        # Load real data if enabled OR if using synthetic_training_only mode
        if self.real_enabled or self.synthetic_training_only:
            self._load_real_data()

        if self.synthetic_enabled and self.replicas:
            self._load_synthetic_data()

    def _load_real_data(self):
        """Load real data from train.csv, val.csv, and test.csv."""
        csv_files = ["train.csv", "val.csv"]

        # Load test samples first
        test_csv = self.real_cache_dir / "test.csv"
        test_subjects_dict: dict[str, SubjectInfo] = {}

        if test_csv.exists():
            with open(test_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    subject_id = row["subject_id"]

                    # Create sample record
                    sample = SampleRecord(
                        subject_id=subject_id,
                        filepath=row["filepath"],
                        z_index=int(row["z_index"]),
                        z_bin=int(row["z_bin"]),
                        has_lesion=row["has_lesion"].lower() == "true",
                        source="real",
                    )

                    # Skip negative cases if use_negative_cases is False
                    if not self.use_negative_cases and not sample.has_lesion:
                        continue

                    # Add to test subjects dict
                    if subject_id not in test_subjects_dict:
                        test_subjects_dict[subject_id] = SubjectInfo(subject_id=subject_id)
                    test_subjects_dict[subject_id].add_sample(sample)

            # Update has_lesion_subject for test samples
            for subject in test_subjects_dict.values():
                subject.update_has_lesion_subject()

            # Flatten to test_samples list
            for subject in test_subjects_dict.values():
                self.test_samples.extend(subject.samples)

            logger.info(
                f"Loaded test data: {len(test_subjects_dict)} subjects, "
                f"{len(self.test_samples)} samples"
            )

        # Get test subject IDs for exclusion
        test_subjects = set(test_subjects_dict.keys())
        if self.exclude_test and test_subjects:
            logger.info(f"Excluding {len(test_subjects)} test subjects from k-fold")

        # Load train and val data
        for csv_name in csv_files:
            csv_path = self.real_cache_dir / csv_name
            if not csv_path.exists():
                logger.warning(f"CSV not found: {csv_path}")
                continue

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    subject_id = row["subject_id"]

                    # Skip test subjects if exclude_test is enabled
                    if self.exclude_test and subject_id in test_subjects:
                        continue

                    # Create sample record
                    sample = SampleRecord(
                        subject_id=subject_id,
                        filepath=row["filepath"],
                        z_index=int(row["z_index"]),
                        z_bin=int(row["z_bin"]),
                        has_lesion=row["has_lesion"].lower() == "true",
                        source="real",
                    )

                    # Skip negative cases if use_negative_cases is False
                    if not self.use_negative_cases and not sample.has_lesion:
                        continue

                    # Add to subject
                    if subject_id not in self.real_subjects:
                        self.real_subjects[subject_id] = SubjectInfo(subject_id=subject_id)

                    self.real_subjects[subject_id].add_sample(sample)

        # Update has_lesion_subject for all samples
        for subject in self.real_subjects.values():
            subject.update_has_lesion_subject()

        total_samples = sum(len(s.samples) for s in self.real_subjects.values())
        lesion_subjects = sum(1 for s in self.real_subjects.values() if s.has_lesion)

        logger.info(
            f"Loaded real train/val data: {len(self.real_subjects)} subjects, "
            f"{total_samples} samples, {lesion_subjects} subjects with lesions"
        )

    def _load_synthetic_data(self):
        """Load synthetic data from NPZ replicas."""
        for replica_name in self.replicas:
            replica_path = self.synthetic_dir / replica_name

            if not replica_path.exists():
                logger.warning(f"Replica not found: {replica_path}")
                continue

            self._load_single_replica(replica_path, replica_name)

        # Organize synthetic samples by (zbin, has_lesion) for balancing
        self.synthetic_by_zbin_lesion = defaultdict(list)
        for sample in self.synthetic_samples:
            key = (sample.z_bin, sample.has_lesion)
            self.synthetic_by_zbin_lesion[key].append(sample)

        logger.info(f"Loaded synthetic data: {len(self.synthetic_samples)} samples from {len(self.replicas)} replicas")

    def _load_single_replica(self, replica_path: Path, replica_name: str):
        """Load samples from a single replica NPZ file.

        Args:
            replica_path: Path to the replica NPZ file
            replica_name: Name of the replica file
            
        Note:
            We determine has_lesion from actual mask content, NOT from the 
            conditioning label (lesion_present). The conditioning label indicates
            what was requested during generation, but ~20% of "lesion" samples
            have empty masks (DDPM generation failure). Using actual mask content
            ensures correct balancing and prevents teaching the model incorrect
            image-mask pairs.
        """
        data = np.load(replica_path, allow_pickle=True)

        n_samples = data["images"].shape[0]
        zbins = data["zbin"]
        masks = data["masks"]
        lesion_labels = data["lesion_present"]  # Keep for logging/debugging

        # Extract replica base name without extension
        replica_base = replica_name.replace(".npz", "")
        
        # Track label vs actual mismatch for logging
        n_false_positives = 0  # label=lesion, mask=empty
        n_false_negatives = 0  # label=no-lesion, mask=has-content

        for idx in range(n_samples):
            zbin = int(zbins[idx])
            
            # Determine has_lesion from ACTUAL mask content, not conditioning label
            # Threshold: at least 10 pixels above 0.0 to count as lesion
            mask = masks[idx]
            has_lesion_actual = bool((mask > 0.0).sum() > 10)
            has_lesion_label = bool(lesion_labels[idx])
            
            # Track mismatches
            if has_lesion_label and not has_lesion_actual:
                n_false_positives += 1
            elif not has_lesion_label and has_lesion_actual:
                n_false_negatives += 1

            # Create synthetic subject ID: synth_<replica>_<index>
            subject_id = f"synth_{replica_base}_{idx:05d}"

            # Filepath points to the replica with index info
            # For synthetic, we'll encode replica + index in filepath
            filepath = f"{replica_name}:{idx}"

            sample = SampleRecord(
                subject_id=subject_id,
                filepath=filepath,
                z_index=zbin,  # Use zbin as z_index for synthetic
                z_bin=zbin,
                has_lesion=has_lesion_actual,  # Use ACTUAL mask content
                source="synthetic",
                has_lesion_subject=has_lesion_actual,  # Each synthetic sample is its own "subject"
                replica_name=replica_name,
            )

            # Skip negative cases if use_negative_cases is False
            if not self.use_negative_cases and not has_lesion_actual:
                continue

            self.synthetic_samples.append(sample)
        
        # Log mismatch statistics
        if n_false_positives > 0 or n_false_negatives > 0:
            logger.warning(
                f"Replica {replica_name}: Label-mask mismatch detected. "
                f"False positives (label=lesion, mask=empty): {n_false_positives}, "
                f"False negatives (label=no-lesion, mask=has-content): {n_false_negatives}. "
                f"Using actual mask content for has_lesion flag."
            )

    def _create_folds(self):
        """Create stratified k-fold splits at subject level."""
        subject_ids = sorted(self.real_subjects.keys())

        if self.stratify_by == "has_lesion_subject":
            strat_labels = np.array([
                int(self.real_subjects[sid].has_lesion)
                for sid in subject_ids
            ])
        else:
            strat_labels = np.zeros(len(subject_ids), dtype=int)

        kfold = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.seed,
        )

        self.folds = []
        for train_idx, val_idx in kfold.split(subject_ids, strat_labels):
            train_subjects = [subject_ids[i] for i in train_idx]
            val_subjects = [subject_ids[i] for i in val_idx]
            self.folds.append((train_subjects, val_subjects))

    def _log_summary(self):
        """Log summary of loaded data and configuration."""
        mode = self._get_mode()
        logger.info(f"KFoldPlanner mode: {mode}")
        logger.info(f"  - n_folds: {self.n_folds}")
        logger.info(f"  - real_enabled: {self.real_enabled}")
        logger.info(f"  - synthetic_enabled: {self.synthetic_enabled}")
        logger.info(f"  - use_negative_cases: {self.use_negative_cases}")
        if self.synthetic_enabled:
            logger.info(f"  - synthetic_training_only: {self.synthetic_training_only}")
            logger.info(f"  - replicas: {self.replicas}")
            if not self.synthetic_training_only:
                logger.info(f"  - merging_strategy: {self.merging_strategy}")
        if self.augmentation_multiplier > 0:
            logger.info(f"  - augmentation_multiplier: {self.augmentation_multiplier}")

    def _get_mode(self) -> str:
        """Get the current mode string."""
        if self.synthetic_training_only:
            return "synthetic_training_only"
        elif self.real_enabled and not self.synthetic_enabled:
            return "real_only"
        elif not self.real_enabled and self.synthetic_enabled:
            return "synthetic_only"
        elif self.real_enabled and self.synthetic_enabled:
            return f"real_plus_synthetic_{self.merging_strategy}"
        else:
            return "no_data"

    def get_fold(
        self, fold_idx: int
    ) -> tuple[list[SampleRecord], list[SampleRecord]]:
        """Get train/val samples for a specific fold.

        Args:
            fold_idx: Fold index (0 to n_folds-1)

        Returns:
            (train_samples, val_samples) tuple
        """
        if not self.real_enabled and not self.synthetic_enabled:
            raise ValueError("No data enabled")

        # Get real samples for fold
        train_real, val_real = self._get_fold_real_samples(fold_idx)

        # Add synthetic samples based on mode
        if not self.synthetic_enabled:
            # Mode 1: Real only
            train_samples = train_real
        elif self.synthetic_training_only:
            # Mode 5: Synthetic for training, real for validation
            train_samples = self.synthetic_samples.copy()
        elif not self.real_enabled:
            # Mode 4: Synthetic only
            train_samples, val_real = self._get_synthetic_only_split(fold_idx)
        else:
            # Mode 2 or 3: Real + Synthetic
            if self.merging_strategy == "concat":
                train_samples = self._merge_concat(train_real)
            elif self.merging_strategy == "balance":
                if self.use_negative_cases:
                    train_samples = self._merge_balance(train_real)
                else:
                    train_samples = self._merge_balance_lesions_only(train_real)
            else:
                raise ValueError(f"Unknown merging_strategy: {self.merging_strategy}")

        # Apply augmentation multiplier if configured
        if self.augmentation_multiplier > 0:
            train_samples = self._apply_augmentation_multiplier(train_samples)

        # Validation always uses only real data
        return train_samples, val_real

    def _apply_augmentation_multiplier(
        self, train_samples: list[SampleRecord]
    ) -> list[SampleRecord]:
        """Duplicate training samples based on augmentation multiplier.

        Creates N additional copies of each training sample, where N is the
        augmentation multiplier. Each copy receives a unique subject_id suffix
        (_aug0, _aug1, etc.) and will receive different random augmentations
        during training due to the probabilistic nature of transforms.

        Formula: total = base_samples * (1 + multiplier)
        - multiplier=1: doubles the dataset
        - multiplier=2: triples the dataset

        Args:
            train_samples: Original training samples

        Returns:
            Training samples with augmented copies appended
        """
        n_base = len(train_samples)
        augmented_samples = []

        for copy_idx in range(self.augmentation_multiplier):
            for sample in train_samples:
                # Create a copy with modified subject_id
                aug_sample = SampleRecord(
                    subject_id=f"{sample.subject_id}_aug{copy_idx}",
                    filepath=sample.filepath,
                    z_index=sample.z_index,
                    z_bin=sample.z_bin,
                    has_lesion=sample.has_lesion,
                    source=sample.source,
                    has_lesion_subject=sample.has_lesion_subject,
                    replica_name=sample.replica_name,
                )
                augmented_samples.append(aug_sample)

        n_augmented = len(augmented_samples)
        total = n_base + n_augmented

        logger.info(
            f"Augmentation multiplier ({self.augmentation_multiplier}): "
            f"{n_base} base + {n_augmented} augmented = {total} total training samples"
        )

        return train_samples + augmented_samples

    def _get_fold_real_samples(
        self, fold_idx: int
    ) -> tuple[list[SampleRecord], list[SampleRecord]]:
        """Get real samples for a fold.

        Args:
            fold_idx: Fold index

        Returns:
            (train_samples, val_samples) tuple
        """
        if not self.real_enabled or not self.folds:
            return [], []

        train_subjects, val_subjects = self.folds[fold_idx]

        train_samples = []
        for subject_id in train_subjects:
            train_samples.extend(self.real_subjects[subject_id].samples)

        val_samples = []
        for subject_id in val_subjects:
            val_samples.extend(self.real_subjects[subject_id].samples)

        return train_samples, val_samples

    def _get_synthetic_only_split(
        self, fold_idx: int
    ) -> tuple[list[SampleRecord], list[SampleRecord]]:
        """Get synthetic-only train/val split.

        For synthetic only, we do a simple stratified split on samples.

        Args:
            fold_idx: Fold index (used for reproducibility)

        Returns:
            (train_samples, val_samples) tuple
        """
        # Use fold_idx for reproducibility
        rng = np.random.RandomState(self.seed + fold_idx)

        # Group by (zbin, has_lesion) for stratified split
        all_samples = self.synthetic_samples.copy()

        # Shuffle samples
        rng.shuffle(all_samples)

        # 80/20 split per (zbin, has_lesion) group
        train_samples = []
        val_samples = []

        samples_by_group = defaultdict(list)
        for sample in all_samples:
            key = (sample.z_bin, sample.has_lesion)
            samples_by_group[key].append(sample)

        for key, samples in samples_by_group.items():
            n_val = max(1, int(len(samples) * 0.2))
            val_samples.extend(samples[:n_val])
            train_samples.extend(samples[n_val:])

        return train_samples, val_samples

    def _get_target_synthetic_count(self, n_real: int) -> int:
        """Calculate target synthetic sample count based on replica multiplier.

        When use_negative_cases=False, the number of synthetic samples should be
        proportional to the filtered real data count. Each replica acts as a
        multiplier: n_synthetic = n_real * n_replicas.

        Args:
            n_real: Number of filtered real training samples

        Returns:
            Target number of synthetic samples to add
        """
        return n_real * len(self.replicas)

    def _sample_synthetic_balanced(
        self, target_count: int, seed: int = 42
    ) -> list[SampleRecord]:
        """Sample synthetic samples with balanced z-bin distribution.

        Samples synthetic samples to reach target_count, distributing evenly
        across z-bins to maintain coverage of the z-axis.

        Args:
            target_count: Target number of synthetic samples
            seed: Random seed for reproducibility

        Returns:
            List of sampled synthetic samples
        """
        if target_count <= 0 or not self.synthetic_samples:
            return []

        if target_count >= len(self.synthetic_samples):
            return self.synthetic_samples.copy()

        rng = np.random.RandomState(seed)

        # Get all z-bins
        zbins = sorted(set(s.z_bin for s in self.synthetic_samples))
        n_zbins = len(zbins)

        if n_zbins == 0:
            return []

        # Calculate samples per z-bin
        base_per_zbin = target_count // n_zbins
        extra = target_count % n_zbins

        sampled = []
        used_ids = set()

        # Sample from each z-bin
        for i, zbin in enumerate(zbins):
            # Allocate extra samples to first z-bins
            n_for_zbin = base_per_zbin + (1 if i < extra else 0)

            # Get available samples for this z-bin
            available = [s for s in self.synthetic_by_zbin_lesion.get((zbin, True), [])
                        if id(s) not in used_ids]

            # Shuffle and take up to n_for_zbin
            rng.shuffle(available)
            take = min(n_for_zbin, len(available))

            for s in available[:take]:
                sampled.append(s)
                used_ids.add(id(s))

        # If we still need more samples, take from any remaining
        if len(sampled) < target_count:
            remaining = [s for s in self.synthetic_samples if id(s) not in used_ids]
            rng.shuffle(remaining)
            needed = target_count - len(sampled)
            sampled.extend(remaining[:needed])

        return sampled

    def _merge_concat(
        self, train_real: list[SampleRecord]
    ) -> list[SampleRecord]:
        """Merge real and synthetic samples using concat strategy.

        When use_negative_cases=True: Simply concatenates all synthetic samples.
        When use_negative_cases=False: Limits synthetic to n_real * n_replicas,
        where replicas act as a multiplier of the filtered real data count.

        Args:
            train_real: Real training samples

        Returns:
            Combined training samples
        """
        if self.use_negative_cases:
            # Original behavior: add all synthetic samples
            return train_real + self.synthetic_samples

        # Lesion-only mode: limit synthetic based on replica multiplier
        target_synthetic = self._get_target_synthetic_count(len(train_real))
        synthetic_to_add = self._sample_synthetic_balanced(
            target_synthetic, seed=self.seed
        )

        logger.info(
            f"Concat merge (lesion-only): {len(train_real)} real + "
            f"{len(synthetic_to_add)} synthetic (target: {target_synthetic}, "
            f"{len(self.replicas)} replicas × {len(train_real)} real)"
        )

        return train_real + synthetic_to_add

    def _merge_balance(
        self, train_real: list[SampleRecord]
    ) -> list[SampleRecord]:
        """Merge real and synthetic samples using balance strategy.

        For each z-bin:
        1. Count real lesion and non-lesion samples
        2. If lesion < non-lesion, add synthetic lesion samples to balance
        3. Then add remaining synthetic samples evenly (lesion and non-lesion)

        Args:
            train_real: Real training samples

        Returns:
            Combined training samples
        """
        # Count real samples by (zbin, has_lesion)
        real_counts = defaultdict(int)
        for sample in train_real:
            key = (sample.z_bin, sample.has_lesion)
            real_counts[key] += 1

        # Get all z-bins from real data
        zbins = set(s.z_bin for s in train_real)

        combined = list(train_real)
        used_synthetic = set()

        # Phase 1: Balance lesion vs non-lesion for each z-bin
        for zbin in zbins:
            n_lesion = real_counts[(zbin, True)]
            n_no_lesion = real_counts[(zbin, False)]

            # If minority is lesion, add synthetic lesion samples
            if n_lesion < n_no_lesion:
                deficit = n_no_lesion - n_lesion
                synth_lesion = self.synthetic_by_zbin_lesion.get((zbin, True), [])

                # Add up to deficit synthetic lesion samples
                for i, sample in enumerate(synth_lesion):
                    if i >= deficit:
                        break
                    if id(sample) not in used_synthetic:
                        combined.append(sample)
                        used_synthetic.add(id(sample))

            # If minority is no-lesion, add synthetic no-lesion samples
            elif n_no_lesion < n_lesion:
                deficit = n_lesion - n_no_lesion
                synth_no_lesion = self.synthetic_by_zbin_lesion.get((zbin, False), [])

                for i, sample in enumerate(synth_no_lesion):
                    if i >= deficit:
                        break
                    if id(sample) not in used_synthetic:
                        combined.append(sample)
                        used_synthetic.add(id(sample))

        # Phase 2: Add remaining synthetic samples in balanced pairs
        remaining_synthetic = [s for s in self.synthetic_samples if id(s) not in used_synthetic]

        # Group remaining by (zbin, has_lesion)
        remaining_by_group = defaultdict(list)
        for sample in remaining_synthetic:
            key = (sample.z_bin, sample.has_lesion)
            remaining_by_group[key].append(sample)

        # Add in pairs (lesion, no-lesion) for each z-bin
        for zbin in zbins:
            lesion_remaining = remaining_by_group.get((zbin, True), [])
            no_lesion_remaining = remaining_by_group.get((zbin, False), [])

            # Add pairs
            n_pairs = min(len(lesion_remaining), len(no_lesion_remaining))
            combined.extend(lesion_remaining[:n_pairs])
            combined.extend(no_lesion_remaining[:n_pairs])

        return combined

    def _merge_balance_lesions_only(
        self, train_real: list[SampleRecord]
    ) -> list[SampleRecord]:
        """Merge real and synthetic lesion samples with z-bin balancing.

        When use_negative_cases=False, all samples have lesions.
        This method:
        1. Calculates target synthetic count: n_real * n_replicas
        2. Balances the number of lesion samples across z-bins first
        3. Distributes remaining budget evenly across z-bins

        Args:
            train_real: Real training samples (all have lesions)

        Returns:
            Combined training samples with balanced z-bin distribution
        """
        n_real = len(train_real)

        # Calculate target synthetic count based on replica multiplier
        target_synthetic = self._get_target_synthetic_count(n_real)

        # Count real samples per z-bin
        real_counts_by_zbin = defaultdict(int)
        for sample in train_real:
            real_counts_by_zbin[sample.z_bin] += 1

        # Get all z-bins from real data
        zbins = sorted(real_counts_by_zbin.keys())

        if not zbins:
            # Fallback: just sample target_synthetic from available
            synthetic_to_add = self._sample_synthetic_balanced(
                target_synthetic, seed=self.seed
            )
            return train_real + synthetic_to_add

        # Find max count across z-bins (for initial balancing)
        max_per_zbin = max(real_counts_by_zbin.values())

        combined = list(train_real)
        used_synthetic = set()
        synthetic_added = 0

        # Phase 1: Balance z-bins up to max_per_zbin
        for zbin in zbins:
            if synthetic_added >= target_synthetic:
                break

            current_count = real_counts_by_zbin[zbin]
            deficit = max_per_zbin - current_count

            if deficit <= 0:
                continue

            # Limit by remaining budget
            deficit = min(deficit, target_synthetic - synthetic_added)

            # Get synthetic lesion samples for this z-bin
            synth_samples = self.synthetic_by_zbin_lesion.get((zbin, True), [])

            for sample in synth_samples:
                if deficit <= 0 or synthetic_added >= target_synthetic:
                    break
                if id(sample) not in used_synthetic:
                    combined.append(sample)
                    used_synthetic.add(id(sample))
                    deficit -= 1
                    synthetic_added += 1

        # Phase 2: Distribute remaining budget evenly across z-bins
        remaining_budget = target_synthetic - synthetic_added

        if remaining_budget > 0 and zbins:
            rng = np.random.RandomState(self.seed + 1)

            # Calculate how many more per z-bin
            extra_per_zbin = remaining_budget // len(zbins)
            extra_remainder = remaining_budget % len(zbins)

            for i, zbin in enumerate(zbins):
                if synthetic_added >= target_synthetic:
                    break

                # Allocate extra remainder to first z-bins
                n_to_add = extra_per_zbin + (1 if i < extra_remainder else 0)

                synth_samples = [
                    s for s in self.synthetic_by_zbin_lesion.get((zbin, True), [])
                    if id(s) not in used_synthetic
                ]
                rng.shuffle(synth_samples)

                for sample in synth_samples[:n_to_add]:
                    if synthetic_added >= target_synthetic:
                        break
                    combined.append(sample)
                    used_synthetic.add(id(sample))
                    synthetic_added += 1

        logger.info(
            f"Balance merge (lesion-only): {n_real} real + "
            f"{synthetic_added} synthetic (target: {target_synthetic}, "
            f"{len(self.replicas)} replicas × {n_real} real)"
        )

        return combined

    def plan(self) -> Path:
        """Generate fold CSV files.

        Includes train, val, and test splits for each fold.
        Test samples are the same across all folds (repeated for convenience).

        Returns:
            Path to the generated CSV file
        """
        mode = self._get_mode()
        csv_path = self.output_dir / f"kfold_plan_{mode}.csv"

        all_rows = []

        for fold_idx in range(self.n_folds):
            train_samples, val_samples = self.get_fold(fold_idx)

            # Add train samples
            for sample in train_samples:
                all_rows.append(sample.to_dict(fold_idx, "train"))

            # Add val samples
            for sample in val_samples:
                all_rows.append(sample.to_dict(fold_idx, "val"))

            # Add test samples (same for all folds)
            for sample in self.test_samples:
                all_rows.append(sample.to_dict(fold_idx, "test"))

        # Write CSV
        fieldnames = [
            "fold", "split", "subject_id", "filepath", "z_index", "z_bin",
            "has_lesion_slice", "has_lesion_subject", "source", "replica"
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        logger.info(
            f"Generated fold plan: {csv_path} "
            f"({len(all_rows)} rows, {len(self.test_samples)} test samples per fold)"
        )

        return csv_path

    def get_fold_statistics(self, fold_idx: int) -> dict:
        """Get detailed statistics for a fold.

        Args:
            fold_idx: Fold index

        Returns:
            Dict with fold statistics (includes train, val, and test)
        """
        train_samples, val_samples = self.get_fold(fold_idx)

        stats = {
            "fold": fold_idx,
            "train": self._compute_split_stats(train_samples),
            "val": self._compute_split_stats(val_samples),
            "test": self._compute_split_stats(self.test_samples),
        }

        return stats

    def _compute_split_stats(self, samples: list[SampleRecord]) -> dict:
        """Compute statistics for a split.

        Args:
            samples: List of samples

        Returns:
            Dict with statistics
        """
        if not samples:
            return {
                "total": 0,
                "real": 0,
                "synthetic": 0,
                "lesion": 0,
                "no_lesion": 0,
                "lesion_ratio": 0.0,
                "zbins": {},
            }

        n_real = sum(1 for s in samples if s.source == "real")
        n_synthetic = sum(1 for s in samples if s.source == "synthetic")
        n_lesion = sum(1 for s in samples if s.has_lesion)
        n_no_lesion = sum(1 for s in samples if not s.has_lesion)

        # Per z-bin stats
        zbins_stats = defaultdict(lambda: {"lesion": 0, "no_lesion": 0, "real": 0, "synthetic": 0})
        for s in samples:
            zbins_stats[s.z_bin]["lesion" if s.has_lesion else "no_lesion"] += 1
            zbins_stats[s.z_bin][s.source] += 1

        return {
            "total": len(samples),
            "real": n_real,
            "synthetic": n_synthetic,
            "lesion": n_lesion,
            "no_lesion": n_no_lesion,
            "lesion_ratio": n_lesion / len(samples) if samples else 0.0,
            "zbins": dict(zbins_stats),
        }

    def print_fold_statistics(self, fold_idx: int):
        """Print detailed statistics for a fold.

        Args:
            fold_idx: Fold index
        """
        stats = self.get_fold_statistics(fold_idx)

        print(f"\nFold {fold_idx} Statistics:")
        print("=" * 60)

        for split in ["train", "val", "test"]:
            s = stats[split]
            print(f"\n{split.upper()}:")
            print(f"  Total samples: {s['total']}")
            print(f"  Real: {s['real']}, Synthetic: {s['synthetic']}")
            print(f"  Lesion: {s['lesion']}, No lesion: {s['no_lesion']}")
            print(f"  Lesion ratio: {s['lesion_ratio']:.2%}")

            if s['zbins']:
                print(f"  Z-bins: {len(s['zbins'])} unique bins")

    def generate_zbin_lesion_visualization(
        self,
        output_dir: Path | str | None = None,
    ) -> Path | None:
        """Generate per-fold z-bin lesion distribution visualization.

        Shows the distribution of lesion samples across z-bins for each fold.
        This is particularly useful when use_negative_cases=False to verify
        the balance strategy is working correctly.

        Args:
            output_dir: Output directory for the visualization.
                       Defaults to self.output_dir.

        Returns:
            Path to the saved figure, or None if matplotlib is not available.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np_viz
        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")
            return None

        if output_dir is None:
            output_dir = self.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect data for all folds
        n_folds = self.n_folds

        # Get all z-bins across all folds
        all_zbins: set[int] = set()
        fold_data = []

        for fold_idx in range(n_folds):
            train_samples, val_samples = self.get_fold(fold_idx)

            # Count lesion samples per z-bin (split by source)
            train_real_by_zbin: dict[int, int] = defaultdict(int)
            train_synth_by_zbin: dict[int, int] = defaultdict(int)
            val_by_zbin: dict[int, int] = defaultdict(int)

            for s in train_samples:
                if s.has_lesion:
                    all_zbins.add(s.z_bin)
                    if s.source == "real":
                        train_real_by_zbin[s.z_bin] += 1
                    else:
                        train_synth_by_zbin[s.z_bin] += 1

            for s in val_samples:
                if s.has_lesion:
                    all_zbins.add(s.z_bin)
                    val_by_zbin[s.z_bin] += 1

            fold_data.append({
                "train_real": dict(train_real_by_zbin),
                "train_synth": dict(train_synth_by_zbin),
                "val": dict(val_by_zbin),
            })

        if not all_zbins:
            logger.warning("No lesion samples found, skipping visualization")
            return None

        zbins = sorted(all_zbins)

        # Create figure with one row per fold
        fig, axes = plt.subplots(
            n_folds, 2,
            figsize=(14, 3 * n_folds),
            dpi=100,
            squeeze=False,
        )

        mode_str = "Lesion-Only" if not self.use_negative_cases else "All Slices"
        fig.suptitle(
            f"Per-Fold Z-bin Lesion Distribution ({mode_str})",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        x = np_viz.arange(len(zbins))
        width = 0.6

        for fold_idx, data in enumerate(fold_data):
            # Training plot (stacked: real + synthetic)
            ax_train = axes[fold_idx, 0]

            real_counts = [data["train_real"].get(z, 0) for z in zbins]
            synth_counts = [data["train_synth"].get(z, 0) for z in zbins]

            ax_train.bar(x, real_counts, width, label="Real", color="#2ecc71")
            ax_train.bar(x, synth_counts, width, bottom=real_counts,
                        label="Synthetic", color="#9b59b6")

            ax_train.set_xlabel("Z-bin", fontsize=10)
            ax_train.set_ylabel("Lesion Count", fontsize=10)
            ax_train.set_title(f"Fold {fold_idx} - Train", fontsize=11, fontweight="bold")

            # Show every 5th tick or so
            tick_step = max(1, len(zbins) // 10)
            tick_positions = np_viz.arange(0, len(zbins), tick_step)
            ax_train.set_xticks(tick_positions)
            ax_train.set_xticklabels([zbins[i] for i in tick_positions])
            ax_train.legend(loc="upper right", fontsize=8)
            ax_train.spines["top"].set_visible(False)
            ax_train.spines["right"].set_visible(False)

            # Validation plot
            ax_val = axes[fold_idx, 1]

            val_counts = [data["val"].get(z, 0) for z in zbins]

            ax_val.bar(x, val_counts, width, label="Real", color="#1abc9c")

            ax_val.set_xlabel("Z-bin", fontsize=10)
            ax_val.set_ylabel("Lesion Count", fontsize=10)
            ax_val.set_title(f"Fold {fold_idx} - Validation", fontsize=11, fontweight="bold")
            ax_val.set_xticks(tick_positions)
            ax_val.set_xticklabels([zbins[i] for i in tick_positions])
            ax_val.spines["top"].set_visible(False)
            ax_val.spines["right"].set_visible(False)

        plt.tight_layout()

        # Save figure
        suffix = "lesion_only" if not self.use_negative_cases else "all_slices"
        fig_path = output_dir / f"kfold_zbin_lesion_distribution_{suffix}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"Saved z-bin lesion distribution visualization to: {fig_path}")

        return fig_path
