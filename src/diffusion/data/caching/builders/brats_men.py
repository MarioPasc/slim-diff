"""BraTS-MEN dataset slice cache builder.

This module implements the SliceCacheBuilder for the BraTS-MEN meningioma dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.transforms import Compose
from omegaconf import OmegaConf

from ..base import DatasetConfig, SliceCacheBuilder
from ..registry import register_dataset
from src.diffusion.data.splits import SubjectInfo
from src.diffusion.data.transforms import get_volume_transforms

logger = logging.getLogger(__name__)


# Modality name mapping: config name -> filename suffix
MODALITY_MAP = {
    "t1c": "t1c",  # T1 contrast-enhanced
    "t1n": "t1n",  # T1 native
    "t2f": "t2f",  # T2 FLAIR
    "t2w": "t2w",  # T2 weighted
    # Legacy aliases for backwards compatibility
    "t1": "t1n",
    "t1gd": "t1c",
    "flair": "t2f",
    "t2": "t2w",
}


@register_dataset("brats_men")
class BraTSMenSliceCacheBuilder(SliceCacheBuilder):
    """Slice cache builder for BraTS-MEN meningioma dataset.

    Dataset characteristics:
    - Multi-class segmentation: NCR (1), ED (2), ET (3)
    - Config-driven label merging to binary
    - Four modalities: t1c (contrast), t1n (native), t2f (FLAIR), t2w (T2)
    - Flat directory structure (all subjects in root directory)
    - Programmatic train/val/test split creation

    Directory structure:
        BraTS_Men_Train/
        ├── BraTS-MEN-00001-000/
        │   ├── BraTS-MEN-00001-000-t1c.nii.gz
        │   ├── BraTS-MEN-00001-000-t1n.nii.gz
        │   ├── BraTS-MEN-00001-000-t2f.nii.gz
        │   ├── BraTS-MEN-00001-000-t2w.nii.gz
        │   └── BraTS-MEN-00001-000-seg.nii.gz
        ├── BraTS-MEN-00002-000/
        └── ...
    """

    def __init__(self, cache_config, train_config=None):
        # IMPORTANT: Initialize cache attributes BEFORE super().__init__()
        # because auto_detect_z_range() may be called during base class initialization
        self._all_subjects = None
        self._split_assignments = None

        # Parse BraTS-MEN config before super().__init__() for use in auto_detect_z_range
        brats_cfg = cache_config.datasets.brats_men
        self.merge_labels = brats_cfg.get("merge_labels", {1: 1, 2: 1, 3: 1})

        # Get split configuration
        splits_cfg = brats_cfg.get("splits", {})
        self.train_fraction = splits_cfg.get("train_fraction", 0.7)
        self.val_fraction = splits_cfg.get("val_fraction", 0.15)
        self.test_fraction = splits_cfg.get("test_fraction", 0.15)
        self.split_seed = splits_cfg.get("seed", 42)

        # Dataset subset percentage (for debugging/testing)
        self.dataset_subset_percent = brats_cfg.get("dataset_subset_percent", 100)
        if not (0 < self.dataset_subset_percent <= 100):
            logger.warning(
                f"Invalid dataset_subset_percent={self.dataset_subset_percent}, "
                "using 100% (full dataset)"
            )
            self.dataset_subset_percent = 100

        # Now call parent __init__ (may trigger auto_detect_z_range)
        super().__init__(cache_config, train_config)

        logger.info(f"BraTS-MEN label merging: {self.merge_labels}")
        logger.info(
            f"BraTS-MEN split ratios: train={self.train_fraction}, "
            f"val={self.val_fraction}, test={self.test_fraction}, seed={self.split_seed}"
        )
        if self.dataset_subset_percent < 100:
            logger.info(f"BraTS-MEN dataset subset: {self.dataset_subset_percent}%")
        else:
            logger.info("BraTS-MEN dataset subset: 100% (full dataset)")

    def _discover_all_subjects_from_flat_structure(self) -> list[SubjectInfo]:
        """Discover all BraTS-MEN subjects from flat root directory.

        Returns:
            List of SubjectInfo objects (all subjects, no split assignment yet)
        """
        brats_cfg = self.cache_cfg.datasets.brats_men
        dataset_path = Path(brats_cfg.root_dir)

        if not dataset_path.exists():
            logger.error(f"Dataset root directory not found: {dataset_path}")
            return []

        # Get modality from config and map to filename suffix
        config_modality = brats_cfg.get("modality_name", "t2f")
        modality_suffix = MODALITY_MAP.get(config_modality, config_modality)

        logger.info(f"Discovering BraTS-MEN subjects from: {dataset_path}")
        logger.info(f"Using modality: {config_modality} (file suffix: {modality_suffix})")

        subject_infos = []

        # Iterate through all subdirectories
        for subject_dir in sorted(dataset_path.iterdir()):
            if not subject_dir.is_dir():
                continue

            # Subject ID is the directory name
            subject_id = subject_dir.name

            # Construct paths using the correct modality suffix
            image_path = subject_dir / f"{subject_id}-{modality_suffix}.nii.gz"
            label_path = subject_dir / f"{subject_id}-seg.nii.gz"

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}, skipping subject {subject_id}")
                continue

            if not label_path.exists():
                logger.warning(f"Label not found: {label_path}, skipping subject {subject_id}")
                continue

            info = SubjectInfo(
                subject_id=subject_id,
                image_path=image_path,
                label_path=label_path,
                source="brats_men",
                split=None,  # Will be assigned later
            )
            subject_infos.append(info)

        logger.info(f"Discovered {len(subject_infos)} BraTS-MEN subjects")

        # Apply dataset subset sampling if configured
        if self.dataset_subset_percent < 100:
            rng = np.random.RandomState(self.split_seed)
            n_total = len(subject_infos)
            n_subset = int(n_total * self.dataset_subset_percent / 100)

            # Randomly sample subjects
            indices = rng.choice(n_total, size=n_subset, replace=False)
            subject_infos = [subject_infos[i] for i in sorted(indices)]

            logger.info(
                f"Sampled {self.dataset_subset_percent}% of dataset: "
                f"{n_subset}/{n_total} subjects"
            )

        return subject_infos

    def _create_splits(self, all_subjects: list[SubjectInfo]) -> dict[str, list[SubjectInfo]]:
        """Create train/val/test splits programmatically.

        Args:
            all_subjects: List of all discovered subjects

        Returns:
            Dictionary mapping split name to list of SubjectInfo objects
        """
        if len(all_subjects) == 0:
            logger.warning("No subjects to split")
            return {"train": [], "val": [], "test": []}

        # Set random seed for reproducibility
        rng = np.random.RandomState(self.split_seed)

        # Shuffle subjects
        indices = np.arange(len(all_subjects))
        rng.shuffle(indices)

        # Compute split sizes
        n_total = len(all_subjects)
        n_train = int(n_total * self.train_fraction)
        n_val = int(n_total * self.val_fraction)
        # Remaining goes to test (handles rounding)

        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        # Create split assignments
        splits = {
            "train": [all_subjects[i] for i in train_indices],
            "val": [all_subjects[i] for i in val_indices],
            "test": [all_subjects[i] for i in test_indices],
        }

        # Update split field in SubjectInfo objects
        for split_name, subjects in splits.items():
            for subject in subjects:
                subject.split = split_name

        logger.info(
            f"Created splits: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

        return splits

    def _get_or_create_splits(self) -> dict[str, list[SubjectInfo]]:
        """Get or create splits (with caching).

        Returns:
            Dictionary mapping split name to list of SubjectInfo objects
        """
        if self._all_subjects is None:
            self._all_subjects = self._discover_all_subjects_from_flat_structure()

        if self._split_assignments is None:
            self._split_assignments = self._create_splits(self._all_subjects)

        return self._split_assignments

    def _discover_all_subjects_for_split(self, split: str) -> list[SubjectInfo]:
        """Discover all BraTS-MEN subjects for a specific split.

        This method is called by the base class build_cache() method.

        Args:
            split: "train", "val", or "test"

        Returns:
            List of SubjectInfo objects for the requested split
        """
        splits = self._get_or_create_splits()
        return splits.get(split, [])

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def discover_subjects(
        self,
        split: str,
        dataset_cfg: DatasetConfig,
    ) -> list[SubjectInfo]:
        """Discover BraTS-MEN subjects for a split.

        Args:
            split: "train", "val", or "test"
            dataset_cfg: Dataset configuration (not used, kept for interface)

        Returns:
            List of SubjectInfo objects for the requested split
        """
        splits = self._get_or_create_splits()
        return splits.get(split, [])

    def get_transforms(self, has_label: bool = True) -> Compose:
        """Get MONAI transforms for BraTS-MEN dataset.

        Uses similar pipeline to epilepsy but with multi-class label merging.

        Args:
            has_label: Whether subjects have labels

        Returns:
            MONAI Compose transform
        """
        from src.diffusion.data.transforms import MergeMultiClassLabeld, BinarizeMaskd

        # Convert cache config to format expected by get_volume_transforms
        # Create a minimal config structure
        legacy_cfg = OmegaConf.create(
            {
                "data": {
                    "transforms": self.cache_cfg.transforms,
                }
            }
        )

        base_transforms = get_volume_transforms(legacy_cfg, has_label=has_label)

        # Insert MergeMultiClassLabeld before BinarizeMaskd
        if has_label and self.merge_labels:
            # Get the list of transforms from the Compose object
            transform_list = list(base_transforms.transforms)

            # Find the index of BinarizeMaskd
            binarize_idx = None
            for i, transform in enumerate(transform_list):
                if isinstance(transform, BinarizeMaskd):
                    binarize_idx = i
                    break

            if binarize_idx is not None:
                # Insert MergeMultiClassLabeld before BinarizeMaskd
                merge_transform = MergeMultiClassLabeld(
                    keys=["seg"],
                    merge_map=self.merge_labels,
                )
                transform_list.insert(binarize_idx, merge_transform)

                logger.info(
                    f"Inserted MergeMultiClassLabeld before BinarizeMaskd "
                    f"with mapping: {self.merge_labels}"
                )

                # Create new Compose with modified transform list
                return Compose(transform_list)

        return base_transforms

    def detect_lesion(
        self,
        mask_slice: torch.Tensor | np.ndarray,
    ) -> bool:
        """Multi-class lesion detection with configurable merging.

        Algorithm:
        1. Extract raw labels from mask (values: 0, 1, 2, 3)
        2. Apply merge_labels mapping to convert to binary
        3. Check if any foreground pixels exist

        Note: Assumes mask has already been processed through MergeMultiClassLabeld
        and BinarizeMaskd transforms, so it's already binary.

        Args:
            mask_slice: 2D mask of shape (C, H, W) or (H, W)

        Returns:
            True if lesion pixels detected
        """
        if isinstance(mask_slice, torch.Tensor):
            mask_array = mask_slice.cpu().numpy()
        else:
            mask_array = mask_slice

        if mask_array.ndim == 3:
            mask_array = mask_array[0]  # Take first channel

        # Binary check (mask is already merged and binarized)
        return bool((mask_array > 0).any())

    def filter_slice(
        self,
        image_slice: torch.Tensor,
        mask_slice: torch.Tensor,
        z_idx: int,
        metadata: dict[str, Any],
    ) -> bool:
        """Apply BraTS-MEN specific filtering.

        Similar to epilepsy filtering for now.

        Args:
            image_slice: Image slice (C, H, W)
            mask_slice: Mask slice (C, H, W)
            z_idx: Z-index
            metadata: Computed metadata

        Returns:
            True to keep slice
        """
        # All filtering is already done in the base class
        # Can be extended for BraTS-specific filtering
        return True

    def auto_detect_z_range(self) -> tuple[int, int]:
        """Auto-detect z-range from BraTS-MEN tumor distribution.

        Scans all subjects in the flat directory to find the range of z-indices
        containing tumors.

        Returns:
            Tuple of (min_z, max_z)
        """
        logger.info("Auto-detecting z-range from BraTS-MEN dataset...")

        offset = self.cache_cfg.slice_sampling.get("auto_z_range_offset", 5)

        # Discover all subjects
        if self._all_subjects is None:
            self._all_subjects = self._discover_all_subjects_from_flat_structure()

        if len(self._all_subjects) == 0:
            logger.warning("No subjects found, using default z-range [20, 100]")
            return (20, 100)

        min_z_global = float("inf")
        max_z_global = float("-inf")

        # Get transforms
        transforms = self.get_transforms(has_label=True)

        # Scan subset of subjects for efficiency (use first 100 or all if fewer)
        subjects_to_scan = self._all_subjects[: min(100, len(self._all_subjects))]

        logger.info(f"Scanning {len(subjects_to_scan)} subjects for tumor extent...")

        for subject_info in subjects_to_scan:
            data_dict = {
                "image": str(subject_info.image_path),
                "seg": str(subject_info.label_path),
            }

            try:
                transformed = transforms(data_dict)
                mask_vol = transformed["seg"]

                if isinstance(mask_vol, torch.Tensor):
                    mask_vol = mask_vol.cpu().numpy()

                # Sum over spatial dimensions to get per-slice tumor counts
                # mask_vol shape: (C, H, W, D)
                lesion_per_z = (mask_vol[0] > 0).sum(axis=(0, 1))
                z_indices = np.where(lesion_per_z > 0)[0]

                if len(z_indices) > 0:
                    min_z_global = min(min_z_global, int(z_indices.min()))
                    max_z_global = max(max_z_global, int(z_indices.max()))

            except Exception as e:
                logger.warning(f"Failed to process {subject_info.subject_id}: {e}")
                continue

        if min_z_global == float("inf"):
            logger.warning("No tumors found, using default z-range [20, 100]")
            return (20, 100)

        # Tighten range by removing offset slices from each side
        final_min = int(min_z_global) + offset
        final_max = int(max_z_global) - offset

        # Sanity check: ensure valid range
        if final_min >= final_max:
            logger.warning(
                f"Invalid range after offset [{final_min}, {final_max}]. "
                f"Using detected tumor range [{int(min_z_global)}, {int(max_z_global)}] without offset."
            )
            final_min = int(min_z_global)
            final_max = int(max_z_global)

        logger.info(
            f"Auto-detected z-range: [{final_min}, {final_max}] "
            f"(tumor range: [{int(min_z_global)}, {int(max_z_global)}], "
            f"offset: {offset} slices removed from each side)"
        )

        return (final_min, final_max)

    def filter_collected_slices(
        self,
        slices: list[dict[str, Any]],
        split: str,
    ) -> list[dict[str, Any]]:
        """Filter collected slices to balance lesion/non-lesion distribution.

        For BraTS-MEN, if drop_healthy_patients=True, this drops 50% of non-lesion
        slices per z-bin to reduce the imbalance between lesion and non-lesion slices.

        This is different from epilepsy's interpretation of drop_healthy_patients,
        which drops entire control subjects.

        Args:
            slices: List of slice metadata dictionaries
            split: Split name ("train", "val", or "test")

        Returns:
            Filtered list of slice metadata dictionaries
        """
        if not self.drop_healthy_patients:
            return slices

        # Group slices by z-bin
        slices_by_zbin = {}
        for slice_meta in slices:
            z_bin = slice_meta["z_bin"]
            if z_bin not in slices_by_zbin:
                slices_by_zbin[z_bin] = {"lesion": [], "non_lesion": []}

            if slice_meta["has_lesion"]:
                slices_by_zbin[z_bin]["lesion"].append(slice_meta)
            else:
                slices_by_zbin[z_bin]["non_lesion"].append(slice_meta)

        # Filter 50% of non-lesion slices per z-bin
        filtered_slices = []
        total_dropped = 0

        rng = np.random.RandomState(42)  # Fixed seed for reproducibility

        for z_bin in sorted(slices_by_zbin.keys()):
            bin_data = slices_by_zbin[z_bin]

            # Keep all lesion slices
            filtered_slices.extend(bin_data["lesion"])

            # Keep 50% of non-lesion slices
            non_lesion_slices = bin_data["non_lesion"]
            n_to_keep = len(non_lesion_slices) // 2

            if len(non_lesion_slices) > 0:
                # Randomly select 50% to keep
                indices = rng.choice(
                    len(non_lesion_slices), size=n_to_keep, replace=False
                )
                kept_slices = [non_lesion_slices[i] for i in indices]
                filtered_slices.extend(kept_slices)
                total_dropped += len(non_lesion_slices) - n_to_keep

        logger.info(
            f"BraTS-MEN {split}: Dropped {total_dropped} non-lesion slices "
            f"({len(slices)} → {len(filtered_slices)}, "
            f"{100 * (1 - len(filtered_slices) / len(slices)):.1f}% reduction)"
        )

        return filtered_slices
