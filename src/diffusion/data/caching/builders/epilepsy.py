"""Epilepsy dataset slice cache builder.

This module implements the SliceCacheBuilder for the epilepsy (FCD lesion) dataset.
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
from src.diffusion.data.splits import (
    SubjectInfo,
    create_epilepsy_splits,
    create_control_splits,
)
from src.diffusion.data.transforms import get_volume_transforms, check_brain_content
from src.diffusion.utils.io import discover_subjects, parse_subject_prefix, get_image_path, get_label_path

logger = logging.getLogger(__name__)


@register_dataset("epilepsy")
class EpilepsySliceCacheBuilder(SliceCacheBuilder):
    """Slice cache builder for epilepsy (FCD lesion) dataset.

    Dataset characteristics:
    - Binary lesion segmentation (lesion vs. background)
    - Two datasets: epilepsy (with lesions), control (healthy)
    - nnUNet-style directory structure
    - Fixed z-range [30, 90] (or auto-detected)

    Directory structure:
        Dataset210_MRIe_none/
        ├── imagesTr/
        │   ├── MRIe_001_0000.nii.gz
        │   └── ...
        ├── labelsTr/
        │   ├── MRIe_001.nii.gz
        │   └── ...
        └── imagesTs/ (optional predefined test set)

        Dataset310_MRIcontrol_none/
        ├── imagesTr/
        │   ├── MRIcontrol_001_0000.nii.gz
        │   └── ...
        └── (no labels for control subjects)
    """

    def __init__(self, cache_config, train_config=None):
        super().__init__(cache_config, train_config)

        # Create splits once during initialization
        self._splits = self._create_all_splits()

    def _create_all_splits(self) -> dict[str, Any]:
        """Create epilepsy and control splits."""
        splits = {}

        # Create epilepsy splits
        if "epilepsy" in self.cache_cfg.datasets:
            # Convert cache config to format expected by create_epilepsy_splits
            legacy_cfg = self._convert_to_legacy_format()
            epilepsy_split = create_epilepsy_splits(legacy_cfg)
            splits["epilepsy"] = epilepsy_split
            logger.info(f"Epilepsy splits: {epilepsy_split}")

        # Create control splits
        if "control" in self.cache_cfg.datasets:
            legacy_cfg = self._convert_to_legacy_format()
            target_test_size = splits.get("epilepsy").n_test if "epilepsy" in splits else None
            control_split = create_control_splits(legacy_cfg, target_test_size=target_test_size)
            splits["control"] = control_split
            logger.info(f"Control splits: {control_split}")

        return splits

    def _convert_to_legacy_format(self):
        """Convert new cache config format to legacy format for split functions.

        This is a temporary adapter until we fully refactor the split functions.
        """
        epilepsy_cfg = self.cache_cfg.datasets.get("epilepsy", {})
        control_cfg = self.cache_cfg.datasets.get("control", {})

        legacy_cfg = OmegaConf.create({
            "data": {
                "root_dir": epilepsy_cfg.get("root_dir", ""),
                "epilepsy": epilepsy_cfg.get("epilepsy_dataset", {}),
                "control": control_cfg.get("control_dataset", {}),
                "splits": epilepsy_cfg.get("splits", {}),
                "transforms": self.cache_cfg.transforms,  # Add transforms for get_volume_transforms
            }
        })

        return legacy_cfg

    def _discover_all_subjects_for_split(self, split: str) -> list[SubjectInfo]:
        """Discover all subjects for a split across epilepsy and control datasets.

        Args:
            split: "train", "val", or "test"

        Returns:
            Combined list of SubjectInfo objects
        """
        subject_infos = []

        # Map split names to directory names
        # For epilepsy dataset, check if using predefined test split
        use_predefined_test = False
        if "epilepsy" in self.cache_cfg.datasets:
            epilepsy_cfg = self.cache_cfg.datasets.epilepsy
            use_predefined_test = epilepsy_cfg.get("splits", {}).get("use_predefined_test", False)

        # Determine directory names based on split and configuration
        if split == "test" and use_predefined_test:
            image_dir = "imagesTs"
            label_dir = "labelsTs"
        else:
            # train, val, or test without predefined test set all use training directories
            image_dir = "imagesTr"
            label_dir = "labelsTr"

        # Get epilepsy subjects for this split
        if "epilepsy" in self._splits:
            epilepsy_split = self._splits["epilepsy"]
            epilepsy_subjects = getattr(epilepsy_split, f"{split}_subjects", [])

            epilepsy_cfg = self.cache_cfg.datasets.epilepsy
            dataset_path = Path(epilepsy_cfg.root_dir) / epilepsy_cfg.epilepsy_dataset.name
            prefix = parse_subject_prefix(epilepsy_cfg.epilepsy_dataset.name)
            modality_index = epilepsy_cfg.epilepsy_dataset.modality_index

            for subject_id in epilepsy_subjects:
                image_path = get_image_path(dataset_path, subject_id, modality_index, image_dir)
                label_path = get_label_path(dataset_path, subject_id, label_dir)

                subject_infos.append(SubjectInfo(
                    subject_id=subject_id,
                    image_path=Path(image_path),
                    label_path=Path(label_path) if label_path else None,
                    source="epilepsy",
                    split=split,
                ))

        # Get control subjects for this split
        if "control" in self._splits:
            control_split = self._splits["control"]
            control_subjects = getattr(control_split, f"{split}_subjects", [])

            control_cfg = self.cache_cfg.datasets.control
            dataset_path = Path(control_cfg.root_dir) / control_cfg.control_dataset.name
            prefix = parse_subject_prefix(control_cfg.control_dataset.name)
            modality_index = control_cfg.control_dataset.modality_index

            for subject_id in control_subjects:
                image_path = get_image_path(dataset_path, subject_id, modality_index, image_dir)
                # Control subjects have no labels
                label_path = None

                subject_infos.append(SubjectInfo(
                    subject_id=subject_id,
                    image_path=Path(image_path),
                    label_path=label_path,
                    source="control",
                    split=split,
                ))

        return subject_infos

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def discover_subjects(
        self,
        split: str,
        dataset_cfg: DatasetConfig,
    ) -> list[SubjectInfo]:
        """Discover epilepsy or control subjects for a split.

        Note: This method is not directly used by the epilepsy builder since
        we handle both datasets together in _discover_all_subjects_for_split.
        Kept for interface compliance.
        """
        return self._discover_all_subjects_for_split(split)

    def get_transforms(self, has_label: bool = True) -> Compose:
        """Get MONAI transforms for epilepsy dataset.

        Uses existing transform pipeline from transforms.py.

        Args:
            has_label: Whether subjects have labels (False for control)

        Returns:
            MONAI Compose transform
        """
        # Convert cache config to format expected by get_volume_transforms
        legacy_cfg = self._convert_to_legacy_format()
        return get_volume_transforms(legacy_cfg, has_label=has_label)

    def detect_lesion(
        self,
        mask_slice: torch.Tensor | np.ndarray,
    ) -> bool:
        """Binary lesion detection: any pixel > 0 indicates lesion.

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

        return bool((mask_array > 0).any())

    def filter_slice(
        self,
        image_slice: torch.Tensor,
        mask_slice: torch.Tensor,
        z_idx: int,
        metadata: dict[str, Any],
    ) -> bool:
        """Apply epilepsy-specific filtering.

        Filters:
        1. Brain content check (if enabled) - already done in process_subject
        2. Lesion area threshold - already done in process_subject

        Returns True to keep slice, False to discard.

        Args:
            image_slice: Image slice (C, H, W)
            mask_slice: Mask slice (C, H, W)
            z_idx: Z-index
            metadata: Computed metadata

        Returns:
            True to keep slice
        """
        # All filtering is already done in the base class process_subject method
        # This can be extended for additional epilepsy-specific filtering
        return True

    def auto_detect_z_range(self) -> tuple[int, int]:
        """Auto-detect z-range from epilepsy lesion distribution.

        Algorithm:
        1. Scan all epilepsy subjects (not control)
        2. Load each volume, find min/max z where lesions exist
        3. Tighten the range by removing offset slices from each side
        4. Return (min_z + offset, max_z - offset)

        Returns:
            Tuple of (min_z, max_z)
        """
        logger.info("Auto-detecting z-range from epilepsy dataset...")

        # Get offset from config (default: 5 slices to remove from each side)
        offset = self.cache_cfg.slice_sampling.get("auto_z_range_offset", 5)

        # Get epilepsy dataset config
        if "epilepsy" not in self.cache_cfg.datasets:
            logger.warning("No epilepsy dataset configured, using default z-range [30, 90]")
            return (30, 90)

        epilepsy_cfg = self.cache_cfg.datasets.epilepsy
        dataset_path = Path(epilepsy_cfg.root_dir) / epilepsy_cfg.epilepsy_dataset.name
        prefix = parse_subject_prefix(epilepsy_cfg.epilepsy_dataset.name)
        modality_index = epilepsy_cfg.epilepsy_dataset.modality_index

        # Discover all epilepsy subjects (training + test if available)
        subjects = discover_subjects(
            dataset_path,
            image_dir="imagesTr",
            prefix=prefix,
            modality_index=modality_index,
        )

        # Also check test set if using predefined test
        if epilepsy_cfg.splits.get("use_predefined_test", False):
            test_subjects = discover_subjects(
                dataset_path,
                image_dir="imagesTs",
                prefix=prefix,
                modality_index=modality_index,
            )
            subjects.extend(test_subjects)

        if not subjects:
            logger.warning("No subjects found, using default z-range [30, 90]")
            return (30, 90)

        # Scan for lesion z-range
        min_z_global = float('inf')
        max_z_global = float('-inf')

        # Get transforms
        transforms = self.get_transforms(has_label=True)

        for subject_id in subjects:
            # Construct paths - try training directories first
            image_path = dataset_path / "imagesTr" / f"{subject_id}_{modality_index:04d}.nii.gz"
            label_path = dataset_path / "labelsTr" / f"{subject_id}.nii.gz"

            # If not found, try test directories
            if not image_path.exists():
                image_path = dataset_path / "imagesTs" / f"{subject_id}_{modality_index:04d}.nii.gz"
                label_path = dataset_path / "labelsTs" / f"{subject_id}.nii.gz"

            # Skip if files don't exist
            if not image_path.exists() or not label_path.exists():
                continue

            # Load and transform
            data_dict = {
                "image": str(image_path),
                "seg": str(label_path),
            }

            try:
                transformed = transforms(data_dict)
                mask_vol = transformed["seg"]

                if isinstance(mask_vol, torch.Tensor):
                    mask_vol = mask_vol.cpu().numpy()

                # Find z-indices with lesions
                # mask_vol shape: (C, H, W, D)
                lesion_per_z = (mask_vol[0] > 0).sum(axis=(0, 1))  # (D,)
                z_indices = np.where(lesion_per_z > 0)[0]

                if len(z_indices) > 0:
                    min_z_global = min(min_z_global, int(z_indices.min()))
                    max_z_global = max(max_z_global, int(z_indices.max()))

            except Exception as e:
                logger.warning(f"Failed to process {subject_id} for z-range detection: {e}")
                continue

        if min_z_global == float('inf'):
            logger.warning("No lesions found, using default z-range [30, 90]")
            return (30, 90)

        # Tighten range by removing offset slices from each side
        final_min = int(min_z_global) + offset
        final_max = int(max_z_global) - offset

        # Sanity check: ensure valid range
        if final_min >= final_max:
            logger.warning(
                f"Invalid range after offset [{final_min}, {final_max}]. "
                f"Using detected lesion range [{int(min_z_global)}, {int(max_z_global)}] without offset."
            )
            final_min = int(min_z_global)
            final_max = int(max_z_global)

        logger.info(
            f"Auto-detected z-range: [{final_min}, {final_max}] "
            f"(lesion range: [{int(min_z_global)}, {int(max_z_global)}], "
            f"offset: {offset} slices removed from each side)"
        )

        return (final_min, final_max)
