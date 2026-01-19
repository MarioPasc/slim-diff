"""Abstract base class for dataset-specific slice cache builders.

This module implements the Template Method pattern: shared logic is in concrete
methods, dataset-specific logic is in abstract methods that subclasses must implement.
"""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from monai.transforms import Compose
from omegaconf import DictConfig
from tqdm import tqdm

from src.diffusion.data.splits import SubjectInfo
from src.diffusion.data.transforms import check_brain_content, extract_axial_slice
from src.diffusion.model.embeddings.zpos import quantize_z
from src.diffusion.utils.io import save_sample_npz
from src.diffusion.utils.zbin_priors import compute_zbin_priors, save_zbin_priors

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset-specific configuration extracted from cache_config.yaml."""

    name: str
    root_dir: Path
    modality_index: int | None = None
    modality_name: str | None = None
    merge_labels: dict[int, int] | None = None
    prefix: str | None = None


class SliceCacheBuilder(ABC):
    """Abstract base class for dataset-specific slice cache builders.

    Implements the Template Method pattern: shared logic in concrete methods,
    dataset-specific logic in abstract methods.

    Subclasses must implement:
        - discover_subjects(): Find subjects for a given split
        - get_transforms(): Get MONAI preprocessing pipeline
        - detect_lesion(): Detect if a slice contains lesion
        - filter_slice(): Apply dataset-specific filtering
        - auto_detect_z_range(): Scan dataset for optimal z-range

    The base class provides:
        - build_cache(): Main orchestration
        - process_subject(): 3D→2D slicing pipeline
        - compute_slice_metadata(): Metadata computation
        - write_index_csv(): CSV file creation
        - compute_and_save_zbin_priors(): Brain ROI atlas generation
    """

    def __init__(
        self,
        cache_config: DictConfig,
        train_config: DictConfig | None = None,
    ):
        """Initialize builder from configuration.

        Args:
            cache_config: Cache-specific configuration (from cache_config.yaml)
            train_config: Optional training configuration for inference of
                          parameters like spatial_shape, intensity_norm, etc.
        """
        self.cache_cfg = cache_config
        self.train_cfg = train_config

        # Core parameters
        self.cache_dir = Path(cache_config.cache_dir)
        self.z_bins = cache_config.z_bins

        # Handle auto z-range detection
        z_range = cache_config.slice_sampling.z_range
        if isinstance(z_range, str) and z_range.lower() == "auto":
            logger.info("Auto-detecting z-range from dataset...")
            self.z_range = self.auto_detect_z_range()
            logger.info(f"Auto-detected z-range: {self.z_range}")
        else:
            self.z_range = tuple(z_range)

        # Optional parameters
        self.lesion_area_min_pixels = cache_config.get("lesion_area_min_pixels", 0)
        self.drop_healthy_patients = cache_config.get("drop_healthy_patients", False)

        # Will be set during first slice processing
        self.spatial_shape: tuple[int, int, int] | None = None

        logger.info(
            f"SliceCacheBuilder initialized: "
            f"z_range={self.z_range}, z_bins={self.z_bins}, "
            f"cache_dir={self.cache_dir}"
        )

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def discover_subjects(
        self,
        split: str,
        dataset_cfg: DatasetConfig,
    ) -> list[SubjectInfo]:
        """Discover subjects for a given split and dataset.

        Args:
            split: "train", "val", or "test"
            dataset_cfg: Dataset-specific configuration

        Returns:
            List of SubjectInfo objects with paths and metadata
        """
        pass

    @abstractmethod
    def get_transforms(
        self,
        has_label: bool = True,
    ) -> Compose:
        """Get MONAI transform pipeline for this dataset.

        Args:
            has_label: Whether subjects have labels (False for control/healthy)

        Returns:
            MONAI Compose transform
        """
        pass

    @abstractmethod
    def detect_lesion(
        self,
        mask_slice: torch.Tensor | np.ndarray,
    ) -> bool:
        """Detect if a mask slice contains lesion pixels.

        Implementation varies by dataset:
        - Epilepsy: Binary check (mask > 0)
        - BraTS-MEN: Multi-class merge then check

        Args:
            mask_slice: 2D mask of shape (C, H, W) or (H, W)

        Returns:
            True if lesion pixels detected
        """
        pass

    @abstractmethod
    def filter_slice(
        self,
        image_slice: torch.Tensor,
        mask_slice: torch.Tensor,
        z_idx: int,
        metadata: dict[str, Any],
    ) -> bool:
        """Apply dataset-specific filtering to determine if slice should be kept.

        Args:
            image_slice: Image slice (C, H, W)
            mask_slice: Mask slice (C, H, W)
            z_idx: Z-index of this slice
            metadata: Computed metadata dict

        Returns:
            True if slice should be kept, False to discard
        """
        pass

    @abstractmethod
    def auto_detect_z_range(self) -> tuple[int, int]:
        """Auto-detect optimal z-range from dataset lesion distribution.

        Scans the dataset to find the z-range where lesions appear,
        optionally with configurable offset/margin.

        Returns:
            Tuple of (min_z, max_z)
        """
        pass

    # =========================================================================
    # Concrete Methods - Shared logic across all datasets
    # =========================================================================

    def filter_collected_slices(
        self,
        slices: list[dict[str, Any]],
        split: str,
    ) -> list[dict[str, Any]]:
        """Optional hook to filter slices after collection but before CSV writing.

        Default implementation returns slices unchanged.
        Subclasses can override to implement custom filtering logic.

        For example, BraTS-MEN uses this to balance lesion/non-lesion slices per z-bin,
        while epilepsy uses drop_healthy_patients at the subject level instead.

        Args:
            slices: List of slice metadata dictionaries
            split: Split name ("train", "val", or "test")

        Returns:
            Filtered list of slice metadata dictionaries
        """
        return slices

    def build_cache(self) -> None:
        """Main entry point: build complete slice cache.

        Algorithm:
        1. Create output directories
        2. For each split (train/val/test):
           a. Discover subjects for all configured datasets
           b. Filter by drop_healthy_patients if enabled
           c. Process each subject → extract slices
           d. Write CSV index file
        3. Generate cache statistics
        4. Compute z-bin priors if enabled
        5. Generate bias analysis visualizations
        """
        logger.info("=" * 80)
        logger.info("Building slice cache")
        logger.info("=" * 80)

        # Create output directories
        slices_dir = self.cache_dir / "slices"
        slices_dir.mkdir(parents=True, exist_ok=True)

        # Track statistics
        stats = {
            "total_slices": 0,
            "train_slices": 0,
            "val_slices": 0,
            "test_slices": 0,
            "train_lesion_slices": 0,
            "train_empty_slices": 0,
            "val_lesion_slices": 0,
            "val_empty_slices": 0,
            "test_lesion_slices": 0,
            "test_empty_slices": 0,
            "z_bins": self.z_bins,
        }

        # Process each split
        all_metadata = {"train": [], "val": [], "test": []}

        for split in ["train", "val", "test"]:
            logger.info(f"\nProcessing {split} split...")

            # Discover subjects for this split
            # Note: Subclasses handle dataset-specific discovery
            subjects = self._discover_all_subjects_for_split(split)

            # Filter subjects if needed
            if self.drop_healthy_patients:
                subjects = [s for s in subjects if s.source != "control"]
                logger.info(
                    f"Dropped healthy patients, {len(subjects)} subjects remaining"
                )

            logger.info(f"Found {len(subjects)} subjects in {split} split")

            # Process each subject
            for subject_info in tqdm(subjects, desc=f"Processing {split}"):
                slice_metadata = self.process_subject(subject_info, slices_dir)
                all_metadata[split].extend(slice_metadata)

            # Apply subclass-specific slice filtering (e.g., balance lesion/non-lesion per z-bin)
            all_metadata[split] = self.filter_collected_slices(all_metadata[split], split)

            # Update stats (after filtering)
            for meta in all_metadata[split]:
                stats["total_slices"] += 1
                stats[f"{split}_slices"] += 1
                if meta["has_lesion"]:
                    stats[f"{split}_lesion_slices"] += 1
                else:
                    stats[f"{split}_empty_slices"] += 1

            # Write CSV index for this split
            self.write_index_csv(all_metadata[split], split)

        # Save statistics
        stats_path = self.cache_dir / "cache_stats.yaml"
        with open(stats_path, "w") as f:
            yaml.dump(stats, f)
        logger.info(f"Saved cache statistics to {stats_path}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Cache build complete!")
        logger.info("=" * 80)
        logger.info(f"Total slices: {stats['total_slices']}")
        logger.info(f"  Train: {stats['train_slices']} ({stats['train_lesion_slices']} lesion)")
        logger.info(f"  Val:   {stats['val_slices']} ({stats['val_lesion_slices']} lesion)")
        logger.info(f"  Test:  {stats['test_slices']} ({stats['test_lesion_slices']} lesion)")

        # Compute z-bin priors if enabled
        if self.cache_cfg.get("postprocessing", {}).get("zbin_priors", {}).get("enabled", False):
            logger.info("\nComputing z-bin priors...")
            self.compute_and_save_zbin_priors()

            # Create visualization of z-bin priors alignment
            logger.info("\nCreating z-bin prior visualization...")
            self._create_zbin_prior_visualization()

    def _discover_all_subjects_for_split(self, split: str) -> list[SubjectInfo]:
        """Discover all subjects for a split across all configured datasets.

        This is a helper method that calls the abstract discover_subjects()
        method for each dataset in the config.

        Args:
            split: "train", "val", or "test"

        Returns:
            Combined list of SubjectInfo objects
        """
        # This will be implemented differently by subclasses
        # For now, delegate to the abstract method
        # Subclasses will handle their own dataset config structure
        raise NotImplementedError(
            "Subclasses must implement _discover_all_subjects_for_split or override build_cache"
        )

    def process_subject(
        self,
        subject_info: SubjectInfo,
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        """Process a single subject and extract all valid slices.

        Algorithm:
        1. Load and transform 3D volume
        2. For each z-index in z_range:
           a. Extract 2D slice
           b. Check brain content (filter_empty_brain)
           c. Detect lesion (via detect_lesion())
           d. Compute lesion area
           e. Apply lesion area filtering
           f. Apply dataset-specific filtering (via filter_slice())
           g. Compute metadata (via compute_slice_metadata())
           h. Save slice as .npz
        3. Return list of metadata dicts

        Args:
            subject_info: Subject information
            output_dir: Directory to save slice .npz files

        Returns:
            List of slice metadata dictionaries
        """
        # Get transforms
        has_label = subject_info.label_path is not None
        transforms = self.get_transforms(has_label=has_label)

        # Load and transform volume
        data_dict = {"image": str(subject_info.image_path)}
        if has_label:
            data_dict["seg"] = str(subject_info.label_path)

        try:
            transformed = transforms(data_dict)
        except Exception as e:
            logger.error(f"Failed to transform {subject_info.subject_id}: {e}")
            return []

        image_vol = transformed["image"]
        mask_vol = transformed.get("seg", None)

        # Store spatial shape for later use
        if self.spatial_shape is None:
            self.spatial_shape = tuple(image_vol.shape[1:])  # (H, W, D)

        # Extract slices
        metadata_list = []
        min_z, max_z = self.z_range

        for z_idx in range(min_z, max_z + 1):
            # Extract 2D slice
            image_slice = extract_axial_slice(image_vol, z_idx)
            if mask_vol is not None:
                mask_slice = extract_axial_slice(mask_vol, z_idx)
            else:
                # Create zero mask for control subjects
                mask_slice = torch.zeros_like(image_slice)

            # Filter empty brain slices
            slice_cfg = self.cache_cfg.slice_sampling
            if slice_cfg.filter_empty_brain:
                if not check_brain_content(
                    image_slice,
                    threshold=slice_cfg.brain_threshold,
                    min_fraction=slice_cfg.brain_min_fraction,
                ):
                    continue

            # Detect lesion
            has_lesion = self.detect_lesion(mask_slice)

            # Compute lesion area
            lesion_area_px = self._compute_lesion_area(mask_slice) if has_lesion else 0

            # Filter by lesion area threshold
            if has_lesion and self.lesion_area_min_pixels > 0:
                if lesion_area_px < self.lesion_area_min_pixels:
                    continue

            # Compute metadata
            metadata = self.compute_slice_metadata(
                z_idx, has_lesion, lesion_area_px, subject_info
            )

            # Apply dataset-specific filtering
            if not self.filter_slice(image_slice, mask_slice, z_idx, metadata):
                continue

            # Save slice as .npz
            z_bin = metadata["z_bin"]
            pathology_class = metadata["pathology_class"]
            filename = f"{subject_info.subject_id}_z{z_idx:03d}_bin{z_bin:02d}_c{pathology_class}.npz"
            filepath = output_dir / filename

            # Convert to numpy and save (save_sample_npz expects separate image and mask)
            image_np = image_slice.cpu().numpy()
            mask_np = mask_slice.cpu().numpy()
            save_sample_npz(filepath, image_np, mask_np, metadata)

            # Add filepath to metadata for CSV
            metadata["filepath"] = str(filepath.relative_to(self.cache_dir))
            metadata_list.append(metadata)

        return metadata_list

    def compute_slice_metadata(
        self,
        z_idx: int,
        has_lesion: bool,
        lesion_area_px: int,
        subject_info: SubjectInfo,
    ) -> dict[str, Any]:
        """Compute metadata for a single slice.

        Args:
            z_idx: Z-index of slice
            has_lesion: Whether slice contains lesion
            lesion_area_px: Lesion area in pixels
            subject_info: Subject information

        Returns:
            Metadata dictionary with keys:
            - subject_id, z_index, z_bin, pathology_class, token
            - source, split, has_lesion, lesion_area_px
        """
        z_bin = quantize_z(z_idx, self.z_range, self.z_bins)
        pathology_class = 1 if has_lesion else 0
        token = z_bin + pathology_class * self.z_bins

        return {
            "subject_id": subject_info.subject_id,
            "z_index": int(z_idx),
            "z_bin": int(z_bin),
            "pathology_class": int(pathology_class),
            "token": int(token),
            "source": subject_info.source,
            "split": subject_info.split,
            "has_lesion": bool(has_lesion),
            "lesion_area_px": int(lesion_area_px),
        }

    def write_index_csv(
        self,
        metadata_list: list[dict[str, Any]],
        split: str,
    ) -> None:
        """Write slice metadata to CSV index file for a split.

        Args:
            metadata_list: List of metadata dictionaries
            split: "train", "val", or "test"
        """
        csv_path = self.cache_dir / f"{split}.csv"

        if not metadata_list:
            logger.warning(f"No slices for {split} split, skipping CSV creation")
            return

        # Write CSV
        fieldnames = [
            "filepath",
            "subject_id",
            "z_index",
            "z_bin",
            "pathology_class",
            "token",
            "source",
            "split",
            "has_lesion",
            "lesion_area_px",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_list)

        logger.info(f"Wrote {len(metadata_list)} slices to {csv_path}")

    def compute_and_save_zbin_priors(self) -> None:
        """Compute and save z-bin priors if enabled in config.

        Z-bin priors are brain ROI atlases used for anatomical conditioning
        and post-processing.
        """
        pp_cfg = self.cache_cfg.postprocessing.zbin_priors
        priors_filename = pp_cfg.get("priors_filename", "zbin_priors_brain_roi.npz")
        priors_path = self.cache_dir / priors_filename

        # Check if recomputation is needed
        if priors_path.exists():
            if not self._should_recompute_priors(priors_path, pp_cfg):
                logger.info(f"Z-bin priors already exist at {priors_path}, skipping")
                return

        # Compute priors
        result = compute_zbin_priors(
            cache_dir=self.cache_dir,
            z_bins=self.z_bins,
            z_range=self.z_range,
            prob_threshold=pp_cfg.prob_threshold,
            dilate_radius_px=pp_cfg.dilate_radius_px,
            gaussian_sigma_px=pp_cfg.gaussian_sigma_px,
            min_component_px=pp_cfg.min_component_px,
            n_first_bins=pp_cfg.get("n_first_bins", 0),
            max_components_for_first_bins=pp_cfg.get("max_components_for_first_bins", 1),
        )

        # Save priors (compute_zbin_priors returns {"priors": dict, "metadata": dict})
        save_zbin_priors(
            priors=result["priors"],
            metadata=result["metadata"],
            output_path=priors_path,
        )

        logger.info(f"Saved z-bin priors to {priors_path}")

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _compute_lesion_area(self, mask_slice: torch.Tensor | np.ndarray) -> int:
        """Compute the lesion area in pixels.

        Args:
            mask_slice: 2D mask of shape (C, H, W) or (H, W)

        Returns:
            Number of lesion pixels
        """
        if isinstance(mask_slice, torch.Tensor):
            mask_slice = mask_slice.cpu().numpy()

        if mask_slice.ndim == 3:
            mask_slice = mask_slice[0]  # Take first channel

        # Count pixels > 0 (lesion region)
        return int((mask_slice > 0).sum())

    def _should_recompute_priors(
        self,
        priors_path: Path,
        zbin_cfg: dict[str, Any],
    ) -> bool:
        """Check if priors need recomputing due to file missing or param mismatch.

        Args:
            priors_path: Path to priors file
            zbin_cfg: Z-bin prior config dict

        Returns:
            True if priors should be recomputed
        """
        if not priors_path.exists():
            return True

        # Load metadata and compare params
        try:
            data = np.load(priors_path, allow_pickle=True)
            metadata = data["metadata"].item()

            # Check z_bins match
            if metadata.get("z_bins") != self.z_bins:
                logger.info(
                    f"Z-bins mismatch: file has {metadata.get('z_bins')}, "
                    f"config has {self.z_bins}. Recomputing priors."
                )
                return True

            # Check key params match
            for key in [
                "prob_threshold",
                "dilate_radius_px",
                "gaussian_sigma_px",
                "n_first_bins",
                "max_components_for_first_bins",
            ]:
                stored = metadata.get(key)
                configured = zbin_cfg.get(key)
                if stored != configured:
                    logger.info(
                        f"Parameter mismatch for {key}: "
                        f"file has {stored}, config has {configured}. Recomputing priors."
                    )
                    return True

            return False
        except Exception as e:
            logger.warning(f"Failed to read priors metadata: {e}. Recomputing.")
            return True

    def _create_zbin_prior_visualization(self) -> None:
        """Create visualization of z-bin priors overlayed on samples.

        This automatically creates one 4-panel visualization per z-bin showing how
        z-bin priors align with actual patient data. All visualizations are saved
        to a folder in the cache directory.
        """
        try:
            from src.diffusion.scripts.visualize_patient_with_priors import (
                visualize_all_zbins,
            )

            success = visualize_all_zbins(
                cache_dir=Path(self.cache_dir),
                alpha=0.3,
            )

            if success:
                logger.info("✓ Z-bin prior visualizations created successfully")
            else:
                logger.warning("⚠ Z-bin prior visualizations skipped")

        except ImportError as e:
            logger.warning(f"Could not import visualization module: {e}")
            logger.warning("Skipping z-bin prior visualization")
        except Exception as e:
            logger.warning(f"Failed to create z-bin prior visualization: {e}")
            logger.warning("Continuing with cache build...")
