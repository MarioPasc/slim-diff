"""Segmentation dataset for real + optional synthetic data."""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from src.segmentation.utils.io import load_npz_sample

if TYPE_CHECKING:
    from src.segmentation.data.kfold_planner import SampleRecord

logger = logging.getLogger(__name__)


class PlannedFoldDataset(Dataset):
    """Dataset that loads samples from KFoldPlanner output.

    This dataset is designed to work with the KFoldPlanner class, which handles
    all the logic for combining real and synthetic data with various strategies.

    Features:
    - Loads real data from slice_cache NPZ files
    - Loads synthetic data from replica NPZ files
    - Converts masks from {-1,+1} to {0,1}
    - Returns (image, mask, metadata) tuples
    """

    def __init__(
        self,
        samples: list["SampleRecord"],
        real_cache_dir: Path | str,
        synthetic_dir: Path | str | None = None,
        transform=None,
        mask_threshold: float = 0.0,
    ):
        """Initialize dataset.

        Args:
            samples: List of SampleRecord from KFoldPlanner
            real_cache_dir: Path to slice_cache directory for real samples
            synthetic_dir: Path to synthetic replicas directory
            transform: MONAI transforms to apply
            mask_threshold: Threshold for binarizing mask in {-1,+1} space
        """
        self.samples = samples
        self.real_cache_dir = Path(real_cache_dir)
        self.synthetic_dir = Path(synthetic_dir) if synthetic_dir else None
        self.transform = transform
        self.mask_threshold = mask_threshold

        # Log dataset info
        n_real = sum(1 for s in samples if s.source == "real")
        n_synth = sum(1 for s in samples if s.source == "synthetic")
        logger.info(f"PlannedFoldDataset: {len(samples)} samples ({n_real} real, {n_synth} synthetic)")

    def _load_replica(self, replica_name: str) -> np.lib.npyio.NpzFile:
        """Load replica NPZ file without caching.

        Args:
            replica_name: Name of the replica file

        Returns:
            Loaded NPZ file

        Note:
            Uses mmap_mode='r' for efficient read-only memory mapping.
            No caching is needed as OS-level page cache handles this efficiently,
            and avoiding cache prevents multiprocessing pickle issues with num_workers > 0.
        """
        replica_path = self.synthetic_dir / replica_name
        # Use mmap_mode='r' for efficient memory-mapped access
        # OS page cache will handle repeated access efficiently
        return np.load(replica_path, mmap_mode='r')

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            dict with keys:
                - image: (1, H, W) float32 tensor in [-1, 1]
                - mask: (1, H, W) float32 tensor in {0, 1}
                - subject_id, has_lesion, source, z_bin
        """
        sample = self.samples[idx]

        if sample.source == "real":
            # Load from slice_cache NPZ
            npz_path = self.real_cache_dir / sample.filepath
            data = load_npz_sample(npz_path)
            image = data["image"]  # (128, 128) in [-1, 1]
            mask = data["mask"]    # (128, 128) in {-1, +1}
        else:
            # Load from replica NPZ
            # filepath format: "replica_name.npz:index"
            replica_name, idx_str = sample.filepath.split(":")
            sample_idx = int(idx_str)

            replica_data = self._load_replica(replica_name)
            # Extract data and convert to arrays immediately
            # This allows numpy to close the mmap file descriptor
            image = np.array(replica_data["images"][sample_idx], dtype=np.float32)
            mask = np.array(replica_data["masks"][sample_idx], dtype=np.float32)
            # Close the NPZ file to free resources
            replica_data.close()

        # Convert mask: {-1, +1} -> {0, 1}
        mask_binary = (mask > self.mask_threshold).astype(np.float32)

        # Add channel dimension
        image = image[np.newaxis, ...].astype(np.float32)  # (1, 128, 128)
        mask_binary = mask_binary[np.newaxis, ...].astype(np.float32)  # (1, 128, 128)

        # Convert to torch
        image = torch.from_numpy(image)
        mask_binary = torch.from_numpy(mask_binary)

        result = {
            "image": image,
            "mask": mask_binary,
            "subject_id": sample.subject_id,
            "has_lesion": sample.has_lesion,
            "source": sample.source,
            "z_bin": sample.z_bin,
        }

        # Apply transforms
        if self.transform is not None:
            result = self.transform(result)

        return result


class SegmentationSliceDataset(Dataset):
    """Dataset for segmentation with real + optional synthetic data.

    Features:
    - Loads real data from slice_cache NPZ files
    - Optionally mixes synthetic data at configurable ratio
    - Converts masks from {-1,+1} to {0,1}
    - Returns (image, mask, metadata) tuples

    Note: For new code, prefer using PlannedFoldDataset with KFoldPlanner.
    """

    def __init__(
        self,
        real_cache_dir: Path | str,
        real_csv: str | list[dict],  # CSV filename or list of sample dicts
        synthetic_dir: Path | str | None = None,
        synthetic_csv: str | None = None,
        synthetic_ratio: float = 0.0,
        transform=None,
        mask_threshold: float = 0.0,
    ):
        """Initialize dataset.

        Args:
            real_cache_dir: Path to slice_cache directory
            real_csv: CSV file name or list of sample metadata dicts
            synthetic_dir: Path to synthetic samples directory
            synthetic_csv: Synthetic samples index CSV
            synthetic_ratio: Ratio of synthetic to real (0.0 = real only)
            transform: MONAI transforms to apply
            mask_threshold: Threshold for binarizing mask in {-1,+1} space
        """
        self.real_cache_dir = Path(real_cache_dir)
        self.synthetic_dir = Path(synthetic_dir) if synthetic_dir else None
        self.transform = transform
        self.mask_threshold = mask_threshold

        # Load real samples
        if isinstance(real_csv, str):
            self.real_samples = self._load_samples_from_csv(
                self.real_cache_dir / real_csv
            )
        else:
            # Already a list of dicts
            self.real_samples = real_csv

        # Load synthetic samples if enabled
        self.synthetic_samples = []
        if synthetic_ratio > 0 and self.synthetic_dir is not None:
            self.synthetic_samples = self._load_synthetic_samples(
                self.synthetic_dir, synthetic_csv
            )

        # Mix samples
        self.samples = self._mix_samples(
            self.real_samples,
            self.synthetic_samples,
            synthetic_ratio,
        )

        logger.info(
            f"Dataset created: {len(self.real_samples)} real, "
            f"{len(self.synthetic_samples)} synthetic, "
            f"{len(self.samples)} total (ratio={synthetic_ratio})"
        )

    def _load_samples_from_csv(self, csv_path: Path) -> list[dict]:
        """Load sample metadata from CSV.

        Args:
            csv_path: Path to CSV file

        Returns:
            List of sample metadata dicts
        """
        samples = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    "filepath": self.real_cache_dir / row["filepath"],
                    "subject_id": row["subject_id"],
                    "z_index": int(row["z_index"]),
                    "has_lesion": row["has_lesion"].lower() == "true",
                    "source": "real",
                })

        return samples

    def _load_synthetic_samples(
        self, synth_dir: Path, csv_file: str
    ) -> list[dict]:
        """Load synthetic sample metadata.

        Args:
            synth_dir: Path to synthetic samples directory
            csv_file: CSV filename

        Returns:
            List of sample metadata dicts
        """
        samples = []
        csv_path = synth_dir / csv_file

        if not csv_path.exists():
            logger.warning(f"Synthetic CSV not found: {csv_path}")
            return samples

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Synthetic samples don't have subject_id
                samples.append({
                    "filepath": synth_dir / row["filepath"],
                    "subject_id": row.get("sample_id", "synth_unknown"),
                    "z_index": int(row.get("z_bin", 0)),
                    "has_lesion": int(row["pathology_class"]) == 1,
                    "source": "synthetic",
                })

        return samples

    def _mix_samples(
        self,
        real: list[dict],
        synthetic: list[dict],
        ratio: float,
    ) -> list[dict]:
        """Mix real and synthetic samples at specified ratio.

        Args:
            real: Real samples
            synthetic: Synthetic samples
            ratio: Ratio of synthetic to real

        Returns:
            Mixed list of samples
        """
        if ratio == 0.0 or len(synthetic) == 0:
            return real

        # Compute number of synthetic samples to add
        n_synthetic = int(len(real) * ratio)

        # Randomly sample from synthetic pool (with replacement if needed)
        if n_synthetic > 0:
            if n_synthetic <= len(synthetic):
                sampled_synthetic = random.sample(synthetic, n_synthetic)
            else:
                # Sample with replacement
                sampled_synthetic = random.choices(synthetic, k=n_synthetic)

            return real + sampled_synthetic

        return real

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            dict with keys:
                - image: (1, H, W) float32 tensor in [-1, 1]
                - mask: (1, H, W) float32 tensor in {0, 1}
                - metadata: dict with subject_id, has_lesion, etc.
        """
        sample_meta = self.samples[idx]

        # Load NPZ
        data = load_npz_sample(sample_meta["filepath"])
        image = data["image"]  # (128, 128) in [-1, 1]
        mask = data["mask"]    # (128, 128) in {-1, +1}

        # Convert mask: {-1, +1} -> {0, 1}
        mask_binary = (mask > self.mask_threshold).astype(np.float32)

        # Add channel dimension
        image = image[np.newaxis, ...].astype(np.float32)  # (1, 128, 128)
        mask_binary = mask_binary[np.newaxis, ...].astype(np.float32)  # (1, 128, 128)

        # Convert to torch
        image = torch.from_numpy(image)
        mask_binary = torch.from_numpy(mask_binary)

        sample = {
            "image": image,
            "mask": mask_binary,
            "subject_id": sample_meta["subject_id"],
            "has_lesion": sample_meta["has_lesion"],
            "source": sample_meta["source"],
        }

        # Apply transforms
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
