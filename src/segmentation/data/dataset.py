"""Segmentation dataset for real + optional synthetic data."""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from src.segmentation.utils.io import load_npz_sample

if TYPE_CHECKING:
    from src.segmentation.data.kfold_planner import SampleRecord

logger = logging.getLogger(__name__)

# =============================================================================
# Global Normalization Configuration
# =============================================================================
# Options: "percentile" (keep [-1, 1] from preprocessing) or "zscore" (mean=0, std=1)
IM_NORMALIZATION_APPROACH: Literal["percentile", "zscore"] = "percentile"


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image according to global IM_NORMALIZATION_APPROACH setting.

    Args:
        image: Input image array (H, W) or (C, H, W) in [-1, 1] range.

    Returns:
        Normalized image:
        - "percentile": unchanged (already in [-1, 1])
        - "zscore": per-image z-score normalization (mean=0, std=1)
    """
    if IM_NORMALIZATION_APPROACH == "percentile":
        # Images are already in [-1, 1] from preprocessing
        return image
    elif IM_NORMALIZATION_APPROACH == "zscore":
        # Per-image z-score normalization
        mean = image.mean()
        std = image.std()
        if std < 1e-8:
            # Avoid division by zero for constant images
            return image - mean
        return (image - mean) / std
    else:
        raise ValueError(
            f"Unknown normalization approach: {IM_NORMALIZATION_APPROACH}. "
            "Expected 'percentile' or 'zscore'."
        )


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

        Raises:
            ValueError: If samples is empty or directories don't exist
        """
        # Validate inputs
        if not samples:
            raise ValueError("Cannot create dataset with empty samples list")

        self.samples = samples
        self.real_cache_dir = Path(real_cache_dir)
        self.synthetic_dir = Path(synthetic_dir) if synthetic_dir else None
        self.transform = transform
        self.mask_threshold = mask_threshold

        # Validate directories exist
        n_real = sum(1 for s in samples if s.source == "real")
        n_synth = sum(1 for s in samples if s.source == "synthetic")

        if n_real > 0 and not self.real_cache_dir.exists():
            raise ValueError(f"Real cache directory does not exist: {self.real_cache_dir}")

        if n_synth > 0 and self.synthetic_dir is None:
            raise ValueError(
                f"Dataset contains {n_synth} synthetic samples but synthetic_dir is None. "
                "This usually happens in synthetic_only mode - make sure to pass synthetic_dir."
            )

        if n_synth > 0 and not self.synthetic_dir.exists():
            raise ValueError(f"Synthetic directory does not exist: {self.synthetic_dir}")

        # Cache for replica arrays loaded into RAM
        # Key: replica_name, Value: {"images": ndarray, "masks": ndarray}
        # Preloading entire replicas into RAM avoids slow random access on external drives
        self._replica_cache = {}

        # Preload all unique replicas into RAM for fast access
        if n_synth > 0:
            self._preload_replicas()

        # Log dataset info
        logger.info(f"PlannedFoldDataset: {len(samples)} samples ({n_real} real, {n_synth} synthetic)")

    def _preload_replicas(self):
        """Preload all unique replica files into RAM for fast access.

        Loads entire image and mask arrays from replica NPZ files into memory.
        This avoids slow random access on external drives during training.

        Memory usage: ~1.2GB per replica (9000 images × 128×128 × 2 arrays × 4 bytes)
        Supports up to 10 replicas (~12GB RAM total).
        """
        # Find all unique replica names
        unique_replicas = set()
        for sample in self.samples:
            if sample.source == "synthetic":
                replica_name = sample.filepath.split(":")[0]
                unique_replicas.add(replica_name)

        if not unique_replicas:
            return

        logger.info(f"Preloading {len(unique_replicas)} replica file(s) into RAM...")

        for replica_name in sorted(unique_replicas):
            replica_path = self.synthetic_dir / replica_name

            # Check file size
            file_size_mb = replica_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Loading {replica_name} ({file_size_mb:.1f} MB)...")

            # Load entire arrays into RAM
            with np.load(replica_path, mmap_mode=None) as data:
                # Copy to memory immediately (no memory mapping)
                images = np.array(data["images"], dtype=np.float32)
                masks = np.array(data["masks"], dtype=np.float32)

            self._replica_cache[replica_name] = {
                "images": images,
                "masks": masks,
            }

            logger.info(f"    Loaded {len(images)} samples into RAM")

        total_samples = sum(len(data["images"]) for data in self._replica_cache.values())
        ram_usage_gb = sum(
            data["images"].nbytes + data["masks"].nbytes
            for data in self._replica_cache.values()
        ) / (1024**3)

        logger.info(
            f"Preloading complete: {total_samples} total samples, "
            f"{ram_usage_gb:.2f} GB RAM used"
        )

    def _get_replica_data(self, replica_name: str) -> dict:
        """Get cached replica data arrays.

        Args:
            replica_name: Name of the replica file

        Returns:
            Dict with "images" and "masks" arrays
        """
        if replica_name not in self._replica_cache:
            raise RuntimeError(
                f"Replica {replica_name} not in cache. This should not happen - "
                "replicas should be preloaded in __init__."
            )
        return self._replica_cache[replica_name]

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
        try:
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

                # Get cached replica data (preloaded in RAM)
                replica_data = self._get_replica_data(replica_name)
                # Direct array access - no I/O, data is already in RAM
                image = replica_data["images"][sample_idx]
                mask = replica_data["masks"][sample_idx]

            # Apply image normalization (zscore or percentile)
            image = normalize_image(image)

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

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            logger.error(f"  Sample info: {self.samples[idx]}")
            raise RuntimeError(f"Failed to load sample {idx}: {e}") from e


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

        # Apply image normalization (zscore or percentile)
        image = normalize_image(image)

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
