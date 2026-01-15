"""Patch extraction utilities for audition.

This module extracts lesion-centered patches from both real and synthetic
datasets for training the real vs synthetic classifier.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PatchInfo:
    """Information about an extracted patch."""

    source: Literal["real", "synthetic"]
    z_bin: int
    original_filepath: str
    replica_id: int | None = None
    sample_idx: int | None = None
    bbox: tuple[int, int, int, int] = field(default_factory=tuple)  # y1, y2, x1, x2


@dataclass
class ExtractionStats:
    """Statistics from patch extraction."""

    n_real: int
    n_synthetic: int
    max_lesion_height: int
    max_lesion_width: int
    patch_size: int
    zbin_distribution_real: dict[int, int]
    zbin_distribution_synthetic: dict[int, int]


class PatchExtractor:
    """Extract lesion-centered patches from real and synthetic data.

    Args:
        cfg: Configuration dictionary containing data paths and extraction settings.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.patch_cfg = cfg.data.patch_extraction

        # Paths
        self.real_cache_dir = Path(self.data_cfg.real.cache_dir)
        self.real_slices_dir = self.real_cache_dir / self.data_cfg.real.slices_subdir
        self.synthetic_dir = Path(self.data_cfg.synthetic.replicas_dir)
        self.replica_ids = self.data_cfg.synthetic.replica_ids

        # Settings
        self.padding = self.patch_cfg.padding
        self.min_patch_size = self.patch_cfg.min_patch_size
        self.max_patch_size = self.patch_cfg.max_patch_size
        self.method = self.patch_cfg.method

    def extract_all(
        self,
        output_dir: Path,
        balance_classes: bool = True,
    ) -> ExtractionStats:
        """Extract patches from both datasets.

        Args:
            output_dir: Directory to save extracted patches.
            balance_classes: If True, sample equal number from synthetic.

        Returns:
            Extraction statistics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Scan both datasets to find lesion samples and compute max size
        logger.info("Scanning real dataset for lesion samples...")
        real_samples, real_bboxes = self._scan_real_dataset()
        logger.info(f"Found {len(real_samples)} real lesion samples")

        logger.info("Scanning synthetic dataset for lesion samples...")
        synth_samples, synth_bboxes = self._scan_synthetic_dataset()
        logger.info(f"Found {len(synth_samples)} synthetic lesion samples")

        # Step 2: Compute patch size
        all_bboxes = real_bboxes + synth_bboxes
        if self.method == "dynamic":
            patch_size = self._compute_dynamic_patch_size(all_bboxes)
        else:
            patch_size = self.patch_cfg.fixed_size
        logger.info(f"Using patch size: {patch_size}x{patch_size}")

        # Step 3: Balance classes if requested
        n_real = len(real_samples)
        if balance_classes and len(synth_samples) > n_real:
            logger.info(f"Balancing: sampling {n_real} synthetic samples to match real")
            rng = np.random.default_rng(self.cfg.experiment.seed)
            indices = rng.choice(len(synth_samples), size=n_real, replace=False)
            synth_samples = [synth_samples[i] for i in indices]
            synth_bboxes = [synth_bboxes[i] for i in indices]

        # Step 4: Extract patches
        logger.info("Extracting real patches...")
        real_patches, real_infos = self._extract_patches(
            real_samples, real_bboxes, patch_size, source="real"
        )

        logger.info("Extracting synthetic patches...")
        synth_patches, synth_infos = self._extract_patches(
            synth_samples, synth_bboxes, patch_size, source="synthetic"
        )

        # Step 5: Save patches
        self._save_patches(output_dir / "real_patches.npz", real_patches, real_infos)
        self._save_patches(output_dir / "synthetic_patches.npz", synth_patches, synth_infos)

        # Compute statistics
        stats = self._compute_stats(
            real_patches, synth_patches, real_infos, synth_infos, all_bboxes, patch_size
        )

        # Save statistics
        stats_path = output_dir / "extraction_stats.json"
        self._save_stats(stats_path, stats)

        return stats

    def _scan_real_dataset(self) -> tuple[list[dict], list[tuple[int, int, int, int]]]:
        """Scan real dataset for lesion samples.

        Returns:
            Tuple of (sample_list, bbox_list) where each sample contains
            image, mask, and metadata.
        """
        samples = []
        bboxes = []

        # Load CSV indices
        for csv_file in self.data_cfg.real.csv_files:
            csv_path = self.real_cache_dir / csv_file
            if not csv_path.exists():
                logger.warning(f"CSV file not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            # Filter for lesion slices
            lesion_df = df[df["has_lesion"] == True]  # noqa: E712

            for _, row in tqdm(lesion_df.iterrows(), total=len(lesion_df), desc=f"Scanning {csv_file}"):
                filepath = self.real_slices_dir / Path(row["filepath"]).name
                if not filepath.exists():
                    continue

                data = np.load(filepath)
                mask = data["mask"]

                # Convert mask from {-1, +1} to {0, 1} for bbox computation
                binary_mask = (mask > 0).astype(np.uint8)

                # Skip if no actual lesion pixels
                if binary_mask.sum() == 0:
                    continue

                bbox = self._compute_bbox(binary_mask)
                bboxes.append(bbox)

                samples.append(
                    {
                        "filepath": str(filepath),
                        "z_bin": int(row["z_bin"]),
                        "image": data["image"],
                        "mask": mask,
                    }
                )

        return samples, bboxes

    def _scan_synthetic_dataset(self) -> tuple[list[dict], list[tuple[int, int, int, int]]]:
        """Scan synthetic dataset for lesion samples.

        Returns:
            Tuple of (sample_list, bbox_list).
        """
        samples = []
        bboxes = []

        for replica_id in tqdm(self.replica_ids, desc="Scanning replicas"):
            replica_path = self.synthetic_dir / f"replica_{replica_id:03d}.npz"
            if not replica_path.exists():
                logger.warning(f"Replica not found: {replica_path}")
                continue

            data = np.load(replica_path)
            images = data["images"]
            masks = data["masks"]
            zbins = data["zbin"]
            lesion_present = data["lesion_present"]

            # Filter for lesion samples
            lesion_indices = np.where(lesion_present == 1)[0]

            for idx in lesion_indices:
                mask = masks[idx]

                # Convert mask from {-1, +1} to {0, 1}
                binary_mask = (mask > 0).astype(np.uint8)

                # Skip if no actual lesion pixels (empty generation)
                if binary_mask.sum() == 0:
                    continue

                bbox = self._compute_bbox(binary_mask)
                bboxes.append(bbox)

                samples.append(
                    {
                        "replica_id": replica_id,
                        "sample_idx": int(idx),
                        "z_bin": int(zbins[idx]),
                        "image": images[idx],
                        "mask": mask,
                    }
                )

        return samples, bboxes

    def _compute_bbox(self, binary_mask: np.ndarray) -> tuple[int, int, int, int]:
        """Compute bounding box of non-zero region.

        Args:
            binary_mask: Binary mask array (H, W).

        Returns:
            Tuple of (y1, y2, x1, x2) bounding box coordinates.
        """
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return (int(y1), int(y2), int(x1), int(x2))

    def _compute_dynamic_patch_size(
        self, bboxes: list[tuple[int, int, int, int]]
    ) -> int:
        """Compute patch size based on maximum lesion dimensions.

        Args:
            bboxes: List of bounding boxes (y1, y2, x1, x2).

        Returns:
            Patch size (square).
        """
        max_height = 0
        max_width = 0

        for y1, y2, x1, x2 in bboxes:
            height = y2 - y1 + 1
            width = x2 - x1 + 1
            max_height = max(max_height, height)
            max_width = max(max_width, width)

        # Add padding and round to nearest multiple of 8
        max_dim = max(max_height, max_width) + 2 * self.padding
        patch_size = int(np.ceil(max_dim / 8) * 8)

        # Clamp to configured range
        patch_size = max(self.min_patch_size, min(patch_size, self.max_patch_size))

        logger.info(f"Max lesion size: {max_height}x{max_width}, computed patch size: {patch_size}")

        return patch_size

    def _extract_patches(
        self,
        samples: list[dict],
        bboxes: list[tuple[int, int, int, int]],
        patch_size: int,
        source: Literal["real", "synthetic"],
    ) -> tuple[np.ndarray, list[PatchInfo]]:
        """Extract patches centered on lesions.

        Args:
            samples: List of sample dictionaries.
            bboxes: List of bounding boxes.
            patch_size: Size of square patches.
            source: Source identifier ("real" or "synthetic").

        Returns:
            Tuple of (patches_array, patch_infos) where patches_array has shape
            (N, 2, H, W) with image and mask channels.
        """
        patches = []
        infos = []

        for sample, bbox in tqdm(
            zip(samples, bboxes), total=len(samples), desc=f"Extracting {source} patches"
        ):
            image = sample["image"]
            mask = sample["mask"]

            # Compute lesion centroid from bbox
            y1, y2, x1, x2 = bbox
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2

            # Extract patch centered on lesion
            half_size = patch_size // 2
            patch_image = self._extract_centered_patch(image, cy, cx, patch_size)
            patch_mask = self._extract_centered_patch(mask, cy, cx, patch_size)

            # Stack image and mask as channels
            patch = np.stack([patch_image, patch_mask], axis=0)
            patches.append(patch)

            # Create patch info
            info = PatchInfo(
                source=source,
                z_bin=sample["z_bin"],
                original_filepath=sample.get("filepath", ""),
                replica_id=sample.get("replica_id"),
                sample_idx=sample.get("sample_idx"),
                bbox=bbox,
            )
            infos.append(info)

        patches_array = np.stack(patches, axis=0).astype(np.float32)
        return patches_array, infos

    def _extract_centered_patch(
        self,
        image: np.ndarray,
        cy: int,
        cx: int,
        patch_size: int,
    ) -> np.ndarray:
        """Extract a patch centered at (cy, cx) with padding if needed.

        Args:
            image: Input image (H, W).
            cy: Center y coordinate.
            cx: Center x coordinate.
            patch_size: Size of square patch.

        Returns:
            Extracted patch (patch_size, patch_size).
        """
        h, w = image.shape
        half_size = patch_size // 2

        # Compute extraction coordinates
        y1 = cy - half_size
        y2 = cy + half_size
        x1 = cx - half_size
        x2 = cx + half_size

        # Handle boundary cases with padding
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - h)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - w)

        # Clip coordinates to valid range
        y1_clipped = max(0, y1)
        y2_clipped = min(h, y2)
        x1_clipped = max(0, x1)
        x2_clipped = min(w, x2)

        # Extract valid region
        patch = image[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

        # Pad if needed
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            patch = np.pad(
                patch,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=-1.0,  # Background value
            )

        return patch

    def _save_patches(
        self,
        filepath: Path,
        patches: np.ndarray,
        infos: list[PatchInfo],
    ) -> None:
        """Save patches and metadata to NPZ file.

        Args:
            filepath: Output file path.
            patches: Patches array (N, 2, H, W).
            infos: List of patch info objects.
        """
        # Convert infos to arrays
        z_bins = np.array([info.z_bin for info in infos], dtype=np.int32)
        sources = np.array([info.source for info in infos], dtype="U10")
        filepaths = np.array([info.original_filepath for info in infos], dtype="U256")
        replica_ids = np.array(
            [info.replica_id if info.replica_id is not None else -1 for info in infos],
            dtype=np.int32,
        )
        sample_indices = np.array(
            [info.sample_idx if info.sample_idx is not None else -1 for info in infos],
            dtype=np.int32,
        )

        np.savez_compressed(
            filepath,
            patches=patches,
            z_bins=z_bins,
            sources=sources,
            filepaths=filepaths,
            replica_ids=replica_ids,
            sample_indices=sample_indices,
        )

        logger.info(f"Saved {len(patches)} patches to {filepath}")

    def _compute_stats(
        self,
        real_patches: np.ndarray,
        synth_patches: np.ndarray,
        real_infos: list[PatchInfo],
        synth_infos: list[PatchInfo],
        all_bboxes: list[tuple[int, int, int, int]],
        patch_size: int,
    ) -> ExtractionStats:
        """Compute extraction statistics."""
        # Max lesion dimensions
        max_height = max(y2 - y1 + 1 for y1, y2, _, _ in all_bboxes)
        max_width = max(x2 - x1 + 1 for _, _, x1, x2 in all_bboxes)

        # Z-bin distributions
        real_zbins = {}
        for info in real_infos:
            real_zbins[info.z_bin] = real_zbins.get(info.z_bin, 0) + 1

        synth_zbins = {}
        for info in synth_infos:
            synth_zbins[info.z_bin] = synth_zbins.get(info.z_bin, 0) + 1

        return ExtractionStats(
            n_real=len(real_patches),
            n_synthetic=len(synth_patches),
            max_lesion_height=max_height,
            max_lesion_width=max_width,
            patch_size=patch_size,
            zbin_distribution_real=real_zbins,
            zbin_distribution_synthetic=synth_zbins,
        )

    def _save_stats(self, filepath: Path, stats: ExtractionStats) -> None:
        """Save extraction statistics to JSON."""
        stats_dict = {
            "n_real": stats.n_real,
            "n_synthetic": stats.n_synthetic,
            "max_lesion_height": stats.max_lesion_height,
            "max_lesion_width": stats.max_lesion_width,
            "patch_size": stats.patch_size,
            "zbin_distribution_real": stats.zbin_distribution_real,
            "zbin_distribution_synthetic": stats.zbin_distribution_synthetic,
        }

        with open(filepath, "w") as f:
            json.dump(stats_dict, f, indent=2)

        logger.info(f"Saved extraction stats to {filepath}")
