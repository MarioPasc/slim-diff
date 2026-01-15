"""PyTorch Dataset for audition patches."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AuditionDataset(Dataset):
    """Dataset for real vs synthetic classification.

    Loads pre-extracted patches and returns image+mask patches with labels.
    Label 0 = real, Label 1 = synthetic.

    Args:
        real_patches_path: Path to real patches NPZ file.
        synthetic_patches_path: Path to synthetic patches NPZ file.
        split: Dataset split ("train", "val", or "test").
        split_info: Dictionary containing split indices.
        transform: Optional transform to apply to patches.
    """

    def __init__(
        self,
        real_patches_path: Path,
        synthetic_patches_path: Path,
        split: Literal["train", "val", "test"],
        split_info: dict,
        transform: callable | None = None,
    ) -> None:
        self.split = split
        self.transform = transform

        # Load patches
        logger.info(f"Loading patches for {split} split...")
        real_data = np.load(real_patches_path)
        synth_data = np.load(synthetic_patches_path)

        # Get split indices
        real_indices = split_info[f"real_{split}_indices"]
        synth_indices = split_info[f"synthetic_{split}_indices"]

        # Extract patches for this split
        self.real_patches = real_data["patches"][real_indices]
        self.real_zbins = real_data["z_bins"][real_indices]

        self.synth_patches = synth_data["patches"][synth_indices]
        self.synth_zbins = synth_data["z_bins"][synth_indices]

        # Combine datasets
        self.patches = np.concatenate([self.real_patches, self.synth_patches], axis=0)
        self.zbins = np.concatenate([self.real_zbins, self.synth_zbins], axis=0)

        # Labels: 0 = real, 1 = synthetic
        self.labels = np.concatenate(
            [
                np.zeros(len(self.real_patches), dtype=np.int64),
                np.ones(len(self.synth_patches), dtype=np.int64),
            ]
        )

        # Source tracking for analysis
        self.sources = np.array(
            ["real"] * len(self.real_patches) + ["synthetic"] * len(self.synth_patches)
        )

        logger.info(
            f"{split} split: {len(self.real_patches)} real, "
            f"{len(self.synth_patches)} synthetic"
        )

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            Dictionary with:
                - patch: (2, H, W) tensor (image + mask channels)
                - label: 0 (real) or 1 (synthetic)
                - z_bin: Z-position bin
                - source: "real" or "synthetic"
        """
        patch = self.patches[idx].astype(np.float32)
        label = self.labels[idx]
        z_bin = self.zbins[idx]
        source = self.sources[idx]

        # Apply transform if provided
        if self.transform is not None:
            patch = self.transform(patch)

        # Convert to tensor
        patch = torch.from_numpy(patch)
        label = torch.tensor(label, dtype=torch.long)
        z_bin = torch.tensor(z_bin, dtype=torch.long)

        return {
            "patch": patch,
            "label": label,
            "z_bin": z_bin,
            "source": source,
        }

    @property
    def n_real(self) -> int:
        """Number of real samples."""
        return len(self.real_patches)

    @property
    def n_synthetic(self) -> int:
        """Number of synthetic samples."""
        return len(self.synth_patches)

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training.

        Returns:
            Tensor of shape (2,) with weights for [real, synthetic].
        """
        n_real = self.n_real
        n_synth = self.n_synthetic
        total = n_real + n_synth

        weight_real = total / (2 * n_real)
        weight_synth = total / (2 * n_synth)

        return torch.tensor([weight_real, weight_synth], dtype=torch.float32)


class PatchDataset(Dataset):
    """Simple dataset that loads from a single patches file.

    Useful for inference on new data.

    Args:
        patches_path: Path to patches NPZ file.
        label: Label to assign to all samples (0=real, 1=synthetic).
    """

    def __init__(
        self,
        patches_path: Path,
        label: int,
    ) -> None:
        data = np.load(patches_path)
        self.patches = data["patches"]
        self.zbins = data["z_bins"]
        self.label = label

        logger.info(f"Loaded {len(self.patches)} patches from {patches_path}")

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> dict:
        patch = torch.from_numpy(self.patches[idx].astype(np.float32))
        z_bin = torch.tensor(self.zbins[idx], dtype=torch.long)
        label = torch.tensor(self.label, dtype=torch.long)

        return {
            "patch": patch,
            "label": label,
            "z_bin": z_bin,
        }
