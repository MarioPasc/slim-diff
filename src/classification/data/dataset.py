"""Classification dataset for real vs. synthetic patches."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """Binary classification dataset: real (label=0) vs synthetic (label=1).

    Supports three input modes:
    - "joint": 2-channel input (image + mask)
    - "image_only": 1-channel input (image patch, channel 0)
    - "mask_only": 1-channel input (mask patch, channel 1)

    Tracks sample metadata for downstream XAI analysis:
    - sample_idx: Global index in the concatenated dataset
    - original_idx: Index into the original real_patches or synth_patches array
    - is_real: Whether the sample is from real_patches (True) or synth_patches (False)

    Args:
        real_patches: Real patches array (N_real, 2, H, W).
        synth_patches: Synthetic patches array (N_synth, 2, H, W).
        real_zbins: Z-bin indices for real patches (N_real,).
        synth_zbins: Z-bin indices for synthetic patches (N_synth,).
        input_mode: Channel selection mode.
    """

    def __init__(
        self,
        real_patches: np.ndarray,
        synth_patches: np.ndarray,
        real_zbins: np.ndarray,
        synth_zbins: np.ndarray,
        input_mode: Literal["joint", "image_only", "mask_only"] = "joint",
    ) -> None:
        n_real = len(real_patches)
        n_synth = len(synth_patches)

        # Concatenate real (label=0) and synthetic (label=1)
        self.patches = np.concatenate([real_patches, synth_patches], axis=0)
        self.z_bins = np.concatenate([real_zbins, synth_zbins], axis=0)
        self.labels = np.concatenate([
            np.zeros(n_real, dtype=np.float32),
            np.ones(n_synth, dtype=np.float32),
        ])
        self.input_mode = input_mode

        # Sample tracking for confusion matrix analysis
        self.sample_indices = np.arange(n_real + n_synth, dtype=np.int32)
        self.original_indices = np.concatenate([
            np.arange(n_real, dtype=np.int32),
            np.arange(n_synth, dtype=np.int32),
        ])
        self.is_real = np.concatenate([
            np.ones(n_real, dtype=bool),
            np.zeros(n_synth, dtype=bool),
        ])

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | bool]:
        patch = self.patches[idx]  # (2, H, W)

        if self.input_mode == "image_only":
            patch = patch[0:1]  # (1, H, W)
        elif self.input_mode == "mask_only":
            patch = patch[1:2]  # (1, H, W)
        # "joint" keeps (2, H, W)

        return {
            "patch": torch.from_numpy(patch.copy()).float(),
            "label": torch.tensor(self.labels[idx]).float(),
            "z_bin": int(self.z_bins[idx]),
            # Sample tracking for confusion matrix analysis
            "sample_idx": int(self.sample_indices[idx]),
            "original_idx": int(self.original_indices[idx]),
            "is_real": bool(self.is_real[idx]),
        }

    @property
    def in_channels(self) -> int:
        """Number of input channels based on mode."""
        if self.input_mode == "joint":
            return 2
        return 1

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced sampling.

        Returns:
            Per-sample weight tensor for WeightedRandomSampler.
        """
        n_real = (self.labels == 0).sum()
        n_synth = (self.labels == 1).sum()
        w_real = len(self.labels) / (2.0 * n_real)
        w_synth = len(self.labels) / (2.0 * n_synth)
        weights = np.where(self.labels == 0, w_real, w_synth)
        return torch.from_numpy(weights).float()
