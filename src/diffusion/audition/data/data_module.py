"""Lightning DataModule for audition."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import AuditionDataset

logger = logging.getLogger(__name__)


class AuditionDataModule(pl.LightningDataModule):
    """Lightning DataModule for real vs synthetic classification.

    Handles data loading, splitting with z-bin stratification, and
    balanced sampling.

    Args:
        cfg: Configuration dictionary.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.train_cfg = cfg.training

        # Paths
        self.patches_dir = Path(cfg.output.patches_dir)
        self.splits_dir = Path(cfg.output.splits_dir)
        self.real_patches_path = self.patches_dir / "real_patches.npz"
        self.synthetic_patches_path = self.patches_dir / "synthetic_patches.npz"
        self.split_info_path = self.splits_dir / "split_info.json"

        # Datasets (set in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.split_info = None

    def prepare_data(self) -> None:
        """Create train/val/test splits if they don't exist.

        This is called only on rank 0 in distributed training.
        """
        if self.split_info_path.exists():
            logger.info(f"Loading existing split info from {self.split_info_path}")
            return

        logger.info("Creating new train/val/test splits...")
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        # Load patch metadata
        real_data = np.load(self.real_patches_path)
        synth_data = np.load(self.synthetic_patches_path)

        real_zbins = real_data["z_bins"]
        synth_zbins = synth_data["z_bins"]

        n_real = len(real_zbins)
        n_synth = len(synth_zbins)

        # Create stratified splits
        real_indices = np.arange(n_real)
        synth_indices = np.arange(n_synth)

        split_cfg = self.data_cfg.splitting
        test_ratio = split_cfg.test_ratio
        val_ratio = split_cfg.val_ratio
        random_state = split_cfg.random_state

        # Split real data
        real_train_val, real_test = train_test_split(
            real_indices,
            test_size=test_ratio,
            stratify=real_zbins,
            random_state=random_state,
        )
        val_size_adjusted = val_ratio / (1 - test_ratio)
        real_train, real_val = train_test_split(
            real_train_val,
            test_size=val_size_adjusted,
            stratify=real_zbins[real_train_val],
            random_state=random_state,
        )

        # Split synthetic data (same proportions)
        synth_train_val, synth_test = train_test_split(
            synth_indices,
            test_size=test_ratio,
            stratify=synth_zbins,
            random_state=random_state,
        )
        synth_train, synth_val = train_test_split(
            synth_train_val,
            test_size=val_size_adjusted,
            stratify=synth_zbins[synth_train_val],
            random_state=random_state,
        )

        # Verify z-bin coverage in test set
        self._verify_zbin_coverage(real_zbins[real_test], "real_test")
        self._verify_zbin_coverage(synth_zbins[synth_test], "synthetic_test")

        # Save split info
        split_info = {
            "real_train_indices": real_train.tolist(),
            "real_val_indices": real_val.tolist(),
            "real_test_indices": real_test.tolist(),
            "synthetic_train_indices": synth_train.tolist(),
            "synthetic_val_indices": synth_val.tolist(),
            "synthetic_test_indices": synth_test.tolist(),
            "n_real": n_real,
            "n_synthetic": n_synth,
            "split_ratios": {
                "test": test_ratio,
                "val": val_ratio,
                "train": 1 - test_ratio - val_ratio,
            },
        }

        with open(self.split_info_path, "w") as f:
            json.dump(split_info, f, indent=2)

        logger.info(f"Saved split info to {self.split_info_path}")
        self._log_split_statistics(split_info)

    def _verify_zbin_coverage(self, zbins: np.ndarray, name: str) -> None:
        """Verify that test set has minimum samples per z-bin."""
        min_samples = self.data_cfg.splitting.min_samples_per_zbin
        unique, counts = np.unique(zbins, return_counts=True)

        low_zbins = unique[counts < min_samples]
        if len(low_zbins) > 0:
            logger.warning(
                f"{name}: Z-bins with < {min_samples} samples: {low_zbins.tolist()}"
            )

    def _log_split_statistics(self, split_info: dict) -> None:
        """Log split statistics."""
        logger.info("Split statistics:")
        logger.info(f"  Total real: {split_info['n_real']}")
        logger.info(f"  Total synthetic: {split_info['n_synthetic']}")
        logger.info(
            f"  Train: {len(split_info['real_train_indices'])} real, "
            f"{len(split_info['synthetic_train_indices'])} synthetic"
        )
        logger.info(
            f"  Val: {len(split_info['real_val_indices'])} real, "
            f"{len(split_info['synthetic_val_indices'])} synthetic"
        )
        logger.info(
            f"  Test: {len(split_info['real_test_indices'])} real, "
            f"{len(split_info['synthetic_test_indices'])} synthetic"
        )

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for each stage.

        Args:
            stage: "fit", "validate", "test", or "predict".
        """
        # Load split info
        with open(self.split_info_path) as f:
            self.split_info = json.load(f)

        if stage == "fit" or stage is None:
            self.train_dataset = AuditionDataset(
                real_patches_path=self.real_patches_path,
                synthetic_patches_path=self.synthetic_patches_path,
                split="train",
                split_info=self.split_info,
            )
            self.val_dataset = AuditionDataset(
                real_patches_path=self.real_patches_path,
                synthetic_patches_path=self.synthetic_patches_path,
                split="val",
                split_info=self.split_info,
            )

        if stage == "test" or stage is None:
            self.test_dataset = AuditionDataset(
                real_patches_path=self.real_patches_path,
                synthetic_patches_path=self.synthetic_patches_path,
                split="test",
                split_info=self.split_info,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            pin_memory=self.train_cfg.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            num_workers=self.train_cfg.num_workers,
            pin_memory=self.train_cfg.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            num_workers=self.train_cfg.num_workers,
            pin_memory=self.train_cfg.pin_memory,
        )

    def get_patch_size(self) -> int:
        """Get the patch size from saved patches."""
        data = np.load(self.real_patches_path)
        return data["patches"].shape[-1]
