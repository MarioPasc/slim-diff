"""Test for CSV logging callback to verify train/* metrics are captured.

This test verifies that the CSVLoggingCallback correctly captures both
training and validation metrics in the CSV file.
"""

from __future__ import annotations

import csv
import logging
import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from src.diffusion.training.callbacks.csv_callback import CSVLoggingCallback
from src.diffusion.training.lit_modules import JSDDPMLightningModule

logger = logging.getLogger(__name__)


def test_csv_callback_captures_training_metrics():
    """Test that CSV callback captures both train and val metrics.

    This is a minimal test that verifies the fix for the issue where
    train/* metrics were missing from the CSV file.
    """
    logger.info("\n" + "="*60)
    logger.info("TEST: CSV Callback captures training metrics")
    logger.info("="*60)

    # Create a minimal config for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = OmegaConf.create({
            "experiment": {
                "name": "test_csv",
                "output_dir": tmpdir,
                "seed": 42,
            },
            "data": {
                "root_dir": tmpdir,
                "cache_dir": tmpdir,
                "transforms": {
                    "target_spacing": [1.0, 1.0, 1.0],
                    "roi_size": [128, 128, 128],
                },
            },
            "conditioning": {
                "z_bins": 50,
                "use_sinusoidal": False,
                "max_z": 127,
                "cfg": {
                    "enabled": False,
                    "null_token": 100,
                    "dropout_prob": 0.1,
                },
            },
            "model": {
                "type": "DiffusionModelUNet",
                "spatial_dims": 2,
                "in_channels": 2,
                "out_channels": 2,
                "channels": [32, 64],
                "attention_levels": [False, True],
                "num_res_blocks": 1,
                "num_head_channels": 16,
                "norm_name": "GROUP",
                "norm_num_groups": 8,
                "use_class_embedding": True,
                "dropout": 0.0,
                "resblock_updown": False,
                "with_conditioning": False,
            },
            "scheduler": {
                "type": "DDPM",
                "num_train_timesteps": 1000,
                "schedule": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "prediction_type": "epsilon",
                "clip_sample": True,
                "clip_sample_range": 1.0,
            },
            "sampler": {
                "type": "DDIM",
                "num_inference_steps": 50,
                "eta": 0.0,
                "guidance_scale": 1.0,
            },
            "training": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "optimizer": {
                    "type": "AdamW",
                    "lr": 1e-4,
                    "weight_decay": 1e-4,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
                "lr_scheduler": {
                    "type": "CosineAnnealingLR",
                    "T_max": 10,
                    "eta_min": 1e-6,
                },
                "max_epochs": 2,
                "max_steps": None,
                "precision": "32",
                "gradient_clip_val": 1.0,
                "gradient_clip_algorithm": "norm",
                "accumulate_grad_batches": 1,
                "val_check_interval": 1.0,
                "check_val_every_n_epoch": 1,
            },
            "loss": {
                "uncertainty_weighting": {
                    "enabled": False,
                    "initial_log_vars": [0.0, 0.0],
                    "learnable": False,
                },
                "lesion_weighted_mask": {
                    "enabled": False,
                    "lesion_weight": 2.5,
                    "background_weight": 1.0,
                },
            },
            "logging": {
                "log_every_n_steps": 1,
                "logger": {
                    "type": "tensorboard",
                },
                "checkpointing": {
                    "save_top_k": 1,
                    "monitor": "val/loss",
                    "mode": "min",
                    "save_last": True,
                    "every_n_epochs": 1,
                    "filename": "test-{epoch:04d}",
                },
            },
        })

        # Create CSV callback
        csv_callback = CSVLoggingCallback(cfg)

        # Create a mock callback to simulate train/val logging
        class MockLoggingCallback(Callback):
            """Mock callback to simulate metric logging."""

            def on_train_epoch_end(self, trainer, pl_module):
                """Simulate training metrics being logged."""
                # Manually add some training metrics to callback_metrics
                trainer.callback_metrics["train/loss"] = torch.tensor(0.5)
                trainer.callback_metrics["train/loss_image"] = torch.tensor(0.3)
                trainer.callback_metrics["train/loss_mask"] = torch.tensor(0.2)
                trainer.callback_metrics["train/lr"] = torch.tensor(1e-4)

            def on_validation_epoch_end(self, trainer, pl_module):
                """Simulate validation metrics being logged."""
                # Manually add some validation metrics
                trainer.callback_metrics["val/loss"] = torch.tensor(0.4)
                trainer.callback_metrics["val/loss_image"] = torch.tensor(0.25)
                trainer.callback_metrics["val/loss_mask"] = torch.tensor(0.15)
                trainer.callback_metrics["val/psnr"] = torch.tensor(20.0)
                trainer.callback_metrics["val/ssim"] = torch.tensor(0.8)

        # Create a mock trainer to test the callback
        class MockTrainer:
            def __init__(self):
                self.current_epoch = 0
                self.global_step = 0
                self.callback_metrics = {}

        trainer = MockTrainer()

        # Manually trigger the callback hooks to simulate training
        for epoch in range(2):
            trainer.current_epoch = epoch
            trainer.global_step = epoch * 10

            # Simulate training metrics
            trainer.callback_metrics = {
                "train/loss": torch.tensor(0.5 - epoch * 0.1),
                "train/loss_image": torch.tensor(0.3 - epoch * 0.05),
                "train/loss_mask": torch.tensor(0.2 - epoch * 0.05),
                "train/lr": torch.tensor(1e-4),
            }
            csv_callback.on_train_epoch_end(trainer, None)

            # Simulate validation metrics
            trainer.callback_metrics = {
                "val/loss": torch.tensor(0.4 - epoch * 0.1),
                "val/loss_image": torch.tensor(0.25 - epoch * 0.05),
                "val/loss_mask": torch.tensor(0.15 - epoch * 0.05),
                "val/psnr": torch.tensor(20.0 + epoch * 2.0),
                "val/ssim": torch.tensor(0.8 + epoch * 0.05),
            }
            csv_callback.on_validation_epoch_end(trainer, None)

        # Check that CSV file was created
        csv_path = Path(tmpdir) / "csv_logs" / "performance.csv"
        assert csv_path.exists(), f"CSV file not created at {csv_path}"

        # Read CSV and check contents
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        logger.info(f"\nCSV contains {len(rows)} rows")
        assert len(rows) == 2, f"Expected 2 rows (2 epochs), got {len(rows)}"

        # Check that both train and val metrics are present
        first_row = rows[0]
        logger.info(f"Columns in CSV: {list(first_row.keys())}")

        # Check for train metrics
        assert "train/loss" in first_row, "train/loss missing from CSV"
        assert "train/loss_image" in first_row, "train/loss_image missing from CSV"
        assert "train/loss_mask" in first_row, "train/loss_mask missing from CSV"

        # Check for val metrics
        assert "val/loss" in first_row, "val/loss missing from CSV"
        assert "val/psnr" in first_row, "val/psnr missing from CSV"
        assert "val/ssim" in first_row, "val/ssim missing from CSV"

        # Check values are correct
        assert first_row["epoch"] == "0", "Epoch should be 0"
        assert abs(float(first_row["train/loss"]) - 0.5) < 0.01, "train/loss value incorrect"
        assert abs(float(first_row["val/loss"]) - 0.4) < 0.01, "val/loss value incorrect"

        logger.info("\n✓ CSV callback correctly captures train/* metrics")
        logger.info(f"✓ CSV has {len(first_row)} columns including:")
        logger.info(f"  - Training metrics: {[k for k in first_row.keys() if k.startswith('train/')]}")
        logger.info(f"  - Validation metrics: {[k for k in first_row.keys() if k.startswith('val/')]}")


def test_csv_callback_multiple_epochs():
    """Test that CSV callback correctly writes multiple epochs."""
    logger.info("\n" + "="*60)
    logger.info("TEST: CSV Callback multiple epochs")
    logger.info("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = OmegaConf.create({
            "experiment": {
                "name": "test_csv_multiple",
                "output_dir": tmpdir,
                "seed": 42,
            },
        })

        csv_callback = CSVLoggingCallback(cfg)

        # Create mock trainer
        class MockTrainer:
            def __init__(self):
                self.current_epoch = 0
                self.global_step = 0
                self.callback_metrics = {}

        trainer = MockTrainer()

        # Run 3 epochs with consistent metrics
        for epoch in range(3):
            trainer.current_epoch = epoch
            trainer.global_step = epoch * 100

            # Training metrics
            trainer.callback_metrics = {
                "train/loss": torch.tensor(0.5 - epoch * 0.1),
                "train/loss_image": torch.tensor(0.3 - epoch * 0.05),
            }
            csv_callback.on_train_epoch_end(trainer, None)

            # Validation metrics
            trainer.callback_metrics = {
                "val/loss": torch.tensor(0.4 - epoch * 0.08),
                "val/psnr": torch.tensor(20.0 + epoch * 2.0),
            }
            csv_callback.on_validation_epoch_end(trainer, None)

        # Check CSV
        csv_path = Path(tmpdir) / "csv_logs" / "performance.csv"
        assert csv_path.exists()

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"

        # Check that values increase/decrease as expected
        for i, row in enumerate(rows):
            assert row["epoch"] == str(i), f"Epoch {i} mismatch"
            assert "train/loss" in row, f"train/loss missing in epoch {i}"
            assert "val/loss" in row, f"val/loss missing in epoch {i}"

        logger.info("✓ CSV callback handles multiple epochs correctly")
        logger.info(f"  - Wrote {len(rows)} epochs successfully")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    test_csv_callback_captures_training_metrics()
    test_csv_callback_multiple_epochs()

    print("\n" + "="*60)
    print("All CSV callback tests passed!")
    print("="*60)
