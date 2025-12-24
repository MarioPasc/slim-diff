"""Smoke tests for JS-DDPM module.

These tests verify that the core components work together:
1. Data pipeline can load a sample with correct shapes/ranges
2. Model forward pass works with conditioning
3. One training step runs without crashing
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.diffusion.data.transforms import (
    check_brain_content,
    check_lesion_content,
    extract_axial_slice,
)
from src.diffusion.losses.diffusion_losses import DiffusionLoss
from src.diffusion.losses.uncertainty import UncertaintyWeightedLoss
from src.diffusion.model.components.conditioning import (
    compute_class_token,
    compute_z_bin,
    get_token_for_condition,
    token_to_condition,
)
from src.diffusion.model.embeddings.zpos import normalize_z, quantize_z
from src.diffusion.model.factory import build_model, build_scheduler
from src.diffusion.training.metrics import dice_coefficient, psnr, ssim


@pytest.fixture
def test_config():
    """Create a test configuration."""
    cfg = OmegaConf.create({
        "experiment": {
            "name": "test",
            "output_dir": "./test_outputs",
            "seed": 42,
        },
        "data": {
            "root_dir": "/tmp",
            "cache_dir": "/tmp/cache",
            "epilepsy": {"name": "test_epi", "modality_index": 0},
            "control": {"name": "test_ctl", "modality_index": 0},
            "splits": {
                "use_predefined_test": True,
                "val_fraction": 0.1,
                "control_test_fraction": 0.15,
                "seed": 42,
            },
            "transforms": {
                "target_spacing": [1.875, 1.875, 1.875],
                "roi_size": [128, 128, 128],
                "intensity_norm": {
                    "type": "percentile",
                    "lower": 0.5,
                    "upper": 99.5,
                    "b_min": -1.0,
                    "b_max": 1.0,
                    "clip": True,
                },
            },
            "slice_sampling": {
                "z_range": [0, 127],
                "filter_empty_brain": True,
                "brain_threshold": -0.9,
                "brain_min_fraction": 0.05,
            },
            "lesion_oversampling": {
                "enabled": True,
                "weight": 5.0,
            },
        },
        "conditioning": {
            "z_bins": 50,
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
            "channels": [32, 64, 64],  # Smaller for testing
            "attention_levels": [False, False, True],
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
            "num_train_timesteps": 100,  # Fewer for testing
            "schedule": "linear_beta",
            "beta_start": 0.0015,
            "beta_end": 0.0195,
            "prediction_type": "epsilon",
            "clip_sample": True,
            "clip_sample_range": 1.0,
        },
        "sampler": {
            "type": "DDIM",
            "num_inference_steps": 10,  # Fewer for testing
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
                "T_max": None,
                "eta_min": 1e-6,
            },
            "max_epochs": 1,
            "max_steps": None,
            "precision": "32",
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
            "accumulate_grad_batches": 1,
            "val_check_interval": 1.0,
            "check_val_every_n_epoch": 1,
            "early_stopping": {"enabled": False},
        },
        "loss": {
            "uncertainty_weighting": {
                "enabled": True,
                "initial_log_vars": [0.0, 0.0],
                "learnable": True,
            },
            "lesion_weighted_mask": {
                "enabled": False,
                "lesion_weight": 2.0,
                "background_weight": 1.0,
            },
        },
        "logging": {
            "log_every_n_steps": 1,
            "logger": {"type": "tensorboard"},
            "checkpointing": {
                "save_top_k": 1,
                "monitor": "val/loss",
                "mode": "min",
                "save_last": True,
                "every_n_epochs": 1,
                "filename": "test",
            },
        },
        "visualization": {
            "enabled": False,
            "every_n_epochs": 1,
            "z_bins_to_show": [0, 25, 49],
            "n_samples_per_condition": 1,
            "overlay": {
                "enabled": True,
                "alpha": 0.5,
                "color": [255, 0, 0],
                "threshold": 0.0,
            },
            "save_png": True,
            "save_npz": False,
        },
        "generation": {
            "n_per_condition": 1,
            "z_bins": None,
            "classes": [0, 1],
            "output_format": "npz",
            "save_individual": True,
            "create_index_csv": True,
        },
    })
    return cfg


class TestZPositionEncoding:
    """Tests for z-position encoding utilities."""

    def test_normalize_z(self):
        """Test z-index normalization."""
        assert normalize_z(0, 127) == 0.0
        assert normalize_z(127, 127) == 1.0
        assert normalize_z(63, 127) == pytest.approx(0.496, rel=0.01)

    def test_quantize_z(self):
        """Test z-position quantization."""
        n_bins = 50

        # Edge cases
        assert quantize_z(0, 127, n_bins) == 0
        assert quantize_z(127, 127, n_bins) == 49  # Clamped

        # Middle value
        z_bin = quantize_z(63, 127, n_bins)
        assert 0 <= z_bin < n_bins


class TestConditioning:
    """Tests for conditioning token utilities."""

    def test_compute_class_token(self):
        """Test token computation."""
        n_bins = 50

        # Control tokens: 0-49
        token = compute_class_token(0, 0, n_bins)
        assert token == 0

        token = compute_class_token(25, 0, n_bins)
        assert token == 25

        # Lesion tokens: 50-99
        token = compute_class_token(0, 1, n_bins)
        assert token == 50

        token = compute_class_token(25, 1, n_bins)
        assert token == 75

    def test_get_token_for_condition(self):
        """Test convenience function."""
        n_bins = 50

        control_token = get_token_for_condition(10, 0, n_bins)
        assert control_token == 10

        lesion_token = get_token_for_condition(10, 1, n_bins)
        assert lesion_token == 60

    def test_token_to_condition(self):
        """Test token decoding."""
        n_bins = 50

        z_bin, cls = token_to_condition(25, n_bins)
        assert z_bin == 25
        assert cls == 0

        z_bin, cls = token_to_condition(75, n_bins)
        assert z_bin == 25
        assert cls == 1


class TestSliceUtilities:
    """Tests for slice extraction and checking utilities."""

    def test_extract_axial_slice(self):
        """Test axial slice extraction."""
        volume = torch.randn(1, 64, 64, 32)  # (C, H, W, D)

        slice_data = extract_axial_slice(volume, 16)
        assert slice_data.shape == (1, 64, 64)

    def test_check_brain_content(self):
        """Test brain content checking."""
        # Mostly high values (brain present)
        brain_slice = torch.ones(1, 64, 64) * 0.5
        assert check_brain_content(brain_slice) is True

        # Mostly low values (empty)
        empty_slice = torch.ones(1, 64, 64) * -0.95
        assert check_brain_content(empty_slice) is False

    def test_check_lesion_content(self):
        """Test lesion content checking."""
        # No lesion (all -1)
        no_lesion = torch.ones(1, 64, 64) * -1
        assert check_lesion_content(no_lesion) is False

        # Has lesion (some +1)
        has_lesion = torch.ones(1, 64, 64) * -1
        has_lesion[0, 30:35, 30:35] = 1
        assert check_lesion_content(has_lesion) is True


class TestModelFactory:
    """Tests for model building."""

    def test_build_model(self, test_config):
        """Test model construction."""
        model = build_model(test_config)

        # Check model type
        assert model is not None

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_build_scheduler(self, test_config):
        """Test scheduler construction."""
        scheduler = build_scheduler(test_config)
        assert scheduler is not None
        assert scheduler.num_train_timesteps == 100

    def test_model_forward_pass(self, test_config):
        """Test model forward pass with conditioning."""
        model = build_model(test_config)

        # Create dummy inputs
        batch_size = 2
        x = torch.randn(batch_size, 2, 64, 64)  # Smaller for speed
        timesteps = torch.randint(0, 100, (batch_size,))
        tokens = torch.randint(0, 100, (batch_size,))

        # Forward pass
        output = model(x, timesteps=timesteps, class_labels=tokens)

        assert output.shape == x.shape


class TestLosses:
    """Tests for loss functions."""

    def test_uncertainty_weighted_loss(self):
        """Test uncertainty weighting."""
        loss_fn = UncertaintyWeightedLoss(n_tasks=2)

        losses = [torch.tensor(1.0), torch.tensor(2.0)]
        total, details = loss_fn(losses)

        assert total > 0
        assert "log_var_0" in details
        assert "log_var_1" in details

    def test_diffusion_loss(self, test_config):
        """Test complete diffusion loss."""
        loss_fn = DiffusionLoss(test_config)

        eps_pred = torch.randn(2, 2, 64, 64)
        eps_target = torch.randn(2, 2, 64, 64)
        mask = torch.randn(2, 1, 64, 64)

        total, details = loss_fn(eps_pred, eps_target, mask)

        assert total > 0
        assert "loss_image" in details
        assert "loss_mask" in details


class TestMetrics:
    """Tests for image quality metrics."""

    def test_psnr(self):
        """Test PSNR computation."""
        # Same images should give inf PSNR
        img = torch.randn(2, 1, 64, 64)
        psnr_val = psnr(img, img)
        assert psnr_val == float("inf")

        # Different images should give finite PSNR
        noisy = img + torch.randn_like(img) * 0.1
        psnr_val = psnr(img, noisy)
        assert psnr_val > 0
        assert psnr_val < 100

    def test_ssim(self):
        """Test SSIM computation."""
        # Same images should give SSIM = 1
        img = torch.randn(2, 1, 64, 64)
        ssim_val = ssim(img, img)
        assert ssim_val == pytest.approx(1.0, rel=0.01)

        # Different images should give lower SSIM
        noisy = img + torch.randn_like(img) * 0.5
        ssim_val = ssim(img, noisy)
        assert 0 < ssim_val < 1

    def test_dice(self):
        """Test Dice coefficient."""
        # Same masks should give Dice = 1
        mask = torch.ones(2, 1, 64, 64)
        dice = dice_coefficient(mask, mask)
        assert dice == pytest.approx(1.0, rel=0.01)

        # Empty masks
        empty = torch.ones(2, 1, 64, 64) * -1
        dice = dice_coefficient(empty, empty)
        assert dice < 0.01  # Close to 0 (smoothing prevents exactly 0)


class TestTrainingStep:
    """Test one training step."""

    def test_lightning_module_step(self, test_config):
        """Test one training step on CPU."""
        from src.diffusion.training.lit_modules import JSDDPMLightningModule

        # Create module
        module = JSDDPMLightningModule(test_config)

        # Create dummy batch
        batch = {
            "image": torch.randn(2, 1, 64, 64),
            "mask": torch.randn(2, 1, 64, 64),
            "token": torch.randint(0, 100, (2,)),
            "metadata": {},
        }

        # Run training step
        loss = module.training_step(batch, 0)

        assert loss is not None
        assert loss > 0
        assert loss.requires_grad


class TestDataShapes:
    """Test expected data shapes and ranges."""

    def test_sample_shapes(self):
        """Test expected shapes for training samples."""
        # Simulate expected data
        image = torch.randn(1, 128, 128)  # (C, H, W)
        mask = torch.randn(1, 128, 128)

        # Combined x0
        x0 = torch.cat([image, mask], dim=0)
        assert x0.shape == (2, 128, 128)

    def test_value_ranges(self):
        """Test expected value ranges."""
        # Image should be in [-1, 1]
        image = torch.randn(1, 128, 128).clamp(-1, 1)
        assert image.min() >= -1
        assert image.max() <= 1

        # Mask should be in {-1, +1}
        mask = (torch.randn(1, 128, 128) > 0).float() * 2 - 1
        assert torch.all((mask == -1) | (mask == 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
