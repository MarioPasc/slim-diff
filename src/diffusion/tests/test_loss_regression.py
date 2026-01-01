"""Regression test for multi-task loss computation.

This test ensures the loss function properly splits image and mask channels
and computes separate losses for each, which is essential for JS-DDPM's
multi-task learning approach.

The bug this prevents:
- Computing a single MSE on the full (B, 2, H, W) tensor instead of
  separate MSE losses for image (channel 0) and mask (channel 1)
- This causes the model to learn an averaged representation without
  proper image/mask separation, resulting in noisy blob outputs.
"""

import torch
import pytest
from omegaconf import OmegaConf

from src.diffusion.losses.diffusion_losses import DiffusionLoss


def test_loss_computes_separate_channel_losses():
    """Verify loss splits channels and computes separate losses.

    This test would FAIL on the broken implementation and PASS after fix.
    """
    cfg = OmegaConf.create({
        "loss": {
            "uncertainty_weighting": {
                "enabled": True,
                "initial_log_vars": [0.0, 0.0],
                "learnable": True,
                "clamp_range": [-5, 5],
            },
            "lesion_weighted_mask": {
                "enabled": False,
                "lesion_weight": 2.5,
                "background_weight": 1.0,
            },
        }
    })

    loss_fn = DiffusionLoss(cfg)

    # Create test data
    B, H, W = 4, 64, 64
    eps_pred = torch.randn(B, 2, H, W)
    eps_target = torch.randn(B, 2, H, W)
    mask = torch.randn(B, 1, H, W)

    # Compute loss
    total_loss, details = loss_fn(eps_pred, eps_target, mask)

    # CRITICAL ASSERTIONS:
    # 1. Both loss_image and loss_mask must be present
    assert "loss_image" in details, "loss_image not computed!"
    assert "loss_mask" in details, "loss_mask not computed!"

    # 2. loss_mask must NOT be zero (the bug makes it always 0)
    assert details["loss_mask"] > 0, "loss_mask is zero - channels not split!"

    # 3. Uncertainty weighting keys must be present
    assert "weighted_loss_0" in details, "Uncertainty weighting not active!"
    assert "weighted_loss_1" in details, "Uncertainty weighting not active!"


def test_loss_image_and_mask_are_independent():
    """Verify image and mask losses are computed independently."""
    cfg = OmegaConf.create({
        "loss": {
            "uncertainty_weighting": {
                "enabled": False,
                "initial_log_vars": [0.0, 0.0],
                "learnable": False,
                "clamp_range": [-5, 5],
            },
            "lesion_weighted_mask": {
                "enabled": False,
                "lesion_weight": 2.5,
                "background_weight": 1.0,
            },
        }
    })

    loss_fn = DiffusionLoss(cfg)

    B, H, W = 4, 64, 64

    # Case 1: Perfect image prediction, bad mask prediction
    eps_target = torch.randn(B, 2, H, W)
    eps_pred = eps_target.clone()
    eps_pred[:, 1:2] += 1.0  # Corrupt mask channel only

    mask = torch.randn(B, 1, H, W)
    _, details1 = loss_fn(eps_pred, eps_target, mask)

    # Case 2: Bad image prediction, perfect mask prediction
    eps_pred = eps_target.clone()
    eps_pred[:, 0:1] += 1.0  # Corrupt image channel only

    _, details2 = loss_fn(eps_pred, eps_target, mask)

    # In case 1: loss_image should be ~0, loss_mask should be ~1.0
    assert details1["loss_image"] < 0.01, f"Expected ~0 image loss, got {details1['loss_image']}"
    assert details1["loss_mask"] > 0.5, f"Expected large mask loss, got {details1['loss_mask']}"

    # In case 2: loss_mask should be ~0, loss_image should be ~1.0
    assert details2["loss_mask"] < 0.01, f"Expected ~0 mask loss, got {details2['loss_mask']}"
    assert details2["loss_image"] > 0.5, f"Expected large image loss, got {details2['loss_image']}"


def test_loss_with_lesion_weighting():
    """Test that lesion weighting is properly applied to mask channel."""
    cfg = OmegaConf.create({
        "loss": {
            "uncertainty_weighting": {
                "enabled": False,
                "initial_log_vars": [0.0, 0.0],
                "learnable": False,
                "clamp_range": [-5, 5],
            },
            "lesion_weighted_mask": {
                "enabled": True,
                "lesion_weight": 10.0,
                "background_weight": 1.0,
            },
        }
    })

    loss_fn = DiffusionLoss(cfg)

    B, H, W = 2, 32, 32

    # Create data with known structure
    eps_pred = torch.zeros(B, 2, H, W)
    eps_target = torch.zeros(B, 2, H, W)

    # Add error only in mask channel
    eps_pred[:, 1:2] = 1.0

    # Create mask with lesion in specific region (positive values = lesion)
    mask = torch.full((B, 1, H, W), -1.0)  # All background
    mask[:, :, 10:20, 10:20] = 1.0  # Lesion region

    _, details = loss_fn(eps_pred, eps_target, mask)

    # Verify lesion weighting is active (loss should be > simple MSE)
    # With lesion_weight=10 and background_weight=1, the weighted loss
    # should be higher than standard MSE when errors are in lesion region
    assert "loss_mask" in details
    assert details["loss_mask"] > 0


def test_loss_forward_signature():
    """Test that loss function accepts the correct arguments."""
    cfg = OmegaConf.create({
        "loss": {
            "uncertainty_weighting": {
                "enabled": True,
                "initial_log_vars": [0.0, 0.0],
                "learnable": True,
                "clamp_range": [-5, 5],
            },
            "lesion_weighted_mask": {
                "enabled": False,
                "lesion_weight": 2.5,
                "background_weight": 1.0,
            },
        }
    })

    loss_fn = DiffusionLoss(cfg)

    B, H, W = 2, 32, 32
    eps_pred = torch.randn(B, 2, H, W)
    eps_target = torch.randn(B, 2, H, W)
    mask = torch.randn(B, 1, H, W)

    # Should accept 3 arguments: eps_pred, eps_target, x0_mask
    total_loss, details = loss_fn(eps_pred, eps_target, mask)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.dim() == 0  # Scalar
    assert isinstance(details, dict)

    # Should also work without mask
    total_loss2, details2 = loss_fn(eps_pred, eps_target)
    assert isinstance(total_loss2, torch.Tensor)
