"""Tests for different loss configurations."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.diffusion.losses.diffusion_losses import DiffusionLoss
from src.diffusion.losses.uncertainty import SimpleWeightedLoss, UncertaintyWeightedLoss


def create_test_config(uncertainty_enabled: bool, learnable: bool = True):
    """Create test configuration with specified uncertainty settings."""
    return OmegaConf.create({
        "loss": {
            "uncertainty_weighting": {
                "enabled": uncertainty_enabled,
                "initial_log_vars": [0.0, 0.0],
                "learnable": learnable,
            },
            "lesion_weighted_mask": {
                "enabled": False,
                "lesion_weight": 2.0,
                "background_weight": 1.0,
            },
        },
    })


def test_uncertainty_weighting_enabled():
    """Test that uncertainty weighting is used when enabled."""
    cfg = create_test_config(uncertainty_enabled=True, learnable=True)
    loss_fn = DiffusionLoss(cfg)

    # Check that UncertaintyWeightedLoss is used
    assert isinstance(loss_fn.uncertainty_loss, UncertaintyWeightedLoss)
    assert loss_fn.uncertainty_loss.learnable is True

    # Test forward pass
    eps_pred = torch.randn(2, 2, 64, 64)
    eps_target = torch.randn(2, 2, 64, 64)
    mask = torch.randn(2, 1, 64, 64)

    total_loss, details = loss_fn(eps_pred, eps_target, mask)

    # Check that uncertainty-specific keys are present
    assert "weighted_loss_0" in details
    assert "weighted_loss_1" in details
    assert "log_var_0" in details
    assert "log_var_1" in details
    assert "sigma_0" in details
    assert "sigma_1" in details

    # Verify loss is finite
    assert torch.isfinite(total_loss)


def test_uncertainty_weighting_disabled():
    """Test that SimpleWeightedLoss is used when uncertainty weighting is disabled.

    When uncertainty_weighting.enabled = False, the model uses SimpleWeightedLoss
    with equal weights [1.0, 1.0], which means:
        total_loss = 1.0 * loss_image + 1.0 * loss_mask
                   = loss_image + loss_mask

    This is a simple unweighted sum of the per-channel MSE losses.
    """
    cfg = create_test_config(uncertainty_enabled=False)
    loss_fn = DiffusionLoss(cfg)

    # Check that SimpleWeightedLoss is used
    assert isinstance(loss_fn.uncertainty_loss, SimpleWeightedLoss)

    # Test forward pass
    eps_pred = torch.randn(2, 2, 64, 64)
    eps_target = torch.randn(2, 2, 64, 64)
    mask = torch.randn(2, 1, 64, 64)

    total_loss, details = loss_fn(eps_pred, eps_target, mask)

    # Check that basic loss keys are present
    assert "loss_image" in details
    assert "loss_mask" in details
    assert "weighted_loss_0" in details  # SimpleWeightedLoss still provides these
    assert "weighted_loss_1" in details

    # Check that uncertainty-specific keys are NOT present
    assert "log_var_0" not in details
    assert "log_var_1" not in details
    assert "sigma_0" not in details
    assert "sigma_1" not in details

    # Verify total loss equals sum of weighted losses
    expected_total = details["weighted_loss_0"] + details["weighted_loss_1"]
    torch.testing.assert_close(details["total_loss"], expected_total)

    # Verify weighted losses equal unweighted losses (weight=1.0 for both)
    torch.testing.assert_close(details["weighted_loss_0"], details["loss_0"])
    torch.testing.assert_close(details["weighted_loss_1"], details["loss_1"])

    # Verify loss is finite
    assert torch.isfinite(total_loss)


def test_uncertainty_weighting_not_learnable():
    """Test uncertainty weighting with learnable=False."""
    cfg = create_test_config(uncertainty_enabled=True, learnable=False)
    loss_fn = DiffusionLoss(cfg)

    # Check that UncertaintyWeightedLoss is used but not learnable
    assert isinstance(loss_fn.uncertainty_loss, UncertaintyWeightedLoss)
    assert loss_fn.uncertainty_loss.learnable is False

    # Check that log_vars do not require gradients
    assert not loss_fn.uncertainty_loss.log_vars.requires_grad

    # Test forward pass
    eps_pred = torch.randn(2, 2, 64, 64)
    eps_target = torch.randn(2, 2, 64, 64)
    mask = torch.randn(2, 1, 64, 64)

    total_loss, details = loss_fn(eps_pred, eps_target, mask)

    # Check that uncertainty-specific keys are present
    assert "weighted_loss_0" in details
    assert "weighted_loss_1" in details
    assert "log_var_0" in details
    assert "log_var_1" in details

    # Verify loss is finite
    assert torch.isfinite(total_loss)


def test_loss_compatibility_with_lightning_module():
    """Test that losses work correctly in the Lightning module context."""
    from src.diffusion.training.lit_modules import JSDDPMLightningModule

    # Test with uncertainty weighting enabled
    cfg_enabled = OmegaConf.create({
        "conditioning": {
            "z_bins": 50,
            "use_sinusoidal": False,
            "max_z": 127,
            "cfg": {"enabled": False, "null_token": 100, "dropout_prob": 0.1},
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
            "norm_num_groups": 8,
            "use_class_embedding": True,
            "dropout": 0.0,
            "resblock_updown": False,
            "with_conditioning": False,
        },
        "scheduler": {
            "type": "DDPM",
            "num_train_timesteps": 100,
            "schedule": "linear_beta",
            "beta_start": 0.0015,
            "beta_end": 0.0195,
            "prediction_type": "epsilon",
            "clip_sample": True,
            "clip_sample_range": 1.0,
        },
        "sampler": {
            "type": "DDIM",
            "num_inference_steps": 10,
            "eta": 0.0,
            "guidance_scale": 1.0,
        },
        "training": {
            "batch_size": 2,
            "optimizer": {
                "type": "Adam",
                "lr": 1e-4,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "lr_scheduler": {"type": "none"},
            "max_epochs": 100,
        },
        "loss": {
            "uncertainty_weighting": {
                "enabled": False,  # Disabled
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
            "wandb": {"enabled": False},
        },
        "visualization": {},
        "generation": {},
        "experiment": {
            "name": "test",
            "output_dir": "./test",
            "seed": 42,
        },
        "data": {
            "root_dir": "/tmp",
            "cache_dir": "/tmp/cache",
        },
    })

    module = JSDDPMLightningModule(cfg_enabled)

    # Test that SimpleWeightedLoss is used
    assert isinstance(module.criterion.uncertainty_loss, SimpleWeightedLoss)

    # Test forward pass
    batch = {
        "image": torch.randn(2, 1, 64, 64),
        "mask": torch.randn(2, 1, 64, 64),
        "token": torch.randint(0, 100, (2,)),
        "metadata": {},
    }

    loss = module.training_step(batch, 0)

    # Verify loss is finite
    assert torch.isfinite(loss)

    # Test with uncertainty weighting enabled
    cfg_enabled.loss.uncertainty_weighting.enabled = True
    module_with_uncertainty = JSDDPMLightningModule(cfg_enabled)

    # Test that UncertaintyWeightedLoss is used
    assert isinstance(module_with_uncertainty.criterion.uncertainty_loss, UncertaintyWeightedLoss)

    loss_with_uncertainty = module_with_uncertainty.training_step(batch, 0)

    # Verify loss is finite
    assert torch.isfinite(loss_with_uncertainty)
