"""Tests for uncertainty-weighted loss implementation."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.diffusion.config.validation import validate_config
from src.diffusion.losses.uncertainty import UncertaintyWeightedLoss


def test_uncertainty_loss_details_consistency():
    """Test that loss details match the optimization objective."""
    criterion = UncertaintyWeightedLoss(
        n_tasks=2,
        initial_log_vars=[0.0, 0.0],
        learnable=True,
    )

    # Fixed losses for reproducibility
    losses = [torch.tensor(0.1), torch.tensor(0.2)]
    total_loss, details = criterion(losses)

    # Check all required keys exist
    required_keys = [
        "loss_0",
        "loss_1",
        "weighted_loss_0",
        "weighted_loss_1",
        "log_var_0",
        "log_var_1",
        "sigma_0",
        "sigma_1",
        "total_loss",
    ]
    for key in required_keys:
        assert key in details, f"Missing key: {key}"

    # Check total loss equals sum of weighted losses
    expected_total = details["weighted_loss_0"] + details["weighted_loss_1"]
    torch.testing.assert_close(details["total_loss"], expected_total)

    # Check log_vars are clamped to [-5, 5]
    assert -5.0 <= details["log_var_0"] <= 5.0
    assert -5.0 <= details["log_var_1"] <= 5.0

    # Check get_log_vars_clamped returns same values as in details
    clamped = criterion.get_log_vars_clamped()
    assert clamped is not None
    torch.testing.assert_close(clamped[0], details["log_var_0"])
    torch.testing.assert_close(clamped[1], details["log_var_1"])


def test_uncertainty_loss_gradient_flow():
    """Test that log_vars receive gradients when learnable."""
    criterion = UncertaintyWeightedLoss(
        n_tasks=2,
        initial_log_vars=[0.0, 0.0],
        learnable=True,
    )

    losses = [
        torch.tensor(0.1, requires_grad=True),
        torch.tensor(0.2, requires_grad=True),
    ]
    total_loss, _ = criterion(losses)

    # Backprop
    total_loss.backward()

    # Check log_vars have gradients
    assert criterion.log_vars.grad is not None
    assert criterion.log_vars.grad.shape == (2,)


def test_uncertainty_loss_not_learnable():
    """Test that get_log_vars_clamped returns None when not learnable."""
    criterion = UncertaintyWeightedLoss(
        n_tasks=2,
        initial_log_vars=[0.0, 0.0],
        learnable=False,
    )

    # Should return None
    assert criterion.get_log_vars_clamped() is None

    # log_vars should not require grad
    assert not criterion.log_vars.requires_grad


def test_validation_accepts_sinusoidal():
    """Test that use_sinusoidal=True is now accepted."""
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "use_sinusoidal": True,
                "max_z": 127,
                "z_bins": 50,
                "cfg": {"enabled": False, "null_token": 100},
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
        }
    )

    # Should not raise
    validate_config(cfg)


def test_validation_rejects_invalid_cfg_null_token():
    """Test that out-of-range null token is rejected when CFG enabled."""
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "use_sinusoidal": False,
                "max_z": 127,
                "z_bins": 50,
                "cfg": {
                    "enabled": True,  # CFG enabled
                    "null_token": 101,  # Out of range (should be < 101)
                },
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
        }
    )

    with pytest.raises(ValueError, match="null_token.*out of range"):
        validate_config(cfg)


def test_validation_passes_valid_config():
    """Test that valid config passes validation."""
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "use_sinusoidal": False,
                "max_z": 127,
                "z_bins": 50,
                "cfg": {"enabled": False, "null_token": 100},
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
        }
    )

    # Should not raise
    validate_config(cfg)


def test_validation_warns_max_z_128():
    """Test that max_z=128 triggers a warning."""
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "use_sinusoidal": False,
                "max_z": 128,  # Should warn
                "z_bins": 50,
                "cfg": {"enabled": False, "null_token": 100},
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
        }
    )

    # Should warn but not raise
    with pytest.warns(UserWarning, match="0-indexed slices"):
        validate_config(cfg)
