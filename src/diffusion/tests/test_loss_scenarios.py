"""Comprehensive tests for all loss weighting scenarios.

Tests that normalized MSE formula is correctly applied in all combinations:
- Anatomical priors only
- Lesion weighting only
- Both anatomical + lesion
- With/without Kendall uncertainty weighting
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.diffusion.losses.diffusion_losses import DiffusionLoss


def create_test_config(
    uncertainty: bool = False,
    lesion_weighted: bool = False,
    anatomical_priors: bool = False,
) -> dict:
    """Create test configuration."""
    return OmegaConf.create({
        "loss": {
            "uncertainty_weighting": {
                "enabled": uncertainty,
                "initial_log_vars": [0.0, 0.0],
                "learnable": False,
            },
            "lesion_weighted_mask": {
                "enabled": lesion_weighted,
                "lesion_weight": 2.5,
                "background_weight": 1.0,
            },
            "anatomical_priors_in_train_loss": anatomical_priors,
        }
    })


class TestNormalizedMSEAllScenarios:
    """Test normalized MSE formula in all weighting scenarios."""

    @pytest.fixture
    def batch_data(self):
        """Create test batch with known values."""
        B, H, W = 4, 128, 128

        # Predictions and targets
        eps_pred = torch.randn(B, 2, H, W)
        eps_target = torch.randn(B, 2, H, W)

        # Mask (half lesion, half background)
        x0_mask = torch.ones(B, 1, H, W)
        x0_mask[:, :, :, :W//2] = -1.0  # Background in {-1, +1}

        # Spatial weights (center brain region)
        spatial_weights = torch.ones(B, 1, H, W) * 0.1  # Out-of-brain
        spatial_weights[:, :, H//4:3*H//4, W//4:3*W//4] = 1.0  # In-brain center

        return eps_pred, eps_target, x0_mask, spatial_weights

    def test_scenario_1_no_weighting(self, batch_data):
        """Test standard MSE (no weighting)."""
        eps_pred, eps_target, x0_mask, spatial_weights = batch_data

        cfg = create_test_config(
            uncertainty=False,
            lesion_weighted=False,
            anatomical_priors=False,
        )
        criterion = DiffusionLoss(cfg)

        loss, details = criterion(eps_pred, eps_target, x0_mask, None)

        # Should use standard MSE for both channels
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        assert "loss_image" in details
        assert "loss_mask" in details

    def test_scenario_2_anatomical_only(self, batch_data):
        """Test anatomical priors only."""
        eps_pred, eps_target, x0_mask, spatial_weights = batch_data

        cfg = create_test_config(
            uncertainty=False,
            lesion_weighted=False,
            anatomical_priors=True,
        )
        criterion = DiffusionLoss(cfg)

        loss, details = criterion(eps_pred, eps_target, x0_mask, spatial_weights)

        # Should use spatial_weighted_mse for both channels
        assert loss.item() >= 0
        assert torch.isfinite(loss)

        # Verify normalized formula manually for image channel
        eps_pred_img = eps_pred[:, 0:1]
        eps_target_img = eps_target[:, 0:1]
        mse_img = (eps_pred_img - eps_target_img) ** 2
        weighted_sum = (mse_img * spatial_weights).sum()
        weight_sum = spatial_weights.sum()
        expected_loss_img = weighted_sum / weight_sum

        assert torch.isclose(details["loss_image"], expected_loss_img, rtol=1e-5)

    def test_scenario_3_lesion_only(self, batch_data):
        """Test lesion weighting only."""
        eps_pred, eps_target, x0_mask, spatial_weights = batch_data

        cfg = create_test_config(
            uncertainty=False,
            lesion_weighted=True,
            anatomical_priors=False,
        )
        criterion = DiffusionLoss(cfg)

        loss, details = criterion(eps_pred, eps_target, x0_mask, None)

        # Should use standard MSE for image, lesion_weighted_mse for mask
        assert loss.item() >= 0
        assert torch.isfinite(loss)

        # Verify normalized formula for mask channel
        eps_pred_msk = eps_pred[:, 1:2]
        eps_target_msk = eps_target[:, 1:2]
        lesion_weights = torch.where(
            x0_mask > 0,
            torch.tensor(2.5),
            torch.tensor(1.0),
        )
        mse_msk = (eps_pred_msk - eps_target_msk) ** 2
        weighted_sum = (mse_msk * lesion_weights).sum()
        weight_sum = lesion_weights.sum()
        expected_loss_msk = weighted_sum / weight_sum

        assert torch.isclose(details["loss_mask"], expected_loss_msk, rtol=1e-5)

    def test_scenario_4_anatomical_and_lesion(self, batch_data):
        """Test BOTH anatomical + lesion weighting (CRITICAL TEST)."""
        eps_pred, eps_target, x0_mask, spatial_weights = batch_data

        cfg = create_test_config(
            uncertainty=False,
            lesion_weighted=True,
            anatomical_priors=True,
        )
        criterion = DiffusionLoss(cfg)

        loss, details = criterion(eps_pred, eps_target, x0_mask, spatial_weights)

        # Image: anatomical weights
        # Mask: COMBINED (lesion * anatomical)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

        # Verify image channel uses anatomical weights
        eps_pred_img = eps_pred[:, 0:1]
        eps_target_img = eps_target[:, 0:1]
        mse_img = (eps_pred_img - eps_target_img) ** 2
        weighted_sum_img = (mse_img * spatial_weights).sum()
        weight_sum_img = spatial_weights.sum()
        expected_loss_img = weighted_sum_img / weight_sum_img

        assert torch.isclose(details["loss_image"], expected_loss_img, rtol=1e-5)

        # Verify mask channel uses COMBINED weights
        eps_pred_msk = eps_pred[:, 1:2]
        eps_target_msk = eps_target[:, 1:2]
        lesion_weights = torch.where(
            x0_mask > 0,
            torch.tensor(2.5),
            torch.tensor(1.0),
        )
        combined_weights = lesion_weights * spatial_weights
        mse_msk = (eps_pred_msk - eps_target_msk) ** 2
        weighted_sum_msk = (mse_msk * combined_weights).sum()
        weight_sum_msk = combined_weights.sum()
        expected_loss_msk = weighted_sum_msk / weight_sum_msk

        assert torch.isclose(details["loss_mask"], expected_loss_msk, rtol=1e-5)

    def test_scenario_5_all_enabled(self, batch_data):
        """Test anatomical + lesion + Kendall uncertainty."""
        eps_pred, eps_target, x0_mask, spatial_weights = batch_data

        cfg = create_test_config(
            uncertainty=True,
            lesion_weighted=True,
            anatomical_priors=True,
        )
        criterion = DiffusionLoss(cfg)

        loss, details = criterion(eps_pred, eps_target, x0_mask, spatial_weights)

        # All three weighting schemes active
        assert loss.item() >= 0
        assert torch.isfinite(loss)

        # Should have uncertainty weighting keys
        assert "log_var_0" in details or "weight_0" in details
        assert "loss_image" in details
        assert "loss_mask" in details

    def test_weight_combination_is_multiplicative(self, batch_data):
        """Test that combining lesion + anatomical weights is multiplicative."""
        eps_pred, eps_target, x0_mask, spatial_weights = batch_data

        cfg = create_test_config(
            uncertainty=False,
            lesion_weighted=True,
            anatomical_priors=True,
        )
        criterion = DiffusionLoss(cfg)

        loss, details = criterion(eps_pred, eps_target, x0_mask, spatial_weights)

        # Extract mask channel
        eps_pred_msk = eps_pred[:, 1:2]
        eps_target_msk = eps_target[:, 1:2]

        # Create expected weights
        lesion_weights = torch.where(x0_mask > 0, torch.tensor(2.5), torch.tensor(1.0))
        expected_combined = lesion_weights * spatial_weights

        # Check some pixel values
        # Lesion pixel in brain: 2.5 * 1.0 = 2.5
        # Lesion pixel out-of-brain: 2.5 * 0.1 = 0.25
        # Background pixel in brain: 1.0 * 1.0 = 1.0
        # Background pixel out-of-brain: 1.0 * 0.1 = 0.1

        # Verify the combined weights produce expected loss
        mse_msk = (eps_pred_msk - eps_target_msk) ** 2
        weighted_sum = (mse_msk * expected_combined).sum()
        weight_sum = expected_combined.sum()
        expected_loss = weighted_sum / weight_sum

        assert torch.isclose(details["loss_mask"], expected_loss, rtol=1e-5)

    def test_normalized_formula_prevents_scale_issues(self, batch_data):
        """Test that normalized formula prevents gradient scale issues."""
        eps_pred, eps_target, x0_mask, spatial_weights = batch_data

        # Create two configs with different weight magnitudes
        cfg1 = OmegaConf.create({
            "loss": {
                "uncertainty_weighting": {"enabled": False},
                "lesion_weighted_mask": {
                    "enabled": True,
                    "lesion_weight": 2.0,
                    "background_weight": 1.0,
                },
                "anatomical_priors_in_train_loss": False,
            }
        })

        cfg2 = OmegaConf.create({
            "loss": {
                "uncertainty_weighting": {"enabled": False},
                "lesion_weighted_mask": {
                    "enabled": True,
                    "lesion_weight": 200.0,  # 100x larger
                    "background_weight": 100.0,  # 100x larger
                },
                "anatomical_priors_in_train_loss": False,
            }
        })

        criterion1 = DiffusionLoss(cfg1)
        criterion2 = DiffusionLoss(cfg2)

        loss1, _ = criterion1(eps_pred, eps_target, x0_mask, None)
        loss2, _ = criterion2(eps_pred, eps_target, x0_mask, None)

        # With normalization, scaling all weights by same factor shouldn't change loss
        # (because weights have same relative proportions)
        assert torch.isclose(loss1, loss2, rtol=1e-4), \
            f"Normalized formula should be scale-invariant: {loss1.item()} vs {loss2.item()}"

    def test_gradient_flow_all_scenarios(self, batch_data):
        """Test gradients flow correctly in all scenarios."""
        eps_pred, eps_target, x0_mask, spatial_weights = batch_data

        scenarios = [
            (False, False, False),  # No weighting
            (False, False, True),   # Anatomical only
            (False, True, False),   # Lesion only
            (False, True, True),    # Anatomical + Lesion
            (True, True, True),     # All three
        ]

        for uncertainty, lesion, anatomical in scenarios:
            cfg = create_test_config(uncertainty, lesion, anatomical)
            criterion = DiffusionLoss(cfg)

            # Enable gradients
            eps_pred_grad = eps_pred.clone().requires_grad_(True)

            loss, _ = criterion(
                eps_pred_grad,
                eps_target,
                x0_mask,
                spatial_weights if anatomical else None,
            )

            # Backward pass should work
            loss.backward()

            # Gradients should exist and be finite
            assert eps_pred_grad.grad is not None
            assert torch.isfinite(eps_pred_grad.grad).all()

    def test_zero_weights_edge_case(self):
        """Test edge case where all weights could be zero."""
        B, H, W = 2, 128, 128

        eps_pred = torch.randn(B, 2, H, W)
        eps_target = torch.randn(B, 2, H, W)
        x0_mask = torch.ones(B, 1, H, W) * -1.0  # All background
        spatial_weights = torch.zeros(B, 1, H, W)  # All zero (edge case)

        cfg = create_test_config(
            uncertainty=False,
            lesion_weighted=True,
            anatomical_priors=True,
        )
        criterion = DiffusionLoss(cfg)

        # Should not crash due to division by zero (clamp(min=1e-8))
        loss, details = criterion(eps_pred, eps_target, x0_mask, spatial_weights)

        assert torch.isfinite(loss)

    def test_consistency_across_batch_sizes(self):
        """Test that loss computation is consistent across batch sizes."""
        # Small batch
        eps_pred_small = torch.randn(2, 2, 64, 64)
        eps_target_small = torch.randn(2, 2, 64, 64)
        x0_mask_small = torch.randn(2, 1, 64, 64).sign()
        spatial_weights_small = torch.rand(2, 1, 64, 64)

        # Large batch (same data repeated)
        eps_pred_large = eps_pred_small.repeat(4, 1, 1, 1)
        eps_target_large = eps_target_small.repeat(4, 1, 1, 1)
        x0_mask_large = x0_mask_small.repeat(4, 1, 1, 1)
        spatial_weights_large = spatial_weights_small.repeat(4, 1, 1, 1)

        cfg = create_test_config(
            uncertainty=False,
            lesion_weighted=True,
            anatomical_priors=True,
        )
        criterion = DiffusionLoss(cfg)

        loss_small, _ = criterion(
            eps_pred_small, eps_target_small, x0_mask_small, spatial_weights_small
        )
        loss_large, _ = criterion(
            eps_pred_large, eps_target_large, x0_mask_large, spatial_weights_large
        )

        # Losses should be identical (normalized formula aggregates over batch)
        assert torch.isclose(loss_small, loss_large, rtol=1e-5)


class TestLpNormModes:
    """Test Lp norm loss modes."""

    def create_lp_norm_config(
        self,
        mode: str = "mse_lp_norm",
        p: float = 2.0,
        uncertainty: bool = False,
        lesion_weighted: bool = False,
    ) -> dict:
        """Create test configuration for Lp norm modes."""
        return OmegaConf.create({
            "loss": {
                "mode": mode,
                "lp_norm": {
                    "p": p,
                },
                "uncertainty_weighting": {
                    "enabled": uncertainty,
                    "initial_log_vars": [0.0, 0.0],
                    "learnable": False,
                },
                "group_uncertainty_weighting": {
                    "enabled": True,
                    "initial_log_vars": [0.0, 0.0],
                    "learnable": False,
                    "intra_group_weights": [1.0, 1.0, 1.0],
                },
                "ffl": {
                    "loss_weight": 1.0,
                    "alpha": 1.0,
                    "patch_factor": 1,
                    "ave_spectrum": False,
                    "log_matrix": False,
                    "batch_matrix": False,
                },
                "lesion_weighted_image": {
                    "enabled": False,
                    "lesion_weight": 1.2,
                    "background_weight": 1.0,
                },
                "lesion_weighted_mask": {
                    "enabled": lesion_weighted,
                    "lesion_weight": 2.5,
                    "background_weight": 1.0,
                },
            }
        })

    def test_lp_norm_p2_equals_mse(self):
        """Test that p=2 gives same result as MSE."""
        from src.diffusion.losses.diffusion_losses import lp_norm_loss

        pred = torch.randn(4, 1, 64, 64)
        target = torch.randn(4, 1, 64, 64)

        lp_loss = lp_norm_loss(pred, target, p=2.0)
        mse_loss = torch.nn.functional.mse_loss(pred, target)

        assert torch.isclose(lp_loss, mse_loss, rtol=1e-5), \
            f"p=2 Lp norm should equal MSE: {lp_loss.item()} vs {mse_loss.item()}"

    def test_lp_norm_p1_is_mae(self):
        """Test that p=1 gives MAE."""
        from src.diffusion.losses.diffusion_losses import lp_norm_loss

        pred = torch.randn(4, 1, 64, 64)
        target = torch.randn(4, 1, 64, 64)

        lp_loss = lp_norm_loss(pred, target, p=1.0)
        mae_loss = torch.nn.functional.l1_loss(pred, target)

        assert torch.isclose(lp_loss, mae_loss, rtol=1e-5), \
            f"p=1 Lp norm should equal MAE: {lp_loss.item()} vs {mae_loss.item()}"

    def test_lesion_weighted_lp_norm(self):
        """Test lesion weighting works with Lp norm."""
        from src.diffusion.losses.diffusion_losses import lesion_weighted_lp_norm

        pred = torch.randn(4, 1, 64, 64)
        target = torch.randn(4, 1, 64, 64)
        mask = torch.ones(4, 1, 64, 64)
        mask[:, :, :, :32] = -1.0  # Left half is background

        loss = lesion_weighted_lp_norm(pred, target, mask, p=2.0, lesion_weight=2.0)

        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_mse_lp_norm_mode_initialization(self):
        """Test mse_lp_norm mode initializes correctly."""
        cfg = self.create_lp_norm_config(mode="mse_lp_norm", p=1.5)
        criterion = DiffusionLoss(cfg)

        assert criterion.mode == "mse_lp_norm"
        assert criterion.lp_p == 1.5

    def test_mse_lp_norm_forward(self):
        """Test mse_lp_norm forward pass."""
        cfg = self.create_lp_norm_config(mode="mse_lp_norm", p=1.5)
        criterion = DiffusionLoss(cfg)

        eps_pred = torch.randn(4, 2, 64, 64)
        eps_target = torch.randn(4, 2, 64, 64)
        x0_mask = torch.randn(4, 1, 64, 64).sign()

        loss, details = criterion(eps_pred, eps_target, x0_mask)

        assert loss.item() >= 0
        assert torch.isfinite(loss)
        assert "loss_image" in details
        assert "loss_mask" in details
        assert details["lp_p"] == 1.5

    def test_mse_lp_norm_ffl_groups_mode(self):
        """Test mse_lp_norm_ffl_groups mode works."""
        cfg = self.create_lp_norm_config(mode="mse_lp_norm_ffl_groups", p=1.5)
        criterion = DiffusionLoss(cfg)

        eps_pred = torch.randn(4, 2, 64, 64)
        eps_target = torch.randn(4, 2, 64, 64)
        x0_mask = torch.randn(4, 1, 64, 64).sign()
        x0 = torch.randn(4, 2, 64, 64)
        x0_pred = torch.randn(4, 2, 64, 64)

        loss, details = criterion(eps_pred, eps_target, x0_mask, x0=x0, x0_pred=x0_pred)

        assert loss.item() >= 0
        assert torch.isfinite(loss)
        assert "loss_image" in details
        assert "loss_mask" in details
        assert "loss_ffl" in details
        assert details["lp_p"] == 1.5

    def test_mse_lp_norm_with_lesion_weighting(self):
        """Test mse_lp_norm mode with lesion weighting."""
        cfg = self.create_lp_norm_config(
            mode="mse_lp_norm",
            p=1.5,
            lesion_weighted=True,
        )
        criterion = DiffusionLoss(cfg)

        eps_pred = torch.randn(4, 2, 64, 64)
        eps_target = torch.randn(4, 2, 64, 64)
        x0_mask = torch.randn(4, 1, 64, 64).sign()

        loss, details = criterion(eps_pred, eps_target, x0_mask)

        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_gradient_flow_lp_norm(self):
        """Test gradients flow correctly in Lp norm modes."""
        cfg = self.create_lp_norm_config(mode="mse_lp_norm", p=1.5)
        criterion = DiffusionLoss(cfg)

        eps_pred = torch.randn(4, 2, 64, 64, requires_grad=True)
        eps_target = torch.randn(4, 2, 64, 64)
        x0_mask = torch.randn(4, 1, 64, 64).sign()

        loss, _ = criterion(eps_pred, eps_target, x0_mask)
        loss.backward()

        assert eps_pred.grad is not None
        assert torch.isfinite(eps_pred.grad).all()
