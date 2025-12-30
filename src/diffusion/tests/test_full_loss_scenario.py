"""Comprehensive test for full loss scenario: Kendall + Lesion + Anatomical.

This test validates that when ALL three weighting mechanisms are enabled:
1. Lesion-weighted mask loss
2. Anatomical priors in training loss
3. Kendall uncertainty weighting

The normalized MSE formula is correctly applied at each level.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.diffusion.losses.diffusion_losses import DiffusionLoss
from src.diffusion.losses.uncertainty import UncertaintyWeightedLoss


def test_full_scenario_kendall_lesion_anatomical():
    """Test complete loss computation with all three weighting schemes.

    This is the CRITICAL test that validates:
    1. Image channel: Anatomical weights + Kendall
    2. Mask channel: (Lesion * Anatomical) weights + Kendall
    3. Final loss: Kendall combines both channels

    Formula breakdown:
    - loss_img = sum(w_anat * (pred - target)²) / sum(w_anat)
    - loss_msk = sum(w_lesion * w_anat * (pred - target)²) / sum(w_lesion * w_anat)
    - total = (loss_img / (2*σ_img²) + loss_msk / (2*σ_msk²)
              + 0.5*log(σ_img²) + 0.5*log(σ_msk²))
    """
    # Setup controlled test data
    torch.manual_seed(42)
    B, H, W = 2, 128, 128

    eps_pred = torch.randn(B, 2, H, W)
    eps_target = torch.randn(B, 2, H, W)

    # Create clear spatial structure
    x0_mask = torch.ones(B, 1, H, W) * -1.0  # All background
    x0_mask[:, :, H//4:3*H//4, W//4:3*W//4] = 1.0  # Center square is lesion

    spatial_weights = torch.ones(B, 1, H, W) * 0.1  # Out-of-brain
    spatial_weights[:, :, H//3:2*H//3, W//3:2*W//3] = 1.0  # Center region is in-brain

    # Configure all three weighting schemes
    cfg = OmegaConf.create({
        "loss": {
            "uncertainty_weighting": {
                "enabled": True,
                "initial_log_vars": [-1.0, -2.0],  # Non-zero for clear effect
                "learnable": False,  # Fixed for predictable test
            },
            "lesion_weighted_mask": {
                "enabled": True,
                "lesion_weight": 2.5,
                "background_weight": 1.0,
            },
            "anatomical_priors_in_train_loss": True,
        }
    })

    criterion = DiffusionLoss(cfg)

    # Compute loss using the full pipeline
    loss_total, details = criterion(eps_pred, eps_target, x0_mask, spatial_weights)

    # === MANUAL COMPUTATION FOR VERIFICATION ===

    # Step 1: Compute image channel loss (anatomical weights only)
    eps_pred_img = eps_pred[:, 0:1]
    eps_target_img = eps_target[:, 0:1]
    mse_img = (eps_pred_img - eps_target_img) ** 2

    # Normalized spatial-weighted MSE
    weighted_sum_img = (mse_img * spatial_weights).sum()
    weight_sum_img = spatial_weights.sum()
    loss_img_expected = weighted_sum_img / weight_sum_img.clamp(min=1e-8)

    print(f"\n=== IMAGE CHANNEL ===")
    print(f"Computed loss_img: {details['loss_image'].item():.6f}")
    print(f"Expected loss_img: {loss_img_expected.item():.6f}")

    assert torch.isclose(details["loss_image"], loss_img_expected, rtol=1e-5), \
        f"Image loss mismatch: {details['loss_image'].item()} != {loss_img_expected.item()}"

    # Step 2: Compute mask channel loss (COMBINED lesion * anatomical weights)
    eps_pred_msk = eps_pred[:, 1:2]
    eps_target_msk = eps_target[:, 1:2]
    mse_msk = (eps_pred_msk - eps_target_msk) ** 2

    # Create lesion weight map
    lesion_weights = torch.where(
        x0_mask > 0,
        torch.tensor(2.5),
        torch.tensor(1.0),
    )

    # COMBINE: lesion * anatomical (multiplicative)
    combined_weights = lesion_weights * spatial_weights

    # Normalized combined-weighted MSE
    weighted_sum_msk = (mse_msk * combined_weights).sum()
    weight_sum_msk = combined_weights.sum()
    loss_msk_expected = weighted_sum_msk / weight_sum_msk.clamp(min=1e-8)

    print(f"\n=== MASK CHANNEL ===")
    print(f"Computed loss_msk: {details['loss_mask'].item():.6f}")
    print(f"Expected loss_msk: {loss_msk_expected.item():.6f}")

    # Verify combined weights have expected structure
    print(f"\nCombined weight statistics:")
    print(f"  - Min weight: {combined_weights.min().item():.3f}")
    print(f"  - Max weight: {combined_weights.max().item():.3f}")
    print(f"  - Unique values: {torch.unique(combined_weights).tolist()}")

    # Expected unique values:
    # - Lesion in brain: 2.5 * 1.0 = 2.5
    # - Lesion out-of-brain: 2.5 * 0.1 = 0.25
    # - Background in brain: 1.0 * 1.0 = 1.0
    # - Background out-of-brain: 1.0 * 0.1 = 0.1
    expected_unique = {0.1, 0.25, 1.0, 2.5}
    actual_unique = set(torch.unique(combined_weights).tolist())

    # Allow for floating point variations
    for val in actual_unique:
        assert any(abs(val - exp) < 0.01 for exp in expected_unique), \
            f"Unexpected weight value: {val}"

    assert torch.isclose(details["loss_mask"], loss_msk_expected, rtol=1e-5), \
        f"Mask loss mismatch: {details['loss_mask'].item()} != {loss_msk_expected.item()}"

    # Step 3: Verify Kendall uncertainty weighting
    # Get log vars from the criterion
    log_vars = criterion.get_log_vars()
    assert log_vars is not None, "Kendall weighting should provide log_vars"

    log_var_img = log_vars[0]
    log_var_msk = log_vars[1]

    print(f"\n=== KENDALL UNCERTAINTY ===")
    print(f"log_var_img: {log_var_img.item():.6f}")
    print(f"log_var_msk: {log_var_msk.item():.6f}")

    # Expected log_vars from config (not learnable)
    assert torch.isclose(log_var_img, torch.tensor(-1.0), rtol=1e-5)
    assert torch.isclose(log_var_msk, torch.tensor(-2.0), rtol=1e-5)

    # Compute expected total loss with Kendall formula
    # L = exp(-log_var) * loss + 0.5 * log_var
    # where log_var = log(σ²)

    precision_img = torch.exp(-log_var_img)  # exp(-log_var)
    precision_msk = torch.exp(-log_var_msk)

    weighted_loss_img = precision_img * loss_img_expected + 0.5 * log_var_img
    weighted_loss_msk = precision_msk * loss_msk_expected + 0.5 * log_var_msk

    total_expected = weighted_loss_img + weighted_loss_msk

    print(f"\n=== TOTAL LOSS ===")
    print(f"Computed total: {loss_total.item():.6f}")
    print(f"Expected total: {total_expected.item():.6f}")
    print(f"  - weighted_loss_img: {weighted_loss_img.item():.6f}")
    print(f"  - weighted_loss_msk: {weighted_loss_msk.item():.6f}")

    assert torch.isclose(loss_total, total_expected, rtol=1e-4), \
        f"Total loss mismatch: {loss_total.item()} != {total_expected.item()}"

    # Step 4: Verify gradient flow through entire pipeline
    print(f"\n=== GRADIENT FLOW ===")

    eps_pred_grad = eps_pred.clone().requires_grad_(True)
    loss_grad, _ = criterion(eps_pred_grad, eps_target, x0_mask, spatial_weights)
    loss_grad.backward()

    assert eps_pred_grad.grad is not None, "Gradients should exist"
    assert torch.isfinite(eps_pred_grad.grad).all(), "Gradients should be finite"

    print(f"Gradient statistics:")
    print(f"  - Mean: {eps_pred_grad.grad.mean().item():.6e}")
    print(f"  - Std: {eps_pred_grad.grad.std().item():.6e}")
    print(f"  - Max abs: {eps_pred_grad.grad.abs().max().item():.6e}")

    # Step 5: Verify weights are correctly combined (detailed check)
    print(f"\n=== WEIGHT COMBINATION VERIFICATION ===")

    # Pick specific pixel locations and verify weights
    # Center pixel (in-brain, lesion)
    h_center, w_center = H//2, W//2
    is_lesion_center = x0_mask[0, 0, h_center, w_center] > 0
    spatial_weight_center = spatial_weights[0, 0, h_center, w_center]
    combined_center = combined_weights[0, 0, h_center, w_center]

    expected_center = (2.5 if is_lesion_center else 1.0) * spatial_weight_center.item()

    print(f"Center pixel (h={h_center}, w={w_center}):")
    print(f"  - Is lesion: {is_lesion_center}")
    print(f"  - Spatial weight: {spatial_weight_center.item():.3f}")
    print(f"  - Combined weight: {combined_center.item():.3f}")
    print(f"  - Expected: {expected_center:.3f}")

    assert abs(combined_center.item() - expected_center) < 0.01

    # Corner pixel (out-of-brain, background)
    h_corner, w_corner = 10, 10
    is_lesion_corner = x0_mask[0, 0, h_corner, w_corner] > 0
    spatial_weight_corner = spatial_weights[0, 0, h_corner, w_corner]
    combined_corner = combined_weights[0, 0, h_corner, w_corner]

    expected_corner = (2.5 if is_lesion_corner else 1.0) * spatial_weight_corner.item()

    print(f"\nCorner pixel (h={h_corner}, w={w_corner}):")
    print(f"  - Is lesion: {is_lesion_corner}")
    print(f"  - Spatial weight: {spatial_weight_corner.item():.3f}")
    print(f"  - Combined weight: {combined_corner.item():.3f}")
    print(f"  - Expected: {expected_corner:.3f}")

    assert abs(combined_corner.item() - expected_corner) < 0.01

    print("\n=== ALL CHECKS PASSED ✓ ===")
    print("Full scenario (Kendall + Lesion + Anatomical) is correctly implemented!")


def test_full_scenario_with_learnable_kendall():
    """Test full scenario with learnable Kendall parameters."""
    torch.manual_seed(42)
    B, H, W = 2, 64, 64

    eps_pred = torch.randn(B, 2, H, W)
    eps_target = torch.randn(B, 2, H, W)
    x0_mask = torch.randn(B, 1, H, W).sign()
    spatial_weights = torch.rand(B, 1, H, W)

    cfg = OmegaConf.create({
        "loss": {
            "uncertainty_weighting": {
                "enabled": True,
                "initial_log_vars": [0.0, 0.0],
                "learnable": True,  # Learnable
            },
            "lesion_weighted_mask": {
                "enabled": True,
                "lesion_weight": 2.0,
                "background_weight": 1.0,
            },
            "anatomical_priors_in_train_loss": True,
        }
    })

    criterion = DiffusionLoss(cfg)

    # Get log vars - note: get_log_vars() returns detached values
    # Check that underlying parameters exist and require grad
    assert hasattr(criterion.uncertainty_loss, 'log_vars')
    log_vars_param = criterion.uncertainty_loss.log_vars
    assert isinstance(log_vars_param, torch.nn.Parameter), "Should be nn.Parameter when learnable"
    assert log_vars_param.requires_grad, "Learnable Kendall params should require grad"

    # Forward pass
    loss, details = criterion(eps_pred, eps_target, x0_mask, spatial_weights)

    assert torch.isfinite(loss)
    assert loss.requires_grad, "Loss should require gradients"

    # Backward to update log_vars
    loss.backward()

    # Log vars parameter should have gradients
    assert log_vars_param.grad is not None
    assert torch.isfinite(log_vars_param.grad).all()

    print("Learnable Kendall scenario passed ✓")


def test_full_scenario_edge_cases():
    """Test edge cases in full scenario."""

    # Case 1: All weights zero (should not crash)
    B, H, W = 2, 32, 32
    eps_pred = torch.randn(B, 2, H, W)
    eps_target = torch.randn(B, 2, H, W)
    x0_mask = -torch.ones(B, 1, H, W)  # All background
    spatial_weights = torch.zeros(B, 1, H, W)  # All zero

    cfg = OmegaConf.create({
        "loss": {
            "uncertainty_weighting": {
                "enabled": True,
                "initial_log_vars": [0.0, 0.0],
                "learnable": False
            },
            "lesion_weighted_mask": {
                "enabled": True,
                "lesion_weight": 2.0,
                "background_weight": 1.0
            },
            "anatomical_priors_in_train_loss": True,
        }
    })

    criterion = DiffusionLoss(cfg)
    loss, _ = criterion(eps_pred, eps_target, x0_mask, spatial_weights)

    assert torch.isfinite(loss), "Should handle zero weights without NaN"

    # Case 2: Extreme weight values
    spatial_weights_extreme = torch.ones(B, 1, H, W) * 1000.0
    loss_extreme, _ = criterion(eps_pred, eps_target, x0_mask, spatial_weights_extreme)

    assert torch.isfinite(loss_extreme), "Should handle extreme weights"

    # Case 3: All lesion
    x0_mask_all_lesion = torch.ones(B, 1, H, W)
    loss_all_lesion, _ = criterion(eps_pred, eps_target, x0_mask_all_lesion, spatial_weights)

    assert torch.isfinite(loss_all_lesion)

    print("Edge cases passed ✓")


if __name__ == "__main__":
    print("="*80)
    print("Testing Full Loss Scenario: Kendall + Lesion + Anatomical")
    print("="*80)

    test_full_scenario_kendall_lesion_anatomical()
    print("\n" + "="*80 + "\n")

    test_full_scenario_with_learnable_kendall()
    print("\n" + "="*80 + "\n")

    test_full_scenario_edge_cases()
    print("\n" + "="*80)
    print("ALL FULL SCENARIO TESTS PASSED ✓✓✓")
    print("="*80)
