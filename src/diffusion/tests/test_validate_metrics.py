"""Test for validating metrics (PSNR, SSIM, Dice, HD95).

This test creates synthetic data with known properties and verifies that
the metrics computed by MetricsCalculator produce expected results.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from src.diffusion.training.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class TestMetricsValidation:
    """Test suite for validating metrics computation."""

    @pytest.fixture
    def metrics_calc(self):
        """Create a MetricsCalculator instance."""
        return MetricsCalculator(data_range=2.0)

    def test_perfect_reconstruction(self, metrics_calc):
        """Test metrics with perfect reconstruction (pred == target)."""
        logger.info("\n" + "="*60)
        logger.info("TEST: Perfect reconstruction")
        logger.info("="*60)

        # Create identical images and masks
        B, H, W = 4, 128, 128
        image = torch.randn(B, 1, H, W) * 0.5  # Random image in [-1, 1] range
        mask = torch.randn(B, 1, H, W).sign()  # Random binary mask

        # Compute metrics (pred == target)
        metrics = metrics_calc.compute_all(
            pred_image=image,
            target_image=image,
            pred_mask=mask,
            target_mask=mask,
        )

        logger.info(f"PSNR: {metrics['psnr']:.2f} dB (expected: inf or very high)")
        logger.info(f"SSIM: {metrics['ssim']:.4f} (expected: 1.0)")
        logger.info(f"Dice: {metrics['dice']:.4f} (expected: 1.0)")
        logger.info(f"HD95: {metrics['hd95']:.4f} (expected: 0.0 or nan)")

        # Assertions
        assert metrics['psnr'] > 40, f"PSNR should be very high for perfect match, got {metrics['psnr']}"
        assert metrics['ssim'] > 0.99, f"SSIM should be ~1.0 for perfect match, got {metrics['ssim']}"
        assert metrics['dice'] > 0.99, f"Dice should be 1.0 for perfect match, got {metrics['dice']}"

        logger.info("✓ Perfect reconstruction test passed")

    def test_random_noise(self, metrics_calc):
        """Test metrics with added noise."""
        logger.info("\n" + "="*60)
        logger.info("TEST: Random noise")
        logger.info("="*60)

        B, H, W = 4, 128, 128

        # Create target
        target_image = torch.randn(B, 1, H, W) * 0.5
        target_mask = (torch.randn(B, 1, H, W) > 0).float() * 2 - 1  # Binary [-1, 1]

        # Add noise to create predictions
        noise_level = 0.1
        pred_image = target_image + torch.randn_like(target_image) * noise_level

        # Slightly perturbed mask
        pred_mask = target_mask.clone()
        # Flip 5% of pixels
        flip_mask = torch.rand_like(pred_mask) < 0.05
        pred_mask[flip_mask] = -pred_mask[flip_mask]

        # Compute metrics
        metrics = metrics_calc.compute_all(
            pred_image=pred_image,
            target_image=target_image,
            pred_mask=pred_mask,
            target_mask=target_mask,
        )

        logger.info(f"PSNR: {metrics['psnr']:.2f} dB")
        logger.info(f"SSIM: {metrics['ssim']:.4f}")
        logger.info(f"Dice: {metrics['dice']:.4f}")
        logger.info(f"HD95: {metrics['hd95']:.4f}")

        # Sanity checks
        assert 0 < metrics['psnr'] < 50, f"PSNR out of reasonable range: {metrics['psnr']}"
        assert 0 < metrics['ssim'] < 1, f"SSIM should be in (0, 1), got {metrics['ssim']}"
        assert 0.5 < metrics['dice'] < 1.0, f"Dice should be high with 5% noise, got {metrics['dice']}"

        logger.info("✓ Random noise test passed")

    def test_empty_masks(self, metrics_calc):
        """Test metrics with empty masks (control samples)."""
        logger.info("\n" + "="*60)
        logger.info("TEST: Empty masks (control samples)")
        logger.info("="*60)

        B, H, W = 4, 128, 128

        # Create images
        target_image = torch.randn(B, 1, H, W) * 0.5
        pred_image = target_image + torch.randn_like(target_image) * 0.05

        # Empty masks (all -1)
        target_mask = torch.ones(B, 1, H, W) * -1.0
        pred_mask = torch.ones(B, 1, H, W) * -1.0

        # Compute metrics
        metrics = metrics_calc.compute_all(
            pred_image=pred_image,
            target_image=target_image,
            pred_mask=pred_mask,
            target_mask=target_mask,
        )

        logger.info(f"PSNR: {metrics['psnr']:.2f} dB")
        logger.info(f"SSIM: {metrics['ssim']:.4f}")
        logger.info(f"Dice: {metrics['dice']:.4f}")
        logger.info(f"HD95: {metrics['hd95']}")

        # For empty masks, Dice may be nan or 1.0 depending on MONAI version
        # HD95 should be nan (no surfaces to compute distance)
        assert np.isnan(metrics['dice']) or metrics['dice'] > 0.99, \
            f"Dice should be nan or 1.0 for matching empty masks, got {metrics['dice']}"
        assert np.isnan(metrics['hd95']), f"HD95 should be nan for empty masks, got {metrics['hd95']}"

        logger.info("✓ Empty masks test passed")

    def test_full_masks(self, metrics_calc):
        """Test metrics with full masks (all lesion)."""
        logger.info("\n" + "="*60)
        logger.info("TEST: Full masks (all lesion)")
        logger.info("="*60)

        B, H, W = 4, 128, 128

        # Create images
        target_image = torch.randn(B, 1, H, W) * 0.5
        pred_image = target_image.clone()

        # Full masks (all +1)
        target_mask = torch.ones(B, 1, H, W)
        pred_mask = torch.ones(B, 1, H, W)

        # Compute metrics
        metrics = metrics_calc.compute_all(
            pred_image=pred_image,
            target_image=target_image,
            pred_mask=pred_mask,
            target_mask=target_mask,
        )

        logger.info(f"PSNR: {metrics['psnr']:.2f} dB")
        logger.info(f"SSIM: {metrics['ssim']:.4f}")
        logger.info(f"Dice: {metrics['dice']:.4f}")
        logger.info(f"HD95: {metrics['hd95']:.4f}")

        # For identical full masks
        assert metrics['dice'] > 0.99, f"Dice should be 1.0 for matching full masks, got {metrics['dice']}"
        # HD95 may be nan if scipy is not installed
        assert np.isnan(metrics['hd95']) or metrics['hd95'] < 1.0, \
            f"HD95 should be ~0 or nan (if scipy missing) for identical masks, got {metrics['hd95']}"

        logger.info("✓ Full masks test passed")

    def test_partial_overlap(self, metrics_calc):
        """Test metrics with partial mask overlap."""
        logger.info("\n" + "="*60)
        logger.info("TEST: Partial mask overlap")
        logger.info("="*60)

        B, H, W = 1, 128, 128

        # Create images
        target_image = torch.randn(B, 1, H, W) * 0.5
        pred_image = target_image.clone()

        # Create masks with known overlap
        target_mask = torch.ones(B, 1, H, W) * -1.0
        pred_mask = torch.ones(B, 1, H, W) * -1.0

        # Target: circle in center
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        center_y, center_x = H // 2, W // 2
        radius = 30
        target_circle = ((y - center_y)**2 + (x - center_x)**2) < radius**2
        target_mask[0, 0, target_circle] = 1.0

        # Pred: circle offset by 10 pixels
        pred_center_y, pred_center_x = center_y + 10, center_x + 10
        pred_circle = ((y - pred_center_y)**2 + (x - pred_center_x)**2) < radius**2
        pred_mask[0, 0, pred_circle] = 1.0

        # Compute metrics
        metrics = metrics_calc.compute_all(
            pred_image=pred_image,
            target_image=target_image,
            pred_mask=pred_mask,
            target_mask=target_mask,
        )

        logger.info(f"PSNR: {metrics['psnr']:.2f} dB")
        logger.info(f"SSIM: {metrics['ssim']:.4f}")
        logger.info(f"Dice: {metrics['dice']:.4f} (expected: 0.5-0.8)")
        logger.info(f"HD95: {metrics['hd95']:.4f} (expected: ~10-15)")

        # With 10 pixel offset, Dice should be moderate
        assert 0.3 < metrics['dice'] < 0.9, f"Dice unexpected for partial overlap: {metrics['dice']}"
        # HD95 should be roughly the offset distance (or nan if scipy not installed)
        if not np.isnan(metrics['hd95']):
            assert 5 < metrics['hd95'] < 25, f"HD95 unexpected for 10px offset: {metrics['hd95']}"
        else:
            logger.warning("HD95 is nan (scipy may not be installed)")

        logger.info("✓ Partial overlap test passed")

    def test_value_ranges(self, metrics_calc):
        """Test that metrics handle different value ranges correctly."""
        logger.info("\n" + "="*60)
        logger.info("TEST: Value ranges")
        logger.info("="*60)

        B, H, W = 2, 128, 128

        # Test with extreme values in [-1, 1] range
        target_image = torch.ones(B, 1, H, W) * 0.8
        pred_image = torch.ones(B, 1, H, W) * 0.7

        target_mask = torch.ones(B, 1, H, W) * -1.0
        pred_mask = torch.ones(B, 1, H, W) * -1.0

        # Compute metrics
        metrics = metrics_calc.compute_all(
            pred_image=pred_image,
            target_image=target_image,
            pred_mask=pred_mask,
            target_mask=target_mask,
        )

        logger.info(f"PSNR: {metrics['psnr']:.2f} dB")
        logger.info(f"SSIM: {metrics['ssim']:.4f}")

        # PSNR should be finite and reasonable
        assert not np.isnan(metrics['psnr']), "PSNR should not be nan"
        assert not np.isinf(metrics['psnr']), "PSNR should not be inf for different values"
        assert 0 < metrics['ssim'] <= 1, f"SSIM out of range: {metrics['ssim']}"

        logger.info("✓ Value ranges test passed")

    def test_batch_consistency(self, metrics_calc):
        """Test that metrics are consistent across batch sizes."""
        logger.info("\n" + "="*60)
        logger.info("TEST: Batch consistency")
        logger.info("="*60)

        H, W = 128, 128

        # Create single sample
        target_image_single = torch.randn(1, 1, H, W) * 0.5
        pred_image_single = target_image_single + torch.randn(1, 1, H, W) * 0.1
        target_mask_single = (torch.randn(1, 1, H, W) > 0).float() * 2 - 1
        pred_mask_single = target_mask_single.clone()

        # Compute metrics for single sample
        metrics_single = metrics_calc.compute_all(
            pred_image=pred_image_single,
            target_image=target_image_single,
            pred_mask=pred_mask_single,
            target_mask=target_mask_single,
        )

        # Replicate to batch of 4
        target_image_batch = target_image_single.repeat(4, 1, 1, 1)
        pred_image_batch = pred_image_single.repeat(4, 1, 1, 1)
        target_mask_batch = target_mask_single.repeat(4, 1, 1, 1)
        pred_mask_batch = pred_mask_single.repeat(4, 1, 1, 1)

        # Compute metrics for batch
        metrics_batch = metrics_calc.compute_all(
            pred_image=pred_image_batch,
            target_image=target_image_batch,
            pred_mask=pred_mask_batch,
            target_mask=target_mask_batch,
        )

        logger.info(f"Single - PSNR: {metrics_single['psnr']:.2f}, Dice: {metrics_single['dice']:.4f}")
        logger.info(f"Batch  - PSNR: {metrics_batch['psnr']:.2f}, Dice: {metrics_batch['dice']:.4f}")

        # Metrics should be the same (since all samples are identical)
        assert abs(metrics_single['psnr'] - metrics_batch['psnr']) < 0.1, \
            f"PSNR inconsistent: {metrics_single['psnr']} vs {metrics_batch['psnr']}"
        assert abs(metrics_single['ssim'] - metrics_batch['ssim']) < 0.01, \
            f"SSIM inconsistent: {metrics_single['ssim']} vs {metrics_batch['ssim']}"
        assert abs(metrics_single['dice'] - metrics_batch['dice']) < 0.01, \
            f"Dice inconsistent: {metrics_single['dice']} vs {metrics_batch['dice']}"

        logger.info("✓ Batch consistency test passed")


def test_metrics_on_real_range():
    """Test metrics with values in the actual training range [-1, 1]."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Real training value range [-1, 1]")
    logger.info("="*60)

    metrics_calc = MetricsCalculator(data_range=2.0)

    B, H, W = 4, 128, 128

    # Create realistic data in [-1, 1] range
    target_image = torch.randn(B, 1, H, W).clamp(-1, 1)
    pred_image = target_image + torch.randn_like(target_image) * 0.15

    # Create realistic masks (binary in [-1, 1])
    target_mask_prob = torch.rand(B, 1, H, W)
    target_mask = (target_mask_prob > 0.5).float() * 2 - 1

    pred_mask_prob = target_mask_prob + torch.randn_like(target_mask_prob) * 0.1
    pred_mask = (pred_mask_prob > 0.5).float() * 2 - 1

    # Compute metrics
    metrics = metrics_calc.compute_all(
        pred_image=pred_image,
        target_image=target_image,
        pred_mask=pred_mask,
        target_mask=target_mask,
    )

    logger.info(f"PSNR: {metrics['psnr']:.2f} dB (typical range: 15-30)")
    logger.info(f"SSIM: {metrics['ssim']:.4f} (typical range: 0.4-0.9)")
    logger.info(f"Dice: {metrics['dice']:.4f} (typical range: 0.6-0.95)")
    logger.info(f"HD95: {metrics['hd95']:.4f}")

    # Sanity checks for realistic training scenario
    assert 10 < metrics['psnr'] < 40, f"PSNR out of typical range: {metrics['psnr']}"
    assert 0.3 < metrics['ssim'] < 1.0, f"SSIM out of typical range: {metrics['ssim']}"
    assert 0.4 < metrics['dice'] < 1.0, f"Dice out of typical range: {metrics['dice']}"

    logger.info("✓ Real training range test passed")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Run tests
    test_metrics_on_real_range()

    print("\n" + "="*60)
    print("Running pytest test suite...")
    print("="*60)
    pytest.main([__file__, "-v", "-s"])
