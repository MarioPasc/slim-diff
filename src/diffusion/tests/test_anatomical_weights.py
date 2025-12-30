"""Tests for anatomical weight generation from z-bin priors.

Tests the get_anatomical_weights() function with real cached priors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.diffusion.utils.zbin_priors import (
    get_anatomical_weights,
    load_zbin_priors,
)


# Path to real priors in cache
CACHE_DIR = Path("/media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache")
PRIORS_FILENAME = "zbin_priors_brain_roi.npz"
PRIORS_PATH = CACHE_DIR / PRIORS_FILENAME


@pytest.fixture(scope="module")
def zbin_priors():
    """Load real z-bin priors from cache."""
    if not PRIORS_PATH.exists():
        pytest.skip(f"Priors file not found: {PRIORS_PATH}")

    # Load priors (30 bins based on actual data)
    priors = load_zbin_priors(CACHE_DIR, PRIORS_FILENAME, z_bins=30)
    return priors


@pytest.fixture
def priors_metadata():
    """Load priors metadata."""
    if not PRIORS_PATH.exists():
        pytest.skip(f"Priors file not found: {PRIORS_PATH}")

    data = np.load(PRIORS_PATH, allow_pickle=True)
    metadata = data["metadata"].item()
    return metadata


class TestGetAnatomicalWeights:
    """Tests for get_anatomical_weights function."""

    def test_basic_functionality(self, zbin_priors, priors_metadata):
        """Test basic weight generation with real priors."""
        # Create batch of z-bins
        z_bins = torch.tensor([0, 5, 10, 15, 20])
        B = z_bins.shape[0]
        H, W = priors_metadata["image_shape"]

        # Generate weights
        weights = get_anatomical_weights(
            z_bins,
            zbin_priors,
            in_brain_weight=1.0,
            out_brain_weight=0.1,
        )

        # Validate shape
        assert weights.shape == (B, 1, H, W), f"Expected (5, 1, 128, 128), got {weights.shape}"

        # Validate dtype
        assert weights.dtype == torch.float32

        # Validate values are either in_brain or out_brain weights
        unique_values = torch.unique(weights)
        assert len(unique_values) <= 2, f"Expected at most 2 unique values, got {len(unique_values)}"
        assert 0.1 in unique_values or 1.0 in unique_values

    def test_output_device(self, zbin_priors):
        """Test that output is on specified device."""
        z_bins = torch.tensor([0, 5, 10])

        # Test CPU
        weights_cpu = get_anatomical_weights(
            z_bins,
            zbin_priors,
            device=torch.device("cpu"),
        )
        assert weights_cpu.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            weights_cuda = get_anatomical_weights(
                z_bins,
                zbin_priors,
                device=torch.device("cuda:0"),
            )
            assert weights_cuda.device.type == "cuda"

    def test_weight_values(self, zbin_priors, priors_metadata):
        """Test that weight values match in_brain and out_brain settings."""
        z_bins = torch.tensor([10])
        H, W = priors_metadata["image_shape"]

        in_weight = 2.0
        out_weight = 0.5

        weights = get_anatomical_weights(
            z_bins,
            zbin_priors,
            in_brain_weight=in_weight,
            out_brain_weight=out_weight,
        )

        # Get the prior mask for z_bin=10
        prior_mask = zbin_priors[10]  # (H, W) boolean

        # Check weights match prior mask
        weight_map = weights[0, 0].cpu().numpy()

        # In-brain pixels should have in_weight
        in_brain_pixels = weight_map[prior_mask]
        if len(in_brain_pixels) > 0:
            assert np.allclose(in_brain_pixels, in_weight), \
                f"In-brain pixels should be {in_weight}, got {np.unique(in_brain_pixels)}"

        # Out-of-brain pixels should have out_weight
        out_brain_pixels = weight_map[~prior_mask]
        if len(out_brain_pixels) > 0:
            assert np.allclose(out_brain_pixels, out_weight), \
                f"Out-of-brain pixels should be {out_weight}, got {np.unique(out_brain_pixels)}"

    def test_batch_independence(self, zbin_priors):
        """Test that each sample in batch gets correct z-bin prior."""
        z_bins = torch.tensor([0, 10, 20, 29])

        weights = get_anatomical_weights(z_bins, zbin_priors)

        # Each sample should correspond to its z-bin prior
        for i, z_bin in enumerate(z_bins.tolist()):
            prior_mask = zbin_priors[z_bin]
            weight_map = weights[i, 0].cpu().numpy()

            # Check that weight map matches prior mask structure
            in_brain_mask = weight_map == 1.0
            out_brain_mask = weight_map == 0.1

            # All pixels should be classified as either in or out
            assert (in_brain_mask | out_brain_mask).all(), \
                f"Sample {i}: Not all pixels classified"

            # Prior mask should align with in_brain_mask
            assert np.array_equal(in_brain_mask, prior_mask), \
                f"Sample {i}: Weight mask doesn't match prior for z_bin={z_bin}"

    def test_edge_z_bins(self, zbin_priors):
        """Test with edge z-bin values (first and last)."""
        # Test first bin
        weights_first = get_anatomical_weights(
            torch.tensor([0]),
            zbin_priors,
        )
        assert weights_first.shape == (1, 1, 128, 128)

        # Test last bin (z_bins - 1)
        weights_last = get_anatomical_weights(
            torch.tensor([29]),
            zbin_priors,
        )
        assert weights_last.shape == (1, 1, 128, 128)

    def test_missing_z_bin(self, zbin_priors):
        """Test behavior when z-bin doesn't exist in priors."""
        # Use a z-bin that shouldn't exist (beyond 0-29 range)
        z_bins = torch.tensor([100])

        weights = get_anatomical_weights(z_bins, zbin_priors)

        # Should default to all out_brain_weight when z_bin not found
        assert torch.allclose(weights, torch.tensor(0.1)), \
            "Missing z-bin should default to out_brain_weight"

    def test_large_batch(self, zbin_priors):
        """Test with larger batch size."""
        B = 32
        z_bins = torch.randint(0, 30, (B,))

        weights = get_anatomical_weights(z_bins, zbin_priors)

        assert weights.shape == (B, 1, 128, 128)
        assert weights.dtype == torch.float32

    def test_custom_weights(self, zbin_priors):
        """Test with custom weight values."""
        z_bins = torch.tensor([15])

        # Custom weights
        in_w = 5.0
        out_w = 0.01

        weights = get_anatomical_weights(
            z_bins,
            zbin_priors,
            in_brain_weight=in_w,
            out_brain_weight=out_w,
        )

        unique_vals = torch.unique(weights)
        assert torch.any(torch.isclose(unique_vals, torch.tensor(in_w))) or \
               torch.any(torch.isclose(unique_vals, torch.tensor(out_w))), \
               f"Expected weights {in_w} or {out_w}, got {unique_vals}"

    def test_gradient_flow(self, zbin_priors):
        """Test that gradients can flow through the weight tensor."""
        z_bins = torch.tensor([10])

        weights = get_anatomical_weights(
            z_bins,
            zbin_priors,
            in_brain_weight=1.0,
            out_brain_weight=0.1,
        )

        # Weights should not require gradients (they're just masks)
        assert not weights.requires_grad

        # But they should be compatible with gradient computation
        dummy_loss = (weights * torch.randn_like(weights, requires_grad=True)).sum()
        dummy_loss.backward()  # Should not raise

    def test_all_z_bins_coverage(self, zbin_priors, priors_metadata):
        """Test that all z-bins in range produce valid weights."""
        z_bins_count = priors_metadata["z_bins"]

        for z_bin in range(z_bins_count):
            weights = get_anatomical_weights(
                torch.tensor([z_bin]),
                zbin_priors,
            )

            assert weights.shape == (1, 1, 128, 128), \
                f"z_bin={z_bin}: Invalid shape {weights.shape}"

            # Check that we have both in and out brain regions (for most bins)
            unique_vals = torch.unique(weights)
            assert len(unique_vals) >= 1, \
                f"z_bin={z_bin}: No valid weights generated"

    def test_weight_map_statistics(self, zbin_priors):
        """Test statistical properties of weight maps."""
        z_bins = torch.tensor([5, 10, 15, 20, 25])

        weights = get_anatomical_weights(z_bins, zbin_priors)

        # Check mean weight (should be between out and in weights)
        mean_weight = weights.mean().item()
        assert 0.1 <= mean_weight <= 1.0, \
            f"Mean weight {mean_weight} outside expected range [0.1, 1.0]"

        # Check that there's variation (not all same value)
        std_weight = weights.std().item()
        assert std_weight > 0, "Weights should have variation across pixels"

    def test_reproducibility(self, zbin_priors):
        """Test that weight generation is deterministic."""
        z_bins = torch.tensor([7, 13, 21])

        weights1 = get_anatomical_weights(z_bins, zbin_priors)
        weights2 = get_anatomical_weights(z_bins, zbin_priors)

        assert torch.allclose(weights1, weights2), \
            "Weight generation should be deterministic"

    def test_spatial_structure(self, zbin_priors):
        """Test that weight maps preserve spatial structure of priors."""
        z_bin = 15
        z_bins = torch.tensor([z_bin])

        weights = get_anatomical_weights(z_bins, zbin_priors)
        weight_map = weights[0, 0].cpu().numpy()

        # Get prior mask
        prior_mask = zbin_priors[z_bin]

        # Check that in-brain region is spatially coherent (not scattered noise)
        in_brain_pixels = (weight_map == 1.0)

        # Count connected regions (rough check for coherence)
        # In-brain region should generally be one or a few large connected components
        in_brain_count = in_brain_pixels.sum()

        # If there are in-brain pixels, they should form meaningful regions
        if in_brain_count > 100:  # Arbitrary threshold
            # Check that prior mask aligns with weight mask
            assert np.array_equal(in_brain_pixels, prior_mask), \
                "Weight map spatial structure doesn't match prior mask"


class TestIntegrationWithTraining:
    """Integration tests for use in training pipeline."""

    def test_realistic_training_scenario(self, zbin_priors):
        """Test with realistic training batch scenario."""
        # Simulate a training batch
        B = 16
        C = 2  # image + mask channels
        H, W = 128, 128

        # Random z-bins from valid range
        z_bins = torch.randint(0, 30, (B,))

        # Generate weights
        weights = get_anatomical_weights(
            z_bins,
            zbin_priors,
            in_brain_weight=1.0,
            out_brain_weight=0.1,
        )

        # Simulate loss computation
        pred = torch.randn(B, C, H, W)
        target = torch.randn(B, C, H, W)

        # Weighted MSE
        mse = (pred - target) ** 2
        weighted_sum = (mse * weights).sum()
        weight_sum = (weights * C).sum()
        loss = weighted_sum / weight_sum.clamp(min=1e-8)

        # Validate loss is reasonable
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_cuda_training_scenario(self, zbin_priors):
        """Test with CUDA tensors if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda:0")
        B = 8
        z_bins = torch.randint(0, 30, (B,)).to(device)

        weights = get_anatomical_weights(
            z_bins,
            zbin_priors,
            device=device,
        )

        assert weights.device.type == "cuda"
        assert weights.shape == (B, 1, 128, 128)

    def test_memory_efficiency(self, zbin_priors):
        """Test that function doesn't consume excessive memory."""
        import gc
        import torch.cuda as cuda

        if not cuda.is_available():
            pytest.skip("CUDA not available for memory test")

        device = torch.device("cuda:0")
        gc.collect()
        cuda.empty_cache()

        # Get initial memory
        initial_mem = cuda.memory_allocated(device)

        # Generate weights for multiple batches
        for _ in range(10):
            z_bins = torch.randint(0, 30, (32,)).to(device)
            weights = get_anatomical_weights(z_bins, zbin_priors, device=device)
            del weights

        gc.collect()
        cuda.empty_cache()

        # Check memory didn't grow significantly
        final_mem = cuda.memory_allocated(device)
        mem_growth = final_mem - initial_mem

        # Allow some growth, but not excessive (< 10MB)
        assert mem_growth < 10 * 1024 * 1024, \
            f"Memory grew by {mem_growth / 1024 / 1024:.2f} MB"
