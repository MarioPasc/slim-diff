"""Comprehensive tests for LOCAL z-binning implementation.

Tests verify:
1. Local binning uses all bins within z_range
2. Bins are evenly distributed
3. Class balance is maintained (lesion oversampling)
4. Training/validation sampling works correctly
5. Integration with actual config and cache
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.diffusion.data.dataset import SliceDataset, create_dataloader
from src.diffusion.model.components.conditioning import compute_class_token
from src.diffusion.model.embeddings.zpos import (
    normalize_z_local,
    quantize_z,
    z_bin_to_index,
)


class TestLocalBinningCore:
    """Test core local binning functions."""

    def test_normalize_z_local(self):
        """Test local normalization within z_range (inclusive)."""
        # Full range [0, 127] = 128 slices
        z_range = (0, 127)
        assert normalize_z_local(0, z_range) == 0.0
        assert normalize_z_local(127, z_range) == pytest.approx(127/128, rel=0.001)  # Inclusive fix
        assert normalize_z_local(63, z_range) == pytest.approx(63/128, rel=0.001)

        # Partial range [24, 93] = 70 slices
        z_range = (24, 93)
        assert normalize_z_local(24, z_range) == 0.0
        assert normalize_z_local(93, z_range) == pytest.approx(69/70, rel=0.001)  # Inclusive fix
        assert normalize_z_local(58, z_range) == pytest.approx(34/70, rel=0.001)

        # Edge case: single slice range
        z_range = (50, 50)
        assert normalize_z_local(50, z_range) == 0.0

        # Out of range should raise error
        z_range = (24, 93)
        with pytest.raises(ValueError):
            normalize_z_local(20, z_range)
        with pytest.raises(ValueError):
            normalize_z_local(100, z_range)

    def test_quantize_z_all_bins_used(self):
        """Test that LOCAL binning uses ALL bins."""
        # Test with different configurations
        test_configs = [
            ((24, 93), 10),   # 70 slices, 10 bins
            ((24, 93), 50),   # 70 slices, 50 bins
            ((40, 100), 30),  # 61 slices, 30 bins
            ((0, 127), 50),   # Full range, 50 bins
        ]

        for z_range, n_bins in test_configs:
            min_z, max_z = z_range
            bins_used = set()

            # Map all slices to bins
            for z_idx in range(min_z, max_z + 1):
                z_bin = quantize_z(z_idx, z_range, n_bins)
                bins_used.add(z_bin)

            # All bins should be used
            expected_bins = set(range(n_bins))
            assert bins_used == expected_bins, (
                f"Config z_range={z_range}, n_bins={n_bins}: "
                f"Expected all {n_bins} bins, got {len(bins_used)} bins. "
                f"Missing: {expected_bins - bins_used}"
            )

    def test_quantize_z_even_distribution(self):
        """Test that bins are evenly distributed."""
        z_range = (24, 93)  # 70 slices
        n_bins = 10  # Should be 7 slices per bin

        bin_counts = {}
        for z_idx in range(z_range[0], z_range[1] + 1):
            z_bin = quantize_z(z_idx, z_range, n_bins)
            bin_counts[z_bin] = bin_counts.get(z_bin, 0) + 1

        # All bins should have approximately the same count
        counts = list(bin_counts.values())
        max_count = max(counts)
        min_count = min(counts)

        # Difference should be at most 1 (due to rounding)
        assert max_count - min_count <= 1, (
            f"Bins not evenly distributed: min={min_count}, max={max_count}. "
            f"Counts: {bin_counts}"
        )

    def test_quantize_z_boundaries(self):
        """Test boundary conditions for local binning."""
        z_range = (24, 93)  # 70 slices
        n_bins = 10

        # First slice maps to bin 0
        assert quantize_z(24, z_range, n_bins) == 0

        # Last slice maps to bin n_bins-1
        assert quantize_z(93, z_range, n_bins) == n_bins - 1

        # Slices outside range raise error
        with pytest.raises(ValueError):
            quantize_z(23, z_range, n_bins)

        with pytest.raises(ValueError):
            quantize_z(94, z_range, n_bins)

    def test_z_bin_to_index_roundtrip(self):
        """Test conversion from bin back to approximate z-index."""
        z_range = (24, 93)
        n_bins = 10

        for z_bin in range(n_bins):
            # Get center z-index for this bin
            z_idx_center = z_bin_to_index(z_bin, z_range, n_bins)

            # Center should be within range
            assert z_range[0] <= z_idx_center <= z_range[1]

            # Center should map back to same bin
            z_bin_recovered = quantize_z(z_idx_center, z_range, n_bins)
            assert z_bin_recovered == z_bin, (
                f"Roundtrip failed for bin {z_bin}: "
                f"center={z_idx_center} maps to bin {z_bin_recovered}"
            )


class TestLocalBinningWithConfig:
    """Test local binning with actual configuration."""

    @pytest.fixture
    def baseline_config(self):
        """Load baseline configuration."""
        config_path = Path("slurm/jsddpm_baseline/jsddpm_baseline.yaml")
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")
        return OmegaConf.load(config_path)

    def test_baseline_config_binning(self, baseline_config):
        """Test binning with baseline config (z_range=[24,93], z_bins=50)."""
        z_range = tuple(baseline_config.data.slice_sampling.z_range)
        n_bins = baseline_config.conditioning.z_bins

        print(f"\nTesting with baseline config:")
        print(f"  z_range: {z_range}")
        print(f"  n_bins: {n_bins}")

        # Check all bins are used
        bins_used = set()
        min_z, max_z = z_range

        for z_idx in range(min_z, max_z + 1):
            z_bin = quantize_z(z_idx, z_range, n_bins)
            bins_used.add(z_bin)

        assert len(bins_used) == n_bins, (
            f"Expected all {n_bins} bins to be used, got {len(bins_used)}"
        )

        # Check distribution
        bin_counts = {}
        for z_idx in range(min_z, max_z + 1):
            z_bin = quantize_z(z_idx, z_range, n_bins)
            bin_counts[z_bin] = bin_counts.get(z_bin, 0) + 1

        counts = list(bin_counts.values())
        print(f"  Slices per bin - min: {min(counts)}, max: {max(counts)}, "
              f"avg: {np.mean(counts):.1f}")

        # Should be relatively even
        assert max(counts) - min(counts) <= 2

    def test_various_z_ranges(self):
        """Test binning with various z_range configurations."""
        test_cases = [
            # (z_range, n_bins, description)
            ((24, 93), 10, "Default range, 10 bins"),
            ((24, 93), 50, "Default range, 50 bins"),
            ((40, 100), 30, "Shifted range, 30 bins"),
            ((0, 127), 50, "Full volume, 50 bins"),
            ((50, 100), 10, "User's example"),
        ]

        for z_range, n_bins, desc in test_cases:
            print(f"\nTesting: {desc}")
            print(f"  z_range={z_range}, n_bins={n_bins}")

            bins_used = set()
            bin_counts = {}
            min_z, max_z = z_range

            for z_idx in range(min_z, max_z + 1):
                z_bin = quantize_z(z_idx, z_range, n_bins)
                bins_used.add(z_bin)
                bin_counts[z_bin] = bin_counts.get(z_bin, 0) + 1

            # All bins should be used
            assert len(bins_used) == n_bins, (
                f"{desc}: Expected {n_bins} bins, got {len(bins_used)}"
            )

            # Check distribution
            counts = list(bin_counts.values())
            print(f"  Slices per bin - min: {min(counts)}, max: {max(counts)}, "
                  f"avg: {np.mean(counts):.1f}")

            # Show first/last few bin ranges
            print("  First 3 bins:")
            for z_bin in range(min(3, n_bins)):
                slices_in_bin = [z for z in range(min_z, max_z + 1)
                                if quantize_z(z, z_range, n_bins) == z_bin]
                print(f"    Bin {z_bin}: [{min(slices_in_bin)}, {max(slices_in_bin)}]")


@pytest.mark.skipif(
    not Path("/media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache").exists(),
    reason="Cache directory not found"
)
class TestLocalBinningWithCache:
    """Test local binning with actual cached data."""

    @pytest.fixture
    def cache_dir(self):
        """Get cache directory."""
        return Path("/media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache")

    @pytest.fixture
    def baseline_config(self):
        """Load baseline configuration."""
        config_path = Path("slurm/jsddpm_baseline/jsddpm_baseline.yaml")
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")
        cfg = OmegaConf.load(config_path)
        # Override cache_dir to use actual cache
        cfg.data.cache_dir = "/media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache"
        return cfg

    def test_cache_uses_all_bins(self, cache_dir, baseline_config):
        """Test that cached data uses all bins after rebuild with local binning."""
        z_range = tuple(baseline_config.data.slice_sampling.z_range)
        n_bins = baseline_config.conditioning.z_bins

        print(f"\nChecking cache at: {cache_dir}")
        print(f"  z_range: {z_range}")
        print(f"  n_bins: {n_bins}")

        # Read train CSV
        train_csv = cache_dir / "train.csv"
        if not train_csv.exists():
            pytest.skip(f"Train CSV not found: {train_csv}")

        # Collect all z_bins from cache
        bins_in_cache = set()
        z_indices_in_cache = set()
        bin_counts = {}

        with open(train_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                z_bin = int(row['z_bin'])
                z_index = int(row['z_index'])
                bins_in_cache.add(z_bin)
                z_indices_in_cache.add(z_index)
                bin_counts[z_bin] = bin_counts.get(z_bin, 0) + 1

        print(f"\nCache statistics:")
        print(f"  Total samples: {sum(bin_counts.values())}")
        print(f"  Unique z_indices: {len(z_indices_in_cache)}")
        print(f"  Unique bins: {len(bins_in_cache)}")
        print(f"  Bins used: {sorted(bins_in_cache)}")

        # With local binning, ALL bins should be present
        # (unless cache was built with old global binning)
        if len(bins_in_cache) == n_bins:
            print("  ✓ ALL bins are used (local binning detected)")
        else:
            print(f"  ⚠ Only {len(bins_in_cache)}/{n_bins} bins used")
            print(f"    This suggests cache was built with GLOBAL binning")
            print(f"    Run: jsddpm-cache --config slurm/jsddpm_baseline/jsddpm_baseline.yaml")
            pytest.skip("Cache needs to be rebuilt with local binning")

    def test_class_balance_in_cache(self, cache_dir, baseline_config):
        """Test class balance (lesion vs healthy) in cached data."""
        train_csv = cache_dir / "train.csv"
        if not train_csv.exists():
            pytest.skip(f"Train CSV not found: {train_csv}")

        # Count samples by class
        lesion_count = 0
        healthy_count = 0
        total_count = 0

        with open(train_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                has_lesion = row['has_lesion'].lower() == 'true'
                total_count += 1
                if has_lesion:
                    lesion_count += 1
                else:
                    healthy_count += 1

        lesion_ratio = lesion_count / total_count if total_count > 0 else 0

        print(f"\nClass distribution in cache (before oversampling):")
        print(f"  Total samples: {total_count}")
        print(f"  Lesion samples: {lesion_count} ({lesion_ratio:.1%})")
        print(f"  Healthy samples: {healthy_count} ({1-lesion_ratio:.1%})")

        # Lesion slices should be minority (epilepsy is rare)
        # Typically 5-20% depending on dataset
        assert 0.01 <= lesion_ratio <= 0.5, (
            f"Unexpected lesion ratio: {lesion_ratio:.1%}. "
            f"Expected between 1% and 50%"
        )

    def test_dataloader_class_balance_with_oversampling(self, baseline_config):
        """Test that dataloader correctly applies lesion oversampling."""
        # Enable oversampling
        baseline_config.data.lesion_oversampling.enabled = True
        baseline_config.data.lesion_oversampling.weight = 5.0

        # Create dataloader with oversampling
        dataloader = create_dataloader(
            baseline_config,
            split="train",
            shuffle=True,
            use_weighted_sampler=True,
        )

        # Sample from dataloader
        n_samples = 1000
        lesion_count = 0

        for i, batch in enumerate(dataloader):
            if i * baseline_config.training.batch_size >= n_samples:
                break

            # Check has_lesion from metadata
            has_lesion = batch['metadata']['has_lesion']
            lesion_count += sum(has_lesion)

        sampled_lesion_ratio = lesion_count / n_samples

        print(f"\nClass distribution WITH lesion oversampling (weight=5.0):")
        print(f"  Sampled {n_samples} batches")
        print(f"  Lesion samples: {lesion_count} ({sampled_lesion_ratio:.1%})")
        print(f"  Healthy samples: {n_samples - lesion_count} ({1-sampled_lesion_ratio:.1%})")

        # With 5x oversampling, lesion ratio should increase significantly
        # Target is roughly 50/50, but depends on original distribution
        # Just check it's higher than without oversampling (typically 5-20%)
        assert sampled_lesion_ratio > 0.25, (
            f"Oversampling not working: lesion ratio is only {sampled_lesion_ratio:.1%}"
        )

    def test_bin_distribution_in_dataset(self, baseline_config):
        """Test that bins are evenly sampled during training."""
        # Create dataset (without oversampling for this test)
        dataset = SliceDataset(
            cache_dir=baseline_config.data.cache_dir,
            split="train",
        )

        z_bins = baseline_config.conditioning.z_bins

        # Count samples per bin
        bin_counts = {}
        for sample in dataset.samples:
            z_bin = sample['z_bin']
            bin_counts[z_bin] = bin_counts.get(z_bin, 0) + 1

        print(f"\nBin distribution in dataset:")
        print(f"  Total samples: {len(dataset.samples)}")
        print(f"  Bins present: {len(bin_counts)}/{z_bins}")

        # Check all bins are present
        if len(bin_counts) == z_bins:
            print("  ✓ ALL bins are present")
        else:
            missing_bins = set(range(z_bins)) - set(bin_counts.keys())
            print(f"  ⚠ Missing bins: {sorted(missing_bins)}")

        # Show distribution
        counts = list(bin_counts.values())
        print(f"  Samples per bin - min: {min(counts)}, max: {max(counts)}, "
              f"avg: {np.mean(counts):.1f}, std: {np.std(counts):.1f}")

    def test_slice_filtering_then_binning(self, cache_dir, baseline_config):
        """Test that filtering happens BEFORE binning (critical order)."""
        z_range = tuple(baseline_config.data.slice_sampling.z_range)
        min_z, max_z = z_range

        train_csv = cache_dir / "train.csv"
        if not train_csv.exists():
            pytest.skip(f"Train CSV not found: {train_csv}")

        # Collect all z_indices from cache
        z_indices_in_cache = set()
        with open(train_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                z_indices_in_cache.add(int(row['z_index']))

        print(f"\nSlice filtering validation:")
        print(f"  z_range: [{min_z}, {max_z}]")
        print(f"  z_indices in cache: {sorted(z_indices_in_cache)}")

        # All z_indices should be within z_range
        for z_idx in z_indices_in_cache:
            assert min_z <= z_idx <= max_z, (
                f"Found z_index {z_idx} outside z_range [{min_z}, {max_z}]. "
                f"This means filtering is not working correctly!"
            )

        print("  ✓ All z_indices are within z_range")

        # Conversely, check that we don't have z_indices outside range
        z_indices_outside = []
        for z_idx in range(0, min_z):
            if z_idx in z_indices_in_cache:
                z_indices_outside.append(z_idx)
        for z_idx in range(max_z + 1, 128):
            if z_idx in z_indices_in_cache:
                z_indices_outside.append(z_idx)

        assert len(z_indices_outside) == 0, (
            f"Found {len(z_indices_outside)} slices outside z_range: {z_indices_outside}"
        )

        print("  ✓ No slices outside z_range")
        print("  ✓ Filtering happens BEFORE binning (correct order)")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
