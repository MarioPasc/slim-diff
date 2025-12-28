#!/usr/bin/env python
"""Verification script for LOCAL z-binning implementation.

This script:
1. Verifies the local binning implementation is correct
2. Checks if cache needs rebuilding
3. Provides instructions for next steps
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.model.embeddings.zpos import quantize_z


def check_implementation():
    """Verify local binning implementation."""
    print("=" * 80)
    print("VERIFICATION 1: Local Binning Implementation")
    print("=" * 80)

    # Test cases
    test_cases = [
        # (z_range, n_bins, z_index, expected_bin)
        ((24, 93), 10, 24, 0),   # First slice → bin 0
        ((24, 93), 10, 93, 9),   # Last slice → bin 9
        ((24, 93), 10, 58, 4),   # Middle → bin 4
        ((50, 100), 10, 50, 0),  # User's example
        ((50, 100), 10, 100, 9),
    ]

    all_pass = True
    for z_range, n_bins, z_index, expected_bin in test_cases:
        try:
            actual_bin = quantize_z(z_index, z_range, n_bins)
            status = "✓" if actual_bin == expected_bin else "✗"
            print(f"{status} z_range={z_range}, z={z_index} → bin {actual_bin} "
                  f"(expected {expected_bin})")

            if actual_bin != expected_bin:
                all_pass = False
        except Exception as e:
            print(f"✗ ERROR: {e}")
            all_pass = False

    if all_pass:
        print("\n✅ LOCAL binning implementation: CORRECT")
    else:
        print("\n❌ LOCAL binning implementation: FAILED")
        return False

    # Test that all bins are used
    print(f"\nChecking that ALL bins are used:")
    z_range = (24, 93)
    n_bins = 50

    bins_used = set()
    for z_idx in range(z_range[0], z_range[1] + 1):
        z_bin = quantize_z(z_idx, z_range, n_bins)
        bins_used.add(z_bin)

    if len(bins_used) == n_bins:
        print(f"✅ All {n_bins} bins are used with z_range={z_range}")
    else:
        print(f"❌ Only {len(bins_used)}/{n_bins} bins used")
        return False

    return True


def check_cache(config_path: Path):
    """Check if cache needs rebuilding."""
    print("\n" + "=" * 80)
    print("VERIFICATION 2: Cache Status")
    print("=" * 80)

    # Load config
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False

    cfg = OmegaConf.load(config_path)
    cache_dir = Path(cfg.data.cache_dir)
    z_range = tuple(cfg.data.slice_sampling.z_range)
    n_bins = cfg.conditioning.z_bins

    print(f"Config: {config_path}")
    print(f"Cache dir: {cache_dir}")
    print(f"z_range: {z_range}")
    print(f"n_bins: {n_bins}")

    # Check if cache exists
    train_csv = cache_dir / "train.csv"
    if not train_csv.exists():
        print(f"\n⚠️  Cache not found at {cache_dir}")
        print(f"   Run: jsddpm-cache --config {config_path}")
        return False

    # Read cache and check bins
    bins_in_cache = set()
    z_indices_in_cache = set()

    with open(train_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bins_in_cache.add(int(row['z_bin']))
            z_indices_in_cache.add(int(row['z_index']))

    print(f"\nCache statistics:")
    print(f"  Bins in cache: {len(bins_in_cache)}/{n_bins}")
    print(f"  Z-indices in cache: {sorted(z_indices_in_cache)}")

    # Check 1: All bins should be present
    cache_ok = True
    if len(bins_in_cache) == n_bins:
        print(f"\n✅ ALL {n_bins} bins are used (LOCAL binning)")
    else:
        missing_bins = set(range(n_bins)) - bins_in_cache
        print(f"\n❌ Only {len(bins_in_cache)}/{n_bins} bins used (GLOBAL binning)")
        print(f"   Missing bins: {sorted(missing_bins)}")
        print(f"   Cache was built with OLD global binning")
        cache_ok = False

    # Check 2: All z_indices should be within z_range
    min_z, max_z = z_range
    z_indices_outside = [z for z in z_indices_in_cache if z < min_z or z > max_z]

    if len(z_indices_outside) == 0:
        print(f"✅ All z-indices are within z_range [{min_z}, {max_z}]")
    else:
        print(f"❌ Found {len(z_indices_outside)} z-indices outside z_range:")
        print(f"   Outside range: {sorted(z_indices_outside)}")
        cache_ok = False

    return cache_ok


def print_rebuild_instructions(config_path: Path):
    """Print instructions for rebuilding cache."""
    print("\n" + "=" * 80)
    print("NEXT STEPS: Rebuild Cache")
    print("=" * 80)
    print(f"""
The cache needs to be rebuilt with the new LOCAL binning implementation.

Commands:
=========

# Activate environment
conda activate jsddpm

# Rebuild cache
jsddpm-cache --config {config_path}

# Verify cache is correct
python scripts/verify_local_binning.py

# After verification passes, you can train
jsddpm-train --config {config_path}

NOTE: Existing checkpoints are INCOMPATIBLE (token semantics changed).
      You must train from scratch.
""")


def main():
    """Main verification script."""
    print("\n" + "=" * 80)
    print("LOCAL Z-Binning Implementation Verification")
    print("=" * 80)

    # Check implementation
    impl_ok = check_implementation()

    if not impl_ok:
        print("\n❌ Implementation check FAILED")
        print("   Please review the code changes")
        return 1

    # Check cache
    config_path = Path("slurm/jsddpm_baseline/jsddpm_baseline.yaml")
    cache_ok = check_cache(config_path)

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Implementation: {'✅ PASS' if impl_ok else '❌ FAIL'}")
    print(f"Cache Status:   {'✅ OK' if cache_ok else '⚠️  REBUILD NEEDED'}")

    if impl_ok and cache_ok:
        print("\n🎉 Everything is ready!")
        print("   You can start training with the new local binning")
        return 0
    elif impl_ok and not cache_ok:
        print_rebuild_instructions(config_path)
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
