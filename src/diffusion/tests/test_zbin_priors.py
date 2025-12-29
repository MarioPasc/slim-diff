"""Tests for z-bin anatomical ROI priors.

Tests offline prior computation and online post-processing functionality.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.diffusion.utils.zbin_priors import (
    apply_zbin_prior_postprocess,
    compute_brain_foreground_mask,
    compute_zbin_priors,
    load_zbin_priors,
    otsu_threshold,
    save_zbin_priors,
)


class TestOtsuThreshold:
    """Tests for Otsu threshold implementation."""

    def test_otsu_bimodal_distribution(self) -> None:
        """Test Otsu correctly identifies threshold for bimodal data."""
        # Create bimodal distribution
        low_mode = np.random.normal(0.2, 0.05, 500)
        high_mode = np.random.normal(0.8, 0.05, 500)
        data = np.concatenate([low_mode, high_mode])

        threshold = otsu_threshold(data)

        # Threshold should be between the modes
        assert 0.3 < threshold < 0.7, f"Expected threshold between 0.3-0.7, got {threshold}"

    def test_otsu_empty_data(self) -> None:
        """Test Otsu handles empty data gracefully."""
        data = np.array([])
        threshold = otsu_threshold(data)
        assert threshold == 0.5  # Default fallback

    def test_otsu_nan_values(self) -> None:
        """Test Otsu handles NaN values."""
        data = np.array([0.2, 0.3, np.nan, 0.7, 0.8, np.nan])
        threshold = otsu_threshold(data)
        assert np.isfinite(threshold)


class TestComputeBrainForegroundMask:
    """Tests for brain foreground mask computation."""

    def test_simple_brain_blob(self) -> None:
        """Test mask computation on simple brain-like blob."""
        # Create image with bright center (brain) and dark background
        image = np.zeros((128, 128), dtype=np.float32)
        # Add bright circular region in center
        y, x = np.ogrid[:128, :128]
        center = (64, 64)
        radius = 40
        mask_region = (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius**2
        image[mask_region] = 0.8
        image += np.random.normal(0, 0.05, image.shape)

        result = compute_brain_foreground_mask(
            image, gaussian_sigma_px=0.7, min_component_px=100
        )

        assert result is not None
        assert result.shape == image.shape
        # Center should be foreground
        assert result[64, 64]
        # Corners should be background
        assert not result[0, 0]
        assert not result[0, 127]

    def test_empty_slice(self) -> None:
        """Test that empty/uniform slices return None."""
        image = np.ones((128, 128), dtype=np.float32) * 0.5
        result = compute_brain_foreground_mask(
            image, gaussian_sigma_px=0.7, min_component_px=100
        )
        # Should return None because p99 <= p1 (uniform image)
        # or component too small
        assert result is None or result.sum() == 0

    def test_small_component_filtered(self) -> None:
        """Test that small components are filtered out."""
        image = np.zeros((128, 128), dtype=np.float32)
        # Add very small bright region
        image[60:65, 60:65] = 1.0  # Only 25 pixels

        result = compute_brain_foreground_mask(
            image, gaussian_sigma_px=0.7, min_component_px=500
        )

        # Should return None because component is too small
        assert result is None


class TestZbinPriorsOffline:
    """Tests for offline prior computation."""

    def test_zbin_priors_offline_build_smoke(self) -> None:
        """Test that priors can be computed from synthetic cache."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            slices_dir = cache_dir / "slices"
            slices_dir.mkdir(parents=True)

            z_bins = 5
            z_range = (0, 4)

            # Create synthetic cached slices
            metadata_list = []
            for z_bin in range(z_bins):
                for sample_idx in range(3):  # 3 samples per bin
                    # Create brain-like image
                    image = np.zeros((64, 64), dtype=np.float32)
                    y, x = np.ogrid[:64, :64]
                    # Slightly different brain region per z_bin
                    center = (32 + z_bin, 32)
                    radius = 20 - z_bin  # Smaller at higher z
                    mask_region = (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius**2
                    image[mask_region] = 0.7 + np.random.uniform(-0.1, 0.1)
                    image += np.random.normal(0, 0.05, image.shape)

                    mask = np.zeros_like(image)

                    # Save slice
                    filename = f"sub001_z{z_bin:03d}_bin{z_bin:02d}_c0_s{sample_idx}.npz"
                    filepath = slices_dir / filename
                    np.savez_compressed(
                        filepath,
                        image=image,
                        mask=mask,
                        z_bin=z_bin,
                        z_index=z_bin,
                    )

                    metadata_list.append({
                        "filepath": f"slices/{filename}",
                        "z_bin": z_bin,
                        "z_index": z_bin,
                        "subject_id": "sub001",
                        "pathology_class": 0,
                        "token": z_bin,
                        "source": "test",
                        "has_lesion": "False",
                    })

            # Write CSV index
            csv_path = cache_dir / "train.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metadata_list[0].keys()))
                writer.writeheader()
                writer.writerows(metadata_list)

            # Compute priors
            result = compute_zbin_priors(
                cache_dir=cache_dir,
                z_bins=z_bins,
                z_range=z_range,
                prob_threshold=0.20,
                dilate_radius_px=2,
                gaussian_sigma_px=0.7,
                min_component_px=50,
            )

            # Verify priors
            assert "priors" in result
            assert "metadata" in result
            priors = result["priors"]

            # Check all bins have priors
            for z_bin in range(z_bins):
                assert z_bin in priors, f"Missing prior for z_bin {z_bin}"
                assert priors[z_bin].shape == (64, 64)
                assert priors[z_bin].dtype == np.bool_

            # Save and reload
            priors_path = cache_dir / "test_priors.npz"
            save_zbin_priors(priors, result["metadata"], priors_path)

            assert priors_path.exists()

            # Load and verify
            loaded_priors = load_zbin_priors(cache_dir, "test_priors.npz", z_bins)
            assert len(loaded_priors) == z_bins
            for z_bin in range(z_bins):
                assert z_bin in loaded_priors
                np.testing.assert_array_equal(priors[z_bin], loaded_priors[z_bin])


class TestZbinPriorsOnline:
    """Tests for online post-processing."""

    def test_zbin_prior_postprocess_removes_outside_noise(self) -> None:
        """Test that post-processing zeros out regions outside ROI."""
        # Create image with central "brain blob" + random bright specks outside
        image = np.zeros((64, 64), dtype=np.float32)
        lesion = np.zeros((64, 64), dtype=np.float32)

        # Add central brain region
        y, x = np.ogrid[:64, :64]
        center = (32, 32)
        radius = 20
        brain_region = (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius**2
        image[brain_region] = 0.7

        # Add lesion in brain
        lesion[30:35, 30:35] = 1.0

        # Add noise specks OUTSIDE the brain (in corners)
        image[0:5, 0:5] = 0.9  # Corner speck
        image[0:5, 59:64] = 0.8
        image[59:64, 0:5] = 0.85
        image[59:64, 59:64] = 0.75

        # Add lesion noise outside brain
        lesion[0:3, 0:3] = 0.5

        # Create prior ROI that covers brain but not corners
        prior = np.zeros((64, 64), dtype=bool)
        prior[10:54, 10:54] = True  # Central region

        priors = {0: prior}

        # Apply post-processing
        img_clean, lesion_clean = apply_zbin_prior_postprocess(
            image, lesion, z_bin=0, priors=priors,
            gaussian_sigma_px=0.7, min_component_px=50, fallback="prior"
        )

        # Verify corners are zeroed out
        assert img_clean[0, 0] == 0.0, "Corner should be zeroed"
        assert img_clean[0, 63] == 0.0, "Corner should be zeroed"
        assert img_clean[63, 0] == 0.0, "Corner should be zeroed"
        assert img_clean[63, 63] == 0.0, "Corner should be zeroed"

        # Verify lesion noise outside brain is zeroed
        assert lesion_clean[0, 0] == 0.0, "Lesion noise outside brain should be zeroed"

        # Verify brain region is preserved
        # Note: The actual brain mask from Otsu may differ slightly from the input
        # but the central region should have non-zero values
        assert img_clean[32, 32] != 0.0, "Brain center should be preserved"

    def test_fallback_prior_mode(self) -> None:
        """Test that fallback='prior' uses prior as mask when thresholding fails."""
        # Create uniform image (will fail Otsu)
        image = np.ones((64, 64), dtype=np.float32) * 0.5
        lesion = np.ones((64, 64), dtype=np.float32) * 0.3

        # Create prior
        prior = np.zeros((64, 64), dtype=bool)
        prior[20:44, 20:44] = True

        priors = {0: prior}

        img_clean, lesion_clean = apply_zbin_prior_postprocess(
            image, lesion, z_bin=0, priors=priors,
            gaussian_sigma_px=0.7, min_component_px=50, fallback="prior"
        )

        # With fallback="prior", prior region should be preserved
        assert img_clean[32, 32] != 0.0, "Prior region should be preserved"
        # Outside prior should be zeroed
        assert img_clean[0, 0] == 0.0, "Outside prior should be zeroed"

    def test_fallback_empty_mode(self) -> None:
        """Test that fallback='empty' zeros everything when thresholding fails."""
        # Create uniform image (will fail Otsu)
        image = np.ones((64, 64), dtype=np.float32) * 0.5
        lesion = np.ones((64, 64), dtype=np.float32) * 0.3

        # Create prior
        prior = np.zeros((64, 64), dtype=bool)
        prior[20:44, 20:44] = True

        priors = {0: prior}

        img_clean, lesion_clean = apply_zbin_prior_postprocess(
            image, lesion, z_bin=0, priors=priors,
            gaussian_sigma_px=0.7, min_component_px=50, fallback="empty"
        )

        # With fallback="empty", everything should be zeroed
        assert img_clean.sum() == 0.0, "Everything should be zeroed"
        assert lesion_clean.sum() == 0.0, "Everything should be zeroed"

    def test_missing_zbin_returns_unchanged(self) -> None:
        """Test that missing z_bin in priors returns unchanged input."""
        image = np.random.rand(64, 64).astype(np.float32)
        lesion = np.random.rand(64, 64).astype(np.float32)

        priors = {0: np.ones((64, 64), dtype=bool)}  # Only z_bin 0

        # Request z_bin 5 which doesn't exist
        img_clean, lesion_clean = apply_zbin_prior_postprocess(
            image.copy(), lesion.copy(), z_bin=5, priors=priors,
            gaussian_sigma_px=0.7, min_component_px=50, fallback="prior"
        )

        # Should return unchanged
        np.testing.assert_array_equal(image, img_clean)
        np.testing.assert_array_equal(lesion, lesion_clean)


class TestToggleDisablesChanges:
    """Test that enabled=false preserves original output."""

    def test_toggle_disables_all_changes(self) -> None:
        """Test that with enabled=false, postprocessing is skipped.

        Note: This tests the conditional logic by checking that when
        use_zbin_priors=False, the original values are preserved.
        In practice, this is controlled by the config flag.
        """
        # Create test image and lesion
        original_image = np.random.rand(64, 64).astype(np.float32)
        original_lesion = np.random.rand(64, 64).astype(np.float32)

        # Simulate disabled case: when postprocessing is disabled,
        # the calling code won't call apply_zbin_prior_postprocess at all.
        # But if it were called, we can verify behavior with an empty priors dict.

        # When priors is None or z_bin not in priors, should return unchanged
        priors = {}  # Empty priors simulates "effectively disabled"

        img_copy = original_image.copy()
        lesion_copy = original_lesion.copy()

        img_result, lesion_result = apply_zbin_prior_postprocess(
            img_copy, lesion_copy, z_bin=0, priors=priors,
            gaussian_sigma_px=0.7, min_component_px=50, fallback="prior"
        )

        # With empty priors dict, input should be unchanged
        np.testing.assert_array_equal(original_image, img_result)
        np.testing.assert_array_equal(original_lesion, lesion_result)


class TestLoadSavePriors:
    """Tests for saving and loading priors."""

    def test_load_validates_zbins(self) -> None:
        """Test that load raises error on z_bins mismatch."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            # Save priors with z_bins=5
            priors = {i: np.ones((64, 64), dtype=bool) for i in range(5)}
            metadata = {"z_bins": 5}
            save_zbin_priors(priors, metadata, cache_dir / "priors.npz")

            # Try to load with z_bins=10 - should raise
            with pytest.raises(ValueError, match="Z-bins mismatch"):
                load_zbin_priors(cache_dir, "priors.npz", z_bins=10)

    def test_load_missing_file_raises(self) -> None:
        """Test that load raises FileNotFoundError for missing file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            with pytest.raises(FileNotFoundError):
                load_zbin_priors(cache_dir, "nonexistent.npz", z_bins=5)
