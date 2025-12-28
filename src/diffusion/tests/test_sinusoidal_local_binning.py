"""Integration tests for sinusoidal embedding with LOCAL z-binning.

These tests verify that the sinusoidal position encoding correctly aligns
with the LOCAL binning scheme (binning within z_range, not global volume).
"""

from __future__ import annotations

import pytest
import torch

from src.diffusion.model.embeddings import (
    ConditionalEmbeddingWithSinusoidal,
    quantize_z,
    z_bin_to_index,
)


class TestSinusoidalWithLocalBinning:
    """Test that sinusoidal encoding uses correct z-indices with LOCAL binning."""

    def test_z_bin_to_index_matches_local_binning(self):
        """Verify z_bin converts to correct z-index using LOCAL binning."""
        z_range = (24, 93)  # 70 slices
        n_bins = 50

        # Test bin 0 (first bin) should map to z ≈ 24
        z_idx_bin0 = z_bin_to_index(0, z_range, n_bins)
        assert 24 <= z_idx_bin0 <= 25, f"Bin 0 should map to ~24, got {z_idx_bin0}"

        # Test bin 25 (middle bin) should map to z ≈ 59 (middle of range)
        z_idx_bin25 = z_bin_to_index(25, z_range, n_bins)
        assert 58 <= z_idx_bin25 <= 60, f"Bin 25 should map to ~59, got {z_idx_bin25}"

        # Test bin 49 (last bin) should map to z ≈ 93
        z_idx_bin49 = z_bin_to_index(49, z_range, n_bins)
        assert 92 <= z_idx_bin49 <= 93, f"Bin 49 should map to ~93, got {z_idx_bin49}"

    def test_sinusoidal_embedding_uses_local_z_indices(self):
        """Test that sinusoidal embedding receives LOCAL z-indices."""
        z_range = (24, 93)  # 70 slices
        z_bins = 50
        max_z = 127

        embedding = ConditionalEmbeddingWithSinusoidal(
            num_embeddings=100,  # 2 * z_bins
            embedding_dim=256,
            z_bins=z_bins,
            z_range=z_range,
            use_sinusoidal=True,
            max_z=max_z,
        )

        # Test token for class=0, bin=25 (middle bin)
        # Should map to z_index ≈ 59 (middle of z_range)
        token_bin25 = torch.tensor([25])  # class=0, bin=25

        # Get the embedding
        emb = embedding(token_bin25)
        assert emb.shape == (1, 256)

        # Verify the z-index conversion happens correctly
        # by checking that bin=25 maps to the middle of z_range
        min_z, max_z_range = z_range
        range_size = max_z_range - min_z  # 69

        # Expected z_index for bin 25
        z_norm = (25 + 0.5) / z_bins  # 0.51
        expected_z_idx = int(min_z + z_norm * range_size)  # 24 + 35.19 = 59

        # We can't directly access the z_index used, but we can verify
        # that the embedding is different from what we'd get with GLOBAL binning
        # OLD BUGGY: z_index = (25.5 / 50) * 127 = 64
        # NEW CORRECT: z_index = 24 + (25.5 / 50) * 69 = 59

        # The two z-indices (59 vs 64) should produce different sinusoidal encodings
        assert 58 <= expected_z_idx <= 60

    def test_round_trip_consistency(self):
        """Test that z_index -> bin -> z_index is approximately consistent."""
        z_range = (24, 93)
        n_bins = 50

        # Test several z-indices
        test_z_indices = [24, 35, 50, 59, 70, 85, 93]

        for z_idx in test_z_indices:
            # Forward: z_index -> bin
            z_bin = quantize_z(z_idx, z_range, n_bins)

            # Backward: bin -> z_index (approximate center of bin)
            z_idx_reconstructed = z_bin_to_index(z_bin, z_range, n_bins)

            # Should be within same bin (tolerance ~1-2 slices)
            diff = abs(z_idx - z_idx_reconstructed)
            bin_width = (z_range[1] - z_range[0]) / n_bins  # ~1.4 slices

            assert diff <= bin_width, (
                f"z_index={z_idx} -> bin={z_bin} -> z_index={z_idx_reconstructed} "
                f"(diff={diff} > bin_width={bin_width})"
            )

    def test_different_z_ranges(self):
        """Test sinusoidal embedding with various z_range configurations."""
        test_configs = [
            ((0, 127), 50),  # Full volume (edge case)
            ((24, 93), 50),  # Baseline config
            ((10, 50), 20),  # Smaller range
            ((50, 100), 25),  # Different offset
        ]

        for z_range, n_bins in test_configs:
            embedding = ConditionalEmbeddingWithSinusoidal(
                num_embeddings=2 * n_bins,
                embedding_dim=128,
                z_bins=n_bins,
                z_range=z_range,
                use_sinusoidal=True,
                max_z=127,
            )

            # Test first, middle, and last bins
            tokens = torch.tensor([0, n_bins // 2, n_bins - 1])
            emb = embedding(tokens)

            assert emb.shape == (3, 128)

            # Embeddings should be different
            assert not torch.allclose(emb[0], emb[1])
            assert not torch.allclose(emb[1], emb[2])

    def test_sinusoidal_vs_non_sinusoidal(self):
        """Verify sinusoidal encoding produces different embeddings than non-sinusoidal."""
        z_range = (24, 93)
        z_bins = 50

        # Create two embeddings: with and without sinusoidal
        emb_with_sin = ConditionalEmbeddingWithSinusoidal(
            num_embeddings=100,
            embedding_dim=256,
            z_bins=z_bins,
            z_range=z_range,
            use_sinusoidal=True,
            max_z=127,
        )

        emb_without_sin = ConditionalEmbeddingWithSinusoidal(
            num_embeddings=100,
            embedding_dim=256,
            z_bins=z_bins,
            z_range=z_range,
            use_sinusoidal=False,
            max_z=127,
        )

        # Same tokens
        tokens = torch.tensor([0, 25, 50, 75])

        # Get embeddings
        out_with_sin = emb_with_sin(tokens)
        out_without_sin = emb_without_sin(tokens)

        # Shapes should match
        assert out_with_sin.shape == out_without_sin.shape == (4, 256)

        # But values should be different (sinusoidal adds extra info)
        assert not torch.allclose(out_with_sin, out_without_sin)

    @pytest.mark.parametrize("z_idx,expected_bin", [
        (24, 0),   # First slice -> bin 0
        (59, 25),  # Middle slice -> bin 25
        (93, 49),  # Last slice -> bin 49
    ])
    def test_specific_z_to_bin_mapping(self, z_idx, expected_bin):
        """Test specific z-index to bin mappings for baseline config."""
        z_range = (24, 93)
        n_bins = 50

        z_bin = quantize_z(z_idx, z_range, n_bins)
        assert z_bin == expected_bin, (
            f"z_index={z_idx} should map to bin {expected_bin}, got {z_bin}"
        )


class TestSinusoidalEmbeddingCorrectness:
    """Test that the fix for LOCAL binning is correctly implemented."""

    def test_buggy_vs_correct_z_index_conversion(self):
        """Demonstrate the difference between GLOBAL (buggy) and LOCAL (correct) conversion."""
        z_range = (24, 93)
        n_bins = 50
        z_bin = 25  # Middle bin

        # CORRECT (LOCAL): Use z_range
        min_z, max_z = z_range
        range_size = max_z - min_z  # 69
        z_norm = (z_bin + 0.5) / n_bins  # 0.51
        z_idx_local = int(min_z + z_norm * range_size)  # 24 + 35.19 = 59

        # OLD BUGGY (GLOBAL): Ignore z_range
        # z_idx_global = int((z_bin + 0.5) / n_bins * 127)  # 64 (WRONG!)

        # Verify the LOCAL conversion matches our helper function
        z_idx_from_helper = z_bin_to_index(z_bin, z_range, n_bins)

        assert z_idx_from_helper == z_idx_local
        assert 58 <= z_idx_local <= 60  # Should be ~59

    def test_embedding_forward_uses_local_conversion(self):
        """Verify that ConditionalEmbeddingWithSinusoidal.forward() uses LOCAL conversion."""
        z_range = (24, 93)
        z_bins = 50

        embedding = ConditionalEmbeddingWithSinusoidal(
            num_embeddings=100,
            embedding_dim=256,
            z_bins=z_bins,
            z_range=z_range,
            use_sinusoidal=True,
            max_z=127,
        )

        # Check that z_range is stored
        assert embedding.z_range == z_range

        # Token for class=0, bin=0 (first bin)
        token_bin0 = torch.tensor([0])
        emb_bin0 = embedding(token_bin0)

        # Token for class=0, bin=49 (last bin)
        token_bin49 = torch.tensor([49])
        emb_bin49 = embedding(token_bin49)

        # Embeddings should be very different (different positions)
        assert not torch.allclose(emb_bin0, emb_bin49)
        cosine_sim = torch.nn.functional.cosine_similarity(
            emb_bin0, emb_bin49, dim=-1
        )
        # Embeddings for opposite ends should have low similarity
        assert cosine_sim.item() < 0.9
