"""Unit tests for the Enhanced Anatomical Encoder components.

Tests for:
- RotaryPositionEmbedding2D
- FPNBackbone
- PriorMapLoader
- EnhancedAnatomicalPriorEncoder
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.diffusion.model.components.rotary_embedding import (
    RotaryPositionEmbedding2D,
    LearnedPositionEmbedding2D,
    SinusoidalPositionEmbedding2D,
    build_position_embedding_2d,
)
from src.diffusion.model.components.fpn_backbone import (
    FPNBackbone,
    SimpleCNNBackbone,
    build_backbone,
)
from src.diffusion.model.components.prior_map_loader import (
    PriorMapLoader,
    load_prior_map_loader,
)
from src.diffusion.model.components.anatomical_encoder import (
    EnhancedAnatomicalPriorEncoder,
    build_enhanced_anatomical_encoder,
)


class TestRotaryPositionEmbedding2D:
    """Tests for 2D Rotary Position Embedding."""

    def test_output_shape(self):
        """Verify RoPE maintains input shape."""
        embed_dim = 256
        rope = RotaryPositionEmbedding2D(embed_dim=embed_dim, max_h=40, max_w=40)

        x = torch.randn(2, 1600, embed_dim)  # (B, 40*40, embed_dim)
        out = rope(x, h=40, w=40)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_embed_dim_divisibility(self):
        """Verify embed_dim must be divisible by 4."""
        with pytest.raises(ValueError, match="divisible by 4"):
            RotaryPositionEmbedding2D(embed_dim=255)

    def test_different_spatial_sizes(self):
        """Verify RoPE works with different spatial sizes."""
        rope = RotaryPositionEmbedding2D(embed_dim=128, max_h=64, max_w=64)

        # Test with smaller spatial size
        x = torch.randn(1, 400, 128)  # 20*20 = 400
        out = rope(x, h=20, w=20)
        assert out.shape == x.shape

        # Test with different aspect ratio
        x = torch.randn(1, 640, 128)  # 20*32 = 640
        out = rope(x, h=20, w=32)
        assert out.shape == x.shape

    def test_rotation_properties(self):
        """Verify RoPE applies proper rotation (output differs from input)."""
        rope = RotaryPositionEmbedding2D(embed_dim=64, max_h=10, max_w=10)
        x = torch.randn(1, 100, 64)

        out = rope(x, h=10, w=10)

        # Output should differ from input (rotation applied)
        assert not torch.allclose(x, out), "RoPE should modify the input"

        # Output should have similar norm (rotation preserves magnitude approximately)
        x_norm = x.norm(dim=-1).mean()
        out_norm = out.norm(dim=-1).mean()
        assert torch.abs(x_norm - out_norm) / x_norm < 0.2, \
            "RoPE should approximately preserve norm"


class TestLearnedPositionEmbedding2D:
    """Tests for Learned 2D Position Embedding."""

    def test_output_shape(self):
        """Verify learned embedding maintains input shape."""
        embed_dim = 256
        pe = LearnedPositionEmbedding2D(embed_dim=embed_dim, max_h=40, max_w=40)

        x = torch.randn(2, 1600, embed_dim)
        out = pe(x, h=40, w=40)

        assert out.shape == x.shape

    def test_additive(self):
        """Verify learned embedding is additive."""
        pe = LearnedPositionEmbedding2D(embed_dim=64, max_h=10, max_w=10)

        x = torch.zeros(1, 100, 64)
        out = pe(x, h=10, w=10)

        # With zero input, output should be just the positional encoding
        assert not torch.allclose(out, x), "Should add positional information"


class TestSinusoidalPositionEmbedding2D:
    """Tests for Sinusoidal 2D Position Embedding."""

    def test_output_shape(self):
        """Verify sinusoidal embedding maintains input shape."""
        embed_dim = 256
        pe = SinusoidalPositionEmbedding2D(embed_dim=embed_dim, max_h=40, max_w=40)

        x = torch.randn(2, 1600, embed_dim)
        out = pe(x, h=40, w=40)

        assert out.shape == x.shape

    def test_deterministic(self):
        """Verify sinusoidal encoding is deterministic."""
        pe = SinusoidalPositionEmbedding2D(embed_dim=64, max_h=10, max_w=10)

        x = torch.randn(1, 100, 64)
        out1 = pe(x, h=10, w=10)
        out2 = pe(x, h=10, w=10)

        assert torch.allclose(out1, out2), "Sinusoidal PE should be deterministic"


class TestBuildPositionEmbedding2D:
    """Tests for the position embedding factory function."""

    def test_build_rope(self):
        """Test building RoPE."""
        pe = build_position_embedding_2d("rope", embed_dim=256, max_h=40, max_w=40)
        assert isinstance(pe, RotaryPositionEmbedding2D)

    def test_build_sinusoidal(self):
        """Test building sinusoidal."""
        pe = build_position_embedding_2d("sinusoidal", embed_dim=256)
        assert isinstance(pe, SinusoidalPositionEmbedding2D)

    def test_build_learned(self):
        """Test building learned."""
        pe = build_position_embedding_2d("learned", embed_dim=256)
        assert isinstance(pe, LearnedPositionEmbedding2D)

    def test_unknown_type(self):
        """Test error on unknown type."""
        with pytest.raises(ValueError, match="Unknown"):
            build_position_embedding_2d("unknown", embed_dim=256)


class TestFPNBackbone:
    """Tests for FPN Backbone."""

    def test_output_shape_downsample_4(self):
        """Verify FPN achieves 4x downsampling."""
        fpn = FPNBackbone(
            in_channels=2,
            hidden_dims=(64, 128),
            out_channels=256,
            downsample_factor=4,
        )

        x = torch.randn(2, 2, 160, 160)
        out = fpn(x)

        expected_shape = (2, 256, 40, 40)  # 160/4 = 40
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

    def test_output_shape_downsample_8(self):
        """Verify FPN achieves 8x downsampling with more stages."""
        fpn = FPNBackbone(
            in_channels=1,
            hidden_dims=(32, 64, 128),
            out_channels=128,
            downsample_factor=8,
        )

        x = torch.randn(2, 1, 160, 160)
        out = fpn(x)

        expected_shape = (2, 128, 20, 20)  # 160/8 = 20
        assert out.shape == expected_shape

    def test_multi_channel_input(self):
        """Verify FPN works with multi-channel tissue input."""
        fpn = FPNBackbone(
            in_channels=5,  # 5 tissue channels
            hidden_dims=(64, 128),
            out_channels=256,
            downsample_factor=4,
        )

        x = torch.randn(2, 5, 160, 160)
        out = fpn(x)

        assert out.shape == (2, 256, 40, 40)


class TestSimpleCNNBackbone:
    """Tests for Simple CNN Backbone."""

    def test_output_shape(self):
        """Verify SimpleCNN achieves correct downsampling."""
        cnn = SimpleCNNBackbone(
            in_channels=1,
            hidden_dims=(32, 64, 128),
            out_channels=256,
            downsample_factor=8,
        )

        x = torch.randn(2, 1, 160, 160)
        out = cnn(x)

        expected_shape = (2, 256, 20, 20)
        assert out.shape == expected_shape


class TestBuildBackbone:
    """Tests for backbone factory function."""

    def test_build_fpn(self):
        """Test building FPN backbone."""
        backbone = build_backbone(
            "fpn",
            in_channels=1,
            hidden_dims=(64, 128),
            out_channels=256,
            downsample_factor=4,
        )
        assert isinstance(backbone, FPNBackbone)

    def test_build_simple(self):
        """Test building simple CNN backbone."""
        backbone = build_backbone(
            "simple",
            in_channels=1,
            hidden_dims=(32, 64, 128),
            out_channels=256,
            downsample_factor=8,
        )
        assert isinstance(backbone, SimpleCNNBackbone)


class TestPriorMapLoader:
    """Tests for Prior Map Loader."""

    @pytest.fixture
    def binary_priors_file(self, tmp_path):
        """Create a temporary binary priors file."""
        filepath = tmp_path / "zbin_priors.npz"

        # Create binary priors for 5 z-bins
        priors = {}
        for z_bin in range(5):
            mask = np.random.rand(160, 160) > 0.3
            priors[f"bin_{z_bin}"] = mask.astype(np.uint8)

        np.savez(filepath, **priors)
        return filepath

    @pytest.fixture
    def multichannel_priors_file(self, tmp_path):
        """Create a temporary multi-channel priors file."""
        filepath = tmp_path / "tissue_priors.npz"

        # Create 3-channel tissue priors for 5 z-bins
        priors = {}
        for z_bin in range(5):
            # Shape: (3, 160, 160) for 3 tissue classes
            prior = np.random.rand(3, 160, 160).astype(np.float32)
            # Normalize so channels sum to 1
            prior = prior / prior.sum(axis=0, keepdims=True)
            priors[f"prior_{z_bin}"] = prior

        np.savez(filepath, **priors)
        return filepath

    def test_load_binary_format(self, binary_priors_file):
        """Test loading binary format priors."""
        loader = PriorMapLoader(
            cache_dir=binary_priors_file.parent,
            filename=binary_priors_file.name,
            n_bins=5,
            channel_mapping={0: "background", 1: "brain"},
        )

        assert loader.format == "binary"
        assert len(loader) == 5
        assert loader.n_channels == 2

    def test_load_multichannel_format(self, multichannel_priors_file):
        """Test loading multi-channel format priors."""
        loader = PriorMapLoader(
            cache_dir=multichannel_priors_file.parent,
            filename=multichannel_priors_file.name,
            n_bins=5,
            channel_mapping={0: "bg", 1: "wm", 2: "gm"},
        )

        assert loader.format == "multichannel"
        assert len(loader) == 5
        assert loader.n_channels == 3

    def test_get_tensor(self, binary_priors_file):
        """Test getting tensors for a batch of z-bins."""
        loader = PriorMapLoader(
            cache_dir=binary_priors_file.parent,
            filename=binary_priors_file.name,
            n_bins=5,
            channel_mapping={0: "brain"},
        )

        tensor = loader.get_tensor([0, 2, 4], device="cpu", normalize=True)

        assert tensor.shape == (3, 1, 160, 160)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_channel_names(self, binary_priors_file):
        """Test channel names property."""
        loader = PriorMapLoader(
            cache_dir=binary_priors_file.parent,
            filename=binary_priors_file.name,
            n_bins=5,
            channel_mapping={0: "background", 1: "brain"},
        )

        assert loader.channel_names == ["background", "brain"]


class TestEnhancedAnatomicalPriorEncoder:
    """Tests for Enhanced Anatomical Prior Encoder."""

    def test_binary_input(self):
        """Verify encoder works with 1-channel binary input."""
        encoder = EnhancedAnatomicalPriorEncoder(
            in_channels=1,
            embed_dim=256,
            downsample_factor=4,
            positional_encoding="rope",
            use_fpn=True,
            input_size=(160, 160),
        )

        x = torch.randn(2, 1, 160, 160)
        out = encoder(x)

        # 160/4 = 40, so seq_len = 40*40 = 1600
        expected_shape = (2, 1600, 256)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

    def test_2channel_input(self):
        """Verify encoder works with 2-channel (background + brain) input."""
        encoder = EnhancedAnatomicalPriorEncoder(
            in_channels=2,
            embed_dim=256,
            downsample_factor=4,
            positional_encoding="rope",
            use_fpn=True,
        )

        x = torch.randn(2, 2, 160, 160)
        out = encoder(x)

        assert out.shape == (2, 1600, 256)

    def test_tissue_input(self):
        """Verify encoder works with 5-channel tissue input."""
        encoder = EnhancedAnatomicalPriorEncoder(
            in_channels=5,
            embed_dim=256,
            downsample_factor=4,
            positional_encoding="rope",
            use_fpn=True,
        )

        x = torch.randn(2, 5, 160, 160)
        out = encoder(x)

        assert out.shape == (2, 1600, 256)

    def test_downsample_8(self):
        """Verify encoder works with 8x downsampling."""
        encoder = EnhancedAnatomicalPriorEncoder(
            in_channels=1,
            embed_dim=256,
            hidden_dims=(32, 64, 128),
            downsample_factor=8,
            positional_encoding="sinusoidal",
            use_fpn=False,  # Use simple CNN for 8x
            input_size=(160, 160),
        )

        x = torch.randn(2, 1, 160, 160)
        out = encoder(x)

        # 160/8 = 20, so seq_len = 20*20 = 400
        expected_shape = (2, 400, 256)
        assert out.shape == expected_shape

    def test_all_positional_encodings(self):
        """Test all positional encoding types work."""
        for pos_type in ["rope", "sinusoidal", "learned"]:
            encoder = EnhancedAnatomicalPriorEncoder(
                in_channels=1,
                embed_dim=256,
                downsample_factor=4,
                positional_encoding=pos_type,
            )

            x = torch.randn(1, 1, 160, 160)
            out = encoder(x)

            assert out.shape == (1, 1600, 256), f"Failed for pos_type={pos_type}"


class TestBuildEnhancedAnatomicalEncoder:
    """Tests for the enhanced encoder factory function."""

    def test_build_from_config(self):
        """Test building encoder from config dict."""
        from omegaconf import OmegaConf

        encoder_cfg = OmegaConf.create({
            "version": "enhanced",
            "channel_mapping": {0: "background", 1: "brain"},
            "architecture": {
                "downsample_factor": 4,
                "use_fpn": True,
                "hidden_dims": [64, 128],
                "embed_dim": 256,
                "positional_encoding": "rope",
                "norm_num_groups": 8,
            },
        })

        encoder = build_enhanced_anatomical_encoder(
            encoder_cfg,
            cross_attention_dim=256,
            input_size=(160, 160),
        )

        assert isinstance(encoder, EnhancedAnatomicalPriorEncoder)
        assert encoder.in_channels == 2
        assert encoder.embed_dim == 256
        assert encoder.seq_len == 1600


class TestEncoderIntegrationWithUNet:
    """Tests for encoder output compatibility with UNet cross-attention."""

    def test_output_compatible_with_cross_attention(self):
        """Verify encoder output shape is compatible with UNet cross-attention."""
        encoder = EnhancedAnatomicalPriorEncoder(
            in_channels=1,
            embed_dim=256,  # Must match UNet cross_attention_dim
            downsample_factor=4,
        )

        x = torch.randn(4, 1, 160, 160)
        context = encoder(x)

        # Cross-attention expects (B, seq_len, embed_dim)
        assert context.dim() == 3
        assert context.shape[0] == 4  # Batch size
        assert context.shape[2] == 256  # embed_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
