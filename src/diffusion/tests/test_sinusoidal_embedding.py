"""Tests for sinusoidal position encoding."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.diffusion.model.embeddings import ConditionalEmbeddingWithSinusoidal
from src.diffusion.model.factory import build_model


def test_conditional_embedding_forward():
    """Test ConditionalEmbeddingWithSinusoidal forward pass."""
    embedding = ConditionalEmbeddingWithSinusoidal(
        num_embeddings=100,  # 2 * 50 z_bins
        embedding_dim=256,
        z_bins=50,
        use_sinusoidal=True,
        max_z=127,
    )

    # Test various tokens
    # Token = z_bin + pathology_class * z_bins
    tokens = torch.tensor([
        0,   # class=0, z_bin=0
        25,  # class=0, z_bin=25
        49,  # class=0, z_bin=49
        50,  # class=1, z_bin=0
        75,  # class=1, z_bin=25
        99,  # class=1, z_bin=49
    ])

    embeddings = embedding(tokens)

    # Check output shape
    assert embeddings.shape == (6, 256)

    # Check embeddings are different for different tokens
    assert not torch.allclose(embeddings[0], embeddings[1])
    assert not torch.allclose(embeddings[0], embeddings[3])


def test_conditional_embedding_null_token():
    """Test ConditionalEmbeddingWithSinusoidal handles null token."""
    embedding = ConditionalEmbeddingWithSinusoidal(
        num_embeddings=101,  # 2 * 50 z_bins + 1 for CFG
        embedding_dim=256,
        z_bins=50,
        use_sinusoidal=True,
        max_z=127,
    )

    # Null token is 100 (2 * z_bins)
    tokens = torch.tensor([0, 50, 100])

    embeddings = embedding(tokens)

    # Check output shape
    assert embeddings.shape == (3, 256)

    # Null token embedding should be different from others
    assert not torch.allclose(embeddings[0], embeddings[2])
    assert not torch.allclose(embeddings[1], embeddings[2])


def test_conditional_embedding_without_sinusoidal():
    """Test ConditionalEmbeddingWithSinusoidal with use_sinusoidal=False."""
    embedding = ConditionalEmbeddingWithSinusoidal(
        num_embeddings=100,
        embedding_dim=256,
        z_bins=50,
        use_sinusoidal=False,  # Disable sinusoidal encoding
        max_z=127,
    )

    tokens = torch.tensor([0, 25, 50, 75])
    embeddings = embedding(tokens)

    # Check output shape
    assert embeddings.shape == (4, 256)

    # Should still produce different embeddings
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_build_model_with_sinusoidal():
    """Test building model with use_sinusoidal=True."""
    cfg = OmegaConf.create({
        "conditioning": {
            "z_bins": 50,
            "use_sinusoidal": True,
            "max_z": 127,
            "cfg": {
                "enabled": False,
                "null_token": 100,
            },
        },
        "model": {
            "type": "DiffusionModelUNet",
            "spatial_dims": 2,
            "in_channels": 2,
            "out_channels": 2,
            "channels": [32, 64],
            "attention_levels": [False, True],
            "num_res_blocks": 1,
            "num_head_channels": 16,
            "norm_num_groups": 8,
            "use_class_embedding": True,
            "dropout": 0.0,
            "resblock_updown": False,
            "with_conditioning": False,
        },
    })

    model = build_model(cfg)

    # Check that class_embedding was replaced
    assert isinstance(model.class_embedding, ConditionalEmbeddingWithSinusoidal)
    assert model.class_embedding.use_sinusoidal is True

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 2, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    tokens = torch.randint(0, 100, (batch_size,))

    output = model(x, timesteps, class_labels=tokens)

    # Check output shape
    assert output.shape == (batch_size, 2, 64, 64)


def test_build_model_without_sinusoidal():
    """Test building model with use_sinusoidal=False."""
    cfg = OmegaConf.create({
        "conditioning": {
            "z_bins": 50,
            "use_sinusoidal": False,
            "max_z": 127,
            "cfg": {
                "enabled": False,
                "null_token": 100,
            },
        },
        "model": {
            "type": "DiffusionModelUNet",
            "spatial_dims": 2,
            "in_channels": 2,
            "out_channels": 2,
            "channels": [32, 64],
            "attention_levels": [False, True],
            "num_res_blocks": 1,
            "num_head_channels": 16,
            "norm_num_groups": 8,
            "use_class_embedding": True,
            "dropout": 0.0,
            "resblock_updown": False,
            "with_conditioning": False,
        },
    })

    model = build_model(cfg)

    # Check that class_embedding is NOT replaced (standard MONAI embedding)
    assert not isinstance(model.class_embedding, ConditionalEmbeddingWithSinusoidal)
    assert type(model.class_embedding).__name__ == "Embedding"

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 2, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    tokens = torch.randint(0, 100, (batch_size,))

    output = model(x, timesteps, class_labels=tokens)

    # Check output shape
    assert output.shape == (batch_size, 2, 64, 64)
