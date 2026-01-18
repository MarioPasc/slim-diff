"""Anatomical Prior Encoder for cross-attention conditioning.

This module provides a lightweight CNN encoder that transforms anatomical
z-bin priors into spatial context embeddings suitable for cross-attention
in the diffusion UNet.

Design Rationale:
-----------------
The anatomical prior is a binary mask indicating expected brain regions at
each z-bin (axial slice position). Instead of concatenating this mask as an
input channel, cross-attention allows the model to *selectively attend* to
spatial regions of the prior, enabling:

1. **Multi-scale guidance**: Cross-attention at different UNet levels allows
   resolution-appropriate spatial guidance.
2. **Learned selectivity**: Attention weights can adaptively focus on ambiguous
   regions (boundaries) vs. confident regions.
3. **Reduced input burden**: Avoids further increasing input channels when
   combined with self-conditioning.

Architecture:
------------
The encoder uses a lightweight CNN to extract spatial features, then flattens
them into a sequence of "spatial tokens" for cross-attention:

    Input: (B, 1, H, W) prior mask in [-1, 1]
    ↓
    CNN backbone (3-4 conv layers with downsampling)
    ↓
    Spatial features: (B, C, H', W')
    ↓
    Flatten + 2D positional encoding
    ↓
    Output: (B, H'*W', embed_dim) context for cross-attention

Each output token represents a spatial region of the input mask, allowing
the UNet's attention layers to focus on specific anatomical regions.

References:
----------
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion
  Models" (Stable Diffusion) - cross-attention for text conditioning
- Vaswani et al., "Attention Is All You Need" - positional encodings
- Park et al., "Semantic Image Synthesis with Spatially-Adaptive Normalization"
  (SPADE) - spatial conditioning for semantic maps
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Sinusoidal2DPositionalEncoding(nn.Module):
    """2D sinusoidal positional encoding for spatial tokens.

    Generates fixed positional encodings based on (x, y) grid positions,
    allowing the model to understand spatial relationships between tokens.
    Uses sine/cosine functions at different frequencies for each dimension.
    """

    def __init__(
        self,
        embed_dim: int,
        max_h: int = 32,
        max_w: int = 32,
    ) -> None:
        """Initialize the positional encoding.

        Args:
            embed_dim: Embedding dimension (must be divisible by 4).
            max_h: Maximum height in tokens.
            max_w: Maximum width in tokens.
        """
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D PE"

        self.embed_dim = embed_dim
        self.max_h = max_h
        self.max_w = max_w

        # Precompute positional encodings
        pe = self._build_pe(max_h, max_w, embed_dim)
        self.register_buffer("pe", pe)  # (max_h, max_w, embed_dim)

    def _build_pe(
        self, max_h: int, max_w: int, embed_dim: int
    ) -> torch.Tensor:
        """Build 2D positional encoding tensor.

        Args:
            max_h: Maximum height.
            max_w: Maximum width.
            embed_dim: Embedding dimension.

        Returns:
            Positional encoding tensor of shape (max_h, max_w, embed_dim).
        """
        dim_per_axis = embed_dim // 2  # Half for height, half for width

        # Create position grids
        y_pos = torch.arange(max_h).unsqueeze(1)  # (max_h, 1)
        x_pos = torch.arange(max_w).unsqueeze(1)  # (max_w, 1)

        # Frequency terms
        div_term = torch.exp(
            torch.arange(0, dim_per_axis, 2) * (-math.log(10000.0) / dim_per_axis)
        )

        # Compute encodings for each axis
        pe_y = torch.zeros(max_h, dim_per_axis)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term)

        pe_x = torch.zeros(max_w, dim_per_axis)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)

        # Combine: (max_h, 1, dim/2) + (1, max_w, dim/2) -> broadcast
        pe_y = pe_y.unsqueeze(1).expand(max_h, max_w, dim_per_axis)
        pe_x = pe_x.unsqueeze(0).expand(max_h, max_w, dim_per_axis)

        # Concatenate to form full encoding
        pe = torch.cat([pe_y, pe_x], dim=-1)  # (max_h, max_w, embed_dim)

        return pe

    def forward(self, h: int, w: int) -> torch.Tensor:
        """Get positional encodings for a given spatial size.

        Args:
            h: Height in tokens.
            w: Width in tokens.

        Returns:
            Positional encodings of shape (h*w, embed_dim).
        """
        # Slice to requested size and flatten
        pe = self.pe[:h, :w, :]  # (h, w, embed_dim)
        return pe.reshape(h * w, self.embed_dim)


class Learned2DPositionalEncoding(nn.Module):
    """Learned 2D positional encoding for spatial tokens.

    Uses learnable embeddings for each (x, y) position, allowing the
    model to learn optimal spatial representations for the task.
    """

    def __init__(
        self,
        embed_dim: int,
        max_h: int = 32,
        max_w: int = 32,
    ) -> None:
        """Initialize the positional encoding.

        Args:
            embed_dim: Embedding dimension.
            max_h: Maximum height in tokens.
            max_w: Maximum width in tokens.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_h = max_h
        self.max_w = max_w

        # Separate embeddings for rows and columns (more parameter efficient)
        self.row_embed = nn.Embedding(max_h, embed_dim // 2)
        self.col_embed = nn.Embedding(max_w, embed_dim // 2)

        # Initialize with small values
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)

    def forward(self, h: int, w: int) -> torch.Tensor:
        """Get positional encodings for a given spatial size.

        Args:
            h: Height in tokens.
            w: Width in tokens.

        Returns:
            Positional encodings of shape (h*w, embed_dim).
        """
        # Get row and column embeddings
        row_indices = torch.arange(h, device=self.row_embed.weight.device)
        col_indices = torch.arange(w, device=self.col_embed.weight.device)

        row_emb = self.row_embed(row_indices)  # (h, embed_dim/2)
        col_emb = self.col_embed(col_indices)  # (w, embed_dim/2)

        # Create grid of positional embeddings
        row_emb = row_emb.unsqueeze(1).expand(h, w, -1)  # (h, w, embed_dim/2)
        col_emb = col_emb.unsqueeze(0).expand(h, w, -1)  # (h, w, embed_dim/2)

        # Concatenate and flatten
        pe = torch.cat([row_emb, col_emb], dim=-1)  # (h, w, embed_dim)
        return pe.reshape(h * w, self.embed_dim)


class AnatomicalPriorEncoder(nn.Module):
    """Lightweight CNN encoder for anatomical prior masks.

    Transforms anatomical z-bin priors into spatial context embeddings
    suitable for cross-attention conditioning in the diffusion UNet.

    The encoder processes the input mask through a series of convolutional
    layers with downsampling, then flattens the spatial features into a
    sequence of tokens. Each token represents a region of the original
    mask and can be attended to by the UNet's cross-attention layers.

    Args:
        in_channels: Number of input channels (1 for binary mask).
        embed_dim: Output embedding dimension (must match cross_attention_dim).
        hidden_dims: Tuple of hidden channel dimensions for conv layers.
        downsample_factor: Total spatial downsampling factor (power of 2).
        positional_encoding: Type of positional encoding ("sinusoidal" or "learned").
        input_size: Expected input spatial size (H, W) for positional encoding.
        norm_num_groups: Number of groups for GroupNorm.

    Example:
        >>> encoder = AnatomicalPriorEncoder(
        ...     in_channels=1,
        ...     embed_dim=256,
        ...     hidden_dims=(32, 64, 128),
        ...     downsample_factor=8,
        ...     input_size=(128, 128),
        ... )
        >>> prior = torch.randn(4, 1, 128, 128)  # (B, 1, H, W)
        >>> context = encoder(prior)
        >>> context.shape
        torch.Size([4, 256, 256])  # (B, seq_len=16*16, embed_dim)
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        hidden_dims: tuple[int, ...] = (32, 64, 128),
        downsample_factor: int = 8,
        positional_encoding: Literal["sinusoidal", "learned"] = "sinusoidal",
        input_size: tuple[int, int] = (128, 128),
        norm_num_groups: int = 8,
    ) -> None:
        """Initialize the encoder."""
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.downsample_factor = downsample_factor
        self.input_size = input_size

        # Calculate output spatial size
        self.output_h = input_size[0] // downsample_factor
        self.output_w = input_size[1] // downsample_factor
        self.seq_len = self.output_h * self.output_w

        # Build convolutional backbone
        self.backbone = self._build_backbone(
            in_channels, hidden_dims, downsample_factor, norm_num_groups
        )

        # Final projection to embed_dim
        self.proj = nn.Conv2d(hidden_dims[-1], embed_dim, kernel_size=1)

        # Positional encoding
        if positional_encoding == "sinusoidal":
            self.pos_encoding = Sinusoidal2DPositionalEncoding(
                embed_dim, self.output_h, self.output_w
            )
        elif positional_encoding == "learned":
            self.pos_encoding = Learned2DPositionalEncoding(
                embed_dim, self.output_h, self.output_w
            )
        else:
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")

        # Log encoder configuration
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"AnatomicalPriorEncoder: {n_params:,} params, "
            f"output shape (B, {self.seq_len}, {embed_dim}), "
            f"positional_encoding={positional_encoding}"
        )

    def _build_backbone(
        self,
        in_channels: int,
        hidden_dims: tuple[int, ...],
        downsample_factor: int,
        norm_num_groups: int,
    ) -> nn.Sequential:
        """Build the convolutional backbone.

        Constructs a series of conv layers that progressively downsample
        the spatial dimensions while increasing channel depth.

        Args:
            in_channels: Number of input channels.
            hidden_dims: Tuple of hidden channel dimensions.
            downsample_factor: Target downsampling factor.
            norm_num_groups: Number of groups for GroupNorm.

        Returns:
            Sequential module containing the backbone layers.
        """
        # Calculate number of downsampling stages needed
        n_stages = int(math.log2(downsample_factor))
        assert 2 ** n_stages == downsample_factor, "downsample_factor must be power of 2"
        assert len(hidden_dims) >= n_stages, "Need at least as many hidden_dims as downsample stages"

        layers = []
        prev_channels = in_channels

        for i in range(n_stages):
            out_channels = hidden_dims[i]

            # Ensure norm_num_groups divides out_channels
            groups = min(norm_num_groups, out_channels)
            while out_channels % groups != 0:
                groups -= 1

            layers.extend([
                nn.Conv2d(
                    prev_channels, out_channels,
                    kernel_size=3, stride=2, padding=1
                ),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
            ])
            prev_channels = out_channels

        # Add remaining hidden dims without downsampling (if any)
        for i in range(n_stages, len(hidden_dims)):
            out_channels = hidden_dims[i]

            groups = min(norm_num_groups, out_channels)
            while out_channels % groups != 0:
                groups -= 1

            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
            ])
            prev_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode anatomical prior mask into cross-attention context.

        Args:
            x: Anatomical prior mask, shape (B, 1, H, W) with values in [-1, 1].
               1 indicates in-brain regions, -1 indicates out-of-brain.

        Returns:
            Context embeddings for cross-attention, shape (B, seq_len, embed_dim).
            seq_len = (H/downsample_factor) * (W/downsample_factor).
        """
        B = x.shape[0]

        # CNN backbone: (B, 1, H, W) -> (B, hidden_dims[-1], H', W')
        features = self.backbone(x)

        # Project to embed_dim: (B, hidden_dims[-1], H', W') -> (B, embed_dim, H', W')
        features = self.proj(features)

        # Get actual spatial dimensions (may differ from expected if input size varies)
        _, _, h, w = features.shape

        # Flatten spatial dimensions: (B, embed_dim, H', W') -> (B, H'*W', embed_dim)
        features = features.flatten(2).transpose(1, 2)  # (B, seq_len, embed_dim)

        # Add positional encoding
        pos_enc = self.pos_encoding(h, w)  # (seq_len, embed_dim)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)  # (B, seq_len, embed_dim)
        features = features + pos_enc.to(features.device, features.dtype)

        return features


def build_anatomical_encoder(
    cfg,
    cross_attention_dim: int,
) -> AnatomicalPriorEncoder:
    """Factory function to build AnatomicalPriorEncoder from config.

    Args:
        cfg: Configuration object with model.anatomical_encoder settings.
        cross_attention_dim: The cross_attention_dim from the UNet.

    Returns:
        Configured AnatomicalPriorEncoder instance.
    """
    # Get encoder config, with defaults
    encoder_cfg = cfg.model.get("anatomical_encoder", {})

    # Default hidden dims based on cross_attention_dim
    default_hidden = (32, 64, min(128, cross_attention_dim))

    encoder = AnatomicalPriorEncoder(
        in_channels=1,
        embed_dim=cross_attention_dim,
        hidden_dims=tuple(encoder_cfg.get("hidden_dims", default_hidden)),
        downsample_factor=encoder_cfg.get("downsample_factor", 8),
        positional_encoding=encoder_cfg.get("positional_encoding", "sinusoidal"),
        input_size=tuple(cfg.data.transforms.roi_size[:2]),  # (H, W) from ROI
        norm_num_groups=encoder_cfg.get("norm_num_groups", 8),
    )

    return encoder
