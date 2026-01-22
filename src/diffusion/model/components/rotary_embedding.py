"""2D Rotary Position Embedding (RoPE) for spatial features.

Implements 2D rotary position encoding following Su et al. (2021) "RoFormer:
Enhanced Transformer with Rotary Position Embedding", extended to 2D grids.

Reference:
    https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding for spatial tokens.

    Unlike additive positional encodings, RoPE encodes position through
    rotation of feature vectors. This provides better relative position
    awareness and generalizes better to unseen sequence lengths.

    For 2D grids, we split the embedding dimension into 4 parts:
    - First half for y-axis encoding (split into sin/cos pairs)
    - Second half for x-axis encoding (split into sin/cos pairs)

    Args:
        embed_dim: Embedding dimension (must be divisible by 4).
        max_h: Maximum height for precomputed frequencies.
        max_w: Maximum width for precomputed frequencies.
        base: Base for frequency computation (default 10000.0).
    """

    def __init__(
        self,
        embed_dim: int,
        max_h: int = 64,
        max_w: int = 64,
        base: float = 10000.0,
    ):
        super().__init__()

        if embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")

        self.embed_dim = embed_dim
        self.max_h = max_h
        self.max_w = max_w
        self.base = base

        # Dimension per axis (y and x each get half)
        self.dim_per_axis = embed_dim // 2

        # Precompute frequency tables
        self._precompute_freqs()

    def _precompute_freqs(self) -> None:
        """Precompute sin/cos frequency tables for y and x axes.

        For RoPE, we rotate pairs of dimensions. For dim_per_axis dimensions,
        we need dim_per_axis/2 frequency values, but we repeat each for the
        pair of dimensions it rotates.
        """
        # Number of dimension pairs per axis
        half_dim = self.dim_per_axis // 2

        # Compute frequency for each dimension pair
        freqs = 1.0 / (
            self.base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        )

        # Position indices
        y_pos = torch.arange(self.max_h, dtype=torch.float32)
        x_pos = torch.arange(self.max_w, dtype=torch.float32)

        # Outer product: (max_pos, half_dim)
        y_freqs = torch.outer(y_pos, freqs)  # (max_h, half_dim)
        x_freqs = torch.outer(x_pos, freqs)  # (max_w, half_dim)

        # Compute sin and cos
        y_sin = y_freqs.sin()  # (max_h, half_dim)
        y_cos = y_freqs.cos()  # (max_h, half_dim)
        x_sin = x_freqs.sin()  # (max_w, half_dim)
        x_cos = x_freqs.cos()  # (max_w, half_dim)

        # Repeat each value for both dimensions in the pair
        # (max_h, half_dim) -> (max_h, dim_per_axis)
        y_sin = y_sin.repeat_interleave(2, dim=-1)
        y_cos = y_cos.repeat_interleave(2, dim=-1)
        x_sin = x_sin.repeat_interleave(2, dim=-1)
        x_cos = x_cos.repeat_interleave(2, dim=-1)

        # Register as buffers (not trainable parameters)
        self.register_buffer("y_sin", y_sin, persistent=False)
        self.register_buffer("y_cos", y_cos, persistent=False)
        self.register_buffer("x_sin", x_sin, persistent=False)
        self.register_buffer("x_cos", x_cos, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input.

        For RoPE, we rotate pairs of dimensions using a rotation matrix:
        [cos(θ), -sin(θ)]   [x1]
        [sin(θ),  cos(θ)] × [x2]

        This is equivalent to: [x1*cos - x2*sin, x1*sin + x2*cos]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Apply 2D rotary position embedding.

        Args:
            x: Input tensor of shape (B, seq_len, embed_dim) where seq_len = h * w.
            h: Height of the spatial grid.
            w: Width of the spatial grid.

        Returns:
            Position-encoded tensor of same shape (B, seq_len, embed_dim).
        """
        B, seq_len, D = x.shape
        assert seq_len == h * w, f"seq_len {seq_len} != h*w {h}*{w}"
        assert D == self.embed_dim, f"embed_dim mismatch: {D} != {self.embed_dim}"

        # Reshape to grid: (B, h, w, D)
        x = x.view(B, h, w, D)

        # Split into y and x components
        # y_part: first half, x_part: second half
        y_part = x[..., : self.dim_per_axis]
        x_part = x[..., self.dim_per_axis :]

        # Get sin/cos for this grid size
        # y positions: broadcast across w dimension
        y_sin = self.y_sin[:h].unsqueeze(1)  # (h, 1, dim_per_axis)
        y_cos = self.y_cos[:h].unsqueeze(1)  # (h, 1, dim_per_axis)

        # x positions: broadcast across h dimension
        x_sin = self.x_sin[:w].unsqueeze(0)  # (1, w, dim_per_axis)
        x_cos = self.x_cos[:w].unsqueeze(0)  # (1, w, dim_per_axis)

        # Apply rotation to y_part based on y position
        y_rotated = y_part * y_cos + self._rotate_half(y_part) * y_sin

        # Apply rotation to x_part based on x position
        x_rotated = x_part * x_cos + self._rotate_half(x_part) * x_sin

        # Concatenate back
        out = torch.cat([y_rotated, x_rotated], dim=-1)

        # Reshape back to sequence: (B, h*w, D)
        out = out.view(B, seq_len, D)

        return out

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, max_h={self.max_h}, max_w={self.max_w}"


class LearnedPositionEmbedding2D(nn.Module):
    """Learned 2D position embedding (fallback alternative to RoPE).

    Args:
        embed_dim: Embedding dimension.
        max_h: Maximum height.
        max_w: Maximum width.
    """

    def __init__(self, embed_dim: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_h = max_h
        self.max_w = max_w

        # Separate embeddings for y and x, then combine
        self.y_embed = nn.Embedding(max_h, embed_dim // 2)
        self.x_embed = nn.Embedding(max_w, embed_dim // 2)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Apply learned 2D position embedding (additive).

        Args:
            x: Input tensor of shape (B, seq_len, embed_dim).
            h: Height of the spatial grid.
            w: Width of the spatial grid.

        Returns:
            Position-encoded tensor of same shape.
        """
        B, seq_len, D = x.shape
        assert seq_len == h * w

        # Create position indices
        y_pos = torch.arange(h, device=x.device)
        x_pos = torch.arange(w, device=x.device)

        # Get embeddings: (h, D//2) and (w, D//2)
        y_emb = self.y_embed(y_pos)  # (h, D//2)
        x_emb = self.x_embed(x_pos)  # (w, D//2)

        # Broadcast to grid: (h, w, D//2) each
        y_emb = y_emb.unsqueeze(1).expand(-1, w, -1)  # (h, w, D//2)
        x_emb = x_emb.unsqueeze(0).expand(h, -1, -1)  # (h, w, D//2)

        # Concatenate and flatten: (h*w, D)
        pos_emb = torch.cat([y_emb, x_emb], dim=-1).view(seq_len, D)

        # Add to input (broadcast over batch)
        return x + pos_emb.unsqueeze(0)

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, max_h={self.max_h}, max_w={self.max_w}"


class SinusoidalPositionEmbedding2D(nn.Module):
    """Fixed sinusoidal 2D position embedding (additive).

    Similar to the original Transformer positional encoding but extended to 2D.

    Args:
        embed_dim: Embedding dimension.
        max_h: Maximum height.
        max_w: Maximum width.
        temperature: Temperature for frequency scaling (default 10000).
    """

    def __init__(
        self,
        embed_dim: int,
        max_h: int = 64,
        max_w: int = 64,
        temperature: float = 10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_h = max_h
        self.max_w = max_w

        # Precompute embeddings
        pe = self._build_pe(max_h, max_w, embed_dim, temperature)
        self.register_buffer("pe", pe, persistent=False)

    def _build_pe(
        self, max_h: int, max_w: int, embed_dim: int, temperature: float
    ) -> torch.Tensor:
        """Build 2D sinusoidal position embedding."""
        # Half dims for y, half for x
        half_dim = embed_dim // 2

        # Frequency scaling
        dim_t = torch.arange(half_dim // 2, dtype=torch.float32)
        dim_t = temperature ** (2 * dim_t / half_dim)

        # Position grids
        y_pos = torch.arange(max_h, dtype=torch.float32)
        x_pos = torch.arange(max_w, dtype=torch.float32)

        # Compute embeddings
        y_embed = y_pos.unsqueeze(1) / dim_t.unsqueeze(0)  # (max_h, half_dim//2)
        x_embed = x_pos.unsqueeze(1) / dim_t.unsqueeze(0)  # (max_w, half_dim//2)

        # Interleave sin/cos
        y_pe = torch.stack([y_embed.sin(), y_embed.cos()], dim=-1).flatten(1)  # (max_h, half_dim)
        x_pe = torch.stack([x_embed.sin(), x_embed.cos()], dim=-1).flatten(1)  # (max_w, half_dim)

        # Combine into grid: (max_h, max_w, embed_dim)
        y_pe = y_pe.unsqueeze(1).expand(-1, max_w, -1)
        x_pe = x_pe.unsqueeze(0).expand(max_h, -1, -1)
        pe = torch.cat([y_pe, x_pe], dim=-1)

        return pe  # (max_h, max_w, embed_dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Apply sinusoidal 2D position embedding (additive).

        Args:
            x: Input tensor of shape (B, seq_len, embed_dim).
            h: Height of the spatial grid.
            w: Width of the spatial grid.

        Returns:
            Position-encoded tensor of same shape.
        """
        B, seq_len, D = x.shape
        assert seq_len == h * w

        # Get relevant portion and flatten
        pe = self.pe[:h, :w, :].reshape(seq_len, D)

        # Add to input
        return x + pe.unsqueeze(0)

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, max_h={self.max_h}, max_w={self.max_w}"


def build_position_embedding_2d(
    pos_type: str,
    embed_dim: int,
    max_h: int = 64,
    max_w: int = 64,
    **kwargs,
) -> nn.Module:
    """Factory function for 2D position embeddings.

    Args:
        pos_type: Type of position embedding ("rope", "sinusoidal", "learned").
        embed_dim: Embedding dimension.
        max_h: Maximum height.
        max_w: Maximum width.
        **kwargs: Additional arguments for specific embedding types.

    Returns:
        Position embedding module.
    """
    if pos_type == "rope":
        return RotaryPositionEmbedding2D(
            embed_dim=embed_dim,
            max_h=max_h,
            max_w=max_w,
            base=kwargs.get("base", 10000.0),
        )
    elif pos_type == "sinusoidal":
        return SinusoidalPositionEmbedding2D(
            embed_dim=embed_dim,
            max_h=max_h,
            max_w=max_w,
            temperature=kwargs.get("temperature", 10000.0),
        )
    elif pos_type == "learned":
        return LearnedPositionEmbedding2D(
            embed_dim=embed_dim,
            max_h=max_h,
            max_w=max_w,
        )
    else:
        raise ValueError(f"Unknown position embedding type: {pos_type}")
