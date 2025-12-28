"""Z-position encoding utilities.

Provides functions for normalizing and quantizing z-position indices
into discrete bins for conditioning.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def normalize_z_local(
    z_index: int,
    z_range: tuple[int, int],
) -> float:
    """Normalize z-index to [0, 1] range within the local z_range.

    Args:
        z_index: Current z-index.
        z_range: (min_z, max_z) range of valid slices (inclusive).

    Returns:
        Normalized z-position in [0, 1] within the range.

    Raises:
        ValueError: If z_index is outside z_range.
    """
    min_z, max_z = z_range

    if z_index < min_z or z_index > max_z:
        raise ValueError(
            f"z_index {z_index} outside z_range [{min_z}, {max_z}]"
        )

    range_size = max_z - min_z
    if range_size == 0:
        return 0.0

    return (z_index - min_z) / range_size


def quantize_z(
    z_index: int,
    z_range: tuple[int, int],
    n_bins: int,
) -> int:
    """Quantize z-position into discrete bins using LOCAL binning within z_range.

    This function bins slices WITHIN the z_range, ensuring all bins are used
    during training and each bin represents a subset of the training slices.

    Example:
        >>> # With z_range=[24, 93] and n_bins=10:
        >>> # Bin 0: slices [24-30]
        >>> # Bin 1: slices [31-37]
        >>> # ...
        >>> # Bin 9: slices [87-93]
        >>> quantize_z(24, z_range=(24, 93), n_bins=10)
        0
        >>> quantize_z(55, z_range=(24, 93), n_bins=10)
        4

    Args:
        z_index: Current z-index.
        z_range: (min_z, max_z) range of valid slices (inclusive).
        n_bins: Number of bins.

    Returns:
        Bin index in [0, n_bins - 1].

    Raises:
        ValueError: If z_index is outside z_range.
    """
    z_norm = normalize_z_local(z_index, z_range)
    z_bin = int(z_norm * n_bins)
    # Clamp to valid range (handles edge case where z_norm == 1.0)
    return min(max(z_bin, 0), n_bins - 1)


def z_bin_to_index(
    z_bin: int,
    z_range: tuple[int, int],
    n_bins: int,
) -> int:
    """Convert z-bin back to approximate z-index using LOCAL binning.

    Args:
        z_bin: Bin index.
        z_range: (min_z, max_z) range of valid slices (inclusive).
        n_bins: Number of bins.

    Returns:
        Approximate z-index (center of bin).

    Example:
        >>> # With z_range=[24, 93] and n_bins=10, bin 5 maps to center of [59-65]
        >>> z_bin_to_index(5, z_range=(24, 93), n_bins=10)
        62
    """
    min_z, max_z = z_range
    range_size = max_z - min_z

    if range_size == 0:
        return min_z

    # Take center of bin in normalized space
    z_norm = (z_bin + 0.5) / n_bins

    # Map back to z_range
    z_index = min_z + int(z_norm * range_size)

    # Clamp to valid range
    return min(max(z_index, min_z), max_z)


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal position encoding for z-position.

    Optional module for encoding z-position as continuous values
    rather than discrete tokens. Not used by default (MONAI's
    num_class_embeds handles discrete conditioning).
    """

    def __init__(
        self,
        dim: int,
        max_positions: int = 128,
    ) -> None:
        """Initialize the encoding.

        Args:
            dim: Embedding dimension.
            max_positions: Maximum number of positions to encode.
        """
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

        # Precompute position encodings
        position = torch.arange(max_positions).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )

        pe = torch.zeros(max_positions, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, z_indices: torch.Tensor) -> torch.Tensor:
        """Get position encodings for z-indices.

        Args:
            z_indices: Tensor of z-indices, shape (B,).

        Returns:
            Position encodings, shape (B, dim).
        """
        return self.pe[z_indices]


class ZPositionEncoder(nn.Module):
    """Encoder for z-position with optional learnable projection.

    Combines z-bin embedding with optional sinusoidal encoding.
    """

    def __init__(
        self,
        n_bins: int,
        embed_dim: int,
        use_sinusoidal: bool = False,
        max_z: int = 128,
    ) -> None:
        """Initialize the encoder.

        Args:
            n_bins: Number of z-bins.
            embed_dim: Output embedding dimension.
            use_sinusoidal: Whether to use sinusoidal encoding.
            max_z: Maximum z-index for sinusoidal encoding.
        """
        super().__init__()
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        self.use_sinusoidal = use_sinusoidal

        # Learnable bin embeddings
        self.bin_embedding = nn.Embedding(n_bins, embed_dim)

        # Optional sinusoidal encoding
        if use_sinusoidal:
            self.sinusoidal = SinusoidalPositionEncoding(embed_dim, max_z)
            self.combine = nn.Linear(embed_dim * 2, embed_dim)

    def forward(
        self,
        z_bins: torch.Tensor,
        z_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode z-positions.

        Args:
            z_bins: Tensor of z-bin indices, shape (B,).
            z_indices: Optional tensor of z-indices for sinusoidal encoding.

        Returns:
            Position embeddings, shape (B, embed_dim).
        """
        # Get bin embeddings
        bin_emb = self.bin_embedding(z_bins)

        if self.use_sinusoidal and z_indices is not None:
            sin_emb = self.sinusoidal(z_indices)
            combined = torch.cat([bin_emb, sin_emb], dim=-1)
            return self.combine(combined)

        return bin_emb


class ConditionalEmbeddingWithSinusoidal(nn.Module):
    """Embedding module that handles both pathology class and z-position.

    Designed to replace MONAI's class_embedding layer when sinusoidal
    encoding is enabled. Handles tokens that encode both z_bin and
    pathology_class: token = z_bin + pathology_class * z_bins.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        z_bins: int,
        z_range: tuple[int, int],
        use_sinusoidal: bool = True,
        max_z: int = 127,
    ) -> None:
        """Initialize the embedding module.

        Args:
            num_embeddings: Total number of token classes (2*z_bins or 2*z_bins+1).
            embedding_dim: Output embedding dimension.
            z_bins: Number of z-position bins.
            z_range: (min_z, max_z) range of valid slices (inclusive) for LOCAL binning.
            use_sinusoidal: Whether to use sinusoidal encoding for z-position.
            max_z: Maximum z-index for sinusoidal encoding (size of sinusoidal lookup table).
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.z_bins = z_bins
        self.z_range = z_range
        self.use_sinusoidal = use_sinusoidal
        self.max_z = max_z

        # Embeddings for pathology classes (control=0, lesion=1)
        self.pathology_embedding = nn.Embedding(2, embedding_dim)

        if use_sinusoidal:
            # Sinusoidal encoding for z-position
            self.z_encoder = ZPositionEncoder(
                n_bins=z_bins,
                embed_dim=embedding_dim,
                use_sinusoidal=True,
                max_z=max_z,
            )
            # Combine pathology and z-position embeddings
            self.combine = nn.Linear(embedding_dim * 2, embedding_dim)
        else:
            # Standard learned embeddings for z-bins
            self.z_embedding = nn.Embedding(z_bins, embedding_dim)
            # Combine pathology and z-position embeddings
            self.combine = nn.Linear(embedding_dim * 2, embedding_dim)

        # Null embedding for CFG (classifier-free guidance)
        self.null_embedding = nn.Parameter(
            torch.randn(1, embedding_dim) * 0.02
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute embeddings from tokens.

        Args:
            tokens: Token indices, shape (B,).
                    tokens = z_bin + pathology_class * z_bins

        Returns:
            Embeddings, shape (B, embedding_dim).
        """
        # Decode tokens into z_bin and pathology_class
        # Handle null token (used for CFG): it's the last token
        is_null = tokens >= (2 * self.z_bins)

        # For null tokens, use dummy values (will be embedded separately)
        safe_tokens = torch.where(is_null, torch.zeros_like(tokens), tokens)

        pathology_class = safe_tokens // self.z_bins  # 0 or 1
        z_bin = safe_tokens % self.z_bins  # 0 to z_bins-1

        # Clamp to valid ranges
        pathology_class = torch.clamp(pathology_class, 0, 1)
        z_bin = torch.clamp(z_bin, 0, self.z_bins - 1)

        # Get pathology embedding
        path_emb = self.pathology_embedding(pathology_class)

        # Get z-position embedding
        if self.use_sinusoidal:
            # Convert z_bin back to approximate z_index for sinusoidal encoding
            # Use LOCAL binning: bins span z_range, not [0, max_z]
            min_z, max_z_range = self.z_range
            range_size = max_z_range - min_z

            # Map bin to center of local range
            z_norm = (z_bin.float() + 0.5) / self.z_bins
            z_indices = (min_z + z_norm * range_size).long()

            # Clamp to z_range and sinusoidal table size
            z_indices = torch.clamp(z_indices, min_z, max_z_range)
            z_indices = torch.clamp(z_indices, 0, self.max_z)

            z_emb = self.z_encoder(z_bin, z_indices)
        else:
            z_emb = self.z_embedding(z_bin)

        # Combine embeddings
        combined = torch.cat([path_emb, z_emb], dim=-1)
        emb = self.combine(combined)

        # Replace null token embeddings
        if is_null.any():
            emb = torch.where(
                is_null.unsqueeze(-1),
                self.null_embedding.expand(emb.shape[0], -1),
                emb,
            )

        return emb
