"""Conditioning utilities for JS-DDPM.

Provides functions for computing conditioning tokens that encode
both z-position and pathology class.
"""

from __future__ import annotations

import torch


def compute_z_bin(
    z_index: int | torch.Tensor,
    z_range: tuple[int, int],
    n_bins: int = 50,
) -> int | torch.Tensor:
    """Compute z-bin from z-index using LOCAL binning within z_range.

    Args:
        z_index: Z-index within z_range.
        z_range: (min_z, max_z) range of valid slices (inclusive).
        n_bins: Number of bins.

    Returns:
        Z-bin index (0 to n_bins - 1).

    Raises:
        ValueError: If z_index is outside z_range.
    """
    min_z, max_z = z_range
    range_size = max_z - min_z

    if isinstance(z_index, torch.Tensor):
        # Check bounds
        if (z_index < min_z).any() or (z_index > max_z).any():
            raise ValueError(
                f"Some z_index values outside z_range [{min_z}, {max_z}]"
            )

        # Normalize within range
        if range_size == 0:
            return torch.zeros_like(z_index, dtype=torch.long)

        z_norm = (z_index.float() - min_z) / range_size
        z_bin = (z_norm * n_bins).long()
        return z_bin.clamp(0, n_bins - 1)
    else:
        # Check bounds
        if z_index < min_z or z_index > max_z:
            raise ValueError(
                f"z_index {z_index} outside z_range [{min_z}, {max_z}]"
            )

        # Normalize within range
        if range_size == 0:
            return 0

        z_norm = (z_index - min_z) / range_size
        z_bin = int(z_norm * n_bins)
        return min(max(z_bin, 0), n_bins - 1)


def compute_class_token(
    z_bin: int | torch.Tensor,
    pathology_class: int | torch.Tensor,
    n_bins: int = 50,
) -> int | torch.Tensor:
    """Compute the conditioning token from z-bin and pathology class.

    Token encoding: token = z_bin + pathology_class * n_bins

    - Tokens 0 to n_bins-1: class 0 (no lesion / control)
    - Tokens n_bins to 2*n_bins-1: class 1 (lesion present)

    Args:
        z_bin: Z-position bin (0 to n_bins - 1).
        pathology_class: 0 for no lesion, 1 for lesion.
        n_bins: Number of z-bins.

    Returns:
        Conditioning token (0 to 2*n_bins - 1).
    """
    if isinstance(z_bin, torch.Tensor):
        return z_bin + pathology_class * n_bins
    else:
        return z_bin + pathology_class * n_bins


def get_token_for_condition(
    z_bin: int,
    pathology_class: int,
    n_bins: int = 50,
) -> int:
    """Get conditioning token for a specific condition.

    Convenience function for generation/visualization.

    Args:
        z_bin: Z-position bin.
        pathology_class: 0 for control/no lesion, 1 for lesion.
        n_bins: Number of z-bins.

    Returns:
        Conditioning token.
    """
    return compute_class_token(z_bin, pathology_class, n_bins)


def token_to_condition(
    token: int | torch.Tensor,
    n_bins: int = 50,
) -> tuple[int | torch.Tensor, int | torch.Tensor]:
    """Decode token back to z_bin and pathology_class.

    Args:
        token: Conditioning token.
        n_bins: Number of z-bins.

    Returns:
        Tuple of (z_bin, pathology_class).
    """
    if isinstance(token, torch.Tensor):
        pathology_class = (token >= n_bins).long()
        z_bin = token % n_bins
        return z_bin, pathology_class
    else:
        pathology_class = 1 if token >= n_bins else 0
        z_bin = token % n_bins
        return z_bin, pathology_class


def get_null_token(n_bins: int = 50) -> int:
    """Get the null token for classifier-free guidance.

    The null token is 2 * n_bins, reserved for unconditional generation.

    Args:
        n_bins: Number of z-bins.

    Returns:
        Null token value.
    """
    return 2 * n_bins


def prepare_cfg_tokens(
    tokens: torch.Tensor,
    null_token: int,
    dropout_prob: float = 0.1,
    training: bool = True,
) -> torch.Tensor:
    """Prepare tokens for classifier-free guidance training.

    Randomly drops tokens to null_token with dropout_prob during training.

    Args:
        tokens: Original conditioning tokens, shape (B,).
        null_token: Value of the null token.
        dropout_prob: Probability of dropping to null token.
        training: Whether in training mode.

    Returns:
        Tokens with some possibly replaced by null_token.
    """
    if not training or dropout_prob <= 0:
        return tokens

    # Create dropout mask
    mask = torch.rand(tokens.shape, device=tokens.device) < dropout_prob
    result = tokens.clone()
    result[mask] = null_token

    return result


def get_visualization_tokens(
    z_bins: list[int],
    n_bins: int = 50,
) -> tuple[list[int], list[int]]:
    """Get tokens for visualization grid.

    Returns tokens for control (class 0) and lesion (class 1)
    conditions at specified z-bins.

    Args:
        z_bins: List of z-bins to visualize.
        n_bins: Number of z-bins.

    Returns:
        Tuple of (control_tokens, lesion_tokens).
    """
    control_tokens = [get_token_for_condition(zb, 0, n_bins) for zb in z_bins]
    lesion_tokens = [get_token_for_condition(zb, 1, n_bins) for zb in z_bins]
    return control_tokens, lesion_tokens
