"""I/O utilities for loading NPZ samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz_sample(npz_path: Path | str) -> dict:
    """Load a sample from NPZ file.

    Args:
        npz_path: Path to NPZ file

    Returns:
        Dictionary with 'image', 'mask', and metadata
    """
    data = np.load(npz_path, allow_pickle=True)

    result = {
        "image": data["image"],  # (128, 128) float32 in [-1, 1]
        "mask": data["mask"],    # (128, 128) float32 in {-1, +1}
    }

    # Load metadata (stored as scalar arrays)
    for key in data.files:
        if key not in ("image", "mask"):
            value = data[key]
            # Convert scalar arrays to Python types
            if hasattr(value, "ndim") and value.ndim == 0:
                result[key] = value.item()
            else:
                result[key] = value

    return result
