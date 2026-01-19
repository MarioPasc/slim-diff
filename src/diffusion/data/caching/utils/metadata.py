"""Metadata computation utilities for slice caching.

Re-exports existing metadata functions for convenience.
"""

# Re-export from existing modules
from src.diffusion.utils.zbin_priors import (
    compute_zbin_priors,
    save_zbin_priors,
    load_zbin_priors,
)

__all__ = [
    "compute_zbin_priors",
    "save_zbin_priors",
    "load_zbin_priors",
]
