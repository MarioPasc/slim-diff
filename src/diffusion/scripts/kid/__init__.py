"""KID (Kernel Inception Distance) evaluation module for synthetic replicas."""

from .kid import (
    compute_global_kid,
    compute_zbin_kid,
    extract_features_batched,
    load_replica,
    load_test_slices,
    preprocess_for_inception,
)

__all__ = [
    "load_test_slices",
    "load_replica",
    "preprocess_for_inception",
    "extract_features_batched",
    "compute_global_kid",
    "compute_zbin_kid",
]
