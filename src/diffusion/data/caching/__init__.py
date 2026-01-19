"""Modular slice caching system for JS-DDPM.

This package provides a dataset-agnostic slice caching architecture using
the Template Method and Registry patterns.

Public API:
    - SliceCacheBuilder: Abstract base class for dataset-specific builders
    - DatasetRegistry: Factory for creating builders
    - get_registry: Get global registry instance

Example:
    from src.diffusion.data.caching import get_registry

    registry = get_registry()
    builder = registry.create("epilepsy", cache_config)
    builder.build_cache()
"""

from .base import SliceCacheBuilder
from .registry import DatasetRegistry, get_registry, register_dataset
from .cli import main

# Import builders to trigger auto-registration via @register_dataset decorator
from . import builders  # noqa: F401

__all__ = [
    "SliceCacheBuilder",
    "DatasetRegistry",
    "get_registry",
    "register_dataset",
    "main",  # For jsddpm-cache CLI entry point
]
