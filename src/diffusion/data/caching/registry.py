"""Dataset registry for slice cache builders.

Implements the Registry and Factory patterns for creating dataset-specific
cache builders.
"""

from __future__ import annotations

import logging
from typing import Type

from omegaconf import DictConfig

from .base import SliceCacheBuilder

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Singleton registry for dataset-specific cache builders.

    This class implements both the Singleton and Factory patterns:
    - Singleton: Only one registry instance exists globally
    - Factory: Creates builder instances from dataset names

    Usage:
        # Get registry instance
        registry = DatasetRegistry()

        # Register builders (usually via decorator)
        registry.register("epilepsy", EpilepsySliceCacheBuilder)

        # Create builder from config
        builder = registry.create("epilepsy", cache_config)
        builder.build_cache()
    """

    _instance: DatasetRegistry | None = None

    def __new__(cls):
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._builders: dict[str, Type[SliceCacheBuilder]] = {}
        return cls._instance

    def register(
        self,
        name: str,
        builder_class: Type[SliceCacheBuilder],
    ) -> None:
        """Register a dataset builder class.

        Args:
            name: Dataset identifier (e.g., "epilepsy", "brats_men")
            builder_class: SliceCacheBuilder subclass

        Raises:
            TypeError: If builder_class doesn't inherit from SliceCacheBuilder
        """
        if not issubclass(builder_class, SliceCacheBuilder):
            raise TypeError(
                f"{builder_class.__name__} must inherit from SliceCacheBuilder"
            )

        self._builders[name] = builder_class
        logger.info(
            f"Registered dataset builder: {name} -> {builder_class.__name__}"
        )

    def create(
        self,
        name: str,
        cache_config: DictConfig,
        train_config: DictConfig | None = None,
    ) -> SliceCacheBuilder:
        """Create a cache builder instance from config.

        This is the factory method that instantiates the appropriate builder
        class based on the dataset name.

        Args:
            name: Dataset identifier
            cache_config: Cache configuration
            train_config: Optional training configuration

        Returns:
            Initialized SliceCacheBuilder instance

        Raises:
            ValueError: If dataset name not registered
        """
        if name not in self._builders:
            available = ", ".join(self._builders.keys())
            raise ValueError(
                f"Unknown dataset: '{name}'. "
                f"Available datasets: {available or 'none'}"
            )

        builder_class = self._builders[name]
        logger.info(f"Creating {builder_class.__name__} for dataset '{name}'")
        return builder_class(cache_config, train_config)

    def list_available(self) -> list[str]:
        """Get list of registered dataset names.

        Returns:
            List of registered dataset identifiers
        """
        return list(self._builders.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a dataset name is registered.

        Args:
            name: Dataset identifier

        Returns:
            True if registered, False otherwise
        """
        return name in self._builders


# Global registry instance
_registry = DatasetRegistry()


def get_registry() -> DatasetRegistry:
    """Get the global dataset registry instance.

    Returns:
        DatasetRegistry singleton instance
    """
    return _registry


def register_dataset(name: str):
    """Decorator for auto-registration of dataset builders.

    This decorator automatically registers a builder class when it's defined,
    eliminating the need for manual registration.

    Usage:
        @register_dataset("epilepsy")
        class EpilepsySliceCacheBuilder(SliceCacheBuilder):
            ...

    Args:
        name: Dataset identifier

    Returns:
        Decorator function
    """

    def decorator(cls: Type[SliceCacheBuilder]):
        _registry.register(name, cls)
        return cls

    return decorator
