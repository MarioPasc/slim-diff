"""Configuration utilities for slice caching.

Provides functions for loading cache configs and migrating from legacy format.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_cache_config(path: str | Path) -> DictConfig:
    """Load and validate cache configuration.

    Args:
        path: Path to cache_config.yaml

    Returns:
        Loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Validate required fields
    validate_cache_config(cfg)

    return cfg


def validate_cache_config(cfg: DictConfig) -> None:
    """Validate cache configuration schema.

    Args:
        cfg: Configuration to validate

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = [
        "dataset_type",
        "cache_dir",
        "z_bins",
        "slice_sampling",
        "datasets",
        "transforms",
    ]

    for field in required_fields:
        if field not in cfg:
            raise ValueError(f"Missing required field in cache config: {field}")

    # Validate dataset_type is a string
    if not isinstance(cfg.dataset_type, str):
        raise ValueError(f"dataset_type must be a string, got {type(cfg.dataset_type)}")

    # Validate z_bins is a positive integer
    if not isinstance(cfg.z_bins, int) or cfg.z_bins <= 0:
        raise ValueError(f"z_bins must be a positive integer, got {cfg.z_bins}")

    # Validate z_range is either "auto" or a list of two integers
    z_range = cfg.slice_sampling.z_range
    if isinstance(z_range, str):
        if z_range.lower() != "auto":
            raise ValueError(f"z_range must be 'auto' or [min, max], got '{z_range}'")
    elif hasattr(z_range, '__len__') and hasattr(z_range, '__getitem__'):  # List-like (list or ListConfig)
        # Convert to list for validation
        z_range_list = list(z_range)
        if len(z_range_list) != 2:
            raise ValueError(f"z_range must have exactly 2 elements, got {len(z_range_list)}")
        if not all(isinstance(z, int) for z in z_range_list):
            raise ValueError(f"z_range values must be integers, got {z_range_list}")
    else:
        raise ValueError(f"z_range must be 'auto' or [min, max], got {type(z_range)}")

    # Validate datasets exists and is not empty
    if not cfg.datasets:
        raise ValueError("datasets section cannot be empty")

    logger.info("Cache config validation passed")


def migrate_legacy_config(legacy_cfg: DictConfig) -> DictConfig:
    """Convert legacy jsddpm.yaml to cache_config.yaml format.

    This allows backwards compatibility with existing epilepsy configs.

    Args:
        legacy_cfg: Old configuration (jsddpm.yaml format)

    Returns:
        New cache configuration

    Example:
        legacy_cfg = OmegaConf.load("src/diffusion/config/jsddpm.yaml")
        cache_cfg = migrate_legacy_config(legacy_cfg)
        builder = registry.create("epilepsy", cache_cfg)
    """
    logger.info("Migrating legacy config to new cache config format")

    # Check if this is actually a legacy config
    if "dataset_type" in legacy_cfg:
        logger.warning("Config appears to be already in new format, returning as-is")
        return legacy_cfg

    # Extract relevant fields from legacy config
    data_cfg = legacy_cfg.data
    cond_cfg = legacy_cfg.conditioning

    cache_cfg = OmegaConf.create({
        "dataset_type": "epilepsy",  # Legacy configs are always epilepsy
        "cache_dir": data_cfg.cache_dir,
        "z_bins": cond_cfg.z_bins,

        "slice_sampling": {
            "z_range": data_cfg.slice_sampling.z_range,
            "auto_z_range_offset": data_cfg.slice_sampling.get(
                "auto_z_range_offset", 5
            ),
            "filter_empty_brain": data_cfg.slice_sampling.filter_empty_brain,
            "brain_threshold": data_cfg.slice_sampling.brain_threshold,
            "brain_min_fraction": data_cfg.slice_sampling.brain_min_fraction,
        },

        "lesion_area_min_pixels": 0,  # Default, can be overridden via CLI
        "drop_healthy_patients": False,  # Default, can be overridden via CLI

        "datasets": {
            "epilepsy": {
                "root_dir": data_cfg.root_dir,
                "epilepsy_dataset": {
                    "name": data_cfg.epilepsy.name,
                    "modality_index": data_cfg.epilepsy.modality_index,
                },
                "control_dataset": {
                    "name": data_cfg.control.name,
                    "modality_index": data_cfg.control.modality_index,
                },
                "splits": {
                    "use_predefined_test": data_cfg.splits.use_predefined_test,
                    "val_fraction": data_cfg.splits.val_fraction,
                    "control_test_fraction": data_cfg.splits.get("control_test_fraction", 0.15),
                    "seed": data_cfg.splits.seed,
                },
            }
        },

        "transforms": data_cfg.transforms,

        "postprocessing": legacy_cfg.get("postprocessing", {}),
    })

    logger.info("Legacy config migration complete")

    return cache_cfg


def load_train_config(path: str | Path) -> DictConfig:
    """Load training configuration.

    This is a simple wrapper around OmegaConf.load for consistency.

    Args:
        path: Path to train_config.yaml (e.g., jsddpm.yaml)

    Returns:
        Loaded configuration
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config file not found: {config_path}")

    return OmegaConf.load(config_path)
