"""Configuration loading and validation for model comparison."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Required fields in the configuration
REQUIRED_FIELDS = [
    "root_results_dir",
    "models",
    "visualizations.enabled",
    "output.output_dir",
]

# Default configuration values
DEFAULT_CONFIG = {
    "visualizations": {
        "enabled": [
            "loss_curves",
            "uncertainty_evolution",
            "timestep_mse_heatmap",
            "reconstruction_comparison",
            "summary_table",
        ],
        "formats": ["png"],
        "dpi": 300,
    },
    "loss_curves": {
        "metrics": [
            "train/loss",
            "val/loss",
            "train/loss_image",
            "train/loss_mask",
        ],
        "smoothing_window": 10,
        "figsize": [14, 10],
    },
    "uncertainty_evolution": {
        "metrics": [
            "train/log_var_mse_group",
            "train/log_var_ffl_group",
            "train/sigma_mse_group",
            "train/sigma_ffl_group",
        ],
        "figsize": [14, 8],
    },
    "timestep_mse_heatmap": {
        "channels": ["image", "mask"],
        "epoch": -1,
        "figsize": [12, 8],
        "cmap": "viridis",
    },
    "reconstruction_comparison": {
        "cache_config_path": "src/diffusion/config/cache/epilepsy.yaml",
        "timestep": 100,
        "n_zbins": 5,
        "min_lesion_pixels": 25,
        "max_lesion_pixels": 500,
        "n_samples_per_bin": 1,
        "seed": 42,
        "use_ema": True,
        "figsize": [20, 12],
        "residual_cmap": "RdBu_r",
        "residual_vmin": -0.5,
        "residual_vmax": 0.5,
        "overlay_alpha": 0.4,
        "overlay_color": [0, 255, 0],
    },
    "summary_table": {
        "metrics": [
            "val/loss",
            "train/loss",
            "val/loss_image",
            "val/loss_mask",
        ],
        "epoch": -1,
        "primary_metric": "val/loss",
        "lower_is_better": True,
    },
    "publication_panel": {
        "figsize": [18, 5],
        "smoothing_window": 10,
        "heatmap_cmap": "viridis",
    },
    "output": {
        "output_dir": "./outputs/model_comparison",
        "timestamp": True,
        "save_data": True,
    },
}


def load_config(config_path: Path) -> DictConfig:
    """Load and validate model comparison configuration.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Validated DictConfig object with defaults applied.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required fields are missing or invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load user config
    user_cfg = OmegaConf.load(config_path)

    # Merge with defaults (user config takes precedence)
    default_cfg = OmegaConf.create(DEFAULT_CONFIG)
    cfg = OmegaConf.merge(default_cfg, user_cfg)

    # Validate required fields
    _validate_required_fields(cfg)

    # Validate model paths
    _validate_model_config(cfg)

    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Models to compare: {list(cfg.models.keys())}")
    logger.info(f"Enabled visualizations: {cfg.visualizations.enabled}")

    return cfg


def _validate_required_fields(cfg: DictConfig) -> None:
    """Validate that all required fields are present.

    Args:
        cfg: Configuration object.

    Raises:
        ValueError: If required fields are missing.
    """
    missing = []
    for field in REQUIRED_FIELDS:
        try:
            value = OmegaConf.select(cfg, field)
            if value is None:
                missing.append(field)
        except Exception:
            missing.append(field)

    if missing:
        raise ValueError(f"Missing required configuration fields: {missing}")


def _validate_model_config(cfg: DictConfig) -> None:
    """Validate model configuration.

    Args:
        cfg: Configuration object.

    Raises:
        ValueError: If model configuration is invalid.
    """
    if not cfg.models:
        raise ValueError("No models specified in configuration")

    if len(cfg.models) < 1:
        raise ValueError("At least one model must be specified")


def validate_model_paths(cfg: DictConfig) -> dict[str, Path]:
    """Validate that all model directories exist and contain required files.

    Args:
        cfg: Configuration object.

    Returns:
        Dictionary mapping model names to validated absolute paths.

    Raises:
        FileNotFoundError: If model directory or required files are missing.
    """
    root_dir = Path(cfg.root_results_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root results directory not found: {root_dir}")

    validated_paths = {}
    missing_items = []

    for model_name, relative_path in cfg.models.items():
        model_dir = root_dir / relative_path

        if not model_dir.exists():
            missing_items.append(f"Model directory not found: {model_dir}")
            continue

        # Check for required files
        csv_path = model_dir / "csv_logs" / "performance.csv"
        if not csv_path.exists():
            missing_items.append(f"Performance CSV not found for {model_name}: {csv_path}")
            continue

        checkpoint_dir = model_dir / "checkpoints"
        if not checkpoint_dir.exists():
            logger.warning(f"No checkpoints directory for {model_name}: {checkpoint_dir}")
            # Not fatal - reconstruction comparison will fail gracefully

        validated_paths[model_name] = model_dir

    if missing_items:
        for item in missing_items:
            logger.error(item)
        if not validated_paths:
            raise FileNotFoundError(
                "No valid model directories found. Check paths in configuration."
            )
        logger.warning(
            f"Continuing with {len(validated_paths)} valid models out of {len(cfg.models)}"
        )

    return validated_paths


def get_model_results_paths(model_dir: Path) -> dict[str, Path]:
    """Get paths to results files for a single model.

    Args:
        model_dir: Path to model results directory.

    Returns:
        Dictionary with keys:
            - 'csv': Path to performance.csv
            - 'histograms_dir': Path to histograms directory
            - 'checkpoint_dir': Path to checkpoints directory
            - 'config': Path to model config YAML (if exists)
    """
    paths = {
        "csv": model_dir / "csv_logs" / "performance.csv",
        "histograms_dir": model_dir / "csv_logs" / "histograms",
        "checkpoint_dir": model_dir / "checkpoints",
    }

    # Try to find config file
    for config_name in ["config.yaml", "*.yaml"]:
        config_files = list(model_dir.glob(config_name))
        if config_files:
            # Prefer config.yaml, otherwise take first match
            for cf in config_files:
                if cf.name == "config.yaml":
                    paths["config"] = cf
                    break
            if "config" not in paths:
                paths["config"] = config_files[0]
            break

    return paths


def get_visualization_config(
    cfg: DictConfig, viz_name: str
) -> DictConfig:
    """Get configuration for a specific visualization.

    Args:
        cfg: Full configuration object.
        viz_name: Visualization name (e.g., 'loss_curves').

    Returns:
        Visualization-specific configuration merged with defaults.
    """
    viz_cfg = OmegaConf.select(cfg, viz_name, default=DictConfig({}))

    # Add common settings
    viz_cfg = OmegaConf.merge(
        viz_cfg,
        {
            "formats": cfg.visualizations.formats,
            "dpi": cfg.visualizations.dpi,
        },
    )

    return viz_cfg
