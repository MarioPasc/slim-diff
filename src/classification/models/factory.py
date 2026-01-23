"""Model factory with registry for classification architectures."""

from __future__ import annotations

import logging
from pathlib import Path

import torch.nn as nn
from omegaconf import OmegaConf, DictConfig

from src.classification.models.simple_cnn import SimpleCNNClassifier, ResNetClassifier

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "simple_cnn": SimpleCNNClassifier,
    "resnet": ResNetClassifier,
}


def build_model(cfg: DictConfig, in_channels: int) -> nn.Module:
    """Build a classifier from the master config.

    Reads the model YAML path from cfg.model.config_path, loads it,
    and instantiates the appropriate model class.

    Args:
        cfg: Master configuration (classification_task.yaml).
        in_channels: Number of input channels (depends on input_mode).

    Returns:
        Instantiated classifier model.
    """
    # Resolve model config path relative to master config's directory
    model_config_path = Path(cfg.model.config_path)
    if not model_config_path.is_absolute():
        # Relative to the classification config directory
        config_dir = Path(__file__).parent.parent / "config"
        model_config_path = config_dir / model_config_path

    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    model_cfg = OmegaConf.load(model_config_path)
    model_type = model_cfg.model.type

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[model_type]

    # Build kwargs from model config, excluding 'type' and overriding in_channels
    kwargs = OmegaConf.to_container(model_cfg.model, resolve=True)
    kwargs.pop("type", None)
    kwargs["in_channels"] = in_channels

    # Convert list-like configs
    if "channels" in kwargs:
        kwargs["channels"] = list(kwargs["channels"])

    model = cls(**kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Built {model_type} classifier: in_channels={in_channels}, "
        f"params={n_params:,}"
    )
    return model
