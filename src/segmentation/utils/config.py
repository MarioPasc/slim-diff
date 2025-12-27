"""Configuration loading and merging utilities."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_and_merge_configs(
    master_path: Path | str,
    model_name: str,
    cli_overrides: dict | None = None,
) -> DictConfig:
    """Load and merge master + model-specific configurations.

    Args:
        master_path: Path to master.yaml
        model_name: Model name (unet, dynunet, unetplusplus, swinunetr)
        cli_overrides: Optional CLI overrides as dict

    Returns:
        Merged configuration
    """
    # Load master config
    master_cfg = OmegaConf.load(master_path)

    # Load model-specific config
    master_dir = Path(master_path).parent
    model_path = master_dir / "models" / f"{model_name}.yaml"

    if not model_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_path}")

    model_cfg = OmegaConf.load(model_path)

    # Merge configurations (model takes precedence)
    cfg = OmegaConf.merge(master_cfg, model_cfg)

    # Apply CLI overrides if provided
    if cli_overrides:
        cli_cfg = OmegaConf.create(cli_overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Resolve interpolations
    OmegaConf.resolve(cfg)

    return cfg
