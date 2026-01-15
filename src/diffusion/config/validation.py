"""Configuration validation for JS-DDPM."""

from __future__ import annotations

import warnings

from omegaconf import DictConfig


def validate_conditioning_config(cfg: DictConfig) -> None:
    """Validate conditioning configuration parameters.

    Raises:
        ValueError: If configuration is invalid.
    """
    cond_cfg = cfg.conditioning

    # 1. Validate max_z (moved from #2)
    max_z = cond_cfg.max_z
    if max_z <= 0:
        raise ValueError(f"conditioning.max_z must be positive, got {max_z}")

    # Warn about common indexing mistake
    if max_z == 128:
        # If dataset uses 0-127 indexing, max_z should be 127
        # This is a warning, not an error, since we don't know the dataset
        warnings.warn(
            "conditioning.max_z is set to 128. If your dataset uses "
            "0-indexed slices (0-127), this should be 127 instead.",
            UserWarning,
        )

    # 2. Validate use_sinusoidal consistency with max_z
    if cond_cfg.use_sinusoidal and max_z <= 0:
        raise ValueError(
            f"When use_sinusoidal=True, max_z must be positive, got {max_z}"
        )

    # 3. Validate CFG null token
    if cond_cfg.cfg.enabled:
        z_bins = cond_cfg.z_bins
        null_token = cond_cfg.cfg.null_token

        # Calculate num_class_embeds (from factory.py logic)
        num_class_embeds = 2 * z_bins + 1  # +1 for CFG

        if null_token >= num_class_embeds:
            raise ValueError(
                f"CFG is enabled but null_token={null_token} is out of range. "
                f"With z_bins={z_bins}, num_class_embeds={num_class_embeds}. "
                f"Set null_token to {2 * z_bins} or increase num_class_embeds."
            )

    # 4. Validate z_bins
    z_bins = cond_cfg.z_bins
    if z_bins <= 0:
        raise ValueError(f"conditioning.z_bins must be positive, got {z_bins}")


def validate_loss_config(cfg: DictConfig) -> None:
    """Validate loss configuration parameters.

    Raises:
        ValueError: If configuration is invalid.
    """
    loss_cfg = cfg.loss

    # Validate uncertainty weighting config
    if loss_cfg.uncertainty_weighting.enabled:
        initial_log_vars = loss_cfg.uncertainty_weighting.initial_log_vars
        if len(initial_log_vars) != 2:
            raise ValueError(
                f"uncertainty_weighting.initial_log_vars must have 2 values "
                f"(image, mask), got {len(initial_log_vars)}"
            )

    # Validate FFL mode dependencies
    loss_mode = loss_cfg.get("mode", "mse_channels")
    if loss_mode == "mse_ffl_groups":
        if "ffl" not in loss_cfg:
            raise ValueError(
                "loss.mode='mse_ffl_groups' requires loss.ffl configuration. "
                "Add loss.ffl section with alpha, loss_weight, etc."
            )
        if "group_uncertainty_weighting" not in loss_cfg:
            raise ValueError(
                "loss.mode='mse_ffl_groups' requires loss.group_uncertainty_weighting configuration."
            )


def validate_model_config(cfg: DictConfig) -> None:
    """Validate model configuration parameters.

    Raises:
        ValueError: If configuration is invalid.
    """
    model_cfg = cfg.model

    # Validate anatomical conditioning dependencies
    use_anatomical = model_cfg.get("anatomical_conditioning", False)
    if use_anatomical:
        pp_cfg = cfg.get("postprocessing", {})
        zbin_cfg = pp_cfg.get("zbin_priors", {})
        priors_enabled = zbin_cfg.get("enabled", False)

        if not priors_enabled:
            raise ValueError(
                "model.anatomical_conditioning=true requires "
                "postprocessing.zbin_priors.enabled=true. "
                "Anatomical conditioning needs priors to be computed and loaded."
            )

        # Verify priors file is specified
        if not zbin_cfg.get("priors_filename"):
            raise ValueError(
                "model.anatomical_conditioning=true requires "
                "postprocessing.zbin_priors.priors_filename to be specified."
            )


def validate_cfg_config(cfg: DictConfig) -> None:
    """Validate classifier-free guidance configuration.

    Raises:
        ValueError: If configuration is invalid.
    """
    cond_cfg = cfg.conditioning
    cfg_enabled = cond_cfg.cfg.enabled

    if cfg_enabled:
        z_bins = cond_cfg.z_bins
        null_token = cond_cfg.cfg.null_token

        # Calculate num_class_embeds
        num_class_embeds = 2 * z_bins + 1  # +1 for CFG

        if null_token >= num_class_embeds:
            raise ValueError(
                f"CFG null_token={null_token} is out of range. "
                f"With z_bins={z_bins}, num_class_embeds={num_class_embeds}. "
                f"Set null_token to {2 * z_bins}."
            )
    else:
        # Warn if null_token is specified but CFG is disabled
        if cond_cfg.cfg.get("null_token") is not None:
            warnings.warn(
                "conditioning.cfg.null_token is specified but cfg.enabled=false. "
                "This parameter will be ignored.",
                UserWarning,
            )


def validate_config(cfg: DictConfig) -> None:
    """Run all configuration validation checks.

    Args:
        cfg: Full configuration object.

    Raises:
        ValueError: If any configuration is invalid.
    """
    validate_conditioning_config(cfg)
    validate_loss_config(cfg)
    validate_model_config(cfg)
    validate_cfg_config(cfg)
