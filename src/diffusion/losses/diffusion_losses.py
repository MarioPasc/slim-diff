"""Diffusion-specific loss functions.

Provides MSE losses for epsilon prediction with optional
per-channel weighting and lesion-aware mask loss.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.diffusion.losses.uncertainty import (
    SimpleWeightedLoss,
    UncertaintyWeightedLoss,
)

logger = logging.getLogger(__name__)


def mse_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute MSE loss per channel.

    Args:
        pred: Predictions, shape (B, C, H, W).
        target: Targets, shape (B, C, H, W).
        reduction: Reduction mode ("mean", "none", "sum").

    Returns:
        Per-channel MSE losses, shape (C,) if reduction="mean" over batch.
    """
    # MSE per pixel
    mse = (pred - target) ** 2

    if reduction == "none":
        return mse

    # Mean over spatial dimensions
    mse = mse.mean(dim=(2, 3))  # (B, C)

    if reduction == "mean":
        return mse.mean(dim=0)  # (C,)
    elif reduction == "sum":
        return mse.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def lesion_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lesion_weight: float = 2.0,
    background_weight: float = 1.0,
) -> torch.Tensor:
    """Compute lesion-weighted MSE for mask channel.

    Gives higher weight to lesion pixels (mask > 0 in {-1, +1} space).

    Args:
        pred: Predictions for mask channel, shape (B, 1, H, W).
        target: Target noise for mask channel, shape (B, 1, H, W).
        mask: Ground truth mask in {-1, +1}, shape (B, 1, H, W).
        lesion_weight: Weight for lesion pixels.
        background_weight: Weight for background pixels.

    Returns:
        Weighted MSE loss (scalar).
    """
    # Create weight map
    # mask > 0 means lesion in {-1, +1} space
    weights = torch.where(
        mask > 0,
        torch.full_like(mask, lesion_weight),
        torch.full_like(mask, background_weight),
    )

    # Compute weighted MSE
    mse = (pred - target) ** 2 * weights
    return mse.mean()


class DiffusionLoss(nn.Module):
    """Complete diffusion loss module.

    Combines per-channel MSE with optional:
    - Uncertainty weighting (Kendall)
    - Lesion-weighted mask loss
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        """Initialize the loss module.

        Args:
            cfg: Configuration object.
        """
        super().__init__()
        self.cfg = cfg
        loss_cfg = cfg.loss

        # Setup uncertainty weighting
        if loss_cfg.uncertainty_weighting.enabled:
            self.loss = UncertaintyWeightedLoss(
                n_tasks=2,
                initial_log_vars=list(loss_cfg.uncertainty_weighting.initial_log_vars),
                learnable=loss_cfg.uncertainty_weighting.learnable,
                clamp_range=loss_cfg.uncertainty_weighting.get("clamp_range", (-5.0, 5.0)),
            )
        else:
            self.loss = SimpleWeightedLoss(weights=[1.0, 1.0])

        # Lesion weighting config
        self.use_lesion_weighting = loss_cfg.lesion_weighted_mask.enabled
        self.lesion_weight = loss_cfg.lesion_weighted_mask.lesion_weight
        self.background_weight = loss_cfg.lesion_weighted_mask.background_weight
        

        logger.info(
            f"DiffusionLoss: "
            f"uncertainty={loss_cfg.uncertainty_weighting.enabled}, "
            f"lesion_weighted={self.use_lesion_weighting}"
        )

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        x0_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute diffusion loss.

        Args:
            eps_pred: Predicted noise, shape (B, 2, H, W).
            eps_target: Target noise, shape (B, 2, H, W).
            x0_mask: Original mask for lesion weighting, shape (B, 1, H, W).

        Returns:
            Tuple of (total_loss, details_dict).
        """
        # Split channels
        eps_pred_img = eps_pred[:, 0:1]
        eps_pred_msk = eps_pred[:, 1:2]
        eps_target_img = eps_target[:, 0:1]
        eps_target_msk = eps_target[:, 1:2]

        # Image channel loss (standard MSE)
        loss_img = F.mse_loss(eps_pred_img, eps_target_img)

        # Mask channel loss
        if self.use_lesion_weighting and x0_mask is not None:
            loss_msk = lesion_weighted_mse(
                eps_pred_msk,
                eps_target_msk,
                x0_mask,
                self.lesion_weight,
                self.background_weight,
            )
        else:
            loss_msk = F.mse_loss(eps_pred_msk, eps_target_msk)

        # Call the self.loss to combine; Kendall or Simple is decided at init
        total_loss, details = self.loss([loss_img, loss_msk])

        # Add named losses to details
        details["loss_image"] = loss_img.detach()
        details["loss_mask"] = loss_msk.detach()

        return total_loss, details

    def get_log_vars(self) -> torch.Tensor | None:
        """Get log variance values if using uncertainty weighting.

        Returns:
            Log variance tensor or None.
        """
        if isinstance(self.loss, UncertaintyWeightedLoss):
            return self.loss.get_log_vars()
        return None