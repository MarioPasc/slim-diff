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
    GroupUncertaintyWeightedLoss,
    SimpleWeightedLoss,
    UncertaintyWeightedLoss,
)
from src.diffusion.losses.focal_frequency_loss import FocalFrequencyLoss

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

    Supports two operational modes controlled by loss.mode:

    1. "mse_channels" (default, original behavior):
       - MSE_image and MSE_mask with per-channel uncertainty weighting
       - 2 learnable log_vars (one per channel)

    2. "mse_ffl_groups":
       - MSE_image, MSE_mask, and FFL with group-level uncertainty weighting
       - 2 learnable log_vars:
         - Group 0: MSE_image + MSE_mask (spatial losses)
         - Group 1: FFL (frequency loss)
       - Requires x0 and x0_pred to be passed to forward()
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

        # Determine operational mode
        self.mode = loss_cfg.get("mode", "mse_channels")

        if self.mode == "mse_channels":
            self._init_mse_channels_mode(loss_cfg)
        elif self.mode == "mse_ffl_groups":
            self._init_mse_ffl_groups_mode(loss_cfg)
        else:
            raise ValueError(f"Unknown loss mode: {self.mode}")

        # Lesion weighting config (applies in both modes)
        # Separate weights for image and mask channels
        self.use_lesion_weighting_image = loss_cfg.lesion_weighted_image.enabled
        self.lesion_weight_image = loss_cfg.lesion_weighted_image.lesion_weight
        self.background_weight_image = loss_cfg.lesion_weighted_image.background_weight

        self.use_lesion_weighting_mask = loss_cfg.lesion_weighted_mask.enabled
        self.lesion_weight_mask = loss_cfg.lesion_weighted_mask.lesion_weight
        self.background_weight_mask = loss_cfg.lesion_weighted_mask.background_weight

        logger.info(
            f"DiffusionLoss: mode={self.mode}, "
            f"lesion_weighted_image={self.use_lesion_weighting_image}, "
            f"lesion_weighted_mask={self.use_lesion_weighting_mask}"
        )

    def _init_mse_channels_mode(self, loss_cfg: DictConfig) -> None:
        """Initialize for original mse_channels mode."""
        if loss_cfg.uncertainty_weighting.enabled:
            self.loss = UncertaintyWeightedLoss(
                n_tasks=2,
                initial_log_vars=list(loss_cfg.uncertainty_weighting.initial_log_vars),
                learnable=loss_cfg.uncertainty_weighting.learnable,
                clamp_range=loss_cfg.uncertainty_weighting.get("clamp_range", (-5.0, 5.0)),
            )
        else:
            self.loss = SimpleWeightedLoss(weights=[1.0, 1.0])

        self.ffl = None
        logger.info(
            f"  mse_channels mode: uncertainty={loss_cfg.uncertainty_weighting.enabled}"
        )

    def _init_mse_ffl_groups_mode(self, loss_cfg: DictConfig) -> None:
        """Initialize for mse_ffl_groups mode with FFL."""
        ffl_cfg = loss_cfg.ffl

        # Create FFL module
        self.ffl = FocalFrequencyLoss(
            loss_weight=ffl_cfg.get("loss_weight", 1.0),
            alpha=ffl_cfg.get("alpha", 1.0),
            patch_factor=ffl_cfg.get("patch_factor", 1),
            ave_spectrum=ffl_cfg.get("ave_spectrum", False),
            log_matrix=ffl_cfg.get("log_matrix", False),
            batch_matrix=ffl_cfg.get("batch_matrix", False),
        )

        # Group uncertainty weighting
        # Group 0: mse_image + mse_mask
        # Group 1: ffl
        group_cfg = loss_cfg.group_uncertainty_weighting
        self.loss = GroupUncertaintyWeightedLoss(
            n_groups=2,
            group_membership=[0, 0, 1],  # [mse_img, mse_mask, ffl]
            initial_log_vars=list(group_cfg.get("initial_log_vars", [0.0, 0.0])),
            learnable=group_cfg.get("learnable", True),
            clamp_range=group_cfg.get("clamp_range", (-5.0, 5.0)),
            intra_group_weights=list(group_cfg.get("intra_group_weights", [1.0, 1.0, 1.0])),
        )

        logger.info(
            f"  mse_ffl_groups mode: FFL alpha={ffl_cfg.get('alpha', 1.0)}, "
            f"group_uncertainty learnable={group_cfg.get('learnable', True)}"
        )

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        x0_mask: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        x0_pred: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute diffusion loss.

        Args:
            eps_pred: Predicted noise, shape (B, 2, H, W).
            eps_target: Target noise, shape (B, 2, H, W).
            x0_mask: Original mask for lesion weighting, shape (B, 1, H, W).
            x0: Original samples (required for mse_ffl_groups mode).
            x0_pred: Predicted x0 (required for mse_ffl_groups mode).

        Returns:
            Tuple of (total_loss, details_dict).
        """
        if self.mode == "mse_channels":
            return self._forward_mse_channels(eps_pred, eps_target, x0_mask)
        else:
            return self._forward_mse_ffl_groups(
                eps_pred, eps_target, x0_mask, x0, x0_pred
            )

    def _forward_mse_channels(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        x0_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Original forward for mse_channels mode."""
        # Split channels
        eps_pred_img = eps_pred[:, 0:1]
        eps_pred_msk = eps_pred[:, 1:2]
        eps_target_img = eps_target[:, 0:1]
        eps_target_msk = eps_target[:, 1:2]

        # Image channel loss (optionally lesion-weighted)
        if self.use_lesion_weighting_image and x0_mask is not None:
            loss_img = lesion_weighted_mse(
                eps_pred_img,
                eps_target_img,
                x0_mask,
                self.lesion_weight_image,
                self.background_weight_image,
            )
        else:
            loss_img = F.mse_loss(eps_pred_img, eps_target_img)

        # Mask channel loss (optionally lesion-weighted)
        if self.use_lesion_weighting_mask and x0_mask is not None:
            loss_msk = lesion_weighted_mse(
                eps_pred_msk,
                eps_target_msk,
                x0_mask,
                self.lesion_weight_mask,
                self.background_weight_mask,
            )
        else:
            loss_msk = F.mse_loss(eps_pred_msk, eps_target_msk)

        # Call the self.loss to combine; Kendall or Simple is decided at init
        total_loss, details = self.loss([loss_img, loss_msk])

        # Add named losses to details
        details["loss_image"] = loss_img.detach()
        details["loss_mask"] = loss_msk.detach()

        return total_loss, details

    def _forward_mse_ffl_groups(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        x0_mask: torch.Tensor | None,
        x0: torch.Tensor | None,
        x0_pred: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward for mse_ffl_groups mode with FFL."""
        if x0 is None or x0_pred is None:
            raise ValueError("mse_ffl_groups mode requires x0 and x0_pred")

        # Split channels for MSE
        eps_pred_img = eps_pred[:, 0:1]
        eps_pred_msk = eps_pred[:, 1:2]
        eps_target_img = eps_target[:, 0:1]
        eps_target_msk = eps_target[:, 1:2]

        # MSE losses (optionally lesion-weighted)
        if self.use_lesion_weighting_image and x0_mask is not None:
            loss_img = lesion_weighted_mse(
                eps_pred_img,
                eps_target_img,
                x0_mask,
                self.lesion_weight_image,
                self.background_weight_image,
            )
        else:
            loss_img = F.mse_loss(eps_pred_img, eps_target_img)

        if self.use_lesion_weighting_mask and x0_mask is not None:
            loss_msk = lesion_weighted_mse(
                eps_pred_msk,
                eps_target_msk,
                x0_mask,
                self.lesion_weight_mask,
                self.background_weight_mask,
            )
        else:
            loss_msk = F.mse_loss(eps_pred_msk, eps_target_msk)

        # FFL on x0_pred vs x0 (image channel only)
        x0_pred_img = x0_pred[:, 0:1]
        x0_img = x0[:, 0:1]
        loss_ffl, ffl_details = self.ffl(x0_pred_img, x0_img)

        # Group uncertainty weighting: [mse_img, mse_msk, ffl]
        total_loss, details = self.loss([loss_img, loss_msk, loss_ffl])

        # Add named losses to details
        details["loss_image"] = loss_img.detach()
        details["loss_mask"] = loss_msk.detach()
        details["loss_ffl"] = loss_ffl.detach()
        details.update(ffl_details)

        return total_loss, details

    def get_log_vars(self) -> torch.Tensor | None:
        """Get log variance values if using uncertainty weighting.

        Returns:
            Log variance tensor or None.
        """
        if isinstance(self.loss, (UncertaintyWeightedLoss, GroupUncertaintyWeightedLoss)):
            return self.loss.get_log_vars()
        return None