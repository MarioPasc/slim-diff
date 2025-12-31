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


def spatial_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    spatial_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE weighted by a spatial map.
    
    CRITICAL FIX: This now normalizes by TOTAL pixels, not the sum of weights.
    This prevents gradient explosion when the mask is sparse.
    """
    mse = (pred - target) ** 2
    
    # Weighted Sum of Errors
    weighted_sum = (mse * spatial_weights).sum()
    
    # Normalize by TOTAL pixels (Batch * C * H * W) to keep gradient scale consistent
    # with standard MSE.
    total_pixels = pred.numel()
    
    return weighted_sum / total_pixels


def lesion_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lesion_weight: float = 10.0,
    background_weight: float = 1.0,
) -> torch.Tensor:
    """Compute MSE with specific weighting for lesion vs background.
    
    CRITICAL FIX: Uses Global Batch Normalization (total_pixels).
    """
    # Create weight map
    weights = torch.ones_like(pred) * background_weight
    weights[mask > 0.5] = lesion_weight

    # Calculate weighted MSE
    return spatial_weighted_mse(pred, target, weights)


class DiffusionLoss(nn.Module):
    """Loss module for diffusion training.

    Computes:
    1. Image noise prediction loss (MSE)
    2. Mask prediction loss (if applicable)
    3. Uncertainty weighting between tasks
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the loss module.

        Args:
            cfg: Configuration object.
        """
        super().__init__()
        self.cfg = cfg
        loss_cfg = cfg.loss

        # 1. Uncertainty Weighting Setup
        self.uncertainty_cfg = loss_cfg.uncertainty_weighting
        if self.uncertainty_cfg.enabled:
            logger.info("Using UncertaintyWeightedLoss (Learnable)")
            self.uncertainty_loss = UncertaintyWeightedLoss(
                n_tasks=2,  # Image + Mask
                initial_log_vars=self.uncertainty_cfg.initial_log_vars,
                learnable=self.uncertainty_cfg.learnable,
                clamp_range=self.uncertainty_cfg.get("clamp_range", (-5.0, 5.0)),
            )
        else:
            logger.info("Using SimpleWeightedLoss (Fixed)")
            self.uncertainty_loss = SimpleWeightedLoss(weights=[1.0, 1.0])

        # 2. Weighting Configuration
        self.lesion_cfg = loss_cfg.lesion_weighted_mask
        if self.lesion_cfg.enabled:
            self.lesion_weight = self.lesion_cfg.lesion_weight
            self.background_weight = self.lesion_cfg.background_weight
            logger.info(
                f"Lesion weighting ENABLED: "
                f"lesion={self.lesion_weight}, background={self.background_weight}"
            )


    def forward(
        self,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
        pred_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        anatomical_weights: torch.Tensor | None = None,
        lesion_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total loss.

        Args:
            pred_noise: Predicted noise (B, C, H, W).
            target_noise: Target noise (B, C, H, W).
            pred_mask: Predicted mask (optional).
            target_mask: Target mask (optional).
            anatomical_weights: Spatial weights for anatomy (optional).
            lesion_mask: Binary mask of lesion (optional).

        Returns:
            Tuple of (total_loss, details_dict).
        """
        # --- 1. Image Loss (Noise Prediction) ---
        
        # LOGIC CHANGE: 
        # Since we are using Input Concatenation, we prefer Standard MSE.
        # We only use weighted loss if specifically targeting LESIONS.
        # We explicitly IGNORE general anatomical weighting for the loss.
        
        if self.lesion_cfg.enabled and lesion_mask is not None:
            # Case A: Lesion Weighting (Safe Version)
            # We apply specific weights to the lesion area, but normalize globally.
            loss_img = lesion_weighted_mse(
                pred_noise,
                target_noise,
                lesion_mask,
                self.lesion_weight,
                self.background_weight,
            )
        else:
            # Case B: Standard MSE (Correct for Input Conditioning)
            # We rely on the input concatenation to guide the model.
            loss_img = F.mse_loss(pred_noise, target_noise)

        # --- 2. Mask Loss (If auxiliary task exists) ---
        loss_msk = torch.tensor(0.0, device=pred_noise.device)
        
        if pred_mask is not None and target_mask is not None:
            # We usually use Standard MSE for the mask prediction task itself
            loss_msk = F.mse_loss(pred_mask, target_mask)

        # --- 3. Combine Losses ---
        # Note: If mask is not present, uncertainty_loss handles the 0.0 correctly
        # assuming the implementation supports it, or we just sum if simple.
        
        if pred_mask is None:
            # Single task mode
            total_loss = loss_img
            details = {"loss_image": loss_img.detach()}
        else:
            # Multi-task mode
            total_loss, details = self.uncertainty_loss([loss_img, loss_msk])
            details["loss_image"] = loss_img.detach()
            details["loss_mask"] = loss_msk.detach()

        return total_loss, details

    def get_log_vars(self) -> torch.Tensor | None:
        """Get log variance values if using uncertainty weighting."""
        if isinstance(self.uncertainty_loss, UncertaintyWeightedLoss):
            return self.uncertainty_loss.get_log_vars()
        return None