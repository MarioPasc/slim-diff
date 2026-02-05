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
from src.diffusion.losses.perceptual_loss import LPIPSLoss, create_lpips_loss

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


def lp_norm_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    p: float = 2.0,
) -> torch.Tensor:
    """Compute Lp norm loss.

    Generalizes MSE (p=2) to arbitrary Lp norms.
    L_p(pred, target) = mean(|pred - target|^p)

    Args:
        pred: Predictions, shape (B, C, H, W).
        target: Targets, shape (B, C, H, W).
        p: Norm order (p=2 is MSE, p=1 is MAE).

    Returns:
        Mean Lp loss (scalar).
    """
    return torch.abs(pred - target).pow(p).mean()


def lesion_weighted_lp_norm(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    p: float = 2.0,
    lesion_weight: float = 2.0,
    background_weight: float = 1.0,
) -> torch.Tensor:
    """Compute lesion-weighted Lp norm loss.

    Gives higher weight to lesion pixels (mask > 0 in {-1, +1} space).

    Args:
        pred: Predictions, shape (B, 1, H, W).
        target: Targets, shape (B, 1, H, W).
        mask: Ground truth mask in {-1, +1}, shape (B, 1, H, W).
        p: Norm order (p=2 is MSE, p=1 is MAE).
        lesion_weight: Weight for lesion pixels.
        background_weight: Weight for background pixels.

    Returns:
        Weighted Lp loss (scalar).
    """
    weights = torch.where(
        mask > 0,
        torch.full_like(mask, lesion_weight),
        torch.full_like(mask, background_weight),
    )
    lp = torch.abs(pred - target).pow(p) * weights
    return lp.mean()


class DiffusionLoss(nn.Module):
    """Complete diffusion loss module.

    Supports multiple operational modes controlled by loss.mode:

    1. "mse_channels" (default, original behavior):
       - MSE_image and MSE_mask with per-channel uncertainty weighting
       - 2 learnable log_vars (one per channel)

    2. "mse_ffl_groups":
       - MSE_image, MSE_mask, and FFL with group-level uncertainty weighting
       - 2 learnable log_vars:
         - Group 0: MSE_image + MSE_mask (spatial losses)
         - Group 1: FFL (frequency loss)
       - Requires x0 and x0_pred to be passed to forward()

    3. "mse_lp_norm":
       - Lp norm loss instead of MSE with per-channel uncertainty weighting
       - 2 learnable log_vars (one per channel)

    4. "mse_lp_norm_ffl_groups":
       - Lp norm with FFL and group-level uncertainty weighting

    pMF-style modes (network predicts x0, loss computed in specified space):

    5. "pmf_x0_loss":
       - Network predicts x0 (sample)
       - Loss computed directly on x0 using Lp norm (p=1.5)
       - No perceptual loss

    6. "pmf_v_loss":
       - Network predicts x0 (sample)
       - Loss computed in velocity space (derived from x0_pred)
       - No perceptual loss

    7. "pmf_x0_loss_lpips":
       - Network predicts x0 (sample)
       - Loss computed directly on x0 using Lp norm (p=1.5)
       - Plus LPIPS perceptual loss on image channel

    8. "pmf_v_loss_lpips":
       - Network predicts x0 (sample)
       - Loss computed in velocity space
       - Plus LPIPS perceptual loss on image channel
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
        elif self.mode == "mse_lp_norm":
            self._init_mse_lp_norm_mode(loss_cfg)
        elif self.mode == "mse_lp_norm_ffl_groups":
            self._init_mse_lp_norm_ffl_groups_mode(loss_cfg)
        elif self.mode == "pmf_x0_loss":
            self._init_pmf_x0_loss_mode(loss_cfg)
        elif self.mode == "pmf_v_loss":
            self._init_pmf_v_loss_mode(loss_cfg)
        elif self.mode == "pmf_x0_loss_lpips":
            self._init_pmf_x0_loss_lpips_mode(loss_cfg)
        elif self.mode == "pmf_v_loss_lpips":
            self._init_pmf_v_loss_lpips_mode(loss_cfg)
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

    def _init_mse_lp_norm_mode(self, loss_cfg: DictConfig) -> None:
        """Initialize for mse_lp_norm mode with Lp norm instead of MSE."""
        lp_cfg = loss_cfg.lp_norm
        self.lp_p = lp_cfg.get("p", 2.0)

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
            f"  mse_lp_norm mode: p={self.lp_p}, "
            f"uncertainty={loss_cfg.uncertainty_weighting.enabled}"
        )

    def _init_mse_lp_norm_ffl_groups_mode(self, loss_cfg: DictConfig) -> None:
        """Initialize for mse_lp_norm_ffl_groups mode with Lp norm + FFL."""
        lp_cfg = loss_cfg.lp_norm
        self.lp_p = lp_cfg.get("p", 2.0)

        ffl_cfg = loss_cfg.ffl
        self.ffl = FocalFrequencyLoss(
            loss_weight=ffl_cfg.get("loss_weight", 1.0),
            alpha=ffl_cfg.get("alpha", 1.0),
            patch_factor=ffl_cfg.get("patch_factor", 1),
            ave_spectrum=ffl_cfg.get("ave_spectrum", False),
            log_matrix=ffl_cfg.get("log_matrix", False),
            batch_matrix=ffl_cfg.get("batch_matrix", False),
        )

        group_cfg = loss_cfg.group_uncertainty_weighting
        self.loss = GroupUncertaintyWeightedLoss(
            n_groups=2,
            group_membership=[0, 0, 1],
            initial_log_vars=list(group_cfg.get("initial_log_vars", [0.0, 0.0])),
            learnable=group_cfg.get("learnable", True),
            clamp_range=group_cfg.get("clamp_range", (-5.0, 5.0)),
            intra_group_weights=list(group_cfg.get("intra_group_weights", [1.0, 1.0, 1.0])),
        )

        logger.info(
            f"  mse_lp_norm_ffl_groups mode: p={self.lp_p}, "
            f"FFL alpha={ffl_cfg.get('alpha', 1.0)}, "
            f"group_uncertainty learnable={group_cfg.get('learnable', True)}"
        )

    def _init_pmf_x0_loss_mode(self, loss_cfg: DictConfig) -> None:
        """Initialize for pmf_x0_loss mode (x0 prediction, x0-space Lp loss)."""
        lp_cfg = loss_cfg.get("lp_norm", {})
        self.lp_p = lp_cfg.get("p", 1.5)  # Default p=1.5 for pMF

        # Group uncertainty weighting for image and mask channels
        group_cfg = loss_cfg.get("group_uncertainty_weighting", {})
        if group_cfg.get("enabled", True):
            self.loss = GroupUncertaintyWeightedLoss(
                n_groups=2,
                group_membership=[0, 1],  # [lp_img, lp_mask] - separate groups
                initial_log_vars=list(group_cfg.get("initial_log_vars", [0.0, 0.0])),
                learnable=group_cfg.get("learnable", True),
                clamp_range=group_cfg.get("clamp_range", (-5.0, 5.0)),
                intra_group_weights=list(group_cfg.get("intra_group_weights", [1.0, 1.0])),
            )
        else:
            self.loss = SimpleWeightedLoss(weights=[1.0, 1.0])

        self.ffl = None
        self.lpips = None

        logger.info(
            f"  pmf_x0_loss mode: p={self.lp_p}, "
            f"group_uncertainty={group_cfg.get('enabled', True)}"
        )

    def _init_pmf_v_loss_mode(self, loss_cfg: DictConfig) -> None:
        """Initialize for pmf_v_loss mode (x0 prediction, v-space L2 loss)."""
        # v-loss uses L2 (MSE), but we store p for consistency
        self.lp_p = 2.0

        # Group uncertainty weighting for image and mask channels
        group_cfg = loss_cfg.get("group_uncertainty_weighting", {})
        if group_cfg.get("enabled", True):
            self.loss = GroupUncertaintyWeightedLoss(
                n_groups=2,
                group_membership=[0, 1],  # [v_img, v_mask] - separate groups
                initial_log_vars=list(group_cfg.get("initial_log_vars", [0.0, 0.0])),
                learnable=group_cfg.get("learnable", True),
                clamp_range=group_cfg.get("clamp_range", (-5.0, 5.0)),
                intra_group_weights=list(group_cfg.get("intra_group_weights", [1.0, 1.0])),
            )
        else:
            self.loss = SimpleWeightedLoss(weights=[1.0, 1.0])

        self.ffl = None
        self.lpips = None

        logger.info(
            f"  pmf_v_loss mode: L2 v-loss, "
            f"group_uncertainty={group_cfg.get('enabled', True)}"
        )

    def _init_pmf_x0_loss_lpips_mode(self, loss_cfg: DictConfig) -> None:
        """Initialize for pmf_x0_loss_lpips mode (x0 loss + LPIPS)."""
        lp_cfg = loss_cfg.get("lp_norm", {})
        self.lp_p = lp_cfg.get("p", 1.5)

        # Group uncertainty weighting: [lp_img, lp_mask, lpips]
        group_cfg = loss_cfg.get("group_uncertainty_weighting", {})
        if group_cfg.get("enabled", True):
            self.loss = GroupUncertaintyWeightedLoss(
                n_groups=3,
                group_membership=[0, 1, 2],  # [lp_img, lp_mask, lpips]
                initial_log_vars=list(group_cfg.get("initial_log_vars", [0.0, 0.0, 0.0])),
                learnable=group_cfg.get("learnable", True),
                clamp_range=group_cfg.get("clamp_range", (-5.0, 5.0)),
                intra_group_weights=list(group_cfg.get("intra_group_weights", [1.0, 1.0, 1.0])),
            )
        else:
            self.loss = SimpleWeightedLoss(weights=[1.0, 1.0, 1.0])

        self.ffl = None

        # Create LPIPS loss from config
        perceptual_cfg = loss_cfg.get("perceptual", {})
        self.lpips = LPIPSLoss(
            weights_path=perceptual_cfg.get("vgg_weights_path"),
            loss_weight=perceptual_cfg.get("loss_weight", 0.1),
            t_threshold=perceptual_cfg.get("t_threshold", 0.5),
            apply_to_mask=perceptual_cfg.get("apply_to_mask", False),
            use_learned_weights=perceptual_cfg.get("use_learned_weights", True),
            lpips_weights_path=perceptual_cfg.get("lpips_weights_path"),
        )

        logger.info(
            f"  pmf_x0_loss_lpips mode: p={self.lp_p}, "
            f"lpips_weight={perceptual_cfg.get('loss_weight', 0.1)}, "
            f"t_threshold={perceptual_cfg.get('t_threshold', 0.5)}"
        )

    def _init_pmf_v_loss_lpips_mode(self, loss_cfg: DictConfig) -> None:
        """Initialize for pmf_v_loss_lpips mode (v-loss + LPIPS)."""
        self.lp_p = 2.0  # v-loss uses L2

        # Group uncertainty weighting: [v_img, v_mask, lpips]
        group_cfg = loss_cfg.get("group_uncertainty_weighting", {})
        if group_cfg.get("enabled", True):
            self.loss = GroupUncertaintyWeightedLoss(
                n_groups=3,
                group_membership=[0, 1, 2],  # [v_img, v_mask, lpips]
                initial_log_vars=list(group_cfg.get("initial_log_vars", [0.0, 0.0, 0.0])),
                learnable=group_cfg.get("learnable", True),
                clamp_range=group_cfg.get("clamp_range", (-5.0, 5.0)),
                intra_group_weights=list(group_cfg.get("intra_group_weights", [1.0, 1.0, 1.0])),
            )
        else:
            self.loss = SimpleWeightedLoss(weights=[1.0, 1.0, 1.0])

        self.ffl = None

        # Create LPIPS loss from config
        perceptual_cfg = loss_cfg.get("perceptual", {})
        self.lpips = LPIPSLoss(
            weights_path=perceptual_cfg.get("vgg_weights_path"),
            loss_weight=perceptual_cfg.get("loss_weight", 0.1),
            t_threshold=perceptual_cfg.get("t_threshold", 0.5),
            apply_to_mask=perceptual_cfg.get("apply_to_mask", False),
            use_learned_weights=perceptual_cfg.get("use_learned_weights", True),
            lpips_weights_path=perceptual_cfg.get("lpips_weights_path"),
        )

        logger.info(
            f"  pmf_v_loss_lpips mode: L2 v-loss, "
            f"lpips_weight={perceptual_cfg.get('loss_weight', 0.1)}, "
            f"t_threshold={perceptual_cfg.get('t_threshold', 0.5)}"
        )

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        x0_mask: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        x0_pred: torch.Tensor | None = None,
        x_t: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        alphas_cumprod: torch.Tensor | None = None,
        num_train_timesteps: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute diffusion loss.

        Args:
            eps_pred: Predicted noise/x0/v depending on mode, shape (B, 2, H, W).
            eps_target: Target noise/x0/v depending on mode, shape (B, 2, H, W).
            x0_mask: Original mask for lesion weighting, shape (B, 1, H, W).
            x0: Original samples (required for FFL and pMF modes).
            x0_pred: Predicted x0 (required for FFL and pMF LPIPS modes).
            x_t: Noisy samples (required for pMF v-loss modes).
            timesteps: Current timesteps (required for pMF modes).
            alphas_cumprod: Scheduler alpha bars (required for pMF v-loss modes).
            num_train_timesteps: Total timesteps T (required for pMF LPIPS modes).

        Returns:
            Tuple of (total_loss, details_dict).
        """
        if self.mode == "mse_channels":
            return self._forward_mse_channels(eps_pred, eps_target, x0_mask)
        elif self.mode == "mse_ffl_groups":
            return self._forward_mse_ffl_groups(
                eps_pred, eps_target, x0_mask, x0, x0_pred
            )
        elif self.mode == "mse_lp_norm":
            return self._forward_mse_lp_norm(eps_pred, eps_target, x0_mask)
        elif self.mode == "mse_lp_norm_ffl_groups":
            return self._forward_mse_lp_norm_ffl_groups(
                eps_pred, eps_target, x0_mask, x0, x0_pred
            )
        elif self.mode == "pmf_x0_loss":
            return self._forward_pmf_x0_loss(x0_pred, x0, x0_mask)
        elif self.mode == "pmf_v_loss":
            return self._forward_pmf_v_loss(
                x0_pred, x0, x0_mask, x_t, timesteps, alphas_cumprod
            )
        elif self.mode == "pmf_x0_loss_lpips":
            return self._forward_pmf_x0_loss_lpips(
                x0_pred, x0, x0_mask, timesteps, num_train_timesteps
            )
        elif self.mode == "pmf_v_loss_lpips":
            return self._forward_pmf_v_loss_lpips(
                x0_pred, x0, x0_mask, x_t, timesteps, alphas_cumprod, num_train_timesteps
            )
        else:
            raise ValueError(f"Unknown loss mode: {self.mode}")

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

    def _forward_mse_lp_norm(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        x0_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward for mse_lp_norm mode using Lp norm."""
        # Split channels
        eps_pred_img = eps_pred[:, 0:1]
        eps_pred_msk = eps_pred[:, 1:2]
        eps_target_img = eps_target[:, 0:1]
        eps_target_msk = eps_target[:, 1:2]

        # Image channel loss (optionally lesion-weighted)
        if self.use_lesion_weighting_image and x0_mask is not None:
            loss_img = lesion_weighted_lp_norm(
                eps_pred_img,
                eps_target_img,
                x0_mask,
                self.lp_p,
                self.lesion_weight_image,
                self.background_weight_image,
            )
        else:
            loss_img = lp_norm_loss(eps_pred_img, eps_target_img, self.lp_p)

        # Mask channel loss (optionally lesion-weighted)
        if self.use_lesion_weighting_mask and x0_mask is not None:
            loss_msk = lesion_weighted_lp_norm(
                eps_pred_msk,
                eps_target_msk,
                x0_mask,
                self.lp_p,
                self.lesion_weight_mask,
                self.background_weight_mask,
            )
        else:
            loss_msk = lp_norm_loss(eps_pred_msk, eps_target_msk, self.lp_p)

        # Combine losses
        total_loss, details = self.loss([loss_img, loss_msk])

        # Add named losses to details
        details["loss_image"] = loss_img.detach()
        details["loss_mask"] = loss_msk.detach()
        details["lp_p"] = self.lp_p

        return total_loss, details

    def _forward_mse_lp_norm_ffl_groups(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        x0_mask: torch.Tensor | None,
        x0: torch.Tensor | None,
        x0_pred: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward for mse_lp_norm_ffl_groups mode with Lp norm + FFL."""
        if x0 is None or x0_pred is None:
            raise ValueError("mse_lp_norm_ffl_groups mode requires x0 and x0_pred")

        # Split channels
        eps_pred_img = eps_pred[:, 0:1]
        eps_pred_msk = eps_pred[:, 1:2]
        eps_target_img = eps_target[:, 0:1]
        eps_target_msk = eps_target[:, 1:2]

        # Lp norm losses (optionally lesion-weighted)
        if self.use_lesion_weighting_image and x0_mask is not None:
            loss_img = lesion_weighted_lp_norm(
                eps_pred_img,
                eps_target_img,
                x0_mask,
                self.lp_p,
                self.lesion_weight_image,
                self.background_weight_image,
            )
        else:
            loss_img = lp_norm_loss(eps_pred_img, eps_target_img, self.lp_p)

        if self.use_lesion_weighting_mask and x0_mask is not None:
            loss_msk = lesion_weighted_lp_norm(
                eps_pred_msk,
                eps_target_msk,
                x0_mask,
                self.lp_p,
                self.lesion_weight_mask,
                self.background_weight_mask,
            )
        else:
            loss_msk = lp_norm_loss(eps_pred_msk, eps_target_msk, self.lp_p)

        # FFL on x0_pred vs x0 (image channel only)
        x0_pred_img = x0_pred[:, 0:1]
        x0_img = x0[:, 0:1]
        loss_ffl, ffl_details = self.ffl(x0_pred_img, x0_img)

        # Group uncertainty weighting: [lp_img, lp_msk, ffl]
        total_loss, details = self.loss([loss_img, loss_msk, loss_ffl])

        # Add named losses to details
        details["loss_image"] = loss_img.detach()
        details["loss_mask"] = loss_msk.detach()
        details["loss_ffl"] = loss_ffl.detach()
        details["lp_p"] = self.lp_p
        details.update(ffl_details)

        return total_loss, details

    def _x0_pred_to_velocity(
        self,
        x0_pred: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        """Convert x0 prediction to velocity space for v-loss.

        In DDPM variance-preserving schedule:
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
            v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x0

        Given x0_pred, we first derive the implied epsilon:
            eps_implied = (x_t - sqrt(alpha_bar_t) * x0_pred) / sqrt(1 - alpha_bar_t)

        Then compute the velocity:
            v_pred = sqrt(alpha_bar_t) * eps_implied - sqrt(1 - alpha_bar_t) * x0_pred

        Args:
            x0_pred: Predicted x0, shape (B, C, H, W).
            x_t: Noisy samples, shape (B, C, H, W).
            timesteps: Current timesteps, shape (B,).
            alphas_cumprod: Alpha cumulative products, shape (T,).

        Returns:
            Predicted velocity, shape (B, C, H, W).
        """
        alpha_bar_t = alphas_cumprod[timesteps]

        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        while alpha_bar_t.dim() < x_t.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

        # Derive implied epsilon from x0_pred
        eps_implied = (x_t - sqrt_alpha_bar * x0_pred) / sqrt_one_minus_alpha_bar

        # Compute velocity
        v_pred = sqrt_alpha_bar * eps_implied - sqrt_one_minus_alpha_bar * x0_pred

        return v_pred

    def _compute_v_target(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the velocity target from x0 and x_t.

        v_target = sqrt(alpha_bar_t) * eps - sqrt(1 - alpha_bar_t) * x0

        where eps = (x_t - sqrt(alpha_bar_t) * x0) / sqrt(1 - alpha_bar_t)

        Args:
            x0: Original samples, shape (B, C, H, W).
            x_t: Noisy samples, shape (B, C, H, W).
            timesteps: Current timesteps, shape (B,).
            alphas_cumprod: Alpha cumulative products, shape (T,).

        Returns:
            Velocity target, shape (B, C, H, W).
        """
        alpha_bar_t = alphas_cumprod[timesteps]

        while alpha_bar_t.dim() < x_t.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

        # Derive epsilon from x0 and x_t
        eps = (x_t - sqrt_alpha_bar * x0) / sqrt_one_minus_alpha_bar

        # Compute velocity target
        v_target = sqrt_alpha_bar * eps - sqrt_one_minus_alpha_bar * x0

        return v_target

    def _forward_pmf_x0_loss(
        self,
        x0_pred: torch.Tensor,
        x0: torch.Tensor,
        x0_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward for pmf_x0_loss mode (x0-space Lp loss).

        Args:
            x0_pred: Predicted x0, shape (B, 2, H, W).
            x0: Target x0, shape (B, 2, H, W).
            x0_mask: Original mask for lesion weighting, shape (B, 1, H, W).

        Returns:
            Tuple of (total_loss, details_dict).
        """
        if x0_pred is None or x0 is None:
            raise ValueError("pmf_x0_loss mode requires x0_pred and x0")

        # Split channels
        x0_pred_img = x0_pred[:, 0:1]
        x0_pred_msk = x0_pred[:, 1:2]
        x0_img = x0[:, 0:1]
        x0_msk = x0[:, 1:2]

        # Image channel loss (optionally lesion-weighted)
        if self.use_lesion_weighting_image and x0_mask is not None:
            loss_img = lesion_weighted_lp_norm(
                x0_pred_img,
                x0_img,
                x0_mask,
                self.lp_p,
                self.lesion_weight_image,
                self.background_weight_image,
            )
        else:
            loss_img = lp_norm_loss(x0_pred_img, x0_img, self.lp_p)

        # Mask channel loss (optionally lesion-weighted)
        if self.use_lesion_weighting_mask and x0_mask is not None:
            loss_msk = lesion_weighted_lp_norm(
                x0_pred_msk,
                x0_msk,
                x0_mask,
                self.lp_p,
                self.lesion_weight_mask,
                self.background_weight_mask,
            )
        else:
            loss_msk = lp_norm_loss(x0_pred_msk, x0_msk, self.lp_p)

        # Combine losses
        total_loss, details = self.loss([loss_img, loss_msk])

        # Add named losses to details
        details["loss_image"] = loss_img.detach()
        details["loss_mask"] = loss_msk.detach()
        details["lp_p"] = self.lp_p
        details["loss_mode"] = "pmf_x0_loss"

        return total_loss, details

    def _forward_pmf_v_loss(
        self,
        x0_pred: torch.Tensor,
        x0: torch.Tensor,
        x0_mask: torch.Tensor | None,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward for pmf_v_loss mode (velocity-space L2 loss).

        Args:
            x0_pred: Predicted x0, shape (B, 2, H, W).
            x0: Target x0, shape (B, 2, H, W).
            x0_mask: Original mask for lesion weighting, shape (B, 1, H, W).
            x_t: Noisy samples, shape (B, 2, H, W).
            timesteps: Current timesteps, shape (B,).
            alphas_cumprod: Alpha cumulative products, shape (T,).

        Returns:
            Tuple of (total_loss, details_dict).
        """
        if x0_pred is None or x0 is None:
            raise ValueError("pmf_v_loss mode requires x0_pred and x0")
        if x_t is None or timesteps is None or alphas_cumprod is None:
            raise ValueError("pmf_v_loss mode requires x_t, timesteps, and alphas_cumprod")

        # Convert x0_pred to velocity
        v_pred = self._x0_pred_to_velocity(x0_pred, x_t, timesteps, alphas_cumprod)

        # Compute velocity target
        v_target = self._compute_v_target(x0, x_t, timesteps, alphas_cumprod)

        # Split channels
        v_pred_img = v_pred[:, 0:1]
        v_pred_msk = v_pred[:, 1:2]
        v_target_img = v_target[:, 0:1]
        v_target_msk = v_target[:, 1:2]

        # Image channel loss (L2, optionally lesion-weighted)
        if self.use_lesion_weighting_image and x0_mask is not None:
            loss_img = lesion_weighted_mse(
                v_pred_img,
                v_target_img,
                x0_mask,
                self.lesion_weight_image,
                self.background_weight_image,
            )
        else:
            loss_img = F.mse_loss(v_pred_img, v_target_img)

        # Mask channel loss (L2, optionally lesion-weighted)
        if self.use_lesion_weighting_mask and x0_mask is not None:
            loss_msk = lesion_weighted_mse(
                v_pred_msk,
                v_target_msk,
                x0_mask,
                self.lesion_weight_mask,
                self.background_weight_mask,
            )
        else:
            loss_msk = F.mse_loss(v_pred_msk, v_target_msk)

        # Combine losses
        total_loss, details = self.loss([loss_img, loss_msk])

        # Add named losses to details
        details["loss_image"] = loss_img.detach()
        details["loss_mask"] = loss_msk.detach()
        details["loss_v_image"] = loss_img.detach()
        details["loss_v_mask"] = loss_msk.detach()
        details["loss_mode"] = "pmf_v_loss"

        return total_loss, details

    def _forward_pmf_x0_loss_lpips(
        self,
        x0_pred: torch.Tensor,
        x0: torch.Tensor,
        x0_mask: torch.Tensor | None,
        timesteps: torch.Tensor,
        num_train_timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward for pmf_x0_loss_lpips mode (x0 Lp loss + LPIPS).

        Args:
            x0_pred: Predicted x0, shape (B, 2, H, W).
            x0: Target x0, shape (B, 2, H, W).
            x0_mask: Original mask for lesion weighting, shape (B, 1, H, W).
            timesteps: Current timesteps, shape (B,).
            num_train_timesteps: Total timesteps T.

        Returns:
            Tuple of (total_loss, details_dict).
        """
        if x0_pred is None or x0 is None:
            raise ValueError("pmf_x0_loss_lpips mode requires x0_pred and x0")
        if timesteps is None or num_train_timesteps is None:
            raise ValueError("pmf_x0_loss_lpips mode requires timesteps and num_train_timesteps")

        # Split channels
        x0_pred_img = x0_pred[:, 0:1]
        x0_pred_msk = x0_pred[:, 1:2]
        x0_img = x0[:, 0:1]
        x0_msk = x0[:, 1:2]

        # Image channel loss (optionally lesion-weighted)
        if self.use_lesion_weighting_image and x0_mask is not None:
            loss_img = lesion_weighted_lp_norm(
                x0_pred_img,
                x0_img,
                x0_mask,
                self.lp_p,
                self.lesion_weight_image,
                self.background_weight_image,
            )
        else:
            loss_img = lp_norm_loss(x0_pred_img, x0_img, self.lp_p)

        # Mask channel loss (optionally lesion-weighted)
        if self.use_lesion_weighting_mask and x0_mask is not None:
            loss_msk = lesion_weighted_lp_norm(
                x0_pred_msk,
                x0_msk,
                x0_mask,
                self.lp_p,
                self.lesion_weight_mask,
                self.background_weight_mask,
            )
        else:
            loss_msk = lp_norm_loss(x0_pred_msk, x0_msk, self.lp_p)

        # LPIPS loss on image channel
        loss_lpips, lpips_details = self.lpips(x0_pred, x0, timesteps, num_train_timesteps)

        # Combine losses: [lp_img, lp_mask, lpips]
        total_loss, details = self.loss([loss_img, loss_msk, loss_lpips])

        # Add named losses to details
        details["loss_image"] = loss_img.detach()
        details["loss_mask"] = loss_msk.detach()
        details["loss_lpips"] = loss_lpips.detach()
        details["lp_p"] = self.lp_p
        details["loss_mode"] = "pmf_x0_loss_lpips"
        details.update(lpips_details)

        return total_loss, details

    def _forward_pmf_v_loss_lpips(
        self,
        x0_pred: torch.Tensor,
        x0: torch.Tensor,
        x0_mask: torch.Tensor | None,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        num_train_timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward for pmf_v_loss_lpips mode (v-loss + LPIPS).

        Args:
            x0_pred: Predicted x0, shape (B, 2, H, W).
            x0: Target x0, shape (B, 2, H, W).
            x0_mask: Original mask for lesion weighting, shape (B, 1, H, W).
            x_t: Noisy samples, shape (B, 2, H, W).
            timesteps: Current timesteps, shape (B,).
            alphas_cumprod: Alpha cumulative products, shape (T,).
            num_train_timesteps: Total timesteps T.

        Returns:
            Tuple of (total_loss, details_dict).
        """
        if x0_pred is None or x0 is None:
            raise ValueError("pmf_v_loss_lpips mode requires x0_pred and x0")
        if x_t is None or timesteps is None or alphas_cumprod is None:
            raise ValueError("pmf_v_loss_lpips mode requires x_t, timesteps, and alphas_cumprod")
        if num_train_timesteps is None:
            raise ValueError("pmf_v_loss_lpips mode requires num_train_timesteps")

        # Convert x0_pred to velocity
        v_pred = self._x0_pred_to_velocity(x0_pred, x_t, timesteps, alphas_cumprod)

        # Compute velocity target
        v_target = self._compute_v_target(x0, x_t, timesteps, alphas_cumprod)

        # Split channels
        v_pred_img = v_pred[:, 0:1]
        v_pred_msk = v_pred[:, 1:2]
        v_target_img = v_target[:, 0:1]
        v_target_msk = v_target[:, 1:2]

        # Image channel loss (L2, optionally lesion-weighted)
        if self.use_lesion_weighting_image and x0_mask is not None:
            loss_img = lesion_weighted_mse(
                v_pred_img,
                v_target_img,
                x0_mask,
                self.lesion_weight_image,
                self.background_weight_image,
            )
        else:
            loss_img = F.mse_loss(v_pred_img, v_target_img)

        # Mask channel loss (L2, optionally lesion-weighted)
        if self.use_lesion_weighting_mask and x0_mask is not None:
            loss_msk = lesion_weighted_mse(
                v_pred_msk,
                v_target_msk,
                x0_mask,
                self.lesion_weight_mask,
                self.background_weight_mask,
            )
        else:
            loss_msk = F.mse_loss(v_pred_msk, v_target_msk)

        # LPIPS loss on image channel (using x0_pred, not v_pred)
        loss_lpips, lpips_details = self.lpips(x0_pred, x0, timesteps, num_train_timesteps)

        # Combine losses: [v_img, v_mask, lpips]
        total_loss, details = self.loss([loss_img, loss_msk, loss_lpips])

        # Add named losses to details
        details["loss_image"] = loss_img.detach()
        details["loss_mask"] = loss_msk.detach()
        details["loss_v_image"] = loss_img.detach()
        details["loss_v_mask"] = loss_msk.detach()
        details["loss_lpips"] = loss_lpips.detach()
        details["loss_mode"] = "pmf_v_loss_lpips"
        details.update(lpips_details)

        return total_loss, details

    def get_log_vars(self) -> torch.Tensor | None:
        """Get log variance values if using uncertainty weighting.

        Returns:
            Log variance tensor or None.
        """
        if isinstance(self.loss, (UncertaintyWeightedLoss, GroupUncertaintyWeightedLoss)):
            return self.loss.get_log_vars()
        return None