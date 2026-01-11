"""Segmentation metrics wrappers for MONAI."""

from __future__ import annotations

import torch
from monai.metrics import DiceMetric as MONAIDice
from monai.metrics import HausdorffDistanceMetric
from omegaconf import DictConfig


class DiceMetric:
    """Wrapper for MONAI Dice metric."""

    def __init__(self, cfg: DictConfig):
        """Initialize Dice metric.

        Args:
            cfg: Configuration with metrics.dice section
        """
        self.metric = MONAIDice(
            include_background=cfg.metrics.dice.include_background,
            reduction=cfg.metrics.dice.reduction,
            get_not_nans=True,  # Always filter NaN values in aggregation
            ignore_empty=True,  # Handle empty ground truth gracefully
        )

    def __call__(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice coefficient.

        Args:
            preds: Binary predictions (B, 1, H, W) in {0, 1}
            targets: Binary ground truth (B, 1, H, W) in {0, 1}

        Returns:
            Dice score (scalar tensor)
        """
        dice = self.metric(preds, targets)

        # Handle empty results or NaN values
        if dice.numel() == 0:
            return torch.tensor(float("nan"), device=preds.device)

        # Filter NaN values before computing mean
        valid_dice = dice[~torch.isnan(dice)]
        if len(valid_dice) == 0:
            return torch.tensor(float("nan"), device=preds.device)

        return valid_dice.mean()

    def reset(self):
        """Reset metric state."""
        self.metric.reset()


class HausdorffDistance95:
    """Wrapper for MONAI HD95 metric."""

    def __init__(self, cfg: DictConfig):
        """Initialize HD95 metric.

        Args:
            cfg: Configuration with metrics.hd95 section
        """
        self.percentile = cfg.metrics.hd95.percentile
        self.spacing = cfg.metrics.hd95.spacing
        # Create metric with spacing for physical distance computation
        self.metric = HausdorffDistanceMetric(
            include_background=False,
            percentile=self.percentile,
            reduction="mean_batch",
        )

    def __call__(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute HD95.

        Args:
            preds: Binary predictions (B, 1, H, W)
            targets: Binary ground truth (B, 1, H, W)

        Returns:
            HD95 distance in physical units (scalar). Returns NaN if no valid samples.
        """
        try:
            # Check if both pred and target have foreground
            has_pred = (preds.sum(dim=[1, 2, 3]) > 0)
            has_target = (targets.sum(dim=[1, 2, 3]) > 0)
            valid_mask = has_pred & has_target

            if not valid_mask.any():
                return torch.tensor(float("nan"), device=preds.device)

            # Compute HD95 with physical spacing
            # MONAI HausdorffDistanceMetric accepts spacing in __call__
            hd95 = self.metric(preds, targets, spacing=self.spacing)

            # Flatten if necessary (B, 1) -> (B,)
            if hd95.dim() > 1:
                hd95 = hd95.flatten()

            # Filter out invalid values using the mask we computed
            # This handles cases where metric returns inf for empty sets
            valid_hd95 = hd95[valid_mask]

            if len(valid_hd95) == 0:
                return torch.tensor(float("nan"), device=preds.device)

            return valid_hd95.mean()

        except Exception:
            # Return NaN on error
            return torch.tensor(float("nan"), device=preds.device)

    def reset(self):
        """Reset metric state."""
        self.metric.reset()
