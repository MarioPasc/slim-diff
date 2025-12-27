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
            get_not_nans=cfg.metrics.dice.get_not_nans,
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
        return dice.mean()

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
        self.metric = HausdorffDistanceMetric(
            include_background=False,
            percentile=cfg.metrics.hd95.percentile,
            reduction="mean_batch",
        )
        self.spacing = cfg.metrics.hd95.spacing

    def __call__(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute HD95.

        Args:
            preds: Binary predictions (B, 1, H, W)
            targets: Binary ground truth (B, 1, H, W)

        Returns:
            HD95 distance (scalar). Returns NaN if no valid samples.
        """
        try:
            # Check if both pred and target have foreground
            has_pred = (preds.sum(dim=[1, 2, 3]) > 0)
            has_target = (targets.sum(dim=[1, 2, 3]) > 0)
            valid_mask = has_pred & has_target

            if not valid_mask.any():
                return torch.tensor(float("nan"), device=preds.device)

            # Compute HD95 only on valid samples
            hd95 = self.metric(preds, targets)

            # Filter out invalid values
            valid_hd95 = hd95[~torch.isnan(hd95)]

            if len(valid_hd95) == 0:
                return torch.tensor(float("nan"), device=preds.device)

            return valid_hd95.mean()

        except Exception:
            # Return NaN on error
            return torch.tensor(float("nan"), device=preds.device)

    def reset(self):
        """Reset metric state."""
        self.metric.reset()
