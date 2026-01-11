"""PyTorch Lightning module for segmentation."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss
from omegaconf import DictConfig

from src.segmentation.metrics.segmentation_metrics import (
    DiceMetric,
    HausdorffDistance95,
)
from src.segmentation.models.factory import build_model


class SegmentationLitModule(pl.LightningModule):
    """Lightning module for lesion segmentation.

    Handles:
    - Model forward pass
    - Loss computation (DiceCE)
    - Metrics (Dice, HD95)
    - Optimizer configuration
    """

    def __init__(self, cfg: DictConfig):
        """Initialize module.

        Args:
            cfg: Configuration object
        """
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        # Build model
        self.model = build_model(cfg)

        # Build loss
        self.criterion = DiceCELoss(
            include_background=cfg.loss.include_background,
            to_onehot_y=cfg.loss.to_onehot_y,
            sigmoid=cfg.loss.sigmoid,
            softmax=cfg.loss.softmax,
            squared_pred=cfg.loss.squared_pred,
            jaccard=cfg.loss.jaccard,
            reduction=cfg.loss.reduction,
            lambda_dice=cfg.loss.lambda_dice,
            lambda_ce=cfg.loss.lambda_ce,
        )

        # Build metrics
        self.dice_metric = DiceMetric(cfg)
        self.hd95_metric = HausdorffDistance95(cfg)


    def forward(self, x):
        """Forward pass.

        Args:
            x: Input images (B, 1, H, W)

        Returns:
            Predictions (B, 1, H, W) logits
        """
        output = self.model(x)
        # Handle models that return list (e.g., UNet++, DynUNet with deep supervision)
        if isinstance(output, list):
            output = output[0]
        return output

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: Dict with 'image' and 'mask'
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        images = batch["image"]  # (B, 1, H, W)
        masks = batch["mask"]    # (B, 1, H, W) in {0, 1}

        # Forward
        preds = self(images)  # (B, 1, H, W) logits

        # Compute loss
        loss = self.criterion(preds, masks)

        # Log
        self.log("train/loss", loss, sync_dist=True, batch_size=images.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch: Dict with 'image' and 'mask'
            batch_idx: Batch index

        Returns:
            Dict of outputs
        """
        images = batch["image"]
        masks = batch["mask"]

        # Forward
        preds = self(images)

        # Compute loss
        loss = self.criterion(preds, masks)

        # Apply sigmoid to get probabilities
        preds_prob = torch.sigmoid(preds)

        # Binarize predictions
        preds_binary = (preds_prob > 0.5).float()

        # Compute metrics
        dice = self.dice_metric(preds_binary, masks)
        hd95 = self.hd95_metric(preds_binary, masks)

        # Log
        B = images.shape[0]
        self.log("val/loss", loss, sync_dist=True, batch_size=B)

        # Handle NaN dice values - log 0.0 instead of NaN to avoid early stopping issues
        # This happens when ground truth has no positive pixels
        if torch.isnan(dice) or torch.isinf(dice):
            # If no lesions in ground truth, set dice based on prediction
            # If pred is also empty, dice should be 1.0; if pred has false positives, dice should be 0.0
            has_pred_positive = preds_binary.sum() > 0
            has_gt_positive = masks.sum() > 0
            if not has_gt_positive and not has_pred_positive:
                dice = torch.tensor(1.0, device=dice.device)  # Both empty = perfect match
            else:
                dice = torch.tensor(0.0, device=dice.device)  # Mismatch

        self.log("val/dice", dice, sync_dist=True, batch_size=B, prog_bar=True)

        if not torch.isnan(hd95) and not torch.isinf(hd95):
            self.log("val/hd95", hd95, sync_dist=True, batch_size=B)

        return {"loss": loss, "dice": dice, "hd95": hd95}

    def test_step(self, batch, batch_idx):
        """Test step.

        Args:
            batch: Dict with 'image' and 'mask'
            batch_idx: Batch index

        Returns:
            Dict of outputs
        """
        images = batch["image"]
        masks = batch["mask"]

        # Forward
        preds = self(images)

        # Compute loss
        loss = self.criterion(preds, masks)

        # Apply sigmoid to get probabilities
        preds_prob = torch.sigmoid(preds)

        # Binarize predictions
        preds_binary = (preds_prob > 0.5).float()

        # Compute metrics
        dice = self.dice_metric(preds_binary, masks)
        hd95 = self.hd95_metric(preds_binary, masks)

        # Log
        B = images.shape[0]
        self.log("test/loss", loss, sync_dist=True, batch_size=B)

        # Handle NaN dice values
        if torch.isnan(dice) or torch.isinf(dice):
            has_pred_positive = preds_binary.sum() > 0
            has_gt_positive = masks.sum() > 0
            if not has_gt_positive and not has_pred_positive:
                dice = torch.tensor(1.0, device=dice.device)
            else:
                dice = torch.tensor(0.0, device=dice.device)

        self.log("test/dice", dice, sync_dist=True, batch_size=B, prog_bar=True)

        if not torch.isnan(hd95) and not torch.isinf(hd95):
            self.log("test/hd95", hd95, sync_dist=True, batch_size=B)

        return {"loss": loss, "dice": dice, "hd95": hd95}

    def configure_optimizers(self):
        """Configure optimizer and scheduler.

        Returns:
            Optimizer configuration dictionary
        """
        opt_cfg = self.cfg.training.optimizer
        lr_cfg = self.cfg.training.lr_scheduler

        # Build optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps,
        )

        # Build scheduler
        if lr_cfg.type == "CosineAnnealingLR":
            T_max = lr_cfg.T_max or self.cfg.training.max_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=lr_cfg.eta_min,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        return {"optimizer": optimizer}
