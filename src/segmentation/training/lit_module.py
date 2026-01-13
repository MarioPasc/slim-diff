"""PyTorch Lightning module for segmentation."""

from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch
from monai.losses import DiceFocalLoss
from omegaconf import DictConfig

from src.segmentation.metrics.segmentation_metrics import (
    DiceMetric,
    HausdorffDistance95,
)
from src.segmentation.models.factory import build_model

logger = logging.getLogger(__name__)


class SegmentationLitModule(pl.LightningModule):
    """Lightning module for lesion segmentation.

    Handles:
    - Model forward pass
    - Loss computation (DiceFocal)
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

        # Check if using DynUNet with deep supervision
        self.use_deep_supervision = (
            cfg.model.name == "DynUNet" and
            cfg.model.get("deep_supervision", False)
        )
        if self.use_deep_supervision:
            self.deep_supr_num = cfg.model.get("deep_supr_num", 1)
            # Weights for deep supervision losses (final output gets higher weight)
            # With deep_supr_num=1, we have 2 outputs total (1 intermediate + 1 final)
            num_outputs = self.deep_supr_num + 1
            # Exponentially increasing weights: [0.5^n, 0.5^(n-1), ..., 0.5^1, 1.0]
            self.deep_supr_weights = [0.5 ** (num_outputs - i) for i in range(num_outputs)]
            # Normalize weights to sum to 1
            weight_sum = sum(self.deep_supr_weights)
            self.deep_supr_weights = [w / weight_sum for w in self.deep_supr_weights]

        # Build loss
        self.criterion = DiceFocalLoss(
            include_background=cfg.loss.include_background,
            to_onehot_y=cfg.loss.to_onehot_y,
            sigmoid=cfg.loss.sigmoid,
            softmax=cfg.loss.softmax,
            squared_pred=cfg.loss.squared_pred,
            jaccard=cfg.loss.jaccard,
            reduction=cfg.loss.reduction,
            gamma=cfg.loss.gamma,
            lambda_dice=cfg.loss.lambda_dice,
            lambda_focal=cfg.loss.lambda_focal,
        )

        # Build metrics
        self.dice_metric = DiceMetric(cfg)
        self.hd95_metric = HausdorffDistance95(cfg)


    def forward(self, x):
        """Forward pass.

        Args:
            x: Input images (B, 1, H, W)

        Returns:
            Predictions:
            - If deep supervision: (B, num_outputs, 1, H, W) tensor with stacked outputs
            - Otherwise: (B, 1, H, W) logits
        """
        output = self.model(x)
        # Handle models that return list (e.g., UNet++)
        # Note: DynUNet with deep supervision returns a tensor, not a list
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
        preds = self(images)  # (B, 1, H, W) or (B, num_outputs, 1, H, W) for deep supervision

        # Compute loss
        if self.use_deep_supervision and preds.dim() == 5:
            # Deep supervision: compute weighted loss for each output
            # preds shape: (B, num_outputs, 1, H, W)
            # Unbind to get list of tensors: [(B, 1, H, W), (B, 1, H, W), ...]
            outputs = torch.unbind(preds, dim=1)

            # Compute loss for each output and weight them
            losses = []
            for i, output in enumerate(outputs):
                loss_i = self.criterion(output, masks)
                weighted_loss_i = self.deep_supr_weights[i] * loss_i
                losses.append(weighted_loss_i)

                # Log individual losses for debugging
                self.log(f"train/loss_output_{i}", loss_i, sync_dist=True, batch_size=images.shape[0])

            # Total weighted loss
            loss = sum(losses)
        else:
            # Standard single output (or deep supervision not returning expected format)
            if self.use_deep_supervision and preds.dim() != 5:
                # Log warning once
                if not hasattr(self, '_deep_super_warning_logged'):
                    logger.warning(
                        f"DynUNet deep supervision enabled but output is {preds.dim()}D, expected 5D. "
                        f"Shape: {preds.shape}. Falling back to standard loss computation. "
                        f"Check MONAI version or DynUNet configuration."
                    )
                    self._deep_super_warning_logged = True
            loss = self.criterion(preds, masks)

        # Log total loss
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
        # Note: DynUNet deep supervision only activates during training mode
        # (self.training=True). During validation, model is in eval mode,
        # so it always returns standard output shape (B, C, H, W, D)
        preds = self(images)  # (B, 1, H, W, D)

        # Standard loss computation (no deep supervision during validation)
        preds_for_metrics = preds
        loss = self.criterion(preds, masks)

        # Apply sigmoid to get probabilities
        preds_prob = torch.sigmoid(preds_for_metrics)

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
        preds = self(images)  # (B, 1, H, W) or (B, num_outputs, 1, H, W)

        # Extract final output for metrics when using deep supervision
        if self.use_deep_supervision and preds.dim() == 5:
            # For testing, only use the final output (last element)
            # preds shape: (B, num_outputs, 1, H, W)
            preds_for_metrics = preds[:, -1, :, :, :]  # (B, 1, H, W)

            # Compute loss using all outputs (same as training)
            outputs = torch.unbind(preds, dim=1)
            losses = []
            for i, output in enumerate(outputs):
                loss_i = self.criterion(output, masks)
                weighted_loss_i = self.deep_supr_weights[i] * loss_i
                losses.append(weighted_loss_i)
            loss = sum(losses)
        else:
            # Standard single output or deep supervision not working as expected
            preds_for_metrics = preds
            loss = self.criterion(preds, masks)

        # Apply sigmoid to get probabilities
        preds_prob = torch.sigmoid(preds_for_metrics)

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
