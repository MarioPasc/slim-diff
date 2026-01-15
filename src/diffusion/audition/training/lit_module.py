"""PyTorch Lightning module for audition classifier training."""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, Precision, Recall

from ..models.classifier import AuditionClassifier

logger = logging.getLogger(__name__)


class AuditionLightningModule(pl.LightningModule):
    """Lightning module for real vs synthetic classifier.

    Handles training, validation, and testing with comprehensive metrics.

    Args:
        cfg: Configuration dictionary.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        # Build model
        self.model = AuditionClassifier.from_config(cfg)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics - instantiate per stage to avoid conflicts
        self._init_metrics()

        # Storage for per-zbin analysis
        self.test_outputs: list[dict] = []

    def _init_metrics(self) -> None:
        """Initialize metrics for each stage."""
        # Training metrics
        self.train_auc = AUROC(task="binary")
        self.train_acc = Accuracy(task="binary")

        # Validation metrics
        self.val_auc = AUROC(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_pr_auc = AveragePrecision(task="binary")

        # Test metrics
        self.test_auc = AUROC(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.test_pr_auc = AveragePrecision(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input patches (B, 2, H, W).

        Returns:
            Logits (B, 1).
        """
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Dictionary with 'patch', 'label', 'z_bin'.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        patches = batch["patch"]
        labels = batch["label"].float().unsqueeze(1)

        logits = self(patches)
        loss = self.criterion(logits, labels)

        # Compute metrics
        probs = torch.sigmoid(logits).squeeze(1)
        targets = batch["label"]

        self.train_auc.update(probs, targets)
        self.train_acc.update(probs, targets)

        # Log
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        self.log("train/auc", self.train_auc.compute(), prog_bar=True)
        self.log("train/acc", self.train_acc.compute())

        self.train_auc.reset()
        self.train_acc.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Dictionary with 'patch', 'label', 'z_bin'.
            batch_idx: Batch index.
        """
        patches = batch["patch"]
        labels = batch["label"].float().unsqueeze(1)

        logits = self(patches)
        loss = self.criterion(logits, labels)

        # Compute metrics
        probs = torch.sigmoid(logits).squeeze(1)
        targets = batch["label"]

        self.val_auc.update(probs, targets)
        self.val_acc.update(probs, targets)
        self.val_pr_auc.update(probs, targets)

        # Log
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        self.log("val/auc", self.val_auc.compute(), prog_bar=True)
        self.log("val/acc", self.val_acc.compute())
        self.log("val/pr_auc", self.val_pr_auc.compute())

        self.val_auc.reset()
        self.val_acc.reset()
        self.val_pr_auc.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step with per-sample storage for z-bin analysis.

        Args:
            batch: Dictionary with 'patch', 'label', 'z_bin', 'source'.
            batch_idx: Batch index.
        """
        patches = batch["patch"]
        labels = batch["label"].float().unsqueeze(1)
        z_bins = batch["z_bin"]

        logits = self(patches)
        loss = self.criterion(logits, labels)

        # Compute metrics
        probs = torch.sigmoid(logits).squeeze(1)
        targets = batch["label"]

        self.test_auc.update(probs, targets)
        self.test_acc.update(probs, targets)
        self.test_pr_auc.update(probs, targets)
        self.test_f1.update(probs, targets)
        self.test_precision.update(probs, targets)
        self.test_recall.update(probs, targets)

        # Store for per-zbin analysis
        self.test_outputs.append(
            {
                "probs": probs.detach().cpu(),
                "labels": targets.detach().cpu(),
                "z_bins": z_bins.detach().cpu(),
            }
        )

        # Log
        self.log("test/loss", loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Log test metrics and aggregate outputs."""
        # Log global metrics
        self.log("test/auc", self.test_auc.compute())
        self.log("test/acc", self.test_acc.compute())
        self.log("test/pr_auc", self.test_pr_auc.compute())
        self.log("test/f1", self.test_f1.compute())
        self.log("test/precision", self.test_precision.compute())
        self.log("test/recall", self.test_recall.compute())

        # Reset metrics
        self.test_auc.reset()
        self.test_acc.reset()
        self.test_pr_auc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()

    def get_test_outputs(self) -> dict:
        """Aggregate test outputs for analysis.

        Returns:
            Dictionary with 'probs', 'labels', 'z_bins' arrays.
        """
        probs = torch.cat([o["probs"] for o in self.test_outputs])
        labels = torch.cat([o["labels"] for o in self.test_outputs])
        z_bins = torch.cat([o["z_bins"] for o in self.test_outputs])

        return {
            "probs": probs.numpy(),
            "labels": labels.numpy(),
            "z_bins": z_bins.numpy(),
        }

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            Optimizer configuration dictionary.
        """
        train_cfg = self.cfg.training

        # Optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        config = {"optimizer": optimizer}

        # Learning rate scheduler
        if train_cfg.scheduler.enabled:
            scheduler_cfg = train_cfg.scheduler

            if scheduler_cfg.type == "reduce_on_plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",  # We're maximizing AUC
                    factor=scheduler_cfg.factor,
                    patience=scheduler_cfg.patience,
                    min_lr=scheduler_cfg.min_lr,
                )
                config["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "monitor": "val/auc",
                    "interval": "epoch",
                }
            elif scheduler_cfg.type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=train_cfg.max_epochs,
                    eta_min=scheduler_cfg.min_lr,
                )
                config["lr_scheduler"] = scheduler

        return config
