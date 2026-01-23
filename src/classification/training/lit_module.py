"""PyTorch Lightning module for real vs. synthetic classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from torchmetrics import AUROC, Accuracy, AveragePrecision

from src.classification.models.factory import build_model


class ClassificationLightningModule(pl.LightningModule):
    """Lightning module for binary classification training.

    Trains a classifier to distinguish real (label=0) from synthetic (label=1)
    lesion patches. Tracks AUC-ROC, accuracy, and PR-AUC.

    Args:
        cfg: Master configuration.
        in_channels: Number of input channels (1 or 2).
        fold_idx: Current fold index for logging.
    """

    def __init__(self, cfg: DictConfig, in_channels: int, fold_idx: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self._fold_idx = fold_idx
        self.save_hyperparameters(ignore=["cfg"])

        # Model
        self.model = build_model(cfg, in_channels=in_channels)

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_auc = AUROC(task="binary")
        self.train_acc = Accuracy(task="binary")

        self.val_auc = AUROC(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_pr_auc = AveragePrecision(task="binary")

        # Store test outputs for post-hoc analysis
        self._test_probs: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []
        self._test_zbins: list[int] = []

    @property
    def fold_idx(self) -> int:
        return self._fold_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self.model(batch["patch"]).squeeze(-1)
        labels = batch["label"]
        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        self.train_auc.update(probs, labels.long())
        self.train_acc.update(preds, labels.long())

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/auc", self.train_auc.compute(), prog_bar=True)
        self.log("train/acc", self.train_acc.compute())
        self.train_auc.reset()
        self.train_acc.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self.model(batch["patch"]).squeeze(-1)
        labels = batch["label"]
        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        self.val_auc.update(probs, labels.long())
        self.val_acc.update(preds, labels.long())
        self.val_pr_auc.update(probs, labels.long())

        self.log("val/loss", loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val/auc", self.val_auc.compute(), prog_bar=True)
        self.log("val/acc", self.val_acc.compute())
        self.log("val/pr_auc", self.val_pr_auc.compute())
        self.val_auc.reset()
        self.val_acc.reset()
        self.val_pr_auc.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        logits = self.model(batch["patch"]).squeeze(-1)
        probs = torch.sigmoid(logits)
        self._test_probs.append(probs.cpu())
        self._test_labels.append(batch["label"].cpu())
        self._test_zbins.extend([int(z) for z in batch["z_bin"]])

    def get_test_outputs(self) -> dict:
        """Retrieve collected test outputs for evaluation.

        Returns:
            Dict with 'probs', 'labels', 'z_bins' as numpy arrays.
        """
        import numpy as np
        probs = torch.cat(self._test_probs).numpy()
        labels = torch.cat(self._test_labels).numpy()
        z_bins = np.array(self._test_zbins)
        return {"probs": probs, "labels": labels, "z_bins": z_bins}

    def clear_test_outputs(self) -> None:
        """Clear stored test outputs."""
        self._test_probs = []
        self._test_labels = []
        self._test_zbins = []

    def configure_optimizers(self) -> dict:
        train_cfg = self.cfg.training

        if train_cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
            )
        elif train_cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_cfg.optimizer}")

        config: dict = {"optimizer": optimizer}

        if train_cfg.scheduler.type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=train_cfg.scheduler.factor,
                patience=train_cfg.scheduler.patience,
                min_lr=train_cfg.scheduler.min_lr,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": train_cfg.early_stopping.monitor,
                "interval": "epoch",
            }
        elif train_cfg.scheduler.type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=train_cfg.max_epochs
            )
            config["lr_scheduler"] = {"scheduler": scheduler, "interval": "epoch"}

        return config
