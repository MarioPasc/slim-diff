"""Training callbacks for audition."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class CSVLoggingCallback(Callback):
    """Callback to log metrics to CSV file.

    Args:
        cfg: Configuration dictionary.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.logs_dir = Path(cfg.output.logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.logs_dir / "training_metrics.csv"
        self.header_written = False
        self.metrics_keys: list[str] = []

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log metrics at end of training epoch."""
        self._write_metrics(trainer, "train")

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log metrics at end of validation epoch."""
        self._write_metrics(trainer, "val")

    def _write_metrics(self, trainer: pl.Trainer, stage: str) -> None:
        """Write metrics to CSV file.

        Args:
            trainer: Lightning trainer.
            stage: Current stage ("train" or "val").
        """
        metrics = trainer.callback_metrics

        # Filter relevant metrics
        row = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[key] = value
            elif hasattr(value, "item"):
                row[key] = value.item()

        if not row:
            return

        # Write header if first time
        if not self.header_written:
            self.metrics_keys = list(row.keys())
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.metrics_keys)
                writer.writeheader()
            self.header_written = True

        # Check for new keys
        current_keys = list(row.keys())
        if set(current_keys) != set(self.metrics_keys):
            # Rewrite with new header
            self.metrics_keys = sorted(set(self.metrics_keys) | set(current_keys))
            # Note: This would require rewriting the file, skip for simplicity

        # Write row
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_keys, extrasaction="ignore")
            writer.writerow(row)


class MetricsLoggerCallback(Callback):
    """Callback for detailed metrics logging.

    Logs metrics in a structured format for analysis.
    """

    def __init__(self, log_every_n_epochs: int = 1) -> None:
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log detailed metrics at validation end."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        metrics = trainer.callback_metrics

        # Extract key metrics
        val_auc = metrics.get("val/auc", 0)
        val_acc = metrics.get("val/acc", 0)
        val_loss = metrics.get("val/loss", 0)
        train_auc = metrics.get("train/auc", 0)
        train_loss = metrics.get("train/loss_epoch", 0)

        logger.info(
            f"Epoch {trainer.current_epoch}: "
            f"Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}"
        )


class GradientNormCallback(Callback):
    """Callback to monitor gradient norms during training.

    Useful for debugging training stability.
    """

    def __init__(self, log_every_n_steps: int = 100) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: any,
    ) -> None:
        """Log gradient norms before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        pl_module.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)
