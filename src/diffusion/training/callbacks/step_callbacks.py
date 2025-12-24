"""Step-level callbacks for JS-DDPM training.

Optional callbacks for monitoring training progress at step level.
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class GradientNormCallback(Callback):
    """Callback to log gradient norms during training.

    Useful for debugging training instabilities.
    """

    def __init__(self, log_every_n_steps: int = 100) -> None:
        """Initialize the callback.

        Args:
            log_every_n_steps: Frequency of gradient norm logging.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._step = 0

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Log gradient norms before optimizer step.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
            optimizer: Optimizer.
        """
        self._step += 1

        if self._step % self.log_every_n_steps != 0:
            return

        # Compute gradient norms
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Log
        pl_module.log("train/grad_norm", total_norm, sync_dist=True)


class LearningRateMonitor(Callback):
    """Monitor and log learning rate at each step.

    Simpler alternative to Lightning's built-in LearningRateMonitor.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        """Initialize the callback.

        Args:
            log_every_n_steps: Logging frequency.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._step = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log learning rate after batch.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
            outputs: Batch outputs.
            batch: Input batch.
            batch_idx: Batch index.
        """
        self._step += 1

        if self._step % self.log_every_n_steps != 0:
            return

        opt = trainer.optimizers[0]
        if isinstance(opt, torch.optim.Optimizer):
            lr = opt.param_groups[0]["lr"]
            pl_module.log("train/learning_rate", lr, sync_dist=True)
