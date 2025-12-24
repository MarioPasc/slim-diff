"""PyTorch Lightning module for JS-DDPM training.

Implements the training, validation, and optimization logic
for the joint-synthesis diffusion model.
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.diffusion.losses.diffusion_losses import DiffusionLoss
from src.diffusion.model.components.conditioning import (
    get_null_token,
    prepare_cfg_tokens,
)
from src.diffusion.model.factory import (
    build_inferer,
    build_model,
    build_scheduler,
    predict_x0,
)
from src.diffusion.training.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class JSDDPMLightningModule(pl.LightningModule):
    """Lightning module for JS-DDPM training.

    Handles:
    - Model forward passes
    - Noise sampling and diffusion steps
    - Loss computation with uncertainty weighting
    - Metric computation on validation
    - Optimizer and scheduler configuration
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the module.

        Args:
            cfg: Configuration object.
        """
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        # Build model components
        self.model = build_model(cfg)
        self.scheduler = build_scheduler(cfg)
        self.inferer = build_inferer(cfg)

        # Build loss
        self.criterion = DiffusionLoss(cfg)

        # Metrics calculator
        self.metrics = MetricsCalculator()

        # CFG settings
        self.use_cfg = cfg.conditioning.cfg.enabled
        self.cfg_dropout = cfg.conditioning.cfg.dropout_prob
        self.null_token = get_null_token(cfg.conditioning.z_bins)

        # Cache alphas_cumprod for x0 prediction
        self._register_scheduler_buffers()

        logger.info(f"Initialized JSDDPMLightningModule with CFG={self.use_cfg}")

    def _register_scheduler_buffers(self) -> None:
        """Register scheduler tensors as buffers for device handling."""
        self.register_buffer(
            "alphas_cumprod",
            self.scheduler.alphas_cumprod.clone(),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Noisy input, shape (B, 2, H, W).
            timesteps: Timesteps, shape (B,).
            class_labels: Conditioning tokens, shape (B,).

        Returns:
            Predicted noise, shape (B, 2, H, W).
        """
        return self.model(x, timesteps=timesteps, class_labels=class_labels)

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for training.

        Args:
            batch_size: Number of samples.

        Returns:
            Timestep indices, shape (B,).
        """
        return torch.randint(
            0,
            self.cfg.scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    def _add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to x0 to get x_t.

        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x0: Original samples, shape (B, C, H, W).
            noise: Gaussian noise, shape (B, C, H, W).
            timesteps: Timesteps, shape (B,).

        Returns:
            Noisy samples x_t.
        """
        alpha_bar_t = self.alphas_cumprod[timesteps]

        # Reshape for broadcasting
        while alpha_bar_t.dim() < x0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t

    def _predict_x0(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x0 from noisy sample and predicted noise.

        Args:
            x_t: Noisy samples.
            eps_pred: Predicted noise.
            timesteps: Timesteps.

        Returns:
            Predicted x0.
        """
        alpha_bar_t = self.alphas_cumprod[timesteps]

        while alpha_bar_t.dim() < x_t.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

        x0_hat = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar
        return x0_hat

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch dictionary with image, mask, token.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        # Get inputs
        image = batch["image"]  # (B, 1, H, W)
        mask = batch["mask"]    # (B, 1, H, W)
        tokens = batch["token"]  # (B,)

        # Concatenate to form x0
        x0 = torch.cat([image, mask], dim=1)  # (B, 2, H, W)
        B = x0.shape[0]

        # Apply CFG dropout if enabled
        if self.use_cfg and self.training:
            tokens = prepare_cfg_tokens(
                tokens,
                self.null_token,
                self.cfg_dropout,
                training=True,
            )

        # Sample timesteps and noise
        timesteps = self._sample_timesteps(B)
        noise = torch.randn_like(x0)

        # Add noise to get x_t
        x_t = self._add_noise(x0, noise, timesteps)

        # Predict noise
        eps_pred = self(x_t, timesteps, tokens)

        # Compute loss
        loss, loss_details = self.criterion(eps_pred, noise, mask)

        # Log metrics
        self.log("train/loss", loss, sync_dist=True, batch_size=B)
        self.log("train/loss_image", loss_details["loss_image"], sync_dist=True, batch_size=B)
        self.log("train/loss_mask", loss_details["loss_mask"], sync_dist=True, batch_size=B)

        # Log uncertainty weights if available
        log_vars = self.criterion.get_log_vars()
        if log_vars is not None:
            self.log("train/log_var_image", log_vars[0], sync_dist=True, batch_size=B)
            self.log("train/log_var_mask", log_vars[1], sync_dist=True, batch_size=B)
            # Also log the actual weights (exp of log_vars)
            self.log("train/weight_image", torch.exp(-log_vars[0]), sync_dist=True, batch_size=B)
            self.log("train/weight_mask", torch.exp(-log_vars[1]), sync_dist=True, batch_size=B)

        return loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Batch dictionary.
            batch_idx: Batch index.

        Returns:
            Dictionary of metrics.
        """
        # Get inputs
        image = batch["image"]  # (B, 1, H, W)
        mask = batch["mask"]    # (B, 1, H, W)
        tokens = batch["token"]  # (B,)

        x0 = torch.cat([image, mask], dim=1)
        B = x0.shape[0]

        # Sample timesteps and noise
        timesteps = self._sample_timesteps(B)
        noise = torch.randn_like(x0)

        # Add noise
        x_t = self._add_noise(x0, noise, timesteps)

        # Predict noise
        eps_pred = self(x_t, timesteps, tokens)

        # Compute loss
        loss, loss_details = self.criterion(eps_pred, noise, mask)

        # Predict x0 for metrics
        x0_hat = self._predict_x0(x_t, eps_pred, timesteps)
        x0_hat_image = x0_hat[:, 0:1]
        x0_hat_mask = x0_hat[:, 1:2]

        # Compute image quality metrics
        metrics = self.metrics.compute_all(
            x0_hat_image, image,
            x0_hat_mask, mask,
        )

        # Log metrics
        self.log("val/loss", loss, sync_dist=True, batch_size=B)
        self.log("val/loss_image", loss_details["loss_image"], sync_dist=True, batch_size=B)
        self.log("val/loss_mask", loss_details["loss_mask"], sync_dist=True, batch_size=B)
        self.log("val/psnr", metrics["psnr"], sync_dist=True, batch_size=B)
        self.log("val/ssim", metrics["ssim"], sync_dist=True, batch_size=B)
        if "dice" in metrics:
            self.log("val/dice", metrics["dice"], sync_dist=True, batch_size=B)
        if "hd95" in metrics:
            self.log("val/hd95", metrics["hd95"], sync_dist=True, batch_size=B)

        # Log uncertainty weights if available
        log_vars = self.criterion.get_log_vars()
        if log_vars is not None:
            self.log("val/log_var_image", log_vars[0], sync_dist=True, batch_size=B)
            self.log("val/log_var_mask", log_vars[1], sync_dist=True, batch_size=B)
            self.log("val/weight_image", torch.exp(-log_vars[0]), sync_dist=True, batch_size=B)
            self.log("val/weight_mask", torch.exp(-log_vars[1]), sync_dist=True, batch_size=B)

        return {"loss": loss, **metrics}

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            Optimizer configuration dictionary.
        """
        opt_cfg = self.cfg.training.optimizer
        lr_cfg = self.cfg.training.lr_scheduler

        # Build optimizer
        optimizer_cls = getattr(torch.optim, opt_cfg.type)
        optimizer = optimizer_cls(
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
        elif lr_cfg.type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                },
            }
        else:
            # Default: no scheduler
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log learning rate
        opt = self.optimizers()
        if isinstance(opt, torch.optim.Optimizer):
            lr = opt.param_groups[0]["lr"]
            self.log("train/lr", lr, sync_dist=True, batch_size=self.cfg.training.batch_size)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch.

        Logs epoch summary for better tracking in cluster environments.
        """
        # Get current epoch
        epoch = self.current_epoch

        # Log epoch number for easier tracking
        logger.info(f"Completed epoch {epoch}")

        # The metrics are already logged via self.log() in validation_step
        # Lightning will automatically aggregate them across batches
        # This hook is just for additional logging/tracking if needed

    def get_model(self) -> nn.Module:
        """Get the underlying diffusion model.

        Returns:
            The DiffusionModelUNet.
        """
        return self.model
