"""Lightning module for BottleneckSharedTwinDDPM training.

Trains a twin DDPM with independent encoders/decoders and a shared,
signal-coupled bottleneck. Mirrors the loss structure of
:class:`IndependentTwinLightningModule`:

- Independent noise draws per channel (image, mask).
- Per-channel Lp loss (p=1.5 image, p=2.0 mask).
- Uncertainty-weighted combination across channels.

The architectural difference is encapsulated in the forward call: the
joint two-channel input is passed once to ``self.model`` (which routes
through both encoders, the shared bottleneck, and both decoders), and
the output is split for per-channel loss computation.
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.diffusion.losses.uncertainty import UncertaintyWeightedLoss
from src.diffusion.model.bottleneck_shared_twin import build_bottleneck_shared_twin
from src.diffusion.model.components.conditioning import (
    get_null_token,
    prepare_cfg_tokens,
)
from src.diffusion.model.factory import build_inferer, build_scheduler
from src.diffusion.training.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class BottleneckSharedTwinLightningModule(pl.LightningModule):
    """Lightning module for ``BottleneckSharedTwinDDPM``.

    Parameters
    ----------
    cfg : DictConfig
        Full experiment configuration.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.model = build_bottleneck_shared_twin(cfg)
        self.scheduler = build_scheduler(cfg)
        self.inferer = build_inferer(cfg)

        self.metrics = MetricsCalculator()

        loss_cfg = cfg.loss
        self.image_p: float = float(loss_cfg.get("image_p", 1.5))
        self.mask_p: float = float(loss_cfg.get("mask_p", 2.0))

        uw_cfg = loss_cfg.get("uncertainty_weighting", {})
        self.uncertainty_loss = UncertaintyWeightedLoss(
            n_tasks=2,
            initial_log_vars=list(uw_cfg.get("initial_log_vars", [0.0, 0.0])),
            learnable=uw_cfg.get("learnable", True),
            clamp_range=tuple(uw_cfg.get("clamp_range", (-5.0, 5.0))),
        )

        self.use_cfg = cfg.conditioning.cfg.enabled
        self.cfg_dropout = cfg.conditioning.cfg.dropout_prob
        self.null_token = get_null_token(cfg.conditioning.z_bins)

        # Callback compatibility (same five attributes as
        # IndependentTwinLightningModule).
        self._use_self_conditioning = False
        self._use_anatomical_conditioning = False
        self._anatomical_method = "concat"
        self._anatomical_encoder = None
        self._zbin_priors = None

        self._register_scheduler_buffers()

    # ------------------------------------------------------------------
    # Scheduler helpers
    # ------------------------------------------------------------------

    def _register_scheduler_buffers(self) -> None:
        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod)

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        T = self.scheduler.num_train_timesteps
        return torch.randint(0, T, (batch_size,), device=self.device, dtype=torch.long)

    def _add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar_t = self.alphas_cumprod[timesteps]
        while alpha_bar_t.dim() < x0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def _get_target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        prediction_type = self.scheduler.prediction_type
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "sample":
            return x0
        if prediction_type == "v_prediction":
            alpha_bar_t = self.alphas_cumprod[timesteps]
            while alpha_bar_t.dim() < x0.dim():
                alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            return (
                torch.sqrt(alpha_bar_t) * noise
                - torch.sqrt(1.0 - alpha_bar_t) * x0
            )
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    # ------------------------------------------------------------------
    # Forward (sampling compatibility)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(x, timesteps=timesteps, context=context, class_labels=tokens)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        image = batch["image"]
        mask = batch["mask"]
        tokens = batch["token"]
        B = image.shape[0]

        if self.use_cfg and self.training:
            tokens = prepare_cfg_tokens(
                tokens, self.null_token, self.cfg_dropout, training=True,
            )

        timesteps = self._sample_timesteps(B)

        # Independent noise draws per branch — see test_independent_noise.
        noise_image = torch.randn_like(image)
        noise_mask = torch.randn_like(mask)

        x_t_image = self._add_noise(image, noise_image, timesteps)
        x_t_mask = self._add_noise(mask, noise_mask, timesteps)

        # Joint forward: the model routes the two channels through
        # independent encoders, the shared bottleneck (signal-coupled),
        # and independent decoders, returning a 2-channel tensor.
        x_t_joint = torch.cat([x_t_image, x_t_mask], dim=1)
        pred_joint = self.model(
            x_t_joint, timesteps=timesteps, class_labels=tokens,
        )
        pred_image = pred_joint[:, 0:1]
        pred_mask = pred_joint[:, 1:2]

        target_image = self._get_target(image, noise_image, timesteps)
        target_mask = self._get_target(mask, noise_mask, timesteps)

        loss_img = torch.mean(torch.abs(pred_image - target_image) ** self.image_p)
        loss_msk = torch.mean(torch.abs(pred_mask - target_mask) ** self.mask_p)

        total_loss, loss_details = self.uncertainty_loss([loss_img, loss_msk])

        bs = self.cfg.training.batch_size
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train/loss_image", loss_img, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train/loss_mask", loss_msk, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        if "log_var_0" in loss_details:
            self.log("train/log_var_image", loss_details["log_var_0"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
            self.log("train/log_var_mask", loss_details["log_var_1"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)

        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        image = batch["image"]
        mask = batch["mask"]
        tokens = batch["token"]
        B = image.shape[0]

        timesteps = self._sample_timesteps(B)

        noise_image = torch.randn_like(image)
        noise_mask = torch.randn_like(mask)

        x_t_image = self._add_noise(image, noise_image, timesteps)
        x_t_mask = self._add_noise(mask, noise_mask, timesteps)

        x_t_joint = torch.cat([x_t_image, x_t_mask], dim=1)
        pred_joint = self.model(
            x_t_joint, timesteps=timesteps, class_labels=tokens,
        )
        pred_image = pred_joint[:, 0:1]
        pred_mask = pred_joint[:, 1:2]

        target_image = self._get_target(image, noise_image, timesteps)
        target_mask = self._get_target(mask, noise_mask, timesteps)

        loss_img = torch.mean(torch.abs(pred_image - target_image) ** self.image_p)
        loss_msk = torch.mean(torch.abs(pred_mask - target_mask) ** self.mask_p)
        total_loss, _ = self.uncertainty_loss([loss_img, loss_msk])

        bs = self.cfg.training.batch_size
        self.log("val/loss", total_loss, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("val/loss_image", loss_img, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("val/loss_mask", loss_msk, on_epoch=True, sync_dist=True, batch_size=bs)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        opt_cfg = self.cfg.training.optimizer
        lr_cfg = self.cfg.training.lr_scheduler

        optimizer_cls = getattr(torch.optim, opt_cfg.type)
        optimizer = optimizer_cls(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps,
        )

        if lr_cfg.type == "CosineAnnealingLR":
            T_max = lr_cfg.T_max or self.cfg.training.max_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=lr_cfg.eta_min,
            )
        elif lr_cfg.type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=lr_cfg.get("factor", 0.5),
                patience=lr_cfg.get("patience", 10),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
            }
        else:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_train_epoch_end(self) -> None:
        opt = self.optimizers()
        if isinstance(opt, torch.optim.Optimizer):
            lr = opt.param_groups[0]["lr"]
            self.log(
                "train/lr",
                lr,
                sync_dist=True,
                batch_size=self.cfg.training.batch_size,
            )
