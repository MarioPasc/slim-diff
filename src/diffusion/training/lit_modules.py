"""PyTorch Lightning module for JS-DDPM training.

Implements the training, validation, and optimization logic
for the joint-synthesis diffusion model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from numpy.typing import NDArray
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
    DiffusionSampler,
)
from src.diffusion.training.metrics import MetricsCalculator
from src.diffusion.utils.zbin_priors import (
    apply_postprocess_batch,
    get_anatomical_priors_as_input,
    load_zbin_priors,
)

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

        # Enable Tensor Cores on CUDA devices for better performance
        #if torch.cuda.is_available():
        #    torch.set_float32_matmul_precision('medium')

        # Validate configuration before building components
        from src.diffusion.config.validation import validate_config
        validate_config(cfg)

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

        # Z-bin priors for validation post-processing
        self._zbin_priors: dict[int, NDArray[np.bool_]] | None = None
        pp_cfg = cfg.get("postprocessing", {})
        zbin_cfg = pp_cfg.get("zbin_priors", {})
        self._use_zbin_priors = (
            zbin_cfg.get("enabled", False)
            and "validation" in zbin_cfg.get("apply_to", [])
        )

        # Anatomical conditioning (input concatenation)
        self._use_anatomical_conditioning = cfg.model.get("anatomical_conditioning", False)

        # Load priors if needed for either validation postprocessing or anatomical conditioning
        if self._use_zbin_priors or self._use_anatomical_conditioning:
            self._load_zbin_priors()

        # Sampler for validation metrics (created lazily when device is available)
        self._val_sampler: DiffusionSampler | None = None

        logger.info(
            f"Initialized JSDDPMLightningModule with CFG={self.use_cfg}, "
            f"zbin_priors_postprocess={self._use_zbin_priors}, "
            f"anatomical_conditioning={self._use_anatomical_conditioning}"
        )

    def setup(self, stage: str) -> None:
        """Setup hook called at the beginning of fit/validate/test.

        Logs model architecture and dataset statistics to wandb.

        Args:
            stage: Current stage ('fit', 'validate', 'test').
        """
        if stage == "fit":
            self._log_model_architecture()

    def _log_model_architecture(self) -> None:
        """Log model architecture information to wandb."""
        if not hasattr(self, "logger") or self.logger is None:
            return

        if not hasattr(self.logger, "experiment"):
            return

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Log to wandb summary
        self.logger.experiment.summary["model/total_parameters"] = total_params
        self.logger.experiment.summary["model/trainable_parameters"] = trainable_params

        # Log architecture details
        logger.info(f"Model architecture: {self.cfg.model.type}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Log to wandb config (in case it wasn't already logged)
        self.logger.experiment.config.update({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
        }, allow_val_change=True)

    def _register_scheduler_buffers(self) -> None:
        """Register scheduler tensors as buffers for device handling."""
        self.register_buffer(
            "alphas_cumprod",
            self.scheduler.alphas_cumprod.clone(),
        )

    def _load_zbin_priors(self) -> None:
        """Load z-bin priors from cache for post-processing."""
        pp_cfg = self.cfg.postprocessing.zbin_priors
        cache_dir = Path(self.cfg.data.cache_dir)
        z_bins = self.cfg.conditioning.z_bins

        try:
            self._zbin_priors = load_zbin_priors(
                cache_dir, pp_cfg.priors_filename, z_bins
            )
            logger.info(f"Loaded z-bin priors for {len(self._zbin_priors)} bins")
        except Exception as e:
            logger.warning(f"Failed to load z-bin priors: {e}. Post-processing disabled.")
            self._use_zbin_priors = False
            self._zbin_priors = None

    def _get_val_sampler(self) -> DiffusionSampler:
        """Get or create the diffusion sampler for validation metrics.

        Returns:
            DiffusionSampler configured for inference.
        """
        if self._val_sampler is None:
            self._val_sampler = DiffusionSampler(
                model=self.model,
                scheduler=self.inferer,
                cfg=self.cfg,
                device=self.device,
            )
            logger.info(
                f"Created validation sampler with {self.cfg.sampler.num_inference_steps} "
                f"inference steps for quality metrics"
            )
        return self._val_sampler

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Noisy input, shape (B, C, H, W) where C=2 (no anatomical)
               or C=3 (with anatomical prior concatenated).
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
            self.scheduler.num_train_timesteps,
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
        # Use scheduler's add_noise if available (MONAI 1.0+)
        if hasattr(self.scheduler, "add_noise"):
            return self.scheduler.add_noise(original_samples=x0, noise=noise, timesteps=timesteps)

        alpha_bar_t = self.alphas_cumprod[timesteps]

        # Reshape for broadcasting
        while alpha_bar_t.dim() < x0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t

    def _get_target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Get the target for loss computation based on prediction type.

        Args:
            x0: Original samples.
            noise: Gaussian noise.
            timesteps: Timesteps.

        Returns:
            Target tensor (noise, x0, or v).
        """
        prediction_type = self.scheduler.prediction_type

        if prediction_type == "epsilon":
            return noise
        elif prediction_type == "sample":
            return x0
        elif prediction_type == "v_prediction":
            # v = sqrt(alpha_bar) * noise - sqrt(1-alpha_bar) * x0
            alpha_bar_t = self.alphas_cumprod[timesteps]
            while alpha_bar_t.dim() < x0.dim():
                alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
            
            return sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * x0
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

    def _predict_x0(
        self,
        x_t: torch.Tensor,
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x0 from noisy sample and model output.

        Args:
            x_t: Noisy samples.
            model_output: Model prediction (epsilon, x0, or v).
            timesteps: Timesteps.

        Returns:
            Predicted x0.
        """
        prediction_type = self.scheduler.prediction_type

        if prediction_type == "sample":
            return model_output

        alpha_bar_t = self.alphas_cumprod[timesteps]

        while alpha_bar_t.dim() < x_t.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

        if prediction_type == "epsilon":
            x0_hat = (x_t - sqrt_one_minus_alpha_bar * model_output) / sqrt_alpha_bar
        elif prediction_type == "v_prediction":
            # x0 = sqrt(alpha_bar) * x_t - sqrt(1-alpha_bar) * v
            x0_hat = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * model_output
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
            
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

        # Concatenate anatomical priors if enabled
        if self._use_anatomical_conditioning and self._zbin_priors is not None:
            # Get z_bins from batch metadata
            z_bins_batch = batch["metadata"]["z_bin"]  # list[int] of length B

            # Get anatomical priors as input channel
            anatomical_priors = get_anatomical_priors_as_input(
                z_bins_batch,
                self._zbin_priors,
                device=x_t.device,
            )  # (B, 1, H, W)

            # Concatenate to x_t: (B, 2, H, W) + (B, 1, H, W) -> (B, 3, H, W)
            x_t = torch.cat([x_t, anatomical_priors], dim=1)

        # Predict
        model_output = self(x_t, timesteps, tokens)

        # Get target for loss
        target = self._get_target(x0, noise, timesteps)

        # Compute loss with mask for channel-separated multi-task learning
        loss, loss_details = self.criterion(model_output, target, mask)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
        if "loss_image" in loss_details:
            self.log("train/loss_image", loss_details["loss_image"], on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
        if "loss_mask" in loss_details:
            self.log("train/loss_mask", loss_details["loss_mask"], on_step=True, on_epoch=True, sync_dist=True, batch_size=B)

        # Log weighted loss metrics (always available when using multi-task loss)
        if "weighted_loss_0" in loss_details:
            self.log("train/weighted_loss_image", loss_details["weighted_loss_0"], on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
            self.log("train/weighted_loss_mask", loss_details["weighted_loss_1"], on_step=True, on_epoch=True, sync_dist=True, batch_size=B)

        # Log uncertainty-specific metrics (only when uncertainty weighting is enabled)
        # Note: These keys come from UncertaintyWeightedLoss.forward() when enabled
        if "log_var_0" in loss_details:
            self.log("train/log_var_image", loss_details["log_var_0"], on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
            self.log("train/log_var_mask", loss_details["log_var_1"], on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
            self.log("train/sigma_image", loss_details["sigma_0"], on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
            self.log("train/sigma_mask", loss_details["sigma_1"], on_step=True, on_epoch=True, sync_dist=True, batch_size=B)

            # Log precision (exp(-log_var)) for interpretability
            self.log("train/precision_image", torch.exp(-loss_details["log_var_0"]), on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
            self.log("train/precision_mask", torch.exp(-loss_details["log_var_1"]), on_step=True, on_epoch=True, sync_dist=True, batch_size=B)

        return loss

    def _get_reconstruction_timesteps(self) -> list[int]:
        """Timesteps for denoising-based monitoring.

        Chosen to cover low/mid/high noise regimes for T=1000 while
        avoiding the near-pure-noise endpoint where metrics get noisy.
        """
        T = self.scheduler.num_train_timesteps
        # Scale anchors if T changes
        anchors = [0.05, 0.25, 0.75]  # low, mid, high
        ts = [min(int(round(a * T)), T - 1) for a in anchors]
        # Ensure strictly increasing and within bounds
        ts = sorted(set(max(0, t) for t in ts))
        return ts


    def _denoise_from_timestep(
        self,
        x_t: torch.Tensor,
        start_timestep: int,
        tokens: torch.Tensor,
        anatomical_priors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denoise from a specific timestep back to x0.

        Performs iterative denoising from timestep t down to 0 using the
        scheduler's step function.

        IMPORTANT: When we add noise at timestep t, we need to start denoising
        from a scheduler timestep >= t (not < t) to ensure we cover the full
        denoising trajectory. Otherwise, we'd skip the initial denoising step(s)
        and get poor reconstruction quality.

        Args:
            x_t: Noisy samples at timestep t, shape (B, 2, H, W).
            start_timestep: Starting timestep for denoising.
            tokens: Conditioning tokens, shape (B,).
            anatomical_priors: Optional anatomical priors, shape (B, 1, H, W).

        Returns:
            Denoised samples x0_hat, shape (B, 2, H, W).
        """
        B = x_t.shape[0]

        # Set timesteps for the scheduler
        sampler = self._get_val_sampler()
        sampler.scheduler.set_timesteps(self.sampler_cfg.num_inference_steps)

        # Get all scheduler timesteps (descending order: [999, 994, 989, ..., 4, 0])
        all_timesteps = sampler.scheduler.timesteps

        # CRITICAL FIX: Find the first scheduler timestep that is >= start_timestep
        # This ensures we start denoising from a noise level at least as high as
        # where we added noise. The previous code found t <= start_timestep, which
        # caused us to skip the initial denoising step(s).
        #
        # Example: If start_timestep=50 and scheduler timesteps are [999, 994, ..., 54, 49, ...]
        # - Old behavior: start at 49 (skipping t=54 step)
        # - New behavior: start at 54 (includes full denoising from >= 50)
        start_idx = len(all_timesteps)  # Default: no timesteps to run
        for i, t in enumerate(all_timesteps):
            if t <= start_timestep:
                # Found the first timestep <= start_timestep
                # We want to include the PREVIOUS timestep (the one > start_timestep)
                # to ensure we start from >= start_timestep
                start_idx = max(0, i - 1) if i > 0 else i
                break
        else:
            # All timesteps are > start_timestep, start from beginning
            start_idx = 0

        # If start_timestep is very small and all scheduler timesteps are above it,
        # we might need to just use the last timestep
        if start_idx >= len(all_timesteps):
            # No scheduler timesteps <= start_timestep, likely start_timestep is very low
            # Use all timesteps from the one closest to start_timestep
            for i in range(len(all_timesteps) - 1, -1, -1):
                if all_timesteps[i] >= start_timestep:
                    start_idx = i
                    break

        timesteps_to_run = all_timesteps[start_idx:]

        # Log for debugging (only first batch to avoid spam)
        if B > 0:
            logger.debug(
                f"Denoising from t={start_timestep}: using scheduler timesteps "
                f"[{timesteps_to_run[0].item() if len(timesteps_to_run) > 0 else 'none'}, ..., "
                f"{timesteps_to_run[-1].item() if len(timesteps_to_run) > 0 else 'none'}] "
                f"({len(timesteps_to_run)} steps)"
            )

        current_x = x_t.clone()

        for t in timesteps_to_run:
            timesteps_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

            # Prepare model input
            model_input = current_x
            if self._use_anatomical_conditioning and anatomical_priors is not None:
                model_input = torch.cat([current_x, anatomical_priors], dim=1)

            # Predict noise (no CFG during reconstruction evaluation)
            noise_pred = sampler.model(model_input, timesteps=timesteps_batch, class_labels=tokens)

            # Scheduler step
            if hasattr(sampler.scheduler, 'step'):
                if hasattr(sampler, 'eta'):
                    current_x, _ = sampler.scheduler.step(noise_pred, t, current_x, eta=sampler.eta)
                else:
                    current_x, _ = sampler.scheduler.step(noise_pred, t, current_x)
            else:
                current_x, _ = sampler.scheduler.step(noise_pred, t, current_x)

        return current_x

    @property
    def sampler_cfg(self) -> DictConfig:
        """Get sampler configuration."""
        return self.cfg.sampler

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Validation step.

        Computes:
        1. Validation loss using random timesteps (consistent with training)
        2. Quality metrics (PSNR, SSIM, Dice, HD95) using RECONSTRUCTION at 4 fixed timesteps
           - Reconstruction: noise real x0 to timestep t, denoise back, compare with original x0
           - Timesteps: T/4, T/2, 3T/4, T (where T = num_train_timesteps)
        3. Generation samples for visualization (pure noise -> denoised, not used for metrics)

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
        H, W = image.shape[2], image.shape[3]

        # ========== Loss computation at random timesteps ==========
        # This is consistent with training and monitors denoising performance
        timesteps = self._sample_timesteps(B)
        noise = torch.randn_like(x0)

        # Add noise
        x_t = self._add_noise(x0, noise, timesteps)

        # Concatenate anatomical priors if enabled
        anatomical_priors = None
        if self._use_anatomical_conditioning and self._zbin_priors is not None:
            z_bins_batch = batch["metadata"]["z_bin"]
            anatomical_priors = get_anatomical_priors_as_input(
                z_bins_batch,
                self._zbin_priors,
                device=x_t.device,
            )
            x_t_with_priors = torch.cat([x_t, anatomical_priors], dim=1)
        else:
            x_t_with_priors = x_t

        # Predict
        model_output = self(x_t_with_priors, timesteps, tokens)

        # Get target for loss
        target = self._get_target(x0, noise, timesteps)

        # Compute loss with mask for channel-separated multi-task learning
        loss, loss_details = self.criterion(model_output, target, mask)

        # ========== Quality metrics via RECONSTRUCTION at fixed timesteps ==========
        # Reconstruction-based validation: noise x0 to t, denoise back, compare with original x0
        # This measures true denoising fidelity (same image comparison)
        reconstruction_timesteps = self._get_reconstruction_timesteps()
        
        all_metrics = {}
        with torch.no_grad():
            for t_val in reconstruction_timesteps:
                # Create timestep tensor
                t_tensor = torch.full((B,), t_val, device=self.device, dtype=torch.long)
                
                # Sample fresh noise for this timestep evaluation
                noise_t = torch.randn_like(x0)
                
                # Add noise to get x_t at this specific timestep
                x_t_eval = self._add_noise(x0, noise_t, t_tensor)
                
                # Denoise from t back to 0
                x0_hat = self._denoise_from_timestep(
                    x_t_eval, 
                    t_val, 
                    tokens, 
                    anatomical_priors
                )
                
                # Split into image and mask channels
                recon_image = x0_hat[:, 0:1]  # (B, 1, H, W)
                recon_mask = x0_hat[:, 1:2]   # (B, 1, H, W)
                
                # Apply z-bin prior post-processing before metrics (if enabled)
                if self._use_zbin_priors and self._zbin_priors is not None:
                    z_bins_batch = batch["metadata"]["z_bin"]
                    recon_image, recon_mask = apply_postprocess_batch(
                        recon_image, recon_mask, z_bins_batch,
                        self._zbin_priors, self.cfg
                    )
                
                # Compute metrics comparing reconstruction to ORIGINAL x0
                metrics_t = self.metrics.compute_all(
                    recon_image, image,  # Compare reconstructed vs original
                    recon_mask, mask,
                )
                
                # Store with timestep suffix
                for metric_name, value in metrics_t.items():
                    all_metrics[f"{metric_name}_t{t_val}"] = value

        # ========== Log metrics ==========
        # Loss metrics
        self.log("val/loss", loss, sync_dist=True, batch_size=B)
        if "loss_image" in loss_details:
            self.log("val/loss_image", loss_details["loss_image"], sync_dist=True, batch_size=B)
        if "loss_mask" in loss_details:
            self.log("val/loss_mask", loss_details["loss_mask"], sync_dist=True, batch_size=B)
        
        # Reconstruction metrics at each timestep
        for t_val in reconstruction_timesteps:
            self.log(f"val/psnr_t{t_val}", all_metrics[f"psnr_t{t_val}"], sync_dist=True, batch_size=B)
            self.log(f"val/ssim_t{t_val}", all_metrics[f"ssim_t{t_val}"], sync_dist=True, batch_size=B)
            if f"dice_t{t_val}" in all_metrics:
                self.log(f"val/dice_t{t_val}", all_metrics[f"dice_t{t_val}"], sync_dist=True, batch_size=B)
            if f"hd95_t{t_val}" in all_metrics:
                self.log(f"val/hd95_t{t_val}", all_metrics[f"hd95_t{t_val}"], sync_dist=True, batch_size=B)

        # Log weighted loss metrics (always available when using multi-task loss)
        if "weighted_loss_0" in loss_details:
            self.log("val/weighted_loss_image", loss_details["weighted_loss_0"], sync_dist=True, batch_size=B)
            self.log("val/weighted_loss_mask", loss_details["weighted_loss_1"], sync_dist=True, batch_size=B)

        # Log uncertainty-specific metrics (only when uncertainty weighting is enabled)
        if "log_var_0" in loss_details:
            self.log("val/log_var_image", loss_details["log_var_0"], sync_dist=True, batch_size=B)
            self.log("val/log_var_mask", loss_details["log_var_1"], sync_dist=True, batch_size=B)
            self.log("val/sigma_image", loss_details["sigma_0"], sync_dist=True, batch_size=B)
            self.log("val/sigma_mask", loss_details["sigma_1"], sync_dist=True, batch_size=B)

            # Log precision (exp(-log_var)) for interpretability
            self.log("val/precision_image", torch.exp(-loss_details["log_var_0"]), sync_dist=True, batch_size=B)
            self.log("val/precision_mask", torch.exp(-loss_details["log_var_1"]), sync_dist=True, batch_size=B)

        return {"loss": loss, **all_metrics}

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        When uncertainty weighting is enabled with learnable=True, the optimizer
        will update both the model parameters AND the log variance parameters
        (self.criterion.uncertainty_loss.log_vars).

        Returns:
            Optimizer configuration dictionary.
        """
        opt_cfg = self.cfg.training.optimizer
        lr_cfg = self.cfg.training.lr_scheduler

        # Collect all trainable parameters
        # This includes:
        # 1. Model parameters (DiffusionModelUNet)
        # 2. Uncertainty log_vars (if learnable=True in config)
        params = self.parameters()

        # Optional: Log what's being optimized
        if self.cfg.loss.uncertainty_weighting.enabled and self.cfg.loss.uncertainty_weighting.learnable:
            # Verify log_vars are trainable
            if hasattr(self.criterion, 'uncertainty_loss'):
                log_vars = self.criterion.uncertainty_loss.log_vars
                if log_vars.requires_grad:
                    total_params = sum(p.numel() for p in self.parameters())
                    logger.info(
                        f"Optimizer will update {total_params} parameters "
                        f"including {log_vars.numel()} uncertainty log_vars"
                    )

        # Build optimizer
        optimizer_cls = getattr(torch.optim, opt_cfg.type)
        optimizer = optimizer_cls(
            params,
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
                factor=lr_cfg.get("factor", 0.5),
                patience=lr_cfg.get("patience", 10),
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

        # CRITICAL: Reset MONAI metrics to clear internal CUDA buffers
        # This prevents "CUDA error: initialization error" when DataLoader workers
        # are forked at the start of the next epoch. MONAI metrics cache CUDA tensors
        # internally, which cannot be shared across forked processes.
        self.metrics.reset()

    def get_model(self) -> nn.Module:
        """Get the underlying diffusion model.

        Returns:
            The DiffusionModelUNet.
        """
        return self.model
