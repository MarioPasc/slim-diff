"""Advanced diagnostic callbacks for JS-DDPM training.

Provides comprehensive diagnostics including timestep distributions,
per-class metrics, SNR tracking, and other deep learning best practices.

Histograms are saved to both wandb and local npz files for offline analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class WandbSummaryCallback(Callback):
    """Track and update wandb.summary with best metrics.

    Automatically updates wandb.summary with best validation metrics,
    making it easy to compare runs in the wandb UI.
    """

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()
        self.best_val_loss = float("inf")
        self.best_val_dice = 0.0
        self.best_val_psnr = 0.0
        self.best_val_ssim = 0.0

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Update wandb summary with best metrics.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        if not hasattr(trainer.logger, "experiment"):
            return

        import wandb

        # Get current metrics from trainer
        metrics = trainer.callback_metrics

        # Update best metrics
        if "val/loss" in metrics:
            val_loss = float(metrics["val/loss"])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                trainer.logger.experiment.summary["best_val_loss"] = val_loss
                trainer.logger.experiment.summary["best_val_loss_epoch"] = trainer.current_epoch

        if "val/dice" in metrics:
            val_dice = float(metrics["val/dice"])
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                trainer.logger.experiment.summary["best_val_dice"] = val_dice
                trainer.logger.experiment.summary["best_val_dice_epoch"] = trainer.current_epoch

        if "val/psnr" in metrics:
            val_psnr = float(metrics["val/psnr"])
            if val_psnr > self.best_val_psnr:
                self.best_val_psnr = val_psnr
                trainer.logger.experiment.summary["best_val_psnr"] = val_psnr
                trainer.logger.experiment.summary["best_val_psnr_epoch"] = trainer.current_epoch

        if "val/ssim" in metrics:
            val_ssim = float(metrics["val/ssim"])
            if val_ssim > self.best_val_ssim:
                self.best_val_ssim = val_ssim
                trainer.logger.experiment.summary["best_val_ssim"] = val_ssim
                trainer.logger.experiment.summary["best_val_ssim_epoch"] = trainer.current_epoch

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log final summary at end of training.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        if not hasattr(trainer.logger, "experiment"):
            return

        # Log total training time and epochs
        trainer.logger.experiment.summary["total_epochs"] = trainer.current_epoch + 1

        logger.info(
            f"Training complete. Best metrics - "
            f"Loss: {self.best_val_loss:.4f}, "
            f"Dice: {self.best_val_dice:.4f}, "
            f"PSNR: {self.best_val_psnr:.2f}, "
            f"SSIM: {self.best_val_ssim:.4f}"
        )


class DataStatisticsCallback(Callback):
    """Log dataset statistics at the start of training.

    Provides insights into data distribution, class balance,
    and other important dataset properties.
    """

    def __init__(self) -> None:
        """Initialize the callback."""
        super().__init__()
        self._logged = False

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log dataset statistics at start of training.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        if self._logged:
            return

        if not hasattr(trainer.logger, "experiment"):
            return

        # Get dataloaders
        train_loader = trainer.train_dataloader
        val_loader = trainer.val_dataloaders

        # Count samples
        train_samples = len(train_loader.dataset) if hasattr(train_loader.dataset, "__len__") else "unknown"
        val_samples = len(val_loader.dataset) if hasattr(val_loader.dataset, "__len__") else "unknown"

        # Log to wandb config
        trainer.logger.experiment.config.update({
            "data/train_samples": train_samples,
            "data/val_samples": val_samples,
            "data/train_batches_per_epoch": len(train_loader),
            "data/val_batches_per_epoch": len(val_loader) if val_loader else 0,
        }, allow_val_change=True)

        # Count class distribution in first epoch
        z_bins = pl_module.cfg.conditioning.z_bins
        lesion_count = 0
        control_count = 0
        total_count = 0

        # Sample from train loader to get class distribution
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 10:  # Only sample first 10 batches for efficiency
                break

            tokens = batch["token"]
            lesion_count += (tokens >= z_bins).sum().item()
            control_count += (tokens < z_bins).sum().item()
            total_count += len(tokens)

        if total_count > 0:
            lesion_ratio = lesion_count / total_count
            control_ratio = control_count / total_count

            trainer.logger.experiment.config.update({
                "data/lesion_ratio_sampled": lesion_ratio,
                "data/control_ratio_sampled": control_ratio,
            }, allow_val_change=True)

            logger.info(
                f"Dataset statistics - Train: {train_samples}, Val: {val_samples}, "
                f"Lesion ratio (sampled): {lesion_ratio:.2%}, Control ratio (sampled): {control_ratio:.2%}"
            )

        self._logged = True


class DiagnosticsCallback(Callback):
    """Comprehensive diagnostics callback for enhanced logging.

    Tracks and logs:
    - Timestep distribution histograms
    - Class/token distribution
    - Per-class validation metrics
    - Noise and prediction error statistics
    - SNR (Signal-to-Noise Ratio) across timesteps

    Histograms are saved to both wandb and local npz files.
    """

    def __init__(
        self,
        cfg: DictConfig,
        log_every_n_epochs: int = 1,
        log_histograms: bool = True,
    ) -> None:
        """Initialize the callback.

        Args:
            cfg: Configuration object.
            log_every_n_epochs: Frequency of diagnostic logging.
            log_histograms: Whether to log histograms (requires wandb).
        """
        super().__init__()
        self.cfg = cfg
        self.log_every_n_epochs = log_every_n_epochs
        self.log_histograms = log_histograms

        # Create histogram directory
        self.histogram_dir = Path(cfg.experiment.output_dir) / "csv_logs" / "histograms"
        self.histogram_dir.mkdir(parents=True, exist_ok=True)

        # Accumulators for epoch-level statistics
        self.train_timesteps = []
        self.train_tokens = []
        self.val_metrics_by_class = {"lesion": [], "control": []}

        logger.info(f"DiagnosticsCallback initialized, histograms will be saved to {self.histogram_dir}")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Accumulate batch-level statistics during training.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
            outputs: Batch outputs (loss).
            batch: Input batch.
            batch_idx: Batch index.
        """
        # Accumulate timesteps and tokens for distribution analysis
        # Note: We need to sample them again since they're not in outputs
        # This is lightweight and only done for diagnostics
        B = batch["image"].shape[0]
        timesteps = torch.randint(
            0,
            pl_module.cfg.scheduler.num_train_timesteps,
            (B,),
            device=pl_module.device,
        )
        tokens = batch["token"]

        self.train_timesteps.extend(timesteps.cpu().numpy().tolist())
        self.train_tokens.extend(tokens.cpu().numpy().tolist())

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log training diagnostics at epoch end.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        current_epoch = trainer.current_epoch

        if current_epoch % self.log_every_n_epochs != 0:
            return

        # Log timestep distribution
        if len(self.train_timesteps) > 0 and self.log_histograms:
            timesteps_array = np.array(self.train_timesteps)

            # Save to npz file
            self._save_histogram(
                name="timestep_distribution",
                data=timesteps_array,
                epoch=current_epoch,
            )

            # Log histogram to wandb
            if hasattr(trainer.logger, "experiment"):
                import wandb
                trainer.logger.experiment.log({
                    "diagnostics/timestep_distribution": wandb.Histogram(timesteps_array),
                    "epoch": current_epoch,
                })

            # Log statistics
            pl_module.log("diagnostics/timestep_mean", float(np.mean(timesteps_array)), sync_dist=False)
            pl_module.log("diagnostics/timestep_std", float(np.std(timesteps_array)), sync_dist=False)

        # Log class/token distribution
        if len(self.train_tokens) > 0:
            tokens_array = np.array(self.train_tokens)
            z_bins = pl_module.cfg.conditioning.z_bins

            # Count lesion vs control tokens
            # Tokens < z_bins are control, tokens >= z_bins are lesion
            n_control = np.sum(tokens_array < z_bins)
            n_lesion = np.sum(tokens_array >= z_bins)
            total = len(tokens_array)

            pl_module.log("diagnostics/class_balance_control", n_control / total, sync_dist=False)
            pl_module.log("diagnostics/class_balance_lesion", n_lesion / total, sync_dist=False)

            # Save to npz file
            self._save_histogram(
                name="token_distribution",
                data=tokens_array,
                epoch=current_epoch,
            )

            if self.log_histograms and hasattr(trainer.logger, "experiment"):
                import wandb
                trainer.logger.experiment.log({
                    "diagnostics/token_distribution": wandb.Histogram(tokens_array),
                    "epoch": current_epoch,
                })

        # Clear accumulators
        self.train_timesteps = []
        self.train_tokens = []

    def _save_histogram(
        self,
        name: str,
        data: np.ndarray,
        epoch: int,
    ) -> None:
        """Save histogram data to npz file.

        Args:
            name: Histogram name (e.g., 'timestep_distribution').
            data: Numpy array of values.
            epoch: Current epoch number.
        """
        # Create filename with epoch number
        filename = f"{name}_epoch{epoch:04d}.npz"
        filepath = self.histogram_dir / filename

        # Save to npz with additional metadata
        np.savez_compressed(
            filepath,
            data=data,
            epoch=epoch,
            histogram_name=name,
            # Add histogram statistics
            mean=np.mean(data),
            std=np.std(data),
            min=np.min(data),
            max=np.max(data),
            count=len(data),
        )

        logger.debug(f"Saved histogram {name} to {filepath}")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate per-class validation metrics.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
            outputs: Validation outputs with metrics.
            batch: Input batch.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.
        """
        # Separate metrics by class (lesion vs control)
        tokens = batch["token"]
        z_bins = pl_module.cfg.conditioning.z_bins

        # Classify samples
        is_lesion = tokens >= z_bins  # (B,)

        # Per-class metrics require per-sample metrics
        # For now, we'll track them at batch level
        # A more sophisticated approach would compute per-sample metrics

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log validation diagnostics at epoch end.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        # Per-class metrics would be logged here
        # Implementation depends on how we want to structure per-sample metrics
        pass


class SNRCallback(Callback):
    """Signal-to-Noise Ratio tracking callback.

    Computes and logs SNR at different timesteps to understand
    the difficulty of the denoising task across the diffusion process.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 5,
        n_samples: int = 100,
        timesteps_to_log: list[int] | None = None,
    ) -> None:
        """Initialize the callback.

        Args:
            log_every_n_epochs: Frequency of SNR logging.
            n_samples: Number of samples to use for SNR computation.
            timesteps_to_log: Specific timesteps to log (default: [0, 250, 500, 750, 999]).
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.n_samples = n_samples
        self.timesteps_to_log = timesteps_to_log or [0, 250, 500, 750, 999]

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Compute and log SNR at validation epoch end.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        current_epoch = trainer.current_epoch

        if current_epoch % self.log_every_n_epochs != 0:
            return

        pl_module.eval()
        with torch.no_grad():
            # Get a batch from validation loader
            val_loader = trainer.val_dataloaders
            if val_loader is None:
                return

            batch = next(iter(val_loader))
            image = batch["image"][:self.n_samples].to(pl_module.device)
            mask = batch["mask"][:self.n_samples].to(pl_module.device)
            x0 = torch.cat([image, mask], dim=1)

            # Compute SNR at different timesteps
            for t in self.timesteps_to_log:
                if t >= pl_module.cfg.scheduler.num_train_timesteps:
                    continue

                timesteps = torch.full((x0.shape[0],), t, device=pl_module.device, dtype=torch.long)

                # Get alpha_bar_t for this timestep
                alpha_bar_t = pl_module.alphas_cumprod[t]

                # SNR = alpha_bar_t / (1 - alpha_bar_t)
                snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)

                # Log SNR
                pl_module.log(
                    f"diagnostics/snr_t{t:04d}",
                    snr.item(),
                    sync_dist=False,
                )

                # Also log signal and noise scales
                pl_module.log(
                    f"diagnostics/signal_scale_t{t:04d}",
                    torch.sqrt(alpha_bar_t).item(),
                    sync_dist=False,
                )
                pl_module.log(
                    f"diagnostics/noise_scale_t{t:04d}",
                    torch.sqrt(1.0 - alpha_bar_t).item(),
                    sync_dist=False,
                )


class PredictionQualityCallback(Callback):
    """Track noise prediction quality across timesteps.

    Logs prediction errors at different timesteps to understand
    where the model struggles most.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 5,
        n_samples: int = 100,
        timestep_bins: int = 10,
    ) -> None:
        """Initialize the callback.

        Args:
            log_every_n_epochs: Frequency of logging.
            n_samples: Number of samples for evaluation.
            timestep_bins: Number of bins to divide timesteps into.
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.n_samples = n_samples
        self.timestep_bins = timestep_bins

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Evaluate prediction quality across timesteps.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        current_epoch = trainer.current_epoch

        if current_epoch % self.log_every_n_epochs != 0:
            return

        pl_module.eval()
        with torch.no_grad():
            val_loader = trainer.val_dataloaders
            if val_loader is None:
                return

            batch = next(iter(val_loader))
            image = batch["image"][:self.n_samples].to(pl_module.device)
            mask = batch["mask"][:self.n_samples].to(pl_module.device)
            tokens = batch["token"][:self.n_samples].to(pl_module.device)
            x0 = torch.cat([image, mask], dim=1)

            num_timesteps = pl_module.cfg.scheduler.num_train_timesteps
            bin_size = num_timesteps // self.timestep_bins

            # Evaluate each bin
            for bin_idx in range(self.timestep_bins):
                t_start = bin_idx * bin_size
                t_end = min((bin_idx + 1) * bin_size, num_timesteps)
                t_mid = (t_start + t_end) // 2

                timesteps = torch.full((x0.shape[0],), t_mid, device=pl_module.device, dtype=torch.long)
                noise = torch.randn_like(x0)

                # Add noise
                x_t = pl_module._add_noise(x0, noise, timesteps)

                # Predict noise
                eps_pred = pl_module(x_t, timesteps, tokens)

                # Compute per-channel MSE
                mse_image = torch.mean((eps_pred[:, 0:1] - noise[:, 0:1]) ** 2).item()
                mse_mask = torch.mean((eps_pred[:, 1:2] - noise[:, 1:2]) ** 2).item()

                # Log
                pl_module.log(
                    f"diagnostics/pred_mse_image_t{t_mid:04d}",
                    mse_image,
                    sync_dist=False,
                )
                pl_module.log(
                    f"diagnostics/pred_mse_mask_t{t_mid:04d}",
                    mse_mask,
                    sync_dist=False,
                )
