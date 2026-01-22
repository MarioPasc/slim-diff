"""Anatomical Encoder Monitoring Callback.

Provides logging and visualization of encoder activations during training.
Useful for debugging and understanding how the encoder processes anatomical priors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class AnatomicalEncoderMonitoringCallback(Callback):
    """Monitor anatomical encoder behavior during training.

    Logs activation statistics and optionally visualizes feature maps
    from the encoder backbone. This helps understand:
    - Whether the encoder is learning meaningful representations
    - Per-channel activation patterns (for tissue maps)
    - Feature sparsity and distribution shifts during training

    Args:
        cfg: Configuration object (expects logging settings in encoder config).
        log_every_n_epochs: How often to log statistics.
        log_feature_stats: Whether to log activation mean/std/sparsity.
        log_channel_stats: Whether to log per-channel statistics.
        visualize_features: Whether to generate feature map visualizations.
        n_visualization_samples: Number of samples for visualization.
    """

    def __init__(
        self,
        cfg: DictConfig,
        log_every_n_epochs: int = 5,
        log_feature_stats: bool = True,
        log_channel_stats: bool = True,
        visualize_features: bool = True,
        n_visualization_samples: int = 4,
    ):
        super().__init__()

        self.cfg = cfg
        self.log_every_n_epochs = log_every_n_epochs
        self.log_feature_stats = log_feature_stats
        self.log_channel_stats = log_channel_stats
        self.visualize_features = visualize_features
        self.n_visualization_samples = n_visualization_samples

        # Storage for activations captured by hooks
        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

        # Track which encoder type we're monitoring
        self._encoder_type: Optional[str] = None

    def _register_hooks(self, encoder: nn.Module) -> None:
        """Register forward hooks to capture intermediate activations.

        Args:
            encoder: The anatomical encoder module.
        """
        # Clear existing hooks
        self._clear_hooks()

        # Detect encoder type and register appropriate hooks
        if hasattr(encoder, "backbone"):
            self._encoder_type = type(encoder).__name__

            # Hook the backbone output
            def backbone_hook(module, input, output):
                self._activations["backbone_output"] = output.detach()

            hook = encoder.backbone.register_forward_hook(backbone_hook)
            self._hooks.append(hook)

            # For FPN backbone, hook individual stages if available
            if hasattr(encoder.backbone, "stages"):
                for i, stage in enumerate(encoder.backbone.stages):
                    def stage_hook(module, input, output, stage_idx=i):
                        self._activations[f"stage_{stage_idx}"] = output.detach()
                    hook = stage.register_forward_hook(stage_hook)
                    self._hooks.append(hook)

            # Hook the fusion layer if present (FPN)
            if hasattr(encoder.backbone, "fusion"):
                def fusion_hook(module, input, output):
                    self._activations["fusion"] = output.detach()
                hook = encoder.backbone.fusion.register_forward_hook(fusion_hook)
                self._hooks.append(hook)

            logger.info(
                f"Registered {len(self._hooks)} monitoring hooks for {self._encoder_type}"
            )

    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations.clear()

    def _compute_activation_stats(
        self, tensor: torch.Tensor
    ) -> Dict[str, float]:
        """Compute statistics for an activation tensor.

        Args:
            tensor: Activation tensor (B, C, H, W) or (B, seq_len, dim).

        Returns:
            Dictionary of statistics.
        """
        with torch.no_grad():
            flat = tensor.float().flatten()
            stats = {
                "mean": flat.mean().item(),
                "std": flat.std().item(),
                "min": flat.min().item(),
                "max": flat.max().item(),
                "sparsity": (flat.abs() < 0.01).float().mean().item(),
            }
        return stats

    def _compute_channel_stats(
        self, tensor: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-channel statistics for an activation tensor.

        Args:
            tensor: Activation tensor of shape (B, C, H, W).

        Returns:
            Dictionary mapping channel index to statistics.
        """
        if tensor.dim() != 4:
            return {}

        channel_stats = {}
        with torch.no_grad():
            for c in range(tensor.shape[1]):
                channel_data = tensor[:, c].float().flatten()
                channel_stats[f"channel_{c}"] = {
                    "mean": channel_data.mean().item(),
                    "std": channel_data.std().item(),
                }
        return channel_stats

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Register hooks when training starts."""
        if hasattr(pl_module, "_anatomical_encoder") and pl_module._anatomical_encoder is not None:
            self._register_hooks(pl_module._anatomical_encoder)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Clean up hooks when training ends."""
        self._clear_hooks()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log activation statistics after training batches.

        Only logs at specified frequency to avoid overhead.
        """
        # Only log at the right epoch frequency and on first batch
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        if batch_idx != 0:
            return

        if not self.log_feature_stats:
            return

        if not self._activations:
            return

        # Log statistics for each captured activation
        for name, activation in self._activations.items():
            stats = self._compute_activation_stats(activation)

            for stat_name, value in stats.items():
                pl_module.log(
                    f"encoder/{name}_{stat_name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

            # Per-channel stats (only for 4D tensors, limited channels)
            if self.log_channel_stats and activation.dim() == 4:
                # Limit to avoid too many metrics
                n_channels = min(activation.shape[1], 8)
                for c in range(n_channels):
                    channel_data = activation[:, c].float()
                    pl_module.log(
                        f"encoder/{name}_ch{c}_mean",
                        channel_data.mean().item(),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Generate and log feature map visualizations at validation end."""
        if not self.visualize_features:
            return

        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        if not hasattr(pl_module, "_anatomical_encoder") or pl_module._anatomical_encoder is None:
            return

        # Get a sample of priors to visualize
        try:
            self._log_feature_visualizations(trainer, pl_module)
        except Exception as e:
            logger.warning(f"Failed to generate encoder visualizations: {e}")

    def _log_feature_visualizations(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Generate and log feature map visualizations.

        Creates a grid showing:
        - Input priors (first few channels)
        - Backbone output features (first few channels)
        - Final encoder output (projected to 2D via PCA or mean)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        encoder = pl_module._anatomical_encoder
        device = next(encoder.parameters()).device

        # Get sample priors from the validation dataloader
        # We need to generate some priors to visualize
        if not hasattr(pl_module, "_zbin_priors") or pl_module._zbin_priors is None:
            logger.debug("No z-bin priors available for visualization")
            return

        # Select a few z-bins to visualize
        n_bins = self.cfg.conditioning.z_bins
        sample_zbins = [0, n_bins // 4, n_bins // 2, 3 * n_bins // 4, n_bins - 1]
        sample_zbins = [z for z in sample_zbins if z < n_bins][:self.n_visualization_samples]

        # Generate priors for these z-bins
        priors = []
        for z_bin in sample_zbins:
            if z_bin in pl_module._zbin_priors:
                prior = pl_module._zbin_priors[z_bin]
                priors.append(torch.from_numpy(prior).float())

        if not priors:
            return

        # Stack and prepare input
        # Note: priors shape depends on encoder type (single or multi-channel)
        prior_batch = torch.stack(priors, dim=0).to(device)

        # Ensure correct shape: (B, C, H, W)
        if prior_batch.dim() == 3:
            prior_batch = prior_batch.unsqueeze(1)

        # Forward pass to capture activations
        with torch.no_grad():
            encoder_output = encoder(prior_batch)

        # Create visualization figure
        n_samples = len(priors)
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Input prior (first channel)
            ax = axes[i, 0]
            prior_img = prior_batch[i, 0].cpu().numpy()
            ax.imshow(prior_img, cmap="gray", vmin=-1, vmax=1)
            ax.set_title(f"Input Prior (z_bin={sample_zbins[i]})")
            ax.axis("off")

            # Backbone output (mean across channels)
            ax = axes[i, 1]
            if "backbone_output" in self._activations:
                backbone_feat = self._activations["backbone_output"][i]
                backbone_img = backbone_feat.mean(dim=0).cpu().numpy()
                ax.imshow(backbone_img, cmap="viridis")
                ax.set_title("Backbone Output (mean)")
            ax.axis("off")

            # Encoder output visualization (mean of sequence)
            ax = axes[i, 2]
            # encoder_output shape: (B, seq_len, embed_dim)
            output_viz = encoder_output[i].mean(dim=-1).cpu().numpy()
            # Reshape to 2D if possible
            seq_len = output_viz.shape[0]
            side = int(np.sqrt(seq_len))
            if side * side == seq_len:
                output_viz = output_viz.reshape(side, side)
                ax.imshow(output_viz, cmap="viridis")
            else:
                ax.plot(output_viz)
            ax.set_title("Encoder Output (mean)")
            ax.axis("off")

        plt.tight_layout()

        # Log to WandB if available
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            try:
                import wandb
                trainer.logger.experiment.log({
                    "encoder/feature_maps": wandb.Image(fig),
                    "epoch": trainer.current_epoch,
                })
            except Exception:
                pass

        plt.close(fig)


def build_encoder_monitoring_callback(
    cfg: DictConfig,
) -> Optional[AnatomicalEncoderMonitoringCallback]:
    """Factory function to build encoder monitoring callback.

    Args:
        cfg: Main configuration object.

    Returns:
        Callback instance if enabled, None otherwise.
    """
    # Check if callback is enabled
    callback_cfg = cfg.logging.callbacks.get("anatomical_encoder", {})
    if not callback_cfg.get("enabled", False):
        return None

    # Check if anatomical conditioning is enabled with cross_attention
    if not cfg.model.get("anatomical_conditioning", False):
        return None
    if cfg.model.get("anatomical_conditioning_method", "concat") != "cross_attention":
        return None

    # Try to load settings from encoder config
    try:
        from src.diffusion.model.factory import load_encoder_config
        encoder_cfg = load_encoder_config(cfg)
        logging_cfg = encoder_cfg.get("logging", {})
    except Exception:
        logging_cfg = {}

    callback = AnatomicalEncoderMonitoringCallback(
        cfg=cfg,
        log_every_n_epochs=logging_cfg.get("log_every_n_epochs", 5),
        log_feature_stats=logging_cfg.get("log_feature_stats", True),
        log_channel_stats=logging_cfg.get("log_channel_stats", True),
        visualize_features=logging_cfg.get("visualize_features", True),
        n_visualization_samples=logging_cfg.get("n_visualization_samples", 4),
    )

    logger.info("Built AnatomicalEncoderMonitoringCallback")
    return callback
