"""Epoch-level callbacks for JS-DDPM training.

Provides visualization callback for generating sample grids
at validation time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

from src.diffusion.model.components.conditioning import get_visualization_tokens
from src.diffusion.model.factory import DiffusionSampler

logger = logging.getLogger(__name__)


def to_display_range(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert from [-1, 1] to [0, 1] for display.

    Args:
        x: Input tensor/array in [-1, 1].

    Returns:
        Array in [0, 1].
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return np.clip((x + 1) / 2, 0, 1)


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
    threshold: float = 0.0,
) -> np.ndarray:
    """Create image with mask overlay.

    Args:
        image: Grayscale image in [0, 1], shape (H, W).
        mask: Mask in [-1, 1] or [0, 1], shape (H, W).
        alpha: Overlay transparency.
        color: RGB color for overlay (0-255).
        threshold: Threshold for binarizing mask.

    Returns:
        RGB image with overlay, shape (H, W, 3).
    """
    # Ensure 2D
    if image.ndim == 3:
        image = image.squeeze()
    if mask.ndim == 3:
        mask = mask.squeeze()

    # Convert to RGB
    rgb = np.stack([image, image, image], axis=-1)

    # Binarize mask
    if mask.min() < 0:
        mask = to_display_range(mask)
    binary_mask = mask > (threshold + 1) / 2  # Convert threshold to [0,1]

    # Normalize color
    color_norm = np.array(color, dtype=np.float32) / 255.0

    # Apply overlay
    if binary_mask.any():
        for c in range(3):
            rgb[:, :, c] = np.where(
                binary_mask,
                (1 - alpha) * rgb[:, :, c] + alpha * color_norm[c],
                rgb[:, :, c],
            )

    return rgb


def create_visualization_grid(
    samples: list[torch.Tensor],
    z_bins: list[int],
    cfg: DictConfig,
) -> np.ndarray:
    """Create a 2x5 visualization grid.

    Row 1: Control samples (no lesion)
    Row 2: Epilepsy samples (with lesion overlay)

    Args:
        samples: List of 10 samples (5 control + 5 epilepsy), each (2, H, W).
        z_bins: List of z-bin values used.
        cfg: Configuration for overlay settings.

    Returns:
        Grid image as numpy array, shape (H*2, W*5, 3).
    """
    n_cols = len(z_bins)
    vis_cfg = cfg.visualization.overlay

    # Split into control and epilepsy
    control_samples = samples[:n_cols]
    epilepsy_samples = samples[n_cols:]

    # Get image dimensions
    H, W = control_samples[0].shape[1:]

    # Create grid
    grid = np.zeros((H * 2, W * n_cols, 3), dtype=np.float32)

    # Row 1: Control (just image, no overlay)
    for i, sample in enumerate(control_samples):
        image = to_display_range(sample[0])  # (H, W)
        rgb = np.stack([image, image, image], axis=-1)
        grid[:H, i * W : (i + 1) * W] = rgb

    # Row 2: Epilepsy with mask overlay
    for i, sample in enumerate(epilepsy_samples):
        image = to_display_range(sample[0])  # (H, W)
        mask = sample[1].cpu().numpy()  # (H, W)

        rgb = create_overlay(
            image,
            mask,
            alpha=vis_cfg.alpha,
            color=tuple(vis_cfg.color),
            threshold=vis_cfg.threshold,
        )
        grid[H:, i * W : (i + 1) * W] = rgb

    return grid


def add_labels_to_grid(
    grid: np.ndarray,
    z_bins: list[int],
    figsize: tuple[float, float] = (15, 6),
) -> plt.Figure:
    """Add labels and create matplotlib figure.

    Args:
        grid: Grid image, shape (H*2, W*5, 3).
        z_bins: Z-bin values for column labels.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(grid)
    ax.axis("off")

    # Add row labels
    H = grid.shape[0] // 2
    W = grid.shape[1] // len(z_bins)

    ax.text(
        -10, H // 2, "Control",
        ha="right", va="center", fontsize=12, fontweight="bold"
    )
    ax.text(
        -10, H + H // 2, "Epilepsy",
        ha="right", va="center", fontsize=12, fontweight="bold"
    )

    # Add column labels (z-bins)
    for i, zb in enumerate(z_bins):
        ax.text(
            i * W + W // 2, -10, f"z={zb}",
            ha="center", va="bottom", fontsize=10
        )

    fig.tight_layout()
    return fig


class VisualizationCallback(Callback):
    """Callback for generating visualization grids during training.

    At the end of each validation epoch (or at configured frequency),
    generates a 2x5 grid showing samples at different z-positions
    for control and epilepsy conditions.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the callback.

        Args:
            cfg: Configuration object.
        """
        super().__init__()
        self.cfg = cfg
        self.vis_cfg = cfg.visualization
        self.output_dir = Path(cfg.experiment.output_dir) / "viz"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.z_bins = list(self.vis_cfg.z_bins_to_show)
        self.every_n_epochs = self.vis_cfg.every_n_epochs

        self._sampler: DiffusionSampler | None = None
        logger.info(f"VisualizationCallback initialized, z_bins={self.z_bins}")

    def _get_sampler(
        self,
        pl_module: pl.LightningModule,
    ) -> DiffusionSampler:
        """Get or create the diffusion sampler.

        Args:
            pl_module: Lightning module.

        Returns:
            DiffusionSampler instance.
        """
        if self._sampler is None:
            self._sampler = DiffusionSampler(
                model=pl_module.model,
                scheduler=pl_module.inferer,
                cfg=self.cfg,
                device=pl_module.device,
            )
        return self._sampler

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Generate visualization at end of validation epoch.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        if not self.vis_cfg.enabled:
            return

        current_epoch = trainer.current_epoch

        # Check frequency
        if current_epoch % self.every_n_epochs != 0:
            return

        logger.info(f"Generating visualization for epoch {current_epoch}")

        # Get conditioning tokens
        n_bins = self.cfg.conditioning.z_bins
        control_tokens, lesion_tokens = get_visualization_tokens(
            self.z_bins, n_bins
        )

        # Get sampler
        sampler = self._get_sampler(pl_module)

        # Generate samples
        pl_module.eval()
        samples = []

        with torch.no_grad():
            # Control samples
            for token in control_tokens:
                sample = sampler.sample_single(token)
                samples.append(sample)

            # Epilepsy samples
            for token in lesion_tokens:
                sample = sampler.sample_single(token)
                samples.append(sample)

        # Create grid
        grid = create_visualization_grid(samples, self.z_bins, self.cfg)

        # Save PNG
        if self.vis_cfg.save_png:
            fig = add_labels_to_grid(grid, self.z_bins)
            save_path = self.output_dir / f"epoch_{current_epoch:04d}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved visualization to {save_path}")

        # Log to tensorboard/wandb
        if trainer.logger is not None:
            # Convert to format suitable for logging
            # Grid is in [0, 1], shape (H, W, 3)
            grid_tensor = torch.from_numpy(grid).permute(2, 0, 1)  # (3, H, W)
            trainer.logger.experiment.add_image(
                "samples/grid",
                grid_tensor,
                global_step=current_epoch,
            )


class EMACallback(Callback):
    """Exponential Moving Average callback for model weights.

    Maintains an EMA of model weights for potentially better generation.
    Optional feature - not enabled by default.
    """

    def __init__(
        self,
        decay: float = 0.999,
        update_every: int = 10,
    ) -> None:
        """Initialize EMA callback.

        Args:
            decay: EMA decay rate.
            update_every: Update EMA every N steps.
        """
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self._ema_weights: dict[str, torch.Tensor] | None = None
        self._step = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA weights after training batch.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
            outputs: Batch outputs.
            batch: Input batch.
            batch_idx: Batch index.
        """
        self._step += 1

        if self._step % self.update_every != 0:
            return

        # Initialize EMA weights on first call
        if self._ema_weights is None:
            self._ema_weights = {
                name: param.data.clone()
                for name, param in pl_module.model.named_parameters()
            }
            return

        # Update EMA
        with torch.no_grad():
            for name, param in pl_module.model.named_parameters():
                if name in self._ema_weights:
                    self._ema_weights[name] = (
                        self.decay * self._ema_weights[name]
                        + (1 - self.decay) * param.data
                    )

    def get_ema_weights(self) -> dict[str, torch.Tensor] | None:
        """Get current EMA weights.

        Returns:
            Dictionary of EMA weights or None.
        """
        return self._ema_weights
