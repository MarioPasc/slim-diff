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
from numpy.typing import NDArray
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

from src.diffusion.model.components.conditioning import get_visualization_tokens
from src.diffusion.model.embeddings.zpos import quantize_z
from src.diffusion.model.factory import DiffusionSampler
from src.diffusion.utils.zbin_priors import apply_zbin_prior_postprocess, load_zbin_priors

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

        # Compute valid z_bins from z_range to prevent generating
        # slices outside the training distribution
        self.z_bins = self._compute_valid_z_bins()
        self.every_n_epochs = self.vis_cfg.every_n_epochs

        self._sampler: DiffusionSampler | None = None

        # Z-bin prior post-processing
        self._zbin_priors: dict[int, NDArray[np.bool_]] | None = None
        pp_cfg = cfg.get("postprocessing", {})
        zbin_cfg = pp_cfg.get("zbin_priors", {})
        self._use_zbin_priors = (
            zbin_cfg.get("enabled", False)
            and "visualization" in zbin_cfg.get("apply_to", [])
        )

        if self._use_zbin_priors:
            self._load_zbin_priors()

        logger.info(
            f"VisualizationCallback initialized, z_bins={self.z_bins}, "
            f"zbin_priors={self._use_zbin_priors}"
        )

    def _compute_valid_z_bins(self) -> list[int]:
        """Compute valid z_bins from z_range to match training distribution.

        This ensures visualization only requests slices the model has been
        trained on, preventing extrapolation to unseen z-positions.

        Returns:
            List of valid z_bin indices for visualization.
        """
        # Get configuration
        z_range = self.cfg.data.slice_sampling.z_range
        min_z, max_z = z_range
        n_bins = self.cfg.conditioning.z_bins

        # With LOCAL binning, all bins from 0 to n_bins-1 are valid
        # since we bin within the z_range
        valid_bins = set()
        for z_idx in range(min_z, max_z + 1):
            z_bin = quantize_z(z_idx, tuple(z_range), n_bins)
            valid_bins.add(z_bin)

        valid_bins_sorted = sorted(list(valid_bins))
        logger.info(
            f"Computed {len(valid_bins_sorted)} valid z_bins from "
            f"z_range=[{min_z}, {max_z}]: {valid_bins_sorted} (LOCAL binning)"
        )

        # Verify all bins are used (sanity check for local binning)
        if len(valid_bins_sorted) != n_bins:
            logger.warning(
                f"Expected all {n_bins} bins to be used with LOCAL binning, "
                f"but got {len(valid_bins_sorted)} bins. This may indicate an issue."
            )

        # Get requested z_bins from config
        requested_bins = self.vis_cfg.get("z_bins_to_show", None)

        # If no bins specified or empty, auto-select evenly-spaced bins
        if requested_bins is None or len(requested_bins) == 0:
            n_to_show = 5  # Default to 5 bins
            selected_bins = self._select_evenly_spaced_bins(
                valid_bins_sorted, n_to_show
            )
            logger.info(
                f"No z_bins_to_show specified. Auto-selected {n_to_show} "
                f"evenly-spaced bins: {selected_bins}"
            )
            return selected_bins

        # Filter requested bins to only include valid ones
        filtered_bins = [zb for zb in requested_bins if zb in valid_bins]

        # Warn if some bins were filtered out
        if len(filtered_bins) != len(requested_bins):
            invalid_bins = [zb for zb in requested_bins if zb not in valid_bins]
            logger.warning(
                f"Some z_bins_to_show are outside training range and will be skipped. "
                f"Requested: {requested_bins}, Invalid: {invalid_bins}, "
                f"Using: {filtered_bins}. "
                f"Training data covers z_range=[{min_z}, {max_z}] → "
                f"z_bins={valid_bins_sorted}"
            )

        # If all bins were filtered out, auto-select
        if len(filtered_bins) == 0:
            n_to_show = min(5, len(valid_bins_sorted))
            filtered_bins = self._select_evenly_spaced_bins(
                valid_bins_sorted, n_to_show
            )
            logger.warning(
                f"All requested z_bins_to_show were invalid! "
                f"Auto-selected {n_to_show} bins from valid range: {filtered_bins}"
            )

        return filtered_bins

    def _select_evenly_spaced_bins(
        self, valid_bins: list[int], n_to_show: int
    ) -> list[int]:
        """Select evenly-spaced bins from valid range.

        Args:
            valid_bins: Sorted list of valid z_bins.
            n_to_show: Number of bins to select.

        Returns:
            List of evenly-spaced z_bin indices.
        """
        if len(valid_bins) == 0:
            logger.error("No valid z_bins available! Using z_bin=0 as fallback.")
            return [0]

        if n_to_show >= len(valid_bins):
            return valid_bins

        # Select evenly-spaced indices
        indices = [int(i * (len(valid_bins) - 1) / (n_to_show - 1)) for i in range(n_to_show)]
        return [valid_bins[i] for i in indices]

    def _load_zbin_priors(self) -> None:
        """Load z-bin priors from cache for post-processing."""
        pp_cfg = self.cfg.postprocessing.zbin_priors
        cache_dir = Path(self.cfg.data.cache_dir)
        z_bins = self.cfg.conditioning.z_bins

        try:
            self._zbin_priors = load_zbin_priors(
                cache_dir, pp_cfg.priors_filename, z_bins
            )
            logger.info(f"VisualizationCallback: Loaded z-bin priors for {len(self._zbin_priors)} bins")
        except Exception as e:
            logger.warning(f"Failed to load z-bin priors: {e}. Post-processing disabled.")
            self._use_zbin_priors = False
            self._zbin_priors = None

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

        # Apply z-bin prior post-processing (if enabled)
        if self._use_zbin_priors and self._zbin_priors is not None:
            pp_cfg = self.cfg.postprocessing.zbin_priors
            cleaned_samples = []
            for i, sample in enumerate(samples):
                # Determine z_bin for this sample
                # First half are control, second half are lesion
                z_bin = self.z_bins[i % len(self.z_bins)]

                img = sample[0].cpu().numpy()
                mask = sample[1].cpu().numpy()

                img_clean, mask_clean = apply_zbin_prior_postprocess(
                    img, mask, z_bin, self._zbin_priors,
                    pp_cfg.gaussian_sigma_px,
                    pp_cfg.min_component_px,
                    pp_cfg.fallback,
                )

                # Convert back to tensor
                cleaned_sample = torch.stack([
                    torch.from_numpy(img_clean).to(sample.device),
                    torch.from_numpy(mask_clean).to(sample.device),
                ])
                cleaned_samples.append(cleaned_sample)
            samples = cleaned_samples

        # Create grid
        grid = create_visualization_grid(samples, self.z_bins, self.cfg)

        # Save PNG
        if self.vis_cfg.save_png:
            fig = add_labels_to_grid(grid, self.z_bins)
            save_path = self.output_dir / f"epoch_{current_epoch:04d}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved visualization to {save_path}")

        # Log to wandb
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            # WandB expects images in HWC format as numpy arrays
            import wandb
            trainer.logger.experiment.log({
                "samples/grid": wandb.Image(grid, caption=f"Epoch {current_epoch}"),
                "epoch": current_epoch,
            })


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
