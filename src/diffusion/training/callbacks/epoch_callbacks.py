"""Epoch-level callbacks for JS-DDPM training.

Provides visualization callback for generating sample grids
at validation time.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

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
from src.diffusion.utils.zbin_priors import (
    apply_zbin_prior_postprocess,
    get_anatomical_priors_as_input,
    load_zbin_priors,
)

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

        # Z-bin priors for post-processing and anatomical conditioning
        self._zbin_priors: dict[int, NDArray[np.bool_]] | None = None
        pp_cfg = cfg.get("postprocessing", {})
        zbin_cfg = pp_cfg.get("zbin_priors", {})
        self._use_zbin_priors = (
            zbin_cfg.get("enabled", False)
            and "visualization" in zbin_cfg.get("apply_to", [])
        )

        # Anatomical conditioning (input concatenation)
        self._use_anatomical_conditioning = cfg.model.get("anatomical_conditioning", False)

        # Load priors if needed for either post-processing or anatomical conditioning
        if self._use_zbin_priors or self._use_anatomical_conditioning:
            self._load_zbin_priors()

        logger.info(
            f"VisualizationCallback initialized, z_bins={self.z_bins}, "
            f"zbin_priors_postprocess={self._use_zbin_priors}, "
            f"anatomical_conditioning={self._use_anatomical_conditioning}"
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
            for i, token in enumerate(control_tokens):
                # Get anatomical prior if needed
                anatomical_mask = None
                if self._use_anatomical_conditioning and self._zbin_priors is not None:
                    z_bin = self.z_bins[i]
                    anatomical_mask = get_anatomical_priors_as_input(
                        [z_bin],
                        self._zbin_priors,
                        device=pl_module.device,
                    ).squeeze(0)  # Remove batch dim: (1, 1, H, W) -> (1, H, W)

                sample = sampler.sample_single(token, anatomical_mask=anatomical_mask)
                samples.append(sample)

            # Epilepsy samples
            for i, token in enumerate(lesion_tokens):
                # Get anatomical prior if needed
                anatomical_mask = None
                if self._use_anatomical_conditioning and self._zbin_priors is not None:
                    z_bin = self.z_bins[i]
                    anatomical_mask = get_anatomical_priors_as_input(
                        [z_bin],
                        self._zbin_priors,
                        device=pl_module.device,
                    ).squeeze(0)  # Remove batch dim: (1, 1, H, W) -> (1, H, W)

                sample = sampler.sample_single(token, anatomical_mask=anatomical_mask)
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
                    pp_cfg.get("n_first_bins", 0),
                    pp_cfg.get("max_components_for_first_bins", 1),
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


from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple


class EMACallback(Callback):
    """EMA of pl_module.model parameters (and optionally buffers).

    Properties:
      - Updates on optimizer steps via trainer.global_step (correct with grad accumulation).
      - EMA state stored in FP32, optionally on CPU.
      - update_every counts optimizer steps; decay is interpreted as per-step and corrected if update_every>1.
      - Safe swap/restore for validation; nested-safe ema_scope() for visualization/generation.
      - Checkpointable; optional export of EMA weights to checkpoint["ema_state_dict"] for offline sampling.
    """

    def __init__(
        self,
        *,
        decay: float = 0.999,
        update_every: int = 1,
        update_start_step: int = 0,
        store_on_cpu: bool = True,
        use_buffers: bool = True,
        use_for_validation: bool = True,
        export_to_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        if update_every < 1:
            raise ValueError(f"EMA update_every must be >= 1, got {update_every}")
        if update_start_step < 0:
            raise ValueError(f"EMA update_start_step must be >= 0, got {update_start_step}")

        self.decay = float(decay)
        self.update_every = int(update_every)
        self.update_start_step = int(update_start_step)
        self.store_on_cpu = bool(store_on_cpu)
        self.use_buffers = bool(use_buffers)
        self.use_for_validation = bool(use_for_validation)
        self.export_to_checkpoint = bool(export_to_checkpoint)

        self._ema: Optional[Dict[str, torch.Tensor]] = None
        self._backup: Optional[Dict[str, torch.Tensor]] = None
        self._applied: bool = False
        self._last_global_step: int = 0
        self._num_updates: int = 0

    # -----------------
    # Internals
    # -----------------
    def _model(self, pl_module: pl.LightningModule) -> torch.nn.Module:
        # In your codebase the diffusion UNet is stored here.
        return pl_module.model  # type: ignore[attr-defined]

    def _iter_named_tensors(self, model: torch.nn.Module) -> Iterator[Tuple[str, torch.Tensor, bool]]:
        """Yield (name, tensor, is_float) for parameters and (optionally) buffers."""
        for name, p in model.named_parameters():
            if p is not None:
                yield name, p, torch.is_floating_point(p)
        if self.use_buffers:
            for name, b in model.named_buffers():
                if b is not None:
                    yield name, b, torch.is_floating_point(b)

    def _ema_device(self, reference: torch.Tensor) -> torch.device:
        return torch.device("cpu") if self.store_on_cpu else reference.device

    def _effective_decay(self) -> float:
        # decay is interpreted as per *optimizer step*.
        # If updating every N steps, match the per-step smoothing:
        #   ema_{t+N} = (decay^N)*ema_t + (1-decay^N)*w_{t+N}
        return self.decay if self.update_every == 1 else float(self.decay ** self.update_every)

    def _ensure_initialized(self, pl_module: pl.LightningModule) -> None:
        if self._ema is not None:
            return

        model = self._model(pl_module)
        self._ema = {}

        with torch.no_grad():
            for name, t, is_float in self._iter_named_tensors(model):
                dev = self._ema_device(t)
                if is_float:
                    self._ema[name] = t.detach().to(device=dev, dtype=torch.float32).clone()
                else:
                    # Non-float buffers: keep last value (no averaging)
                    self._ema[name] = t.detach().to(device=dev).clone()

        logger.info(
            "Initialized EMA: %d tensors (use_buffers=%s, store_on_cpu=%s)",
            len(self._ema),
            self.use_buffers,
            self.store_on_cpu,
        )

    @torch.no_grad()
    def _update(self, pl_module: pl.LightningModule) -> None:
        self._ensure_initialized(pl_module)
        assert self._ema is not None

        model = self._model(pl_module)
        d = self._effective_decay()
        one_minus_d = 1.0 - d

        for name, t, is_float in self._iter_named_tensors(model):
            if name not in self._ema:
                # Rare: tensor appears mid-run. Initialize it.
                dev = self._ema_device(t)
                self._ema[name] = (
                    t.detach().to(device=dev, dtype=torch.float32).clone()
                    if is_float
                    else t.detach().to(device=dev).clone()
                )
                continue

            ema_t = self._ema[name]
            if is_float:
                src = t.detach().to(dtype=torch.float32)
                if self.store_on_cpu:
                    src = src.to("cpu")
                ema_t.mul_(d).add_(src, alpha=one_minus_d)
            else:
                # Non-float buffers: copy latest
                src = t.detach()
                if self.store_on_cpu:
                    src = src.to("cpu")
                ema_t.copy_(src)

        self._num_updates += 1

    @torch.no_grad()
    def _apply(self, pl_module: pl.LightningModule) -> bool:
        """Swap current model weights to EMA weights. Returns True if this call applied the swap."""
        if self._applied:
            return False

        self._ensure_initialized(pl_module)
        assert self._ema is not None

        model = self._model(pl_module)
        self._backup = {}

        for name, t, _ in self._iter_named_tensors(model):
            self._backup[name] = t.detach().clone()
            ema_t = self._ema.get(name)
            if ema_t is not None:
                t.copy_(ema_t.to(device=t.device, dtype=t.dtype))

        self._applied = True
        return True

    @torch.no_grad()
    def _restore(self, pl_module: pl.LightningModule) -> None:
        if not self._applied or self._backup is None:
            return

        model = self._model(pl_module)
        for name, t, _ in self._iter_named_tensors(model):
            b = self._backup.get(name)
            if b is not None:
                t.copy_(b.to(device=t.device, dtype=t.dtype))

        self._backup = None
        self._applied = False

    @contextmanager
    def ema_scope(self, pl_module: pl.LightningModule) -> Iterator[None]:
        """Temporarily swap weights to EMA weights (nested-safe)."""
        applied_now = self._apply(pl_module)
        try:
            yield
        finally:
            if applied_now:
                self._restore(pl_module)

    def get_ema_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._ema

    # -----------------
    # Lightning hooks
    # -----------------
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if stage == "fit":
            self._ensure_initialized(pl_module)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # trainer.global_step increments after each optimizer step (handles accumulate_grad_batches).
        gs = int(trainer.global_step)
        if gs <= 0 or gs == self._last_global_step:
            return
        self._last_global_step = gs

        step_idx = gs - 1  # zero-based
        if step_idx < self.update_start_step:
            return
        if step_idx % self.update_every != 0:
            return

        self._update(pl_module)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.use_for_validation:
            self._apply(pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.use_for_validation:
            self._restore(pl_module)

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        # Never leave the model swapped on failure.
        self._restore(pl_module)

    # -----------------
    # Checkpointing
    # -----------------
    def state_dict(self) -> Dict[str, Any]:
        # Storing EMA is required to resume *true* EMA continuation.
        return {
            "ema": self._ema,
            "last_global_step": self._last_global_step,
            "num_updates": self._num_updates,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._ema = state_dict.get("ema", None)
        self._last_global_step = int(state_dict.get("last_global_step", 0))
        self._num_updates = int(state_dict.get("num_updates", 0))

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        if self.export_to_checkpoint and self._ema is not None:
            checkpoint["ema_state_dict"] = self._ema
            checkpoint["ema_meta"] = {
                "decay": self.decay,
                "update_every": self.update_every,
                "update_start_step": self.update_start_step,
                "store_on_cpu": self.store_on_cpu,
                "use_buffers": self.use_buffers,
                "num_updates": self.num_updates,
            }

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        ema_sd = checkpoint.get("ema_state_dict", None)
        if isinstance(ema_sd, dict):
            self._ema = ema_sd
