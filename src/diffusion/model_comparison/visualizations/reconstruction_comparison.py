"""Lesion reconstruction comparison visualization (MANDATORY).

This module generates a grid comparison of lesion reconstruction quality
across multiple trained models.

Layout:
- Rows: Representative z-bins with lesion samples
- First column: Original image with lesion overlay
- Subsequent columns: Model reconstructions (2 sub-columns each)
  - Reconstructed image with lesion boundary
  - Residual map with lesion boundary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from scipy import ndimage

from src.diffusion.data.dataset import SliceDataset
from src.diffusion.model.components.anatomical_encoder import AnatomicalPriorEncoder
from src.diffusion.model.factory import build_scheduler, get_alpha_bar, predict_x0
from src.diffusion.scripts.analyze_lesion_reconstruction_error import (
    add_noise_at_timestep,
    build_model_standalone,
    ensure_config_defaults,
    load_model_from_checkpoint,
    reconstruct_from_noisy,
)
from src.diffusion.utils.zbin_priors import (
    get_anatomical_priors_as_input,
    load_zbin_priors,
)

from ..utils import (
    PLOT_SETTINGS,
    create_boundary_overlay,
    find_best_checkpoint,
    get_model_color,
    to_display_range,
)
from .base import BaseVisualization

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionSample:
    """Data for a single reconstruction sample."""

    subject_id: str
    z_index: int
    z_bin: int
    lesion_pixels: int
    original_image: NDArray[np.float32]  # (H, W) in [-1, 1]
    original_mask: NDArray[np.float32]  # (H, W) in [-1, 1]
    token: int


@dataclass
class ModelReconstruction:
    """Reconstruction results for a single model."""

    model_name: str
    reconstructed_image: NDArray[np.float32]  # (H, W) in [-1, 1]
    reconstructed_mask: NDArray[np.float32]  # (H, W) in [-1, 1]
    residual_image: NDArray[np.float32]  # (H, W) raw difference
    mse_lesion: float
    mse_nonlesion: float


class ReconstructionComparisonVisualization(BaseVisualization):
    """Generate lesion reconstruction comparison across models.

    Creates a grid visualization:
    - Rows = z-bins with representative lesion samples
    - First column = original image with lesion overlay
    - Subsequent columns = model reconstructions (2 sub-cols each)
    """

    name = "reconstruction_comparison"

    def __init__(
        self,
        cfg: DictConfig,
        output_dir: Path,
        model_names: list[str],
    ) -> None:
        """Initialize with configuration.

        Args:
            cfg: Configuration object.
            output_dir: Output directory for plots.
            model_names: List of model names being compared.
        """
        super().__init__(cfg, output_dir, model_names)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model paths for checkpoint loading
        self.model_paths: dict[str, Path] = {}

        # Loaded models cache: (model, scheduler, cfg, actual_in_channels, anatomical_encoder)
        self._loaded_models: dict[str, tuple[torch.nn.Module, torch.nn.Module, DictConfig, int, torch.nn.Module | None]] = {}

    def set_model_paths(self, model_paths: dict[str, Path]) -> None:
        """Set model directory paths for checkpoint loading.

        Args:
            model_paths: Dictionary mapping model_name -> model_dir.
        """
        self.model_paths = model_paths

    def generate(self, data_loader: Any) -> list[Path]:
        """Generate reconstruction comparison visualization.

        Args:
            data_loader: ComparisonDataLoader instance (not directly used here).

        Returns:
            List of saved plot paths.
        """
        saved_paths = []

        # Load cache configuration
        cache_config_path = Path(self.viz_cfg.cache_config_path)
        if not cache_config_path.exists():
            logger.error(f"Cache config not found: {cache_config_path}")
            return saved_paths

        cache_cfg = OmegaConf.load(cache_config_path)
        cache_dir = Path(cache_cfg.cache_dir)

        if not cache_dir.exists():
            logger.error(f"Cache directory not found: {cache_dir}")
            return saved_paths

        # Select representative samples
        samples = self._select_representative_samples(cache_dir)
        if not samples:
            logger.error("No suitable samples found for reconstruction comparison")
            return saved_paths

        logger.info(f"Selected {len(samples)} samples for reconstruction comparison")

        # Reconstruct samples with each model
        all_reconstructions: dict[str, list[ModelReconstruction]] = {}

        for model_name in self.model_names:
            if model_name not in self.model_paths:
                logger.warning(f"No path for model {model_name}, skipping")
                continue

            model_dir = self.model_paths[model_name]
            reconstructions = self._reconstruct_samples(
                model_name=model_name,
                model_dir=model_dir,
                samples=samples,
            )
            if reconstructions:
                all_reconstructions[model_name] = reconstructions

        if not all_reconstructions:
            logger.error("No reconstructions generated for any model")
            return saved_paths

        # Create comparison grid
        fig = self._create_comparison_grid(samples, all_reconstructions)
        saved_paths.extend(self._save_figure(fig, "reconstruction_comparison"))
        self._close_figure(fig)

        # Save reconstruction data
        if self.cfg.output.save_data:
            data_path = self._save_reconstruction_data(samples, all_reconstructions)
            saved_paths.append(data_path)

        logger.info(f"Generated reconstruction comparison with {len(saved_paths)} outputs")
        return saved_paths

    def _select_representative_samples(
        self,
        cache_dir: Path,
    ) -> list[ReconstructionSample]:
        """Select representative samples with lesions for comparison.

        Criteria:
        - Lesion size between min_lesion_pixels and max_lesion_pixels
        - Diverse z-bins (n_zbins total)
        - One sample per z-bin

        Args:
            cache_dir: Path to slice cache directory.

        Returns:
            List of ReconstructionSample objects.
        """
        n_zbins = self.viz_cfg.n_zbins
        min_pixels = self.viz_cfg.min_lesion_pixels
        max_pixels = self.viz_cfg.max_lesion_pixels
        seed = self.viz_cfg.seed

        # Load test dataset
        dataset = SliceDataset(cache_dir, split="test")
        logger.info(f"Loaded test dataset with {len(dataset)} samples")

        # Group samples by z-bin
        zbin_samples: dict[int, list[tuple[int, dict]]] = {}
        for idx in range(len(dataset)):
            sample = dataset[idx]

            # Check if sample has lesion
            mask = sample["mask"].numpy()
            lesion_pixels = int((mask > 0).sum())

            if min_pixels <= lesion_pixels <= max_pixels:
                z_bin = int(sample["z_bin"])
                if z_bin not in zbin_samples:
                    zbin_samples[z_bin] = []
                zbin_samples[z_bin].append((idx, sample, lesion_pixels))

        # Select diverse z-bins
        available_zbins = sorted(zbin_samples.keys())
        if len(available_zbins) < n_zbins:
            logger.warning(
                f"Only {len(available_zbins)} z-bins have suitable samples, "
                f"requested {n_zbins}"
            )
            selected_zbins = available_zbins
        else:
            # Select evenly spaced z-bins
            step = len(available_zbins) // n_zbins
            selected_zbins = [available_zbins[i * step] for i in range(n_zbins)]

        # Select one sample per z-bin
        rng = np.random.default_rng(seed)
        selected_samples = []

        for z_bin in selected_zbins:
            candidates = zbin_samples[z_bin]
            # Random selection
            idx = rng.integers(0, len(candidates))
            sample_idx, sample, lesion_pixels = candidates[idx]

            selected_samples.append(
                ReconstructionSample(
                    subject_id=sample.get("subject_id", f"sample_{sample_idx}"),
                    z_index=int(sample.get("z_index", 0)),
                    z_bin=z_bin,
                    lesion_pixels=lesion_pixels,
                    original_image=sample["image"].numpy()[0],  # (H, W)
                    original_mask=sample["mask"].numpy()[0],  # (H, W)
                    token=int(sample["token"]),
                )
            )

        return selected_samples

    def _reconstruct_samples(
        self,
        model_name: str,
        model_dir: Path,
        samples: list[ReconstructionSample],
    ) -> list[ModelReconstruction]:
        """Reconstruct samples using a specific model.

        Args:
            model_name: Model display name.
            model_dir: Path to model directory.
            samples: Samples to reconstruct.

        Returns:
            List of ModelReconstruction results.
        """
        try:
            # Load model if not cached
            if model_name not in self._loaded_models:
                model, scheduler, cfg, actual_in_channels = self._load_model(model_dir)
                # Load anatomical encoder if needed for cross_attention
                anatomical_encoder = self._load_anatomical_encoder(model_dir, cfg)
                self._loaded_models[model_name] = (model, scheduler, cfg, actual_in_channels, anatomical_encoder)
            else:
                model, scheduler, cfg, actual_in_channels, anatomical_encoder = self._loaded_models[model_name]

            model.eval()
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return []

        timestep = self.viz_cfg.timestep
        reconstructions = []

        # Check for anatomical conditioning
        use_anatomical = cfg.model.get("anatomical_conditioning", False)
        anatomical_method = cfg.model.get("anatomical_conditioning_method", "concat")
        zbin_priors = None

        if use_anatomical:
            # Try to load z-bin priors
            cache_config_path = Path(self.viz_cfg.cache_config_path)
            cache_cfg = OmegaConf.load(cache_config_path)
            cache_dir = Path(cache_cfg.cache_dir)
            priors_filename = "zbin_priors_brain_roi.npz"
            priors_path = cache_dir / priors_filename
            z_bins = cache_cfg.z_bins  # Get z_bins from cache config

            if priors_path.exists():
                zbin_priors = load_zbin_priors(cache_dir, priors_filename, z_bins)
                logger.info(f"Loaded z-bin priors from {priors_path}")
            else:
                logger.warning(f"Z-bin priors not found at {priors_path}")

        # Determine if self-conditioning channels are needed
        # Base channels: 2 (image + mask), optionally +1 for anatomical concat
        base_channels = 2
        if use_anatomical and anatomical_method == "concat":
            base_channels += 1

        needs_self_cond = actual_in_channels > base_channels
        if needs_self_cond:
            num_self_cond_channels = actual_in_channels - base_channels
            logger.info(f"Model expects {actual_in_channels} channels, adding {num_self_cond_channels} self-conditioning dummy channels")

        for sample in samples:
            try:
                # Prepare input tensor
                image = torch.tensor(sample.original_image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                mask = torch.tensor(sample.original_mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                x0 = torch.cat([image, mask], dim=1).to(self.device)  # (1, 2, H, W)
                tokens = torch.tensor([sample.token], device=self.device)

                # Add noise
                x_t, noise = add_noise_at_timestep(x0, scheduler, timestep, self.device)

                # Prepare anatomical priors if needed
                anatomical_priors = None
                if use_anatomical and zbin_priors is not None:
                    # get_anatomical_priors_as_input(z_bins_batch, priors, device)
                    anatomical_priors = get_anatomical_priors_as_input(
                        z_bins_batch=torch.tensor([sample.z_bin], device=self.device),
                        priors=zbin_priors,
                        device=self.device,
                    )

                # Add self-conditioning dummy channels if needed
                # Self-conditioning models expect additional zero channels during inference
                if needs_self_cond:
                    B, C, H, W = x_t.shape
                    self_cond_dummy = torch.zeros(
                        B, num_self_cond_channels, H, W, device=self.device
                    )
                    x_t = torch.cat([x_t, self_cond_dummy], dim=1)

                # Reconstruct
                x0_hat = reconstruct_from_noisy(
                    model=model,
                    x_t=x_t,
                    timestep=timestep,
                    tokens=tokens,
                    scheduler=scheduler,
                    anatomical_priors=anatomical_priors,
                    anatomical_encoder=anatomical_encoder,
                    anatomical_method=anatomical_method,
                    device=self.device,
                )

                # Extract results
                reconstructed_image = x0_hat[0, 0].cpu().numpy()
                reconstructed_mask = x0_hat[0, 1].cpu().numpy()
                residual_image = sample.original_image - reconstructed_image

                # Compute MSE in lesion vs non-lesion regions
                lesion_mask = sample.original_mask > 0
                nonlesion_mask = ~lesion_mask

                sq_error = (sample.original_image - reconstructed_image) ** 2
                mse_lesion = float(sq_error[lesion_mask].mean()) if lesion_mask.any() else 0.0
                mse_nonlesion = float(sq_error[nonlesion_mask].mean()) if nonlesion_mask.any() else 0.0

                reconstructions.append(
                    ModelReconstruction(
                        model_name=model_name,
                        reconstructed_image=reconstructed_image,
                        reconstructed_mask=reconstructed_mask,
                        residual_image=residual_image,
                        mse_lesion=mse_lesion,
                        mse_nonlesion=mse_nonlesion,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to reconstruct sample {sample.subject_id}: {e}")
                continue

        return reconstructions

    def _load_anatomical_encoder(
        self,
        model_dir: Path,
        cfg: DictConfig,
    ) -> torch.nn.Module | None:
        """Load anatomical encoder from checkpoint if using cross_attention method.

        Args:
            model_dir: Path to model directory.
            cfg: Model configuration.

        Returns:
            AnatomicalPriorEncoder if cross_attention is used, None otherwise.
        """
        use_anatomical = cfg.model.get("anatomical_conditioning", False)
        anatomical_method = cfg.model.get("anatomical_conditioning_method", "concat")

        if not use_anatomical or anatomical_method != "cross_attention":
            return None

        try:
            # Get encoder configuration
            encoder_cfg = cfg.model.get("anatomical_encoder", {})
            cross_attention_dim = cfg.model.get("cross_attention_dim", 256)

            # Build the encoder
            anatomical_encoder = AnatomicalPriorEncoder(
                embed_dim=cross_attention_dim,
                hidden_dims=tuple(encoder_cfg.get("hidden_dims", [32, 64, 128])),
                downsample_factor=encoder_cfg.get("downsample_factor", 8),
                input_size=(128, 128),
                positional_encoding=encoder_cfg.get("positional_encoding", "sinusoidal"),
                norm_num_groups=encoder_cfg.get("norm_num_groups", 8),
            )
            anatomical_encoder.to(self.device)
            anatomical_encoder.eval()

            # Find and load checkpoint weights
            checkpoint_path = find_best_checkpoint(model_dir)
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if "state_dict" in ckpt:
                encoder_state = {}
                for k, v in ckpt["state_dict"].items():
                    if k.startswith("_anatomical_encoder."):
                        key_without_prefix = k[21:]  # Remove "_anatomical_encoder." prefix
                        # Fix corrupted key names (missing first character)
                        if key_without_prefix.startswith("ackbone."):
                            key_without_prefix = "b" + key_without_prefix
                        elif key_without_prefix.startswith("roj."):
                            key_without_prefix = "p" + key_without_prefix
                        elif key_without_prefix.startswith("os_encoding."):
                            key_without_prefix = "p" + key_without_prefix
                        encoder_state[key_without_prefix] = v

                if encoder_state:
                    missing, unexpected = anatomical_encoder.load_state_dict(encoder_state, strict=False)
                    if not missing and not unexpected:
                        logger.info(f"Loaded anatomical encoder weights from checkpoint")
                    else:
                        logger.warning(
                            f"Anatomical encoder loaded with missing={len(missing)}, "
                            f"unexpected={len(unexpected)} keys"
                        )
                else:
                    logger.warning("No anatomical encoder weights found in checkpoint")
                    return None

            return anatomical_encoder

        except Exception as e:
            logger.error(f"Failed to load anatomical encoder: {e}")
            return None

    def _load_model(
        self,
        model_dir: Path,
    ) -> tuple[torch.nn.Module, torch.nn.Module, DictConfig, int]:
        """Load model from checkpoint.

        Args:
            model_dir: Path to model directory.

        Returns:
            Tuple of (model, scheduler, config, actual_in_channels).
        """
        # Find config
        config_path = model_dir / "config.yaml"
        if not config_path.exists():
            # Try alternative names
            yaml_files = list(model_dir.glob("*.yaml"))
            if yaml_files:
                config_path = yaml_files[0]
            else:
                raise FileNotFoundError(f"No config found in {model_dir}")

        cfg = OmegaConf.load(config_path)
        cfg = ensure_config_defaults(cfg)

        # Find checkpoint
        checkpoint_path = find_best_checkpoint(model_dir)
        use_ema = self.viz_cfg.use_ema

        # load_model_from_checkpoint returns (model, scheduler, ema_loaded, actual_in_channels)
        model, scheduler, ema_loaded, actual_in_channels = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            device=self.device,
            use_ema=use_ema,
        )

        logger.info(f"Loaded model from {checkpoint_path} (EMA: {ema_loaded})")
        return model, scheduler, cfg, actual_in_channels

    def _create_comparison_grid(
        self,
        samples: list[ReconstructionSample],
        all_reconstructions: dict[str, list[ModelReconstruction]],
    ) -> plt.Figure:
        """Create the comparison grid figure.

        Layout:
        - First column: Original image with lesion overlay
        - Subsequent columns: Model reconstructions (2 sub-cols each)

        Args:
            samples: Original samples.
            all_reconstructions: Dictionary mapping model_name -> reconstructions.

        Returns:
            Matplotlib figure.
        """
        n_samples = len(samples)
        n_models = len(all_reconstructions)

        # Calculate figure dimensions
        # First col: original, then 2 cols per model (recon + residual)
        n_cols = 1 + 2 * n_models
        n_rows = n_samples

        figsize = self._get_figsize((3 * n_cols, 3 * n_rows))
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.15)

        # Get residual colormap settings
        residual_cmap = self.viz_cfg.residual_cmap
        residual_vmin = self.viz_cfg.residual_vmin
        residual_vmax = self.viz_cfg.residual_vmax
        overlay_color = tuple(self.viz_cfg.overlay_color)

        model_names_ordered = list(all_reconstructions.keys())

        for row_idx, sample in enumerate(samples):
            # Column 0: Original with lesion overlay
            ax = fig.add_subplot(gs[row_idx, 0])
            orig_display = to_display_range(sample.original_image)
            overlay = create_boundary_overlay(
                orig_display,
                sample.original_mask,
                color=overlay_color,
                threshold=0.0,
                linewidth=2,
            )
            ax.imshow(overlay)
            ax.set_title(f"Original\nz={sample.z_bin}, {sample.lesion_pixels}px", fontsize=9)
            ax.axis("off")

            # Add row label on left side
            if row_idx == n_samples // 2:
                ax.set_ylabel("Z-bins", fontsize=12, labelpad=20)

            # Model columns
            for model_idx, model_name in enumerate(model_names_ordered):
                reconstructions = all_reconstructions[model_name]
                if row_idx >= len(reconstructions):
                    continue

                recon = reconstructions[row_idx]
                col_base = 1 + 2 * model_idx

                # Reconstructed image with boundary
                ax_recon = fig.add_subplot(gs[row_idx, col_base])
                recon_display = to_display_range(recon.reconstructed_image)
                recon_overlay = create_boundary_overlay(
                    recon_display,
                    sample.original_mask,
                    color=overlay_color,
                    threshold=0.0,
                    linewidth=2,
                )
                ax_recon.imshow(recon_overlay)
                if row_idx == 0:
                    ax_recon.set_title(f"{model_name}\nRecon", fontsize=9)
                ax_recon.axis("off")

                # Residual map with boundary
                ax_resid = fig.add_subplot(gs[row_idx, col_base + 1])

                # Create residual visualization
                residual_vis = self._create_residual_visualization(
                    recon.residual_image,
                    sample.original_mask,
                    cmap=residual_cmap,
                    vmin=residual_vmin,
                    vmax=residual_vmax,
                    overlay_color=overlay_color,
                )
                ax_resid.imshow(residual_vis)
                if row_idx == 0:
                    ax_resid.set_title(f"Residual\nMSE_L={recon.mse_lesion:.4f}", fontsize=9)
                else:
                    ax_resid.set_title(f"MSE_L={recon.mse_lesion:.4f}", fontsize=8)
                ax_resid.axis("off")

        fig.suptitle(
            f"Lesion Reconstruction Comparison (t={self.viz_cfg.timestep})",
            fontsize=PLOT_SETTINGS["title_fontsize"],
            y=1.02,
        )

        return fig

    def _create_residual_visualization(
        self,
        residual: NDArray[np.float32],
        mask: NDArray[np.float32],
        cmap: str = "RdBu_r",
        vmin: float = -0.5,
        vmax: float = 0.5,
        overlay_color: tuple[int, int, int] = (0, 255, 0),
    ) -> NDArray[np.float32]:
        """Create residual map with lesion boundary overlay.

        Args:
            residual: Residual image (original - reconstructed).
            mask: Lesion mask in [-1, 1].
            cmap: Colormap for residual.
            vmin: Minimum value for colormap.
            vmax: Maximum value for colormap.
            overlay_color: RGB color for boundary.

        Returns:
            RGB image with residual and boundary.
        """
        # Normalize residual to [0, 1] for colormap
        residual_norm = np.clip((residual - vmin) / (vmax - vmin), 0, 1)

        # Apply colormap
        cmap_obj = plt.get_cmap(cmap)
        residual_rgb = cmap_obj(residual_norm)[:, :, :3].astype(np.float32)

        # Add boundary overlay
        boundary = self._compute_lesion_boundary(mask)
        color_norm = np.array(overlay_color) / 255.0

        for c in range(3):
            residual_rgb[:, :, c] = np.where(
                boundary,
                color_norm[c],
                residual_rgb[:, :, c],
            )

        return residual_rgb

    def _compute_lesion_boundary(
        self,
        mask: NDArray[np.float32],
        threshold: float = 0.0,
        linewidth: int = 2,
    ) -> NDArray[np.bool_]:
        """Extract lesion boundary for overlay.

        Args:
            mask: Lesion mask in [-1, 1].
            threshold: Binarization threshold.
            linewidth: Width of boundary line.

        Returns:
            Boolean mask of lesion boundary.
        """
        binary_mask = mask > threshold
        eroded = ndimage.binary_erosion(binary_mask, iterations=linewidth)
        boundary = binary_mask & ~eroded
        return boundary

    def _save_reconstruction_data(
        self,
        samples: list[ReconstructionSample],
        all_reconstructions: dict[str, list[ModelReconstruction]],
    ) -> Path:
        """Save reconstruction data to NPZ file.

        Args:
            samples: Original samples.
            all_reconstructions: Model reconstructions.

        Returns:
            Path to saved file.
        """
        data_dir = self.output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        filepath = data_dir / "reconstruction_results.npz"

        # Prepare data
        save_data = {
            "z_bins": np.array([s.z_bin for s in samples]),
            "lesion_pixels": np.array([s.lesion_pixels for s in samples]),
            "original_images": np.stack([s.original_image for s in samples]),
            "original_masks": np.stack([s.original_mask for s in samples]),
            "model_names": np.array(list(all_reconstructions.keys())),
        }

        for model_name, recons in all_reconstructions.items():
            prefix = model_name.replace(" ", "_").lower()
            save_data[f"{prefix}_reconstructed_images"] = np.stack([r.reconstructed_image for r in recons])
            save_data[f"{prefix}_residuals"] = np.stack([r.residual_image for r in recons])
            save_data[f"{prefix}_mse_lesion"] = np.array([r.mse_lesion for r in recons])
            save_data[f"{prefix}_mse_nonlesion"] = np.array([r.mse_nonlesion for r in recons])

        np.savez_compressed(filepath, **save_data)
        logger.info(f"Saved reconstruction data to {filepath}")
        return filepath
