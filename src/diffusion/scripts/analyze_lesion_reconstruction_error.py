#!/usr/bin/env python3
"""Analyze whether DDPM model has higher reconstruction error on lesion pixels.

This script investigates whether the diffusion model struggles more to reconstruct
lesion pixels compared to non-lesion pixels. The hypothesis is that lesion pixels
may have higher L2 reconstruction error due to their rarity and heterogeneity.

Methodology:
1. Load trained DDPM checkpoint and real data from slice cache
2. For each sample with lesions:
   - Add noise at a specified timestep t: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
   - Predict noise using the model: eps_pred = model(x_t, t, token)
   - Reconstruct x_0: x_0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
   - Compute pixel-wise squared error: error = (x_0 - x_0_hat)^2
3. Compare error distributions: lesion pixels vs non-lesion pixels
4. Perform statistical analysis (paired t-test, effect size)
5. Generate visualizations

Output:
- NPZ file with raw results (errors, masks, etc.)
- Statistical report (significance, effect size)
- Visualization plots (distributions, example reconstructions)

Usage:
python -m src.diffusion.scripts.analyze_lesion_reconstruction_error \
    --config slurm/lesion_replication_fix_experiments/jsddpm_complete/jsddpm_complete.yaml \
    --checkpoint /media/mpascual/Sandisk2TB/research/jsddpm/results/epilepsy/updated_loss/anatomical_conditioning/jsddpm_complete/checkpoints/jsddpm-epoch=0422-val_loss=0.0000.ckpt \
    --cache-dir /media/mpascual/Sandisk2TB/research/jsddpm/data/epilepsy/slice_cache \
    --output-dir /media/mpascual/Sandisk2TB/research/jsddpm/results/epilepsy/updated_loss/anatomical_conditioning/jsddpm_complete/lesion_error_analysis \
    --timestep 100 \
    --num-samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from tqdm import tqdm

from src.diffusion.data.dataset import SliceDataset, collate_fn
from src.diffusion.model.factory import (
    build_scheduler,
    get_alpha_bar,
    predict_x0,
)
from src.diffusion.utils.logging import setup_logger
from src.diffusion.utils.seeding import seed_everything
from src.diffusion.utils.zbin_priors import (
    get_anatomical_priors_as_input,
    load_zbin_priors,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SampleResult:
    """Results for a single sample analysis."""

    sample_idx: int
    subject_id: str
    z_index: int
    z_bin: int

    # Original data
    original_image: NDArray[np.float32]  # (H, W)
    original_mask: NDArray[np.float32]  # (H, W)

    # Reconstructed data
    reconstructed_image: NDArray[np.float32]  # (H, W)
    reconstructed_mask: NDArray[np.float32]  # (H, W)

    # Error maps
    image_error: NDArray[np.float32]  # (H, W) - squared error
    mask_error: NDArray[np.float32]  # (H, W) - squared error

    # Summary statistics
    mean_error_lesion: float
    mean_error_nonlesion: float
    n_lesion_pixels: int
    n_nonlesion_pixels: int


@dataclass
class AnalysisResults:
    """Aggregated analysis results across all samples."""

    # Per-sample summaries
    samples: list[SampleResult] = field(default_factory=list)

    # Configuration
    timestep: int = 0
    num_samples: int = 0

    # Aggregated statistics
    all_mean_errors_lesion: list[float] = field(default_factory=list)
    all_mean_errors_nonlesion: list[float] = field(default_factory=list)

    def add_sample(self, result: SampleResult) -> None:
        """Add a sample result."""
        self.samples.append(result)
        self.all_mean_errors_lesion.append(result.mean_error_lesion)
        self.all_mean_errors_nonlesion.append(result.mean_error_nonlesion)


# =============================================================================
# Model Loading
# =============================================================================


def ensure_config_defaults(cfg: DictConfig) -> DictConfig:
    """Ensure config has all required keys with sensible defaults.

    This handles backward compatibility with older configs that may be
    missing newer configuration keys.

    Args:
        cfg: Configuration object.

    Returns:
        Updated configuration with defaults filled in.
    """
    # Make a mutable copy
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Ensure training.self_conditioning exists
    if "training" not in cfg_dict:
        cfg_dict["training"] = {}
    if "self_conditioning" not in cfg_dict["training"]:
        cfg_dict["training"]["self_conditioning"] = {"enabled": False, "probability": 0.5}

    # Ensure lesion_quality_metrics exists
    if "lesion_quality_metrics" not in cfg_dict:
        cfg_dict["lesion_quality_metrics"] = {
            "min_lesion_size_px": 5,
            "intensity_percentile_bg": 50.0,
        }

    # Ensure postprocessing exists
    if "postprocessing" not in cfg_dict:
        cfg_dict["postprocessing"] = {"zbin_priors": {"enabled": False}}

    return OmegaConf.create(cfg_dict)


def build_model_standalone(cfg: DictConfig) -> torch.nn.Module:
    """Build model from config without using full Lightning module.

    This is a simplified version of build_model that handles missing config
    keys gracefully for loading older checkpoints.

    Args:
        cfg: Configuration object.

    Returns:
        DiffusionModelUNet instance.
    """
    from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet

    from src.diffusion.model.embeddings import ConditionalEmbeddingWithSinusoidal

    model_cfg = cfg.model
    cond_cfg = cfg.conditioning
    z_bins = cond_cfg.z_bins

    # Check for conditioning options (with defaults for older configs)
    use_anatomical_conditioning = model_cfg.get("anatomical_conditioning", False)
    anatomical_method = model_cfg.get("anatomical_conditioning_method", "concat")
    use_self_conditioning = cfg.training.get("self_conditioning", {}).get(
        "enabled", False
    )

    # Configure input channels
    in_channels = model_cfg.in_channels

    if use_self_conditioning:
        in_channels += 2
        logger.info(f"Self-Conditioning: input channels -> {in_channels}")

    # Anatomical conditioning: only concat method adds input channels
    # cross_attention method uses context instead
    if use_anatomical_conditioning and anatomical_method == "concat":
        in_channels += 1
        logger.info(f"Anatomical Conditioning (concat): input channels -> {in_channels}")
    elif use_anatomical_conditioning and anatomical_method == "cross_attention":
        logger.info(f"Anatomical Conditioning (cross_attention): using context, no extra input channels")

    # Calculate number of class embeddings
    num_class_embeds = 2 * z_bins
    if cond_cfg.cfg.enabled:
        num_class_embeds += 1

    channels = tuple(model_cfg.channels)
    attention_levels = tuple(model_cfg.attention_levels)

    # Determine cross-attention settings
    # cross_attention method requires with_conditioning=True
    with_conditioning = model_cfg.with_conditioning
    cross_attention_dim = None
    if use_anatomical_conditioning and anatomical_method == "cross_attention":
        with_conditioning = True
        cross_attention_dim = model_cfg.get("cross_attention_dim", channels[-1])

    # Create model
    model = DiffusionModelUNet(
        spatial_dims=model_cfg.spatial_dims,
        in_channels=in_channels,
        out_channels=model_cfg.out_channels,
        channels=channels,
        attention_levels=attention_levels,
        num_res_blocks=model_cfg.num_res_blocks,
        num_head_channels=model_cfg.num_head_channels,
        norm_num_groups=model_cfg.norm_num_groups,
        norm_eps=1e-6,
        resblock_updown=model_cfg.resblock_updown,
        num_class_embeds=num_class_embeds if model_cfg.use_class_embedding else None,
        with_conditioning=with_conditioning,
        cross_attention_dim=cross_attention_dim,
        dropout_cattn=model_cfg.dropout,
    )

    # Replace class embedding with sinusoidal if enabled
    if cond_cfg.use_sinusoidal and model_cfg.use_class_embedding:
        embedding_dim = model.class_embedding.embedding_dim
        z_range = tuple(cfg.data.slice_sampling.z_range)

        custom_embedding = ConditionalEmbeddingWithSinusoidal(
            num_embeddings=num_class_embeds,
            embedding_dim=embedding_dim,
            z_bins=z_bins,
            z_range=z_range,
            use_sinusoidal=True,
            max_z=cond_cfg.max_z,
            use_cfg=cond_cfg.cfg.enabled,
        )
        model.class_embedding = custom_embedding

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Built DiffusionModelUNet: {n_params:,} params")

    return model


def load_model_from_checkpoint(
    checkpoint_path: Path,
    cfg: DictConfig,
    device: str = "cuda",
    use_ema: bool = True,
) -> tuple[torch.nn.Module, torch.nn.Module, bool, int]:
    """Load model from checkpoint with optional EMA weights.

    This function handles older checkpoints that may have different config
    structures by:
    1. Building the model architecture from the current config
    2. Loading weights with strict=False to handle mismatches
    3. Reporting any missing/unexpected keys
    4. Detecting actual input channels from loaded weights

    Args:
        checkpoint_path: Path to Lightning checkpoint.
        cfg: Configuration object.
        device: Device to load model on.
        use_ema: Whether to load EMA weights if available.

    Returns:
        Tuple of (model, scheduler, ema_loaded_flag, actual_in_channels).
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Ensure config has all required defaults
    cfg = ensure_config_defaults(cfg)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build model from config
    model = build_model_standalone(cfg)
    model.to(device)

    # Build scheduler
    scheduler = build_scheduler(cfg)

    # Determine which weights to load
    ema_loaded = False
    state_dict = None
    source = "unknown"

    if use_ema:
        # Try EMA weights first
        if "ema_state_dict" in ckpt and isinstance(ckpt["ema_state_dict"], dict):
            state_dict = ckpt["ema_state_dict"]
            source = "ema_state_dict"
            ema_loaded = True
        elif "callbacks" in ckpt:
            cb_state = ckpt.get("callbacks", {}).get("EMACallback", {})
            if "ema" in cb_state and cb_state["ema"] is not None:
                state_dict = cb_state["ema"]
                source = "callbacks.EMACallback.ema"
                ema_loaded = True

    # Fall back to regular model weights if no EMA found
    if state_dict is None:
        if "state_dict" in ckpt:
            # Lightning checkpoint format: state_dict contains full module state
            # Extract just the model weights (keys starting with "model.")
            full_state = ckpt["state_dict"]
            state_dict = {}
            for k, v in full_state.items():
                if k.startswith("model."):
                    state_dict[k[6:]] = v  # Remove "model." prefix
            source = "state_dict (model.*)"
        else:
            raise ValueError(
                "Checkpoint does not contain 'state_dict' or EMA weights. "
                "Cannot load model."
            )

    # Detect actual input channels from loaded weights
    actual_in_channels = cfg.model.in_channels  # Default from config
    # Try different possible key names for the first conv layer
    conv_in_keys = ["conv_in.0.weight", "conv_in.conv.weight"]
    for key in conv_in_keys:
        if key in state_dict:
            # Shape is [out_channels, in_channels, kernel_h, kernel_w]
            actual_in_channels = state_dict[key].shape[1]
            logger.info(f"Detected actual input channels from checkpoint ({key}): {actual_in_channels}")
            break

    # Load state dict with flexible matching
    logger.info(f"Loading weights from: {source}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    if not missing and not unexpected:
        logger.info("All weights loaded successfully (exact match)")
    elif not missing:
        logger.info(
            f"Weights loaded successfully ({len(unexpected)} unexpected keys ignored)"
        )

    model.eval()

    if use_ema and not ema_loaded:
        logger.warning(
            "EMA weights requested but not found. Using regular model weights."
        )

    return model, scheduler, ema_loaded, actual_in_channels


# =============================================================================
# Data Loading
# =============================================================================


def load_lesion_samples(
    cache_dir: Path,
    split: str = "test",
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load samples that contain lesions from the slice cache.

    Args:
        cache_dir: Path to slice cache directory.
        split: Which split to use (train, val, test).
        max_samples: Maximum number of samples to return.
        seed: Random seed for sampling.

    Returns:
        List of sample dictionaries with lesion pixels.
    """
    dataset = SliceDataset(cache_dir=cache_dir, split=split)

    # Filter to samples with lesions
    lesion_samples = [
        (idx, sample)
        for idx, sample in enumerate(dataset.samples)
        if sample["has_lesion"]
    ]

    logger.info(
        f"Found {len(lesion_samples)} samples with lesions out of {len(dataset)} total"
    )

    # Optionally subsample
    if max_samples is not None and max_samples < len(lesion_samples):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(lesion_samples), size=max_samples, replace=False)
        lesion_samples = [lesion_samples[i] for i in sorted(indices)]
        logger.info(f"Subsampled to {len(lesion_samples)} samples")

    # Load actual data for each sample
    loaded_samples = []
    for idx, sample_meta in tqdm(lesion_samples, desc="Loading samples"):
        sample = dataset[idx]
        sample["dataset_idx"] = idx
        loaded_samples.append(sample)

    return loaded_samples


# =============================================================================
# Reconstruction Analysis
# =============================================================================


def add_noise_at_timestep(
    x0: torch.Tensor,
    scheduler: torch.nn.Module,
    timestep: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add noise to x0 at a specific timestep.

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

    Args:
        x0: Original samples, shape (B, C, H, W).
        scheduler: DDPM scheduler with alphas_cumprod.
        timestep: Target timestep.
        device: Device for tensors.

    Returns:
        Tuple of (x_t, noise) where noise is the added Gaussian noise.
    """
    B = x0.shape[0]
    timesteps = torch.full((B,), timestep, device=device, dtype=torch.long)

    # Sample noise
    noise = torch.randn_like(x0)

    # Get alpha_bar_t
    alpha_bar_t = get_alpha_bar(scheduler, timesteps)

    # Reshape for broadcasting
    while alpha_bar_t.dim() < x0.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)

    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

    # Forward diffusion
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    return x_t, noise


def reconstruct_from_noisy(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    timestep: int,
    tokens: torch.Tensor,
    scheduler: torch.nn.Module,
    anatomical_priors: torch.Tensor | None = None,
    anatomical_encoder: torch.nn.Module | None = None,
    anatomical_method: str = "concat",
    device: str = "cuda",
    n_output_channels: int = 2,
) -> torch.Tensor:
    """Predict x0 from noisy x_t using single-step reconstruction.

    This performs a single forward pass to predict the noise, then uses
    the closed-form formula to estimate x0.

    Args:
        model: Diffusion model.
        x_t: Noisy samples at timestep t, shape (B, C, H, W). May include
            self-conditioning channels.
        timestep: Current timestep.
        tokens: Conditioning tokens, shape (B,).
        scheduler: DDPM scheduler.
        anatomical_priors: Optional anatomical priors, shape (B, 1, H, W).
        anatomical_encoder: Optional encoder for cross_attention method.
        anatomical_method: Anatomical conditioning method ("concat" or "cross_attention").
        device: Device for tensors.
        n_output_channels: Number of output channels (typically 2 for image+mask).

    Returns:
        Reconstructed x0, shape (B, n_output_channels, H, W).
    """
    B = x_t.shape[0]
    timesteps = torch.full((B,), timestep, device=device, dtype=torch.long)

    # Prepare model input and context based on anatomical method
    model_input = x_t
    anatomical_context = None

    if anatomical_priors is not None:
        if anatomical_method == "cross_attention":
            # Encode priors to context for cross-attention
            if anatomical_encoder is not None:
                anatomical_context = anatomical_encoder(anatomical_priors)
            else:
                logger.warning(
                    "cross_attention method requires anatomical_encoder, but None provided. "
                    "Skipping anatomical conditioning."
                )
        else:
            # Concat method: add prior as input channel
            model_input = torch.cat([x_t, anatomical_priors], dim=1)

    # Predict noise
    with torch.no_grad():
        eps_pred = model(
            model_input,
            timesteps=timesteps,
            context=anatomical_context,
            class_labels=tokens,
        )

    # Reconstruct x0 using only the first n_output_channels of x_t
    # (exclude self-conditioning channels for the reconstruction formula)
    x_t_for_recon = x_t[:, :n_output_channels]
    x0_hat = predict_x0(x_t_for_recon, eps_pred, scheduler, timesteps)

    return x0_hat


def compute_pixel_errors(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    mask: torch.Tensor,
    lesion_threshold: float = 0.0,
) -> dict[str, Any]:
    """Compute pixel-wise squared errors separated by lesion/non-lesion regions.

    Args:
        original: Original x0, shape (B, C, H, W).
        reconstructed: Reconstructed x0, shape (B, C, H, W).
        mask: Lesion mask channel, shape (B, 1, H, W) in [-1, 1].
        lesion_threshold: Threshold for binarizing mask (in [-1, 1] space).

    Returns:
        Dictionary with error statistics and maps.
    """
    # Compute squared error
    squared_error = (original - reconstructed) ** 2

    # Image channel error (channel 0)
    image_error = squared_error[:, 0:1]  # (B, 1, H, W)

    # Mask channel error (channel 1)
    mask_error = squared_error[:, 1:2]  # (B, 1, H, W)

    # Binarize lesion mask
    lesion_binary = mask > lesion_threshold  # (B, 1, H, W)

    # Compute statistics per sample
    results = []
    for b in range(original.shape[0]):
        lesion_mask_b = lesion_binary[b, 0]  # (H, W)
        nonlesion_mask_b = ~lesion_mask_b

        img_err_b = image_error[b, 0]  # (H, W)

        # Count pixels
        n_lesion = lesion_mask_b.sum().item()
        n_nonlesion = nonlesion_mask_b.sum().item()

        # Mean error in each region (for image channel)
        if n_lesion > 0:
            mean_lesion = img_err_b[lesion_mask_b].mean().item()
        else:
            mean_lesion = float("nan")

        if n_nonlesion > 0:
            mean_nonlesion = img_err_b[nonlesion_mask_b].mean().item()
        else:
            mean_nonlesion = float("nan")

        results.append(
            {
                "mean_error_lesion": mean_lesion,
                "mean_error_nonlesion": mean_nonlesion,
                "n_lesion_pixels": n_lesion,
                "n_nonlesion_pixels": n_nonlesion,
                "image_error": img_err_b.cpu().numpy(),
                "mask_error": mask_error[b, 0].cpu().numpy(),
            }
        )

    return {
        "per_sample": results,
        "image_error_batch": image_error.cpu().numpy(),
        "mask_error_batch": mask_error.cpu().numpy(),
    }


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_analysis(
    model: torch.nn.Module,
    scheduler: torch.nn.Module,
    samples: list[dict[str, Any]],
    timestep: int,
    cfg: DictConfig,
    zbin_priors: dict[int, NDArray[np.bool_]] | None,
    anatomical_encoder: torch.nn.Module | None = None,
    device: str = "cuda",
    batch_size: int = 16,
    actual_in_channels: int = 2,
) -> AnalysisResults:
    """Run the full reconstruction error analysis.

    Args:
        model: Loaded diffusion model.
        scheduler: DDPM scheduler.
        samples: List of sample dictionaries from dataset.
        timestep: Timestep at which to add noise.
        cfg: Configuration object.
        zbin_priors: Z-bin priors for anatomical conditioning.
        anatomical_encoder: Optional encoder for cross_attention method.
        device: Device for computation.
        batch_size: Batch size for processing.
        actual_in_channels: Actual input channels expected by the model.

    Returns:
        AnalysisResults with all sample results.
    """
    model.eval()
    results = AnalysisResults(timestep=timestep, num_samples=len(samples))

    use_anatomical = cfg.model.get("anatomical_conditioning", False)
    anatomical_method = cfg.model.get("anatomical_conditioning_method", "concat")
    
    # Determine if self-conditioning channels are needed
    base_channels = 2  # image + mask
    if use_anatomical and anatomical_method == "concat":
        base_channels += 1  # +1 for anatomical prior
    
    needs_self_cond = actual_in_channels > base_channels
    if needs_self_cond:
        logger.info(f"Model expects {actual_in_channels} channels, adding {actual_in_channels - base_channels} self-conditioning dummy channels")

    # Process in batches
    for start_idx in tqdm(
        range(0, len(samples), batch_size), desc="Processing batches"
    ):
        batch_samples = samples[start_idx : start_idx + batch_size]
        batch = collate_fn(batch_samples)

        # Move to device
        images = batch["image"].to(device)  # (B, 1, H, W)
        masks = batch["mask"].to(device)  # (B, 1, H, W)
        tokens = batch["token"].to(device)  # (B,)

        # Concatenate to form x0
        x0 = torch.cat([images, masks], dim=1)  # (B, 2, H, W)

        # Add noise
        x_t, noise = add_noise_at_timestep(x0, scheduler, timestep, device)

        # Get anatomical priors if needed
        anatomical_priors = None
        if use_anatomical and zbin_priors is not None:
            z_bins_batch = batch["metadata"]["z_bin"]
            anatomical_priors = get_anatomical_priors_as_input(
                z_bins_batch, zbin_priors, device
            )

        # Add self-conditioning dummy channels if needed
        if needs_self_cond:
            B, C, H, W = x_t.shape
            num_self_cond_channels = actual_in_channels - (C + (1 if use_anatomical and anatomical_method == "concat" else 0))
            self_cond_dummy = torch.zeros(B, num_self_cond_channels, H, W, device=device)
            x_t = torch.cat([x_t, self_cond_dummy], dim=1)

        # Reconstruct
        x0_hat = reconstruct_from_noisy(
            model,
            x_t,
            timestep,
            tokens,
            scheduler,
            anatomical_priors=anatomical_priors,
            anatomical_encoder=anatomical_encoder,
            anatomical_method=anatomical_method,
            device=device,
        )

        # Compute errors
        error_results = compute_pixel_errors(x0, x0_hat, masks)

        # Store results
        for i, sample in enumerate(batch_samples):
            per_sample = error_results["per_sample"][i]

            result = SampleResult(
                sample_idx=sample["dataset_idx"],
                subject_id=sample["subject_id"],
                z_index=sample["z_index"],
                z_bin=sample["z_bin"],
                original_image=images[i, 0].cpu().numpy(),
                original_mask=masks[i, 0].cpu().numpy(),
                reconstructed_image=x0_hat[i, 0].cpu().numpy(),
                reconstructed_mask=x0_hat[i, 1].cpu().numpy(),
                image_error=per_sample["image_error"],
                mask_error=per_sample["mask_error"],
                mean_error_lesion=per_sample["mean_error_lesion"],
                mean_error_nonlesion=per_sample["mean_error_nonlesion"],
                n_lesion_pixels=per_sample["n_lesion_pixels"],
                n_nonlesion_pixels=per_sample["n_nonlesion_pixels"],
            )
            results.add_sample(result)

    return results


# =============================================================================
# Statistical Analysis
# =============================================================================


def perform_statistical_analysis(
    results: AnalysisResults,
) -> dict[str, Any]:
    """Perform statistical analysis on reconstruction errors.

    Tests the hypothesis: Do lesion pixels have higher reconstruction error
    than non-lesion pixels?

    Args:
        results: Analysis results.

    Returns:
        Dictionary with statistical analysis results.
    """
    lesion_errors = np.array(results.all_mean_errors_lesion)
    nonlesion_errors = np.array(results.all_mean_errors_nonlesion)

    # Remove any NaN values
    valid_mask = ~(np.isnan(lesion_errors) | np.isnan(nonlesion_errors))
    lesion_errors = lesion_errors[valid_mask]
    nonlesion_errors = nonlesion_errors[valid_mask]

    n_valid = len(lesion_errors)
    logger.info(f"Performing analysis on {n_valid} valid samples")

    # Descriptive statistics
    mean_lesion = float(np.mean(lesion_errors))
    mean_nonlesion = float(np.mean(nonlesion_errors))
    std_lesion = float(np.std(lesion_errors))
    std_nonlesion = float(np.std(nonlesion_errors))

    # Paired t-test (within-sample comparison)
    t_stat, p_value_ttest = stats.ttest_rel(lesion_errors, nonlesion_errors)

    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, p_value_wilcoxon = stats.wilcoxon(
            lesion_errors, nonlesion_errors, alternative="greater"
        )
    except ValueError:
        # Can fail if all differences are zero
        w_stat, p_value_wilcoxon = float("nan"), float("nan")

    # Effect size: Cohen's d for paired samples
    differences = lesion_errors - nonlesion_errors
    cohens_d = float(np.mean(differences) / np.std(differences))

    # Percentage of samples where lesion error > non-lesion error
    pct_lesion_higher = float(np.mean(differences > 0) * 100)

    # Log-transformed analysis (if errors are positive)
    if np.all(lesion_errors > 0) and np.all(nonlesion_errors > 0):
        log_ratio = np.log(lesion_errors / nonlesion_errors)
        mean_log_ratio = float(np.mean(log_ratio))
        geometric_mean_ratio = float(np.exp(mean_log_ratio))
    else:
        mean_log_ratio = float("nan")
        geometric_mean_ratio = float("nan")

    analysis = {
        "n_samples": int(n_valid),
        "descriptive": {
            "lesion_error": {"mean": mean_lesion, "std": std_lesion},
            "nonlesion_error": {"mean": mean_nonlesion, "std": std_nonlesion},
            "difference": {
                "mean": float(np.mean(differences)),
                "std": float(np.std(differences)),
            },
        },
        "paired_ttest": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value_ttest),
            "significant_at_0.05": bool(p_value_ttest < 0.05),
            "significant_at_0.01": bool(p_value_ttest < 0.01),
        },
        "wilcoxon_test": {
            "w_statistic": float(w_stat),
            "p_value": float(p_value_wilcoxon),
            "significant_at_0.05": bool(p_value_wilcoxon < 0.05),
        },
        "effect_size": {
            "cohens_d": cohens_d,
            "interpretation": interpret_cohens_d(cohens_d),
        },
        "practical_significance": {
            "pct_lesion_higher_error": pct_lesion_higher,
            "mean_log_ratio": mean_log_ratio,
            "geometric_mean_ratio": geometric_mean_ratio,
        },
    }

    return analysis


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value.

    Returns:
        Interpretation string.
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        size = "negligible"
    elif abs_d < 0.5:
        size = "small"
    elif abs_d < 0.8:
        size = "medium"
    else:
        size = "large"

    direction = "higher" if d > 0 else "lower"
    return f"{size} effect ({direction} error in lesion pixels)"


# =============================================================================
# Visualization
# =============================================================================


def to_display_range(x: np.ndarray) -> np.ndarray:
    """Convert from [-1, 1] to [0, 1] for display."""
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
        mask: Mask in [-1, 1], shape (H, W).
        alpha: Overlay transparency.
        color: RGB color for overlay (0-255).
        threshold: Threshold for binarizing mask (in [-1, 1] space).

    Returns:
        RGB image with overlay, shape (H, W, 3).
    """
    # Convert to RGB
    rgb = np.stack([image, image, image], axis=-1)

    # Binarize mask
    binary_mask = mask > threshold

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


def plot_error_distributions(
    results: AnalysisResults,
    stats_results: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot error distribution comparison.

    Args:
        results: Analysis results.
        stats_results: Statistical analysis results.
        output_path: Path to save the plot.
    """
    lesion_errors = np.array(results.all_mean_errors_lesion)
    nonlesion_errors = np.array(results.all_mean_errors_nonlesion)

    # Remove NaN
    valid_mask = ~(np.isnan(lesion_errors) | np.isnan(nonlesion_errors))
    lesion_errors = lesion_errors[valid_mask]
    nonlesion_errors = nonlesion_errors[valid_mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Histogram comparison
    ax = axes[0]
    ax.hist(lesion_errors, bins=30, alpha=0.6, label="Lesion pixels", color="red")
    ax.hist(
        nonlesion_errors, bins=30, alpha=0.6, label="Non-lesion pixels", color="blue"
    )
    ax.axvline(
        np.mean(lesion_errors), color="darkred", linestyle="--", label="Mean (lesion)"
    )
    ax.axvline(
        np.mean(nonlesion_errors),
        color="darkblue",
        linestyle="--",
        label="Mean (non-lesion)",
    )
    ax.set_xlabel("Mean Squared Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Per-Sample Mean Errors")
    ax.legend()

    # 2. Paired comparison scatter
    ax = axes[1]
    ax.scatter(nonlesion_errors, lesion_errors, alpha=0.5, s=20)
    max_val = max(np.max(lesion_errors), np.max(nonlesion_errors))
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="y=x (equal error)")
    ax.set_xlabel("Non-lesion Mean Error")
    ax.set_ylabel("Lesion Mean Error")
    ax.set_title("Paired Error Comparison")
    ax.legend()

    # Add annotation
    pct_above = stats_results["practical_significance"]["pct_lesion_higher_error"]
    ax.text(
        0.05,
        0.95,
        f"{pct_above:.1f}% samples above line",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    # 3. Box plot comparison
    ax = axes[2]
    bp = ax.boxplot(
        [lesion_errors, nonlesion_errors],
        labels=["Lesion", "Non-lesion"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("lightcoral")
    bp["boxes"][1].set_facecolor("lightblue")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Error Distribution by Region")

    # Add significance annotation
    p_val = stats_results["paired_ttest"]["p_value"]
    sig_symbol = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    cohens_d = stats_results["effect_size"]["cohens_d"]
    ax.text(
        0.5,
        0.95,
        f"p={p_val:.2e}, Cohen's d={cohens_d:.3f} ({sig_symbol})",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        verticalalignment="top",
    )

    plt.suptitle(f"Reconstruction Error Analysis (t={results.timestep})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved error distribution plot to {output_path}")


def plot_example_reconstructions(
    results: AnalysisResults,
    output_dir: Path,
    n_examples: int = 6,
    select_supporting: bool = True,
) -> None:
    """Plot example reconstructions showing original, reconstructed, and error.

    Args:
        results: Analysis results.
        output_dir: Directory to save plots.
        n_examples: Number of examples to plot.
        select_supporting: If True, select samples that support the hypothesis
            (lesion error > non-lesion error).
    """
    # Select samples
    if select_supporting:
        # Samples where lesion error > non-lesion error
        supporting = [
            s
            for s in results.samples
            if not np.isnan(s.mean_error_lesion)
            and s.mean_error_lesion > s.mean_error_nonlesion
        ]
        # Sort by effect magnitude
        supporting.sort(
            key=lambda s: s.mean_error_lesion - s.mean_error_nonlesion, reverse=True
        )
        selected = supporting[:n_examples]
        suffix = "supporting"
    else:
        # Random selection
        valid = [s for s in results.samples if not np.isnan(s.mean_error_lesion)]
        selected = valid[:n_examples]
        suffix = "random"

    if len(selected) == 0:
        logger.warning("No valid samples to visualize")
        return

    # Create figure
    n_cols = 4  # Original + overlay, Reconstructed + overlay, Error map, Error overlay
    fig, axes = plt.subplots(len(selected), n_cols, figsize=(4 * n_cols, 4 * len(selected)))

    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    for row, sample in enumerate(selected):
        # Convert to display range
        orig_img = to_display_range(sample.original_image)
        orig_mask = sample.original_mask
        recon_img = to_display_range(sample.reconstructed_image)
        error_map = sample.image_error

        # Normalize error map for display
        error_display = error_map / (error_map.max() + 1e-8)

        # 1. Original with mask overlay
        overlay_orig = create_overlay(orig_img, orig_mask, alpha=0.4, color=(255, 0, 0))
        axes[row, 0].imshow(overlay_orig)
        axes[row, 0].set_title(f"Original + Mask\n(z={sample.z_index})")
        axes[row, 0].axis("off")

        # 2. Reconstructed with mask overlay
        overlay_recon = create_overlay(
            recon_img, orig_mask, alpha=0.4, color=(255, 0, 0)
        )
        axes[row, 1].imshow(overlay_recon)
        axes[row, 1].set_title("Reconstructed + Mask")
        axes[row, 1].axis("off")

        # 3. Error map (heatmap)
        im = axes[row, 2].imshow(error_map, cmap="hot")
        axes[row, 2].set_title(
            f"Squared Error\n"
            f"Lesion: {sample.mean_error_lesion:.4f}\n"
            f"Non-lesion: {sample.mean_error_nonlesion:.4f}"
        )
        axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # 4. Error map with lesion boundary overlay
        # Create RGB error map
        error_rgb = plt.cm.hot(error_display)[:, :, :3]
        # Add lesion boundary
        from scipy import ndimage

        lesion_binary = orig_mask > 0
        if lesion_binary.any():
            boundary = lesion_binary ^ ndimage.binary_erosion(lesion_binary)
            error_rgb[boundary] = [0, 1, 0]  # Green boundary

        axes[row, 3].imshow(error_rgb)
        axes[row, 3].set_title("Error + Lesion Boundary")
        axes[row, 3].axis("off")

    plt.suptitle(
        f"Example Reconstructions (t={results.timestep}, {suffix} cases)", fontsize=14
    )
    plt.tight_layout()

    output_path = output_dir / f"example_reconstructions_{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved example reconstructions to {output_path}")


def generate_summary_report(
    stats_results: dict[str, Any],
    config_info: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate a text summary report.

    Args:
        stats_results: Statistical analysis results.
        config_info: Configuration information.
        output_path: Path to save the report.
    """
    lines = [
        "=" * 80,
        "LESION RECONSTRUCTION ERROR ANALYSIS REPORT",
        "=" * 80,
        "",
        "CONFIGURATION",
        "-" * 40,
        f"  Checkpoint: {config_info['checkpoint_path']}",
        f"  Timestep: {config_info['timestep']}",
        f"  Number of samples: {stats_results['n_samples']}",
        f"  Split: {config_info['split']}",
        "",
        "HYPOTHESIS",
        "-" * 40,
        "  H0: Lesion pixels have equal reconstruction error as non-lesion pixels",
        "  H1: Lesion pixels have HIGHER reconstruction error than non-lesion pixels",
        "",
        "DESCRIPTIVE STATISTICS",
        "-" * 40,
        f"  Lesion mean error:     {stats_results['descriptive']['lesion_error']['mean']:.6f} "
        f"(+/- {stats_results['descriptive']['lesion_error']['std']:.6f})",
        f"  Non-lesion mean error: {stats_results['descriptive']['nonlesion_error']['mean']:.6f} "
        f"(+/- {stats_results['descriptive']['nonlesion_error']['std']:.6f})",
        f"  Mean difference:       {stats_results['descriptive']['difference']['mean']:.6f} "
        f"(+/- {stats_results['descriptive']['difference']['std']:.6f})",
        "",
        "STATISTICAL TESTS",
        "-" * 40,
        "  Paired t-test (two-tailed):",
        f"    t-statistic: {stats_results['paired_ttest']['t_statistic']:.4f}",
        f"    p-value:     {stats_results['paired_ttest']['p_value']:.2e}",
        f"    Significant at alpha=0.05: {stats_results['paired_ttest']['significant_at_0.05']}",
        f"    Significant at alpha=0.01: {stats_results['paired_ttest']['significant_at_0.01']}",
        "",
        "  Wilcoxon signed-rank test (one-tailed, greater):",
        f"    W-statistic: {stats_results['wilcoxon_test']['w_statistic']:.4f}",
        f"    p-value:     {stats_results['wilcoxon_test']['p_value']:.2e}",
        f"    Significant at alpha=0.05: {stats_results['wilcoxon_test']['significant_at_0.05']}",
        "",
        "EFFECT SIZE",
        "-" * 40,
        f"  Cohen's d: {stats_results['effect_size']['cohens_d']:.4f}",
        f"  Interpretation: {stats_results['effect_size']['interpretation']}",
        "",
        "PRACTICAL SIGNIFICANCE",
        "-" * 40,
        f"  % samples with lesion error > non-lesion error: "
        f"{stats_results['practical_significance']['pct_lesion_higher_error']:.1f}%",
        f"  Geometric mean ratio (lesion/non-lesion): "
        f"{stats_results['practical_significance']['geometric_mean_ratio']:.4f}",
        "",
        "CONCLUSION",
        "-" * 40,
    ]

    # Generate conclusion
    p_val = stats_results["paired_ttest"]["p_value"]
    cohens_d = stats_results["effect_size"]["cohens_d"]
    pct = stats_results["practical_significance"]["pct_lesion_higher_error"]

    if p_val < 0.05 and cohens_d > 0:
        lines.append(
            f"  The analysis SUPPORTS the hypothesis that lesion pixels have higher"
        )
        lines.append(f"  reconstruction error (p={p_val:.2e}, Cohen's d={cohens_d:.3f}).")
        lines.append(f"  {pct:.1f}% of samples show higher error in lesion regions.")
    elif p_val >= 0.05:
        lines.append(
            f"  The analysis does NOT support the hypothesis (p={p_val:.2e})."
        )
        lines.append("  No significant difference in reconstruction error was found.")
    else:
        lines.append(
            f"  The analysis shows LOWER error in lesion pixels (p={p_val:.2e},"
        )
        lines.append(f"  Cohen's d={cohens_d:.3f}), contrary to the hypothesis.")

    lines.extend(
        [
            "",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            "=" * 80,
        ]
    )

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    # Also print to console
    print(report)

    logger.info(f"Saved summary report to {output_path}")


# =============================================================================
# Output Saving
# =============================================================================


def save_results_npz(
    results: AnalysisResults,
    stats_results: dict[str, Any],
    config_info: dict[str, Any],
    output_path: Path,
) -> None:
    """Save analysis results to NPZ file.

    Args:
        results: Analysis results.
        stats_results: Statistical analysis results.
        config_info: Configuration information.
        output_path: Path to save NPZ file.
    """
    # Collect arrays
    n = len(results.samples)

    # Per-sample arrays
    sample_indices = np.array([s.sample_idx for s in results.samples], dtype=np.int32)
    z_indices = np.array([s.z_index for s in results.samples], dtype=np.int32)
    z_bins = np.array([s.z_bin for s in results.samples], dtype=np.int32)
    mean_errors_lesion = np.array(results.all_mean_errors_lesion, dtype=np.float32)
    mean_errors_nonlesion = np.array(results.all_mean_errors_nonlesion, dtype=np.float32)
    n_lesion_pixels = np.array(
        [s.n_lesion_pixels for s in results.samples], dtype=np.int32
    )
    n_nonlesion_pixels = np.array(
        [s.n_nonlesion_pixels for s in results.samples], dtype=np.int32
    )

    # Stack images and errors (can be large, but useful for detailed analysis)
    original_images = np.stack(
        [s.original_image for s in results.samples], axis=0
    ).astype(np.float16)
    original_masks = np.stack([s.original_mask for s in results.samples], axis=0).astype(
        np.float16
    )
    reconstructed_images = np.stack(
        [s.reconstructed_image for s in results.samples], axis=0
    ).astype(np.float16)
    reconstructed_masks = np.stack(
        [s.reconstructed_mask for s in results.samples], axis=0
    ).astype(np.float16)
    image_errors = np.stack([s.image_error for s in results.samples], axis=0).astype(
        np.float16
    )

    np.savez_compressed(
        output_path,
        # Metadata
        timestep=results.timestep,
        num_samples=n,
        # Per-sample data
        sample_indices=sample_indices,
        z_indices=z_indices,
        z_bins=z_bins,
        mean_errors_lesion=mean_errors_lesion,
        mean_errors_nonlesion=mean_errors_nonlesion,
        n_lesion_pixels=n_lesion_pixels,
        n_nonlesion_pixels=n_nonlesion_pixels,
        # Images (compressed float16)
        original_images=original_images,
        original_masks=original_masks,
        reconstructed_images=reconstructed_images,
        reconstructed_masks=reconstructed_masks,
        image_errors=image_errors,
        # Stats as JSON string (for complex nested dict)
        stats_json=json.dumps(stats_results),
        config_json=json.dumps(config_info),
    )

    logger.info(
        f"Saved results NPZ to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze DDPM reconstruction error on lesion vs non-lesion pixels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to slice cache directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output files",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=100,
        help="Timestep at which to add noise (default: 100)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to analyze (default: 100)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to use (default: test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="Use EMA weights if available (default: True)",
    )
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Do not use EMA weights",
    )
    parser.add_argument(
        "--n-vis-examples",
        type=int,
        default=6,
        help="Number of visualization examples (default: 6)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup
    setup_logger("jsddpm", level=logging.INFO)
    seed_everything(args.seed)

    # Paths
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = OmegaConf.load(config_path)

    logger.info("=" * 60)
    logger.info("LESION RECONSTRUCTION ERROR ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Timestep: {args.timestep}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Split: {args.split}")

    # Load model
    model, scheduler, ema_loaded, actual_in_channels = load_model_from_checkpoint(
        checkpoint_path, cfg, args.device, args.use_ema
    )

    # Load z-bin priors and anatomical encoder if needed
    use_anatomical = cfg.model.get("anatomical_conditioning", False)
    anatomical_method = cfg.model.get("anatomical_conditioning_method", "concat")
    zbin_priors = None
    anatomical_encoder = None

    if use_anatomical:
        try:
            pp_cfg = cfg.get("postprocessing", {})
            zbin_cfg = pp_cfg.get("zbin_priors", {})
            priors_filename = zbin_cfg.get("priors_filename", "zbin_priors_brain_roi.npz")
            zbin_priors = load_zbin_priors(cache_dir, priors_filename, cfg.conditioning.z_bins)
            logger.info(f"Loaded z-bin priors for anatomical conditioning")
        except Exception as e:
            logger.warning(f"Failed to load z-bin priors: {e}")

        # Load anatomical encoder for cross_attention method
        if anatomical_method == "cross_attention":
            try:
                from src.diffusion.model.components.anatomical_encoder import AnatomicalPriorEncoder

                encoder_cfg = cfg.model.get("anatomical_encoder", {})
                cross_attention_dim = cfg.model.get("cross_attention_dim", 256)

                anatomical_encoder = AnatomicalPriorEncoder(
                    embed_dim=cross_attention_dim,
                    hidden_dims=tuple(encoder_cfg.get("hidden_dims", [32, 64, 128])),
                    downsample_factor=encoder_cfg.get("downsample_factor", 8),
                    input_size=tuple(cfg.data.transforms.roi_size[:2]),
                    positional_encoding=encoder_cfg.get("positional_encoding", "sinusoidal"),
                    norm_num_groups=encoder_cfg.get("norm_num_groups", 8),
                )
                anatomical_encoder.to(args.device)
                anatomical_encoder.eval()

                # Try to load encoder weights from checkpoint
                ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
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
                            logger.warning(f"Anatomical encoder loaded with missing={len(missing)}, unexpected={len(unexpected)} keys")
                    else:
                        logger.warning("No anatomical encoder weights found in checkpoint")

            except Exception as e:
                logger.warning(f"Failed to load anatomical encoder: {e}")

    # Load data
    samples = load_lesion_samples(
        cache_dir, args.split, args.num_samples, args.seed
    )

    if len(samples) == 0:
        logger.error("No lesion samples found. Exiting.")
        return

    # Run analysis
    results = run_analysis(
        model=model,
        scheduler=scheduler,
        samples=samples,
        timestep=args.timestep,
        cfg=cfg,
        zbin_priors=zbin_priors,
        anatomical_encoder=anatomical_encoder,
        device=args.device,
        batch_size=args.batch_size,
        actual_in_channels=actual_in_channels,
    )

    # Statistical analysis
    stats_results = perform_statistical_analysis(results)

    # Configuration info for output
    config_info = {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "cache_dir": str(cache_dir),
        "timestep": args.timestep,
        "num_samples": len(samples),
        "split": args.split,
        "use_ema": args.use_ema,
        "ema_loaded": ema_loaded,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    # Save outputs
    save_results_npz(
        results, stats_results, config_info, output_dir / "analysis_results.npz"
    )

    # Save stats as JSON
    with open(output_dir / "statistical_analysis.json", "w") as f:
        json.dump(stats_results, f, indent=2)

    # Generate visualizations
    plot_error_distributions(
        results, stats_results, output_dir / "error_distributions.png"
    )
    plot_example_reconstructions(
        results, output_dir, n_examples=args.n_vis_examples, select_supporting=True
    )
    plot_example_reconstructions(
        results, output_dir, n_examples=args.n_vis_examples, select_supporting=False
    )

    # Generate summary report
    generate_summary_report(
        stats_results, config_info, output_dir / "summary_report.txt"
    )

    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
