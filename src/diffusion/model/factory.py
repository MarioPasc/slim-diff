"""Model factory for JS-DDPM.

Builds the DiffusionModelUNet, scheduler, and inferer from configuration.
"""

from __future__ import annotations

import logging

import torch
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from monai.networks.schedulers.ddim import DDIMScheduler
from monai.networks.schedulers.ddpm import DDPMScheduler
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> DiffusionModelUNet:
    """Build the DiffusionModelUNet from configuration.

    Args:
        cfg: Configuration object.

    Returns:
        Configured DiffusionModelUNet instance.
    """
    model_cfg = cfg.model
    cond_cfg = cfg.conditioning
    z_bins = cond_cfg.z_bins

    # Calculate number of class embeddings
    # 2 * z_bins for (control, lesion) classes
    # +1 if CFG enabled for null token
    num_class_embeds = 2 * z_bins
    if cond_cfg.cfg.enabled:
        num_class_embeds += 1

    # Build channel configuration
    channels = tuple(model_cfg.channels)
    attention_levels = tuple(model_cfg.attention_levels)

    # Create model
    model = DiffusionModelUNet(
        spatial_dims=model_cfg.spatial_dims,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        channels=channels,
        attention_levels=attention_levels,
        num_res_blocks=model_cfg.num_res_blocks,
        num_head_channels=model_cfg.num_head_channels,
        norm_num_groups=model_cfg.norm_num_groups,
        norm_eps=1e-6,
        resblock_updown=model_cfg.resblock_updown,
        num_class_embeds=num_class_embeds if model_cfg.use_class_embedding else None,
        with_conditioning=model_cfg.with_conditioning,
        dropout_cattn=model_cfg.dropout,
    )

    # Log model info
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Built DiffusionModelUNet: "
        f"{n_params:,} params ({n_trainable:,} trainable), "
        f"channels={channels}, "
        f"num_class_embeds={num_class_embeds}"
    )

    return model


def build_scheduler(cfg: DictConfig) -> DDPMScheduler | DDIMScheduler:
    """Build the diffusion scheduler from configuration.

    Args:
        cfg: Configuration object.

    Returns:
        Configured scheduler instance.
    """
    sched_cfg = cfg.scheduler

    # Common kwargs
    common_kwargs = {
        "num_train_timesteps": sched_cfg.num_train_timesteps,
        "beta_start": sched_cfg.beta_start,
        "beta_end": sched_cfg.beta_end,
        "schedule": sched_cfg.schedule,
        "prediction_type": sched_cfg.prediction_type,
        "clip_sample": sched_cfg.clip_sample,
    }

    if sched_cfg.type == "DDPM":
        scheduler = DDPMScheduler(**common_kwargs)
    elif sched_cfg.type == "DDIM":
        scheduler = DDIMScheduler(**common_kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_cfg.type}")

    logger.info(
        f"Built {sched_cfg.type} scheduler: "
        f"T={sched_cfg.num_train_timesteps}, "
        f"beta=[{sched_cfg.beta_start}, {sched_cfg.beta_end}]"
    )

    return scheduler


def build_inferer(cfg: DictConfig) -> DDIMScheduler:
    """Build the inference scheduler (DDIM) for sampling.

    Args:
        cfg: Configuration object.

    Returns:
        Configured DDIM scheduler for inference.
    """
    sched_cfg = cfg.scheduler
    sampler_cfg = cfg.sampler

    # Build DDIM scheduler for inference
    inferer = DDIMScheduler(
        num_train_timesteps=sched_cfg.num_train_timesteps,
        beta_start=sched_cfg.beta_start,
        beta_end=sched_cfg.beta_end,
        schedule=sched_cfg.schedule,
        prediction_type=sched_cfg.prediction_type,
        clip_sample=sched_cfg.clip_sample,
    )

    logger.info(
        f"Built DDIM inferer: "
        f"inference_steps={sampler_cfg.num_inference_steps}, "
        f"eta={sampler_cfg.eta}"
    )

    return inferer


class DiffusionSampler:
    """Wrapper for DDIM/DDPM sampling with classifier-free guidance.

    Handles the sampling loop and optional CFG.
    """

    def __init__(
        self,
        model: DiffusionModelUNet,
        scheduler: DDIMScheduler | DDPMScheduler,
        cfg: DictConfig,
        device: torch.device | str = "cuda",
    ) -> None:
        """Initialize the sampler.

        Args:
            model: The diffusion model.
            scheduler: The inference scheduler.
            cfg: Configuration object.
            device: Device to run on.
        """
        self.model = model
        self.scheduler = scheduler
        self.sampler_cfg = cfg.sampler
        self.cond_cfg = cfg.conditioning
        self.device = device

        self.num_inference_steps = self.sampler_cfg.num_inference_steps
        self.eta = self.sampler_cfg.eta
        self.guidance_scale = self.sampler_cfg.guidance_scale

        # Null token for CFG
        self.null_token = self.cond_cfg.z_bins * 2 if self.cond_cfg.cfg.enabled else None

    @torch.no_grad()
    def sample(
        self,
        tokens: torch.Tensor,
        shape: tuple[int, ...] = (1, 2, 128, 128),
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate samples using DDIM.

        Args:
            tokens: Conditioning tokens, shape (B,).
            shape: Output shape (B, C, H, W).
            guidance_scale: Override guidance scale.
            generator: Optional random generator.

        Returns:
            Generated samples, shape (B, C, H, W).
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        B = tokens.shape[0]
        if shape[0] != B:
            shape = (B,) + shape[1:]

        # Start from noise
        x_t = torch.randn(shape, device=self.device, generator=generator)

        # Set inference timesteps
        self.scheduler.set_timesteps(self.num_inference_steps)

        # Sampling loop
        for t in self.scheduler.timesteps:
            # Expand timesteps for batch
            timesteps = torch.full((B,), t, device=self.device, dtype=torch.long)

            if guidance_scale > 1.0 and self.null_token is not None:
                # CFG: compute conditioned and unconditioned predictions
                x_t_double = torch.cat([x_t, x_t], dim=0)
                t_double = torch.cat([timesteps, timesteps], dim=0)

                # Tokens: conditioned and null
                null_tokens = torch.full_like(tokens, self.null_token)
                tokens_double = torch.cat([tokens, null_tokens], dim=0)

                # Get predictions
                noise_pred = self.model(
                    x_t_double,
                    timesteps=t_double,
                    class_labels=tokens_double,
                )

                # Split predictions
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)

                # CFG combination
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                # No CFG
                noise_pred = self.model(x_t, timesteps=timesteps, class_labels=tokens)

            # DDIM step
            x_t, _ = self.scheduler.step(noise_pred, t, x_t, eta=self.eta)

        return x_t

    def sample_single(
        self,
        token: int,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate a single sample.

        Args:
            token: Conditioning token.
            guidance_scale: Override guidance scale.
            generator: Optional random generator.

        Returns:
            Generated sample, shape (2, 128, 128).
        """
        tokens = torch.tensor([token], device=self.device, dtype=torch.long)
        sample = self.sample(tokens, (1, 2, 128, 128), guidance_scale, generator)
        return sample[0]


def get_alpha_bar(scheduler: DDPMScheduler, t: torch.Tensor) -> torch.Tensor:
    """Get cumulative product of alphas (alpha_bar) at timestep t.

    Args:
        scheduler: The DDPM scheduler.
        t: Timesteps tensor.

    Returns:
        Alpha bar values at each timestep.
    """
    # MONAI schedulers store alphas_cumprod
    alphas_cumprod = scheduler.alphas_cumprod.to(t.device)
    return alphas_cumprod[t]


def predict_x0(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    scheduler: DDPMScheduler,
    t: torch.Tensor,
) -> torch.Tensor:
    """Predict x0 from x_t and epsilon prediction.

    x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)

    Args:
        x_t: Noisy samples at timestep t.
        eps_pred: Predicted noise.
        scheduler: The DDPM scheduler.
        t: Timesteps.

    Returns:
        Predicted x0.
    """
    alpha_bar_t = get_alpha_bar(scheduler, t)

    # Reshape for broadcasting (B,) -> (B, 1, 1, 1)
    while alpha_bar_t.dim() < x_t.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)

    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

    x0_hat = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar

    return x0_hat
