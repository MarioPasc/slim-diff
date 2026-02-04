"""Model factory for JS-DDPM.

Builds the DiffusionModelUNet, scheduler, and inferer from configuration.

Anatomical Conditioning Methods:
--------------------------------
The factory supports two methods for incorporating anatomical z-bin priors:

1. "concat" (default, original behavior):
   - Concatenates the prior mask as an additional input channel
   - Simple and effective, minimal overhead
   - Input channels: base + self_cond + 1

2. "cross_attention":
   - Encodes the prior via a lightweight CNN into spatial context embeddings
   - Uses cross-attention in the UNet to attend to the prior
   - More expressive: learned, selective, multi-scale spatial guidance
   - Input channels: base + self_cond (no +1)
   - Requires AnatomicalPriorEncoder module

Encoder Versions:
-----------------
When using cross_attention method, two encoder versions are available:

1. "legacy" (original):
   - Simple CNN backbone, 8x downsampling, 400 tokens
   - Sinusoidal or learned positional encoding
   - Single-channel binary input only

2. "enhanced":
   - FPN backbone with multi-scale feature fusion
   - 4x downsampling for 1600 tokens (finer resolution)
   - 2D RoPE for better relative position encoding
   - Flexible multi-channel input (tissue probability maps)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Union

import torch
import torch.nn as nn
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from monai.networks.schedulers.ddim import DDIMScheduler
from monai.networks.schedulers.ddpm import DDPMScheduler
from omegaconf import DictConfig, OmegaConf

from src.diffusion.model.embeddings import ConditionalEmbeddingWithSinusoidal
from src.diffusion.model.components.anatomical_encoder import (
    AnatomicalPriorEncoder,
    EnhancedAnatomicalPriorEncoder,
    build_anatomical_encoder,
    build_enhanced_anatomical_encoder,
)

logger = logging.getLogger(__name__)

# Type alias for conditioning methods
AnatomicalConditioningMethod = Literal["concat", "cross_attention"]

# Type alias for encoder types
EncoderType = Union[AnatomicalPriorEncoder, EnhancedAnatomicalPriorEncoder]


def load_encoder_config(cfg: DictConfig) -> DictConfig:
    """Load anatomical encoder configuration from separate YAML file.

    If model.anatomical_encoder_config is specified, loads that file.
    Otherwise, falls back to inline config in model.anatomical_encoder
    for backward compatibility.

    Args:
        cfg: Main configuration object.

    Returns:
        Encoder configuration (from file or inline).
    """
    encoder_config_path = cfg.model.get("anatomical_encoder_config")

    if encoder_config_path:
        # Load from separate file
        path = Path(encoder_config_path)
        if not path.is_absolute():
            # Resolve relative to current working directory
            path = Path.cwd() / path

        if not path.exists():
            raise FileNotFoundError(
                f"Anatomical encoder config not found: {path}"
            )

        encoder_cfg = OmegaConf.load(path)
        logger.info(f"Loaded encoder config from: {path}")
        return encoder_cfg

    # Fallback to inline config for backward compatibility
    inline_cfg = cfg.model.get("anatomical_encoder", {})
    if inline_cfg:
        logger.info("Using inline encoder config (legacy mode)")
    return OmegaConf.create(inline_cfg)


def build_model(
    cfg: DictConfig,
) -> tuple[DiffusionModelUNet, EncoderType | None]:
    """Build the DiffusionModelUNet and optional AnatomicalPriorEncoder from config.

    Handles two anatomical conditioning methods:
    - "concat": Adds +1 input channel for the prior mask (default)
    - "cross_attention": Enables cross-attention with a separate encoder

    When using cross_attention, supports two encoder versions:
    - "legacy": Original AnatomicalPriorEncoder (simple CNN, 8x downsample)
    - "enhanced": EnhancedAnatomicalPriorEncoder (FPN, RoPE, 4x downsample)

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (model, anatomical_encoder).
        anatomical_encoder is None if anatomical_conditioning is False or method is "concat".
    """
    model_cfg = cfg.model
    cond_cfg = cfg.conditioning
    z_bins = cond_cfg.z_bins

    # 1. Check for Anatomical Conditioning Toggle and Method
    # Default to False if not present to ensure backward compatibility
    use_anatomical_conditioning = model_cfg.get("anatomical_conditioning", False)
    anatomical_method: AnatomicalConditioningMethod = model_cfg.get(
        "anatomical_conditioning_method", "concat"
    )

    # Validate method
    if anatomical_method not in ("concat", "cross_attention"):
        raise ValueError(
            f"Unknown anatomical_conditioning_method: {anatomical_method}. "
            f"Must be 'concat' or 'cross_attention'."
        )

    # 2. Configure Input Channels
    # Base channels (e.g., 2 for FLAIR + Mask, or 1 for FLAIR)
    in_channels = model_cfg.in_channels

    # Check for self-conditioning (adds 2 channels for x0_self_cond)
    use_self_conditioning = cfg.training.self_conditioning.get("enabled", False)
    if use_self_conditioning:
        in_channels += 2  # Add 2 channels for self-conditioned x0 (image + mask)
        logger.info(
            f"Self-Conditioning ENABLED: "
            f"Input channels increased {model_cfg.in_channels} -> {in_channels}"
        )

    # Handle anatomical conditioning based on method
    use_cross_attention_anatomical = False
    cross_attention_dim = None
    anatomical_encoder = None

    if use_anatomical_conditioning:
        if anatomical_method == "concat":
            # Original behavior: add 1 channel for the Anatomical Prior Mask
            in_channels += 1
            logger.info(
                f"Anatomical Conditioning ENABLED (concat method): "
                f"Input channels increased -> {in_channels}"
            )
        elif anatomical_method == "cross_attention":
            # Cross-attention: don't add input channel, use separate encoder
            use_cross_attention_anatomical = True
            # Get cross_attention_dim from config or use default based on model channels
            cross_attention_dim = model_cfg.get(
                "cross_attention_dim",
                model_cfg.channels[-1]  # Default: same as deepest channel
            )
            logger.info(
                f"Anatomical Conditioning ENABLED (cross_attention method): "
                f"cross_attention_dim={cross_attention_dim}"
            )

    if not use_self_conditioning and not use_anatomical_conditioning:
        logger.info("Standard input channels (no conditioning augmentations).")

    # Calculate number of class embeddings
    num_class_embeds = 2 * z_bins
    if cond_cfg.cfg.enabled:
        num_class_embeds += 1

    channels = tuple(model_cfg.channels)
    attention_levels = tuple(model_cfg.attention_levels)

    # 3. Determine with_conditioning flag
    # Enable cross-attention if:
    # - Originally enabled in config (model_cfg.with_conditioning), OR
    # - Using cross-attention for anatomical conditioning
    enable_with_conditioning = model_cfg.with_conditioning or use_cross_attention_anatomical

    # 4. Create Model
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
        with_conditioning=enable_with_conditioning,
        cross_attention_dim=cross_attention_dim if use_cross_attention_anatomical else None,
        dropout_cattn=model_cfg.dropout,
    )

    # Replace class embedding with custom sinusoidal embedding if enabled
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

    # 5. Build AnatomicalPriorEncoder if using cross-attention method
    if use_cross_attention_anatomical:
        # Load encoder config (from separate file or inline)
        encoder_cfg = load_encoder_config(cfg)

        # Determine encoder version
        encoder_version = encoder_cfg.get("version", "legacy")

        if encoder_version == "enhanced":
            # Build enhanced encoder with FPN and RoPE
            input_size = tuple(cfg.data.transforms.roi_size[:2])
            anatomical_encoder = build_enhanced_anatomical_encoder(
                encoder_cfg, cross_attention_dim, input_size
            )
            logger.info(
                f"Built EnhancedAnatomicalPriorEncoder: "
                f"in_channels={anatomical_encoder.in_channels}, "
                f"seq_len={anatomical_encoder.seq_len}"
            )
        else:
            # Build legacy encoder (default for backward compatibility)
            anatomical_encoder = build_anatomical_encoder(cfg, cross_attention_dim)

        encoder_params = sum(p.numel() for p in anatomical_encoder.parameters())
        logger.info(f"Built AnatomicalPriorEncoder ({encoder_version}): {encoder_params:,} params")

    # Log model info
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Built DiffusionModelUNet: {n_params:,} params. "
        f"Anatomical Conditioning: {use_anatomical_conditioning} "
        f"(method: {anatomical_method if use_anatomical_conditioning else 'N/A'})"
    )

    return model, anatomical_encoder


def build_scheduler(cfg: DictConfig) -> DDPMScheduler | DDIMScheduler:
    """Build the diffusion scheduler from configuration.

    Args:
        cfg: Configuration object.

    Returns:
        Configured scheduler instance.
    """
    sched_cfg = cfg.scheduler

    # Base arguments
    kwargs = {
        "num_train_timesteps": sched_cfg.num_train_timesteps,
        "schedule": sched_cfg.schedule,
        "prediction_type": sched_cfg.prediction_type,
        "clip_sample": sched_cfg.clip_sample,
    }

    # Handle clipping range
    if sched_cfg.clip_sample and "clip_sample_range" in sched_cfg:
        kwargs["clip_sample_min"] = -sched_cfg.clip_sample_range
        kwargs["clip_sample_max"] = sched_cfg.clip_sample_range

    # Handle schedule-specific arguments
    if sched_cfg.schedule == "sigmoid_beta":
        kwargs["beta_start"] = sched_cfg.beta_start
        kwargs["beta_end"] = sched_cfg.beta_end
        kwargs["sig_range"] = sched_cfg.sig_range
    elif sched_cfg.schedule != "cosine":
        # linear_beta, scaled_linear_beta
        kwargs["beta_start"] = sched_cfg.beta_start
        kwargs["beta_end"] = sched_cfg.beta_end

    if sched_cfg.type == "DDPM":
        scheduler = DDPMScheduler(**kwargs)
    elif sched_cfg.type == "DDIM":
        scheduler = DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_cfg.type}")

    logger.info(
        f"Built {sched_cfg.type} scheduler: "
        f"T={sched_cfg.num_train_timesteps}, "
        f"schedule={sched_cfg.schedule}"
    )

    return scheduler


def build_inferer(cfg: DictConfig) -> DDPMScheduler | DDIMScheduler:
    """Build inference scheduler from configuration.

    Uses cfg.sampler.type to determine scheduler type (DDIM or DDPM),
    but inherits beta schedule parameters from cfg.scheduler to ensure
    consistency with the training scheduler.

    Args:
        cfg: Configuration object.

    Returns:
        Configured inference scheduler instance.
    """
    sched_cfg = cfg.scheduler
    sampler_cfg = cfg.sampler

    # Base arguments from training scheduler config
    kwargs = {
        "num_train_timesteps": sched_cfg.num_train_timesteps,
        "schedule": sched_cfg.schedule,
        "prediction_type": sched_cfg.prediction_type,
        "clip_sample": sched_cfg.clip_sample,
    }

    # Handle clipping range
    if sched_cfg.clip_sample and "clip_sample_range" in sched_cfg:
        kwargs["clip_sample_min"] = -sched_cfg.clip_sample_range
        kwargs["clip_sample_max"] = sched_cfg.clip_sample_range

    # Handle schedule-specific arguments
    if sched_cfg.schedule == "sigmoid_beta":
        kwargs["beta_start"] = sched_cfg.beta_start
        kwargs["beta_end"] = sched_cfg.beta_end
        kwargs["sig_range"] = sched_cfg.sig_range
    elif sched_cfg.schedule != "cosine":
        # linear_beta, scaled_linear_beta
        kwargs["beta_start"] = sched_cfg.beta_start
        kwargs["beta_end"] = sched_cfg.beta_end

    # Build scheduler based on sampler type (not training scheduler type)
    if sampler_cfg.type == "DDIM":
        inferer = DDIMScheduler(**kwargs)
    elif sampler_cfg.type == "DDPM":
        inferer = DDPMScheduler(**kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_cfg.type}")

    logger.info(
        f"Built {sampler_cfg.type} inferer for sampling: "
        f"T={sched_cfg.num_train_timesteps}, "
        f"schedule={sched_cfg.schedule}"
    )

    return inferer

class DiffusionSampler:
    """Wrapper for DDIM/DDPM sampling with optional anatomical and self conditioning.

    Supports two anatomical conditioning methods:
    - "concat": Concatenates prior mask as input channel
    - "cross_attention": Encodes prior via AnatomicalPriorEncoder and uses cross-attention

    When using cross_attention, supports both legacy and enhanced encoders:
    - Legacy: AnatomicalPriorEncoder (simple CNN, single-channel input)
    - Enhanced: EnhancedAnatomicalPriorEncoder (FPN, RoPE, multi-channel input)
    """

    def __init__(
        self,
        model: DiffusionModelUNet,
        scheduler: DDIMScheduler | DDPMScheduler,
        cfg: DictConfig,
        device: torch.device | str = "cuda",
        anatomical_encoder: EncoderType | None = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            model: The DiffusionModelUNet.
            scheduler: The diffusion scheduler (DDIM or DDPM).
            cfg: Configuration object.
            device: Target device for sampling.
            anatomical_encoder: Optional encoder for cross-attention anatomical conditioning.
                Required if anatomical_conditioning_method is "cross_attention".
                Can be either AnatomicalPriorEncoder (legacy) or
                EnhancedAnatomicalPriorEncoder (enhanced).
        """
        self.model = model
        self.scheduler = scheduler
        self.sampler_cfg = cfg.sampler
        self.cond_cfg = cfg.conditioning
        self.device = device

        # Capture the conditioning flags
        self.use_anatomical_conditioning = cfg.model.get("anatomical_conditioning", False)
        self.anatomical_method: AnatomicalConditioningMethod = cfg.model.get(
            "anatomical_conditioning_method", "concat"
        )
        self.use_self_conditioning = cfg.training.self_conditioning.get("enabled", False)

        # Store anatomical encoder for cross-attention method
        self.anatomical_encoder = anatomical_encoder
        if (
            self.use_anatomical_conditioning
            and self.anatomical_method == "cross_attention"
            and self.anatomical_encoder is None
        ):
            raise ValueError(
                "anatomical_encoder is required when using cross_attention method"
            )

        # Move encoder to device if present
        if self.anatomical_encoder is not None:
            self.anatomical_encoder = self.anatomical_encoder.to(device)
            self.anatomical_encoder.eval()

        self.num_inference_steps = self.sampler_cfg.num_inference_steps
        self.eta = self.sampler_cfg.eta
        self.guidance_scale = self.sampler_cfg.guidance_scale
        self.null_token = self.cond_cfg.z_bins * 2 if self.cond_cfg.cfg.enabled else None

        # Spatial size from config (roi_size is [H, W, D] for 3D volumes, use H, W for 2D slices)
        roi_size = cfg.data.transforms.roi_size
        self.spatial_size: tuple[int, int] = (int(roi_size[0]), int(roi_size[1]))

    @torch.no_grad()
    def sample(
        self,
        tokens: torch.Tensor,
        shape: tuple[int, ...] | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
        anatomical_mask: torch.Tensor | None = None,
        x_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples.

        Args:
            tokens: (B,) conditioning tokens.
            shape: Output shape (B, C, H, W).
            guidance_scale: CFG scale. If None, uses default from config.
            generator: Optional torch.Generator for reproducible noise.
            anatomical_mask: (B, 1, H, W) Tensor. Required if anatomical_conditioning is True.
                For "concat" method: concatenated as input channel.
                For "cross_attention" method: encoded and used as cross-attention context.
            x_T: Optional (B, C, H, W) pre-generated initial noise. If provided,
                 generator is ignored and x_T is used directly. Enables deterministic
                 per-sample seeding for replica generation.

        Returns:
            Generated samples tensor (B, C, H, W).
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        if shape is None:
            H, W = self.spatial_size
            shape = (tokens.shape[0], 2, H, W)

        B = tokens.shape[0]
        if shape[0] != B:
            shape = (B,) + shape[1:]

        # Validation
        if self.use_anatomical_conditioning:
            if anatomical_mask is None:
                raise ValueError("Model expects 'anatomical_mask', but None provided.")
            if anatomical_mask.shape != (B, 1, shape[2], shape[3]):
                raise ValueError(
                    f"Mask shape {anatomical_mask.shape} mismatch. "
                    f"Expected ({B}, 1, {shape[2]}, {shape[3]})"
                )

        # Use pre-generated noise if provided, otherwise generate fresh noise
        if x_T is not None:
            if x_T.shape != shape:
                raise ValueError(f"x_T shape {x_T.shape} != expected {shape}")
            x_t = x_T.to(device=self.device, dtype=torch.float32)
        else:
            # Generate on CPU with generator for reproducibility, then move to device
            # (generator device must match randn device)
            if generator is not None:
                x_t = torch.randn(shape, generator=generator).to(self.device)
            else:
                x_t = torch.randn(shape, device=self.device)
        self.scheduler.set_timesteps(self.num_inference_steps)

        # Initialize self-conditioning signal (zeros for first step, then previous x0 estimate)
        x0_self_cond = torch.zeros_like(x_t) if self.use_self_conditioning else None

        # Pre-compute cross-attention context if using cross_attention method
        anatomical_context = None
        if (
            self.use_anatomical_conditioning
            and self.anatomical_method == "cross_attention"
        ):
            anatomical_context = self.anatomical_encoder(
                anatomical_mask.to(self.device)
            )  # (B, seq_len, embed_dim)

        for t in self.scheduler.timesteps:
            timesteps = torch.full((B,), t, device=self.device, dtype=torch.long)

            # 1. Prepare Model Inputs
            model_input = x_t

            # Self-conditioning: use previous x0 estimate (zeros on first step)
            if self.use_self_conditioning:
                model_input = torch.cat([model_input, x0_self_cond], dim=1)

            # Anatomical conditioning: concat method
            if (
                self.use_anatomical_conditioning
                and self.anatomical_method == "concat"
            ):
                # Concatenate mask to noisy input
                # x_t: [B, C, H, W], mask: [B, 1, H, W] -> input: [B, C+1, H, W]
                model_input = torch.cat([model_input, anatomical_mask], dim=1)

            # 2. Classifier-Free Guidance Logic
            if guidance_scale > 1.0 and self.null_token is not None:
                # Duplicate inputs for [Conditional, Unconditional] batching
                model_input_double = torch.cat([model_input, model_input], dim=0)
                t_double = torch.cat([timesteps, timesteps], dim=0)

                # Prepare tokens
                null_tokens = torch.full_like(tokens, self.null_token)
                tokens_double = torch.cat([tokens, null_tokens], dim=0)

                # Prepare context for cross-attention (if using)
                context_double = None
                if anatomical_context is not None:
                    context_double = torch.cat(
                        [anatomical_context, anatomical_context], dim=0
                    )

                # Forward pass
                noise_pred = self.model(
                    model_input_double,
                    timesteps=t_double,
                    context=context_double,
                    class_labels=tokens_double,
                )

                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                # Standard Forward Pass
                noise_pred = self.model(
                    model_input,
                    timesteps=timesteps,
                    context=anatomical_context,
                    class_labels=tokens,
                )

            # 3. Update self-conditioning signal with current x0 estimate
            if self.use_self_conditioning:
                # Compute x0 estimate from current noise prediction for next iteration
                x0_self_cond = predict_x0(x_t, noise_pred, self.scheduler, timesteps)
                # Clamp to valid range to prevent instability
                x0_self_cond = torch.clamp(x0_self_cond, -1.0, 1.0)

            # 4. Scheduler Step
            if isinstance(self.scheduler, DDIMScheduler):
                x_t, _ = self.scheduler.step(noise_pred, t, x_t, eta=self.eta)
            else:
                x_t, _ = self.scheduler.step(noise_pred, t, x_t)

        return x_t

    def sample_single(
        self,
        token: int,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
        anatomical_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a single sample with optional mask."""
        tokens = torch.tensor([token], device=self.device, dtype=torch.long)

        # Handle single-item batch for mask
        if anatomical_mask is not None and anatomical_mask.dim() == 3:
            anatomical_mask = anatomical_mask.unsqueeze(0)

        H, W = self.spatial_size
        sample = self.sample(
            tokens,
            (1, 2, H, W),
            guidance_scale,
            generator,
            anatomical_mask=anatomical_mask,
        )
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
    model_output: torch.Tensor,
    scheduler: DDPMScheduler,
    t: torch.Tensor,
) -> torch.Tensor:
    """Predict x0 from x_t and model output.

    Supports all prediction types:
    - epsilon: x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
    - sample: x0_hat = model_output (direct prediction)
    - v_prediction: x0_hat = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v_pred

    Args:
        x_t: Noisy samples at timestep t.
        model_output: Model prediction (epsilon, x0, or v depending on prediction_type).
        scheduler: The DDPM/DDIM scheduler (must have prediction_type attribute).
        t: Timesteps.

    Returns:
        Predicted x0.
    """
    prediction_type = scheduler.prediction_type

    # Handle direct sample prediction (no computation needed)
    if prediction_type == "sample":
        return model_output

    alpha_bar_t = get_alpha_bar(scheduler, t)

    # Reshape for broadcasting (B,) -> (B, 1, 1, 1)
    while alpha_bar_t.dim() < x_t.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)

    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

    if prediction_type == "epsilon":
        x0_hat = (x_t - sqrt_one_minus_alpha_bar * model_output) / sqrt_alpha_bar
    elif prediction_type == "v_prediction":
        # v = sqrt(alpha_bar) * noise - sqrt(1-alpha_bar) * x0
        # Solving for x0: x0 = sqrt(alpha_bar) * x_t - sqrt(1-alpha_bar) * v
        x0_hat = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * model_output
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    return x0_hat
