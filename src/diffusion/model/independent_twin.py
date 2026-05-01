"""Independent Twin DDPM: two single-channel DDPMs sharing only the conditioning embedding.

Zero-coupling baseline for the SASHIMI 2026 coupling-continuum ablation.
Each U-Net processes one channel (FLAIR image or lesion mask) independently;
the only shared component is the class_embedding that encodes (z_bin, pathology).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from omegaconf import DictConfig

from src.diffusion.model.embeddings import ConditionalEmbeddingWithSinusoidal

logger = logging.getLogger(__name__)


class IndependentTwinDDPM(nn.Module):
    """Two independent single-channel DDPMs with a shared conditioning embedding.

    Forward interface is identical to DiffusionModelUNet (accepts and returns
    2-channel tensors) so DiffusionSampler and generate_replicas work unmodified.

    Parameters
    ----------
    image_unet : DiffusionModelUNet
        Single-channel U-Net for the FLAIR image (in=1, out=1).
    mask_unet : DiffusionModelUNet
        Single-channel U-Net for the lesion mask (in=1, out=1).
    """

    def __init__(
        self,
        image_unet: DiffusionModelUNet,
        mask_unet: DiffusionModelUNet,
    ) -> None:
        super().__init__()
        self.image_unet = image_unet
        self.mask_unet = mask_unet

    @property
    def cond_embed(self) -> nn.Module:
        """The shared conditioning embedding (lives on image_unet)."""
        return self.image_unet.class_embedding

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass splitting input channels across two independent U-Nets.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, 2, H, W)`` — channel 0 is image,
            channel 1 is mask.
        timesteps : Tensor
            Diffusion timesteps, shape ``(B,)``.
        context : Tensor or None
            Cross-attention context (unused; kept for API compatibility).
        class_labels : Tensor or None
            Conditioning tokens, shape ``(B,)``.

        Returns
        -------
        Tensor
            Concatenated predictions, shape ``(B, 2, H, W)``.
        """
        x_image = x[:, 0:1]
        x_mask = x[:, 1:2]

        out_image = self.image_unet(
            x_image, timesteps=timesteps, context=context, class_labels=class_labels,
        )
        out_mask = self.mask_unet(
            x_mask, timesteps=timesteps, context=context, class_labels=class_labels,
        )

        return torch.cat([out_image, out_mask], dim=1)


def build_independent_twin(cfg: DictConfig) -> IndependentTwinDDPM:
    """Build an IndependentTwinDDPM from configuration.

    Creates two single-channel DiffusionModelUNet instances and one shared
    ConditionalEmbeddingWithSinusoidal, assigning the same Python object
    to both U-Nets' ``class_embedding`` attribute.

    Parameters
    ----------
    cfg : DictConfig
        Full experiment configuration.

    Returns
    -------
    IndependentTwinDDPM
        The constructed model.
    """
    model_cfg = cfg.model
    cond_cfg = cfg.conditioning

    channels = tuple(model_cfg.channels)
    attention_levels = tuple(model_cfg.attention_levels)
    z_bins = cond_cfg.z_bins

    num_class_embeds = 2 * z_bins
    if cond_cfg.cfg.enabled:
        num_class_embeds += 1

    unet_kwargs = dict(
        spatial_dims=model_cfg.spatial_dims,
        in_channels=1,
        out_channels=1,
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

    image_unet = DiffusionModelUNet(**unet_kwargs)
    mask_unet = DiffusionModelUNet(**unet_kwargs)

    if cond_cfg.use_sinusoidal and model_cfg.use_class_embedding:
        embedding_dim = image_unet.class_embedding.embedding_dim
        z_range = tuple(cfg.data.slice_sampling.z_range)

        shared_embed = ConditionalEmbeddingWithSinusoidal(
            num_embeddings=num_class_embeds,
            embedding_dim=embedding_dim,
            z_bins=z_bins,
            z_range=z_range,
            use_sinusoidal=True,
            max_z=cond_cfg.max_z,
            use_cfg=cond_cfg.cfg.enabled,
        )
        image_unet.class_embedding = shared_embed
        mask_unet.class_embedding = shared_embed

    elif model_cfg.use_class_embedding:
        # Non-sinusoidal: share the stock nn.Embedding across both U-Nets
        mask_unet.class_embedding = image_unet.class_embedding

    model = IndependentTwinDDPM(image_unet, mask_unet)

    n_params = sum(p.numel() for p in model.parameters())
    n_params_dedup = len({id(p): p for p in model.parameters()})
    logger.info(
        "Built IndependentTwinDDPM: %s unique params (%s total tensors). "
        "Shared cond_embed: %s params.",
        f"{n_params:,}",
        n_params_dedup,
        f"{sum(p.numel() for p in model.cond_embed.parameters()):,}",
    )

    return model
