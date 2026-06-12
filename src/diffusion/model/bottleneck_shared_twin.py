"""Bottleneck-Shared Twin DDPM: independent enc/dec, shared signal-coupled bottleneck.

Bottleneck-only point on the SLIM-Diff coupling continuum::

    zero coupling      → BOTTLENECK-ONLY → bottleneck-decoupled → full coupling
    (IndependentTwin)    (this variant)    (DecoupledMiddleBlock)  (shared UNet)

Architecture (forward orchestration)
------------------------------------
Two independent ``SplitForwardUNet`` instances process each channel of the
input separately through their own encoder. The deepest features from both
branches are concatenated along the channel dimension, processed by a
single shared ``_BottleneckPath`` middle block (which mixes the two
branches' signals through its convolutional layers), then split back to
feed two independent decoders::

    x[:, 0:1] → enc_img → h_img ┐                       ┌→ h_img' → dec_img → out_img
                                 ├ cat → shared_middle ─┤
    x[:, 1:2] → enc_mask → h_mask ┘                       └→ h_mask' → dec_mask → out_mask

The ``shared_middle`` is the **only** point of cross-branch coupling — by
design, this isolates the contribution of bottleneck-level coupling so
the 2×2 design table from the PRL pivot plan (§4.1) can be filled::

                       Bottleneck shared       Bottleneck independent
    Enc/dec shared     `shared`  (have)        `decoupled`     (have)
    Enc/dec indep.     **this variant**        `zero-coupling` (have)

Forward interface is identical to ``DiffusionModelUNet`` and
``IndependentTwinDDPM`` (accepts and returns 2-channel tensors), so
``DiffusionSampler`` and ``generate_replicas`` work unmodified.

Sharing implementation
----------------------
``class_embedding`` is shared between both UNets via Python object
identity (same trick as ``independent_twin.py``). Each UNet's stock
``middle_block`` is replaced by an ``_IdentityMiddle`` no-op after
construction; the real middle lives on the outer wrapper as
``self.shared_middle`` and is invoked from this module's ``forward``.

Parameter budget
----------------
Per branch encoder/decoder at ``channels=[48,96,180,180]`` ≈ 11.6M.
Shared ``_BottleneckPath`` at ``2C=360`` channels ≈ 7.5M. Two cond
embedding instances share a single backing tensor (≈ 150K). Total
≈ 30.8M — overshoots the 26.9M parity target by ~14%, so by default the
config uses ``channels=[44, 88, 160, 160]`` with the ``bottleneck_shared``
sub-block to land near 26.9M. Final tuning is verified by
``test_bottleneck_shared_twin.py::test_parameter_count``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from monai.networks.nets.diffusion_model_unet import (
    DiffusionModelUNet,
    get_timestep_embedding,
)
from omegaconf import DictConfig

from src.diffusion.model.decoupled_unet import _BottleneckPath
from src.diffusion.model.embeddings import ConditionalEmbeddingWithSinusoidal

logger = logging.getLogger(__name__)


class SplitForwardUNet(DiffusionModelUNet):
    """``DiffusionModelUNet`` exposing encoder and decoder phases as methods.

    The stock ``DiffusionModelUNet.forward`` is monolithic — it runs
    ``conv_in → down_blocks → middle_block → up_blocks → out`` in a single
    method. To insert a cross-branch shared middle block between two
    independent UNets, the encoder and decoder halves are needed as
    separate callables. This subclass exposes them without altering any
    layer or initialisation behaviour, so checkpoints from any
    ``DiffusionModelUNet`` instance load directly.

    Notes
    -----
    The ``middle_block`` attribute is replaced with an ``_IdentityMiddle``
    by ``BottleneckSharedTwinDDPM`` after construction. The real middle
    lives on the parent wrapper.
    """

    def encoder_forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Run time-emb + class-emb + ``conv_in`` + ``down_blocks``.

        Returns
        -------
        h : Tensor
            Deepest feature map ready for the middle block.
        residuals : list of Tensor
            Skip-connection feature maps; consumed in reverse by
            ``decoder_forward``.
        emb : Tensor
            Combined ``(time + class)`` embedding, passed through to the
            decoder so both halves see the same conditioning signal.
        """
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )
            class_emb = self.class_embedding(class_labels).to(dtype=x.dtype)
            emb = emb + class_emb

        h = self.conv_in(x)

        if context is not None and self.with_conditioning is False:
            raise ValueError(
                "model should have with_conditioning = True if context is provided"
            )

        residuals: list[torch.Tensor] = [h]
        for down_block in self.down_blocks:
            h, res_samples = down_block(hidden_states=h, temb=emb, context=context)
            residuals.extend(res_samples)

        return h, residuals, emb

    def decoder_forward(
        self,
        h: torch.Tensor,
        residuals: list[torch.Tensor],
        emb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run ``up_blocks`` (consuming residuals in reverse) and ``out``."""
        for up_block in self.up_blocks:
            idx = -len(up_block.resnets)  # type: ignore[attr-defined]
            res_samples = residuals[idx:]
            residuals = residuals[:idx]
            h = up_block(
                hidden_states=h,
                res_hidden_states_list=res_samples,
                temb=emb,
                context=context,
            )

        return self.out(h)


class _IdentityMiddle(nn.Module):
    """No-op replacement for each child UNet's stock middle block.

    The real middle lives on the outer ``BottleneckSharedTwinDDPM``
    wrapper. This placeholder ensures any code path that happens to call
    a child UNet's ``forward`` directly (e.g. instrumentation in tests)
    does not double-process the bottleneck.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hidden_states


class BottleneckSharedTwinDDPM(nn.Module):
    """Two independent UNets coupled through a single shared bottleneck.

    Parameters
    ----------
    image_unet : SplitForwardUNet
        Single-channel UNet for the FLAIR image (in=1, out=1).
        Its ``middle_block`` is replaced with ``_IdentityMiddle`` here.
    mask_unet : SplitForwardUNet
        Single-channel UNet for the lesion mask (in=1, out=1).
        Its ``middle_block`` is replaced with ``_IdentityMiddle`` here.
    shared_middle : nn.Module
        Module with signature ``(hidden_states, temb, context=None) →
        Tensor``, accepting input of shape ``(B, 2C, H', W')`` and
        returning the same shape. Typically a ``_BottleneckPath``.
    """

    def __init__(
        self,
        image_unet: SplitForwardUNet,
        mask_unet: SplitForwardUNet,
        shared_middle: nn.Module,
    ) -> None:
        super().__init__()
        self.image_unet = image_unet
        self.mask_unet = mask_unet
        self.shared_middle = shared_middle
        self.image_unet.middle_block = _IdentityMiddle()
        self.mask_unet.middle_block = _IdentityMiddle()

    @property
    def cond_embed(self) -> nn.Module:
        """Shared conditioning embedding (lives on ``image_unet``)."""
        return self.image_unet.class_embedding

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Orchestrate enc_img + enc_mask → shared_middle → dec_img + dec_mask.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, 2, H, W)`` — channel 0 is the FLAIR
            image, channel 1 is the lesion mask.
        timesteps : Tensor
            Diffusion timesteps, shape ``(B,)``.
        context : Tensor or None
            Cross-attention context (unused when ``with_conditioning=False``).
        class_labels : Tensor or None
            Conditioning tokens, shape ``(B,)``.

        Returns
        -------
        Tensor
            Concatenated predictions of shape ``(B, 2, H, W)``.
        """
        x_image = x[:, 0:1]
        x_mask = x[:, 1:2]

        h_img, skips_img, emb = self.image_unet.encoder_forward(
            x_image,
            timesteps=timesteps,
            context=context,
            class_labels=class_labels,
        )
        h_msk, skips_msk, _ = self.mask_unet.encoder_forward(
            x_mask,
            timesteps=timesteps,
            context=context,
            class_labels=class_labels,
        )

        c = h_img.shape[1]
        h_joint = torch.cat([h_img, h_msk], dim=1)
        h_joint = self.shared_middle(h_joint, emb, context)
        h_img_post = h_joint[:, :c]
        h_msk_post = h_joint[:, c:]

        out_img = self.image_unet.decoder_forward(
            h_img_post, skips_img, emb, context=context,
        )
        out_msk = self.mask_unet.decoder_forward(
            h_msk_post, skips_msk, emb, context=context,
        )

        return torch.cat([out_img, out_msk], dim=1)


# ---------------------------------------------------------------------
# Helpers for shared-middle hyperparameter selection
# ---------------------------------------------------------------------


def _largest_divisor_le(n: int, bound: int) -> int:
    """Largest positive divisor of ``n`` that is ≤ ``bound``."""
    if bound <= 0:
        raise ValueError(f"bound must be positive, got {bound}.")
    for d in range(bound, 0, -1):
        if n % d == 0:
            return d
    raise RuntimeError("unreachable: 1 divides every integer")


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------


def build_bottleneck_shared_twin(cfg: DictConfig) -> BottleneckSharedTwinDDPM:
    """Build a ``BottleneckSharedTwinDDPM`` from configuration.

    Reads
    -----
    cfg.model.channels, attention_levels, num_res_blocks, num_head_channels,
        norm_num_groups, resblock_updown, use_class_embedding,
        with_conditioning, dropout, spatial_dims
    cfg.model.bottleneck_shared (optional sub-block):
        extra_resnet_blocks : int (default 0)
        norm_num_groups_joint : int | None (default: auto-select)
        num_head_channels_joint : int | None (default: auto-select)
    cfg.conditioning.{z_bins, use_sinusoidal, max_z, cfg.enabled}
    cfg.data.slice_sampling.z_range

    Returns
    -------
    BottleneckSharedTwinDDPM
        With two independent ``SplitForwardUNet`` instances and one
        shared ``_BottleneckPath`` at ``2C`` channels.
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

    image_unet = SplitForwardUNet(**unet_kwargs)
    mask_unet = SplitForwardUNet(**unet_kwargs)

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
        mask_unet.class_embedding = image_unet.class_embedding

    # Shared bottleneck at 2C channels (signal coupling across branches).
    bottleneck_c = int(channels[-1])
    joint_c = 2 * bottleneck_c
    time_embed_dim = int(channels[0]) * 4

    bs_cfg = model_cfg.get("bottleneck_shared", {}) or {}
    extra_resnet_blocks = int(bs_cfg.get("extra_resnet_blocks", 0))

    # GroupNorm groups must divide joint_c. Inherit base norm_num_groups if
    # it divides cleanly; otherwise pick the largest valid divisor ≤ that
    # value (deterministic, never silently changes capacity).
    base_groups = int(model_cfg.norm_num_groups)
    requested_groups = bs_cfg.get("norm_num_groups_joint", None)
    if requested_groups is not None:
        norm_num_groups_joint = int(requested_groups)
        if joint_c % norm_num_groups_joint != 0:
            raise ValueError(
                f"bottleneck_shared.norm_num_groups_joint={norm_num_groups_joint} "
                f"does not divide joint channels {joint_c}."
            )
    elif joint_c % base_groups == 0:
        norm_num_groups_joint = base_groups
    else:
        norm_num_groups_joint = _largest_divisor_le(joint_c, base_groups)

    # Attention head channels must divide joint_c. Same fall-back logic.
    base_heads = int(model_cfg.num_head_channels)
    requested_heads = bs_cfg.get("num_head_channels_joint", None)
    if requested_heads is not None:
        num_head_channels_joint = int(requested_heads)
        if joint_c % num_head_channels_joint != 0:
            raise ValueError(
                f"bottleneck_shared.num_head_channels_joint={num_head_channels_joint} "
                f"does not divide joint channels {joint_c}."
            )
    elif joint_c % base_heads == 0:
        num_head_channels_joint = base_heads
    else:
        num_head_channels_joint = _largest_divisor_le(joint_c, base_heads)

    shared_middle = _BottleneckPath(
        spatial_dims=int(model_cfg.spatial_dims),
        channels=joint_c,
        temb_channels=time_embed_dim,
        norm_num_groups=norm_num_groups_joint,
        norm_eps=1e-6,
        with_conditioning=bool(model_cfg.with_conditioning),
        num_head_channels=num_head_channels_joint,
        transformer_num_layers=int(model_cfg.get("transformer_num_layers", 1)),
        cross_attention_dim=None,
        extra_resnet_blocks=extra_resnet_blocks,
        upcast_attention=bool(model_cfg.get("upcast_attention", False)),
        dropout_cattn=float(model_cfg.get("dropout", 0.0)),
        include_fc=bool(model_cfg.get("include_fc", True)),
        use_combined_linear=bool(model_cfg.get("use_combined_linear", False)),
        use_flash_attention=bool(model_cfg.get("use_flash_attention", False)),
    )

    model = BottleneckSharedTwinDDPM(image_unet, mask_unet, shared_middle)

    unique_params = {id(p): p for p in model.parameters()}
    n_unique = sum(p.numel() for p in unique_params.values())
    n_shared_middle = sum(p.numel() for p in shared_middle.parameters())
    n_cond_embed = sum(p.numel() for p in model.cond_embed.parameters())
    logger.info(
        "Built BottleneckSharedTwinDDPM: %s unique params (%d tensors). "
        "Shared middle at joint_c=%d (groups=%d, heads=%d): %s params. "
        "Shared cond_embed: %s params.",
        f"{n_unique:,}",
        len(unique_params),
        joint_c,
        norm_num_groups_joint,
        num_head_channels_joint,
        f"{n_shared_middle:,}",
        f"{n_cond_embed:,}",
    )

    return model


__all__ = [
    "BottleneckSharedTwinDDPM",
    "SplitForwardUNet",
    "build_bottleneck_shared_twin",
]
