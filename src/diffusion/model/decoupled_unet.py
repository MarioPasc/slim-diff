"""Decoupled-bottleneck variant of MONAI's DiffusionModelUNet middle block.

Context
-------
SLIM-Diff's central architectural claim is that the shared bottleneck (a
single ``ResBlock → Attn → ResBlock`` at the deepest level) acts as a joint
representation that regularises image+mask synthesis in the low-data
regime. This module provides the ICIP 2026 camera-ready ablation that
tests the claim by replacing the shared block with two *independent*
half-channel paths.

Architecture
------------
Given an input feature map ``h`` at ``C = channels[-1]`` channels
(e.g. 256 for the paper config), the decoupled block computes::

    h_split = proj_split(h)                        # Conv 1x1, C -> 2*c
    h_a, h_b = h_split[:, :c], h_split[:, c:]       # split along channels
    h_a = path_a(h_a, temb, context)                # resnet-attn-resnet
    h_b = path_b(h_b, temb, context)                # independent weights
    out = proj_merge(cat([h_a, h_b], dim=1))        # Conv 1x1, 2*c -> C

where ``c = channels_per_path`` (default ``C // 2``). The two paths share
no parameters; both receive the same ``(time + class)`` embedding
``temb`` via their own ``time_emb_proj`` weights -- identical to the
conditioning pathway used in MONAI's stock mid block.

Integration
-----------
The ``build_decoupled_middle_block`` helper is called by
``src.diffusion.model.factory.build_model`` when
``cfg.model.bottleneck_mode == "decoupled"``. The resulting block is
swapped in place of ``model.middle_block``; the rest of MONAI's forward
pass is unchanged, so the returned model remains a
``DiffusionModelUNet`` instance compatible with PyTorch-Lightning
checkpointing.

Camera-ready parameter match
----------------------------
On the paper's reference config (``channels=[64,128,256,256]``,
``norm_num_groups=32``, ``num_res_blocks=2``, ``num_head_channels=32``)
the shared variant totals ``26,894,210`` parameters. The closest
decoupled configuration within the 1 % target is::

    model:
      bottleneck_mode: "decoupled"
      decoupled_bottleneck:
        channels_per_path: null      # resolves to channels[-1] // 2 = 128
        extra_resnet_blocks: 2       # + 2 ResBlocks per path

which yields ``27,029,378`` parameters (Δ = +0.503 %, +135,168).
TASK-03 (``slurm/camera_ready/``) should bake these values into the
decoupled SLURM YAMLs. Measured 2026-04-13 by the sweep in
``docs/icip2026/rebuttal_plans/TASK_01_ARCH_decoupled_bottleneck.md``'s
verification section.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from monai.networks.blocks import SpatialAttentionBlock
from monai.networks.nets.diffusion_model_unet import (
    DiffusionUNetResnetBlock,
    SpatialTransformer,
)
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def count_middle_params(middle_block: nn.Module) -> int:
    """Total parameter count of a single mid block module."""
    return sum(p.numel() for p in middle_block.parameters())


class _BottleneckPath(nn.Module):
    """One independent ``resnet_1 → attention → [extras] → resnet_2`` path.

    Mirrors the per-layer structure of MONAI's ``AttnMidBlock`` /
    ``CrossAttnMidBlock`` but at the reduced per-path channel count.
    ``extra_resnet_blocks`` lets us pad depth to compensate for the
    parameter loss from halving channels, without widening the outer
    bottleneck.
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        temb_channels: int,
        norm_num_groups: int,
        norm_eps: float,
        with_conditioning: bool,
        num_head_channels: int,
        transformer_num_layers: int,
        cross_attention_dim: int | None,
        extra_resnet_blocks: int,
        upcast_attention: bool,
        dropout_cattn: float,
        include_fc: bool,
        use_combined_linear: bool,
        use_flash_attention: bool,
    ) -> None:
        super().__init__()

        self.resnet_1 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=channels,
            out_channels=channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

        if with_conditioning:
            if channels % num_head_channels != 0:
                raise ValueError(
                    f"channels_per_path ({channels}) must be divisible by "
                    f"num_head_channels ({num_head_channels}) when "
                    f"with_conditioning=True."
                )
            self.attention = SpatialTransformer(
                spatial_dims=spatial_dims,
                in_channels=channels,
                num_attention_heads=channels // num_head_channels,
                num_head_channels=num_head_channels,
                num_layers=transformer_num_layers,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                dropout=dropout_cattn,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )
        else:
            self.attention = SpatialAttentionBlock(
                spatial_dims=spatial_dims,
                num_channels=channels,
                num_head_channels=num_head_channels,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )

        self.extra_resnets = nn.ModuleList(
            [
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=channels,
                    out_channels=channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
                for _ in range(int(extra_resnet_blocks))
            ]
        )

        self.resnet_2 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=channels,
            out_channels=channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

        self._with_conditioning = with_conditioning

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.resnet_1(x, temb)
        if self._with_conditioning:
            x = self.attention(x, context=context)
        else:
            x = self.attention(x).contiguous()
        for block in self.extra_resnets:
            x = block(x, temb)
        x = self.resnet_2(x, temb)
        return x


class DecoupledMiddleBlock(nn.Module):
    """Drop-in replacement for MONAI's shared mid block.

    The replacement preserves MONAI's keyword-argument forward signature
    ``forward(hidden_states, temb, context=None)`` so the surrounding
    ``DiffusionModelUNet.forward`` code path is untouched.

    Parameters
    ----------
    spatial_dims, in_channels, temb_channels, norm_num_groups, norm_eps :
        Same meaning as in MONAI's ``get_mid_block`` -- taken from the
        current ``DiffusionModelUNet`` config.
    with_conditioning, num_head_channels, transformer_num_layers,
    cross_attention_dim, upcast_attention, dropout_cattn, include_fc,
    use_combined_linear, use_flash_attention :
        Same plumbing as ``get_mid_block``; forwarded to each path's
        attention module.
    channels_per_path :
        Number of channels used inside each independent path. Defaults
        to ``in_channels // 2`` (``requires in_channels`` even). The
        outer projections expand/contract as needed, so this knob is
        decoupled from the surrounding UNet.
    extra_resnet_blocks :
        Extra ``DiffusionUNetResnetBlock`` layers inserted between
        ``attention`` and ``resnet_2`` of each path. Used to compensate
        the parameter drop from halving channels (see module docstring).
    norm_num_groups_path :
        Optional override for group-norm groups inside the paths. If
        ``None`` (default), inherits from ``norm_num_groups``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_conditioning: bool = False,
        num_head_channels: int = 32,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        channels_per_path: int | None = None,
        extra_resnet_blocks: int = 0,
        norm_num_groups_path: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if channels_per_path is None:
            if in_channels % 2 != 0:
                raise ValueError(
                    f"in_channels ({in_channels}) must be even when "
                    f"channels_per_path is not specified."
                )
            channels_per_path = in_channels // 2
        if channels_per_path <= 0:
            raise ValueError(
                f"channels_per_path must be positive, got {channels_per_path}."
            )
        groups = int(norm_num_groups_path or norm_num_groups)
        if channels_per_path % groups != 0:
            raise ValueError(
                f"channels_per_path ({channels_per_path}) must be divisible "
                f"by norm_num_groups_path ({groups})."
            )
        if spatial_dims not in (2, 3):
            raise ValueError(
                f"spatial_dims must be 2 or 3, got {spatial_dims}."
            )

        self.in_channels = int(in_channels)
        self.channels_per_path = int(channels_per_path)
        self.split_channels = 2 * self.channels_per_path
        self._spatial_dims = int(spatial_dims)

        ConvNd = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.proj_split = ConvNd(
            self.in_channels, self.split_channels, kernel_size=1, bias=True
        )
        self.proj_merge = ConvNd(
            self.split_channels, self.in_channels, kernel_size=1, bias=True
        )

        path_kwargs = dict(
            spatial_dims=spatial_dims,
            channels=self.channels_per_path,
            temb_channels=int(temb_channels),
            norm_num_groups=groups,
            norm_eps=float(norm_eps),
            with_conditioning=bool(with_conditioning),
            num_head_channels=int(num_head_channels),
            transformer_num_layers=int(transformer_num_layers),
            cross_attention_dim=cross_attention_dim,
            extra_resnet_blocks=int(extra_resnet_blocks),
            upcast_attention=bool(upcast_attention),
            dropout_cattn=float(dropout_cattn),
            include_fc=bool(include_fc),
            use_combined_linear=bool(use_combined_linear),
            use_flash_attention=bool(use_flash_attention),
        )
        self.path_a = _BottleneckPath(**path_kwargs)
        self.path_b = _BottleneckPath(**path_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[1] != self.in_channels:
            raise ValueError(
                f"DecoupledMiddleBlock expects {self.in_channels} input "
                f"channels, got {hidden_states.shape[1]}."
            )
        h = self.proj_split(hidden_states)
        c = self.channels_per_path
        h_a = self.path_a(h[:, :c], temb, context)
        h_b = self.path_b(h[:, c:], temb, context)
        h = torch.cat([h_a, h_b], dim=1)
        return self.proj_merge(h)


def build_decoupled_middle_block(
    cfg: DictConfig,
    original_middle_block: nn.Module | None = None,
) -> DecoupledMiddleBlock:
    """Build a ``DecoupledMiddleBlock`` from a SLIM-Diff YAML config.

    Replicates the same ``get_mid_block`` plumbing
    ``factory.build_model`` uses so the block's conditioning behaviour
    matches the original middle block exactly.

    If ``original_middle_block`` is provided, logs the parameter count of
    both variants and their delta -- used by the factory for diagnostic
    output.
    """
    model_cfg = cfg.model
    in_channels = int(model_cfg.channels[-1])
    time_embed_dim = int(model_cfg.channels[0]) * 4  # matches DiffusionModelUNet

    use_cross_attn_anatomical = (
        bool(model_cfg.get("anatomical_conditioning", False))
        and model_cfg.get("anatomical_conditioning_method", "concat")
        == "cross_attention"
    )
    with_conditioning = (
        bool(model_cfg.get("with_conditioning", False))
        or use_cross_attn_anatomical
    )
    cross_attention_dim: int | None = None
    if with_conditioning:
        if use_cross_attn_anatomical:
            cross_attention_dim = int(
                model_cfg.get("cross_attention_dim", in_channels)
            )
        else:
            # Respect explicit cross_attention_dim if the user set
            # with_conditioning=True directly.
            cross_attention_dim = int(
                model_cfg.get("cross_attention_dim", in_channels)
            )

    num_head_channels = model_cfg.num_head_channels
    if isinstance(num_head_channels, (list, tuple)):
        num_head_channels = int(num_head_channels[-1])
    else:
        num_head_channels = int(num_head_channels)

    dec_cfg = model_cfg.get("decoupled_bottleneck", {}) or {}
    channels_per_path = dec_cfg.get("channels_per_path", None)
    extra_resnet_blocks = int(dec_cfg.get("extra_resnet_blocks", 0))
    norm_num_groups_path = dec_cfg.get("norm_num_groups_path", None)

    block = DecoupledMiddleBlock(
        spatial_dims=int(model_cfg.spatial_dims),
        in_channels=in_channels,
        temb_channels=time_embed_dim,
        norm_num_groups=int(model_cfg.norm_num_groups),
        norm_eps=1e-6,
        with_conditioning=with_conditioning,
        num_head_channels=num_head_channels,
        transformer_num_layers=int(model_cfg.get("transformer_num_layers", 1)),
        cross_attention_dim=cross_attention_dim,
        channels_per_path=channels_per_path,
        extra_resnet_blocks=extra_resnet_blocks,
        norm_num_groups_path=norm_num_groups_path,
        upcast_attention=bool(model_cfg.get("upcast_attention", False)),
        dropout_cattn=float(model_cfg.get("dropout", 0.0)),
        include_fc=bool(model_cfg.get("include_fc", True)),
        use_combined_linear=bool(model_cfg.get("use_combined_linear", False)),
        use_flash_attention=bool(model_cfg.get("use_flash_attention", False)),
    )

    if original_middle_block is not None:
        n_orig = count_middle_params(original_middle_block)
        n_new = count_middle_params(block)
        delta = (n_new - n_orig) / n_orig if n_orig else 0.0
        logger.info(
            "DecoupledMiddleBlock built: "
            "%d params (shared baseline %d params, Δ=%+.4f%%, "
            "channels_per_path=%d, extra_resnet_blocks=%d).",
            n_new,
            n_orig,
            100.0 * delta,
            block.channels_per_path,
            extra_resnet_blocks,
        )

    return block


__all__ = [
    "DecoupledMiddleBlock",
    "build_decoupled_middle_block",
    "count_middle_params",
]
