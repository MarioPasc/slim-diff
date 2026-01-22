"""Feature Pyramid Network (FPN) backbone for multi-scale feature extraction.

Implements a lightweight FPN suitable for encoding anatomical prior maps.
The FPN extracts features at multiple scales and fuses them for rich
spatial representations.

Reference:
    Lin et al. (2017) "Feature Pyramid Networks for Object Detection"
    https://arxiv.org/abs/1612.03144
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolution block with normalization and activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        norm_num_groups: Number of groups for GroupNorm.
        activation: Whether to apply activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_num_groups: int = 8,
        activation: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        # Ensure norm_num_groups doesn't exceed out_channels
        actual_groups = min(norm_num_groups, out_channels)
        self.norm = nn.GroupNorm(actual_groups, out_channels)
        self.activation = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for downsampling (1 or 2).
        norm_num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_num_groups: int = 8,
    ):
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_num_groups=norm_num_groups,
        )
        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_num_groups=norm_num_groups,
            activation=False,
        )

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(min(norm_num_groups, out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.activation(out)


class FPNBackbone(nn.Module):
    """Feature Pyramid Network backbone for multi-scale feature extraction.

    This is a lightweight FPN designed for encoding anatomical prior maps.
    It uses a bottom-up pathway to extract features at multiple scales,
    then a top-down pathway with lateral connections to combine them.

    The total downsampling factor is 2^(num_stages), so for 2 stages we get 4x.

    Args:
        in_channels: Number of input channels (e.g., 1 for binary, 5 for tissue).
        hidden_dims: Tuple of hidden dimensions for each stage.
        out_channels: Number of output channels (embed_dim).
        downsample_factor: Total downsampling factor (must be power of 2).
        norm_num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: Tuple[int, ...] = (64, 128),
        out_channels: int = 256,
        downsample_factor: int = 4,
        norm_num_groups: int = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.downsample_factor = downsample_factor
        self.norm_num_groups = norm_num_groups

        # Calculate number of stages from downsample_factor
        # downsample_factor = 2^num_stages
        import math
        self.num_stages = int(math.log2(downsample_factor))
        assert 2 ** self.num_stages == downsample_factor, \
            f"downsample_factor must be power of 2, got {downsample_factor}"
        assert len(hidden_dims) >= self.num_stages, \
            f"hidden_dims must have at least {self.num_stages} elements"

        # ===== Bottom-up pathway =====
        # Initial convolution (no downsampling)
        self.stem = ConvBlock(
            in_channels,
            hidden_dims[0],
            kernel_size=7,
            stride=1,
            padding=3,
            norm_num_groups=norm_num_groups,
        )

        # Downsampling stages
        self.stages = nn.ModuleList()
        prev_channels = hidden_dims[0]
        for i in range(self.num_stages):
            stage_channels = hidden_dims[i] if i < len(hidden_dims) else hidden_dims[-1]
            self.stages.append(
                ResidualBlock(
                    prev_channels,
                    stage_channels,
                    stride=2,  # Each stage downsamples by 2x
                    norm_num_groups=norm_num_groups,
                )
            )
            prev_channels = stage_channels

        # ===== Lateral connections (1x1 convs) =====
        # Project each stage output to out_channels
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_stages):
            stage_channels = hidden_dims[i] if i < len(hidden_dims) else hidden_dims[-1]
            self.lateral_convs.append(
                nn.Conv2d(stage_channels, out_channels, kernel_size=1)
            )

        # ===== Top-down pathway =====
        # 3x3 convs after addition to reduce aliasing
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_stages):
            self.fpn_convs.append(
                ConvBlock(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_num_groups=norm_num_groups,
                )
            )

        # ===== Final fusion =====
        # Combine all FPN levels into a single output at lowest resolution
        self.fusion = nn.Sequential(
            ConvBlock(
                out_channels * self.num_stages,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_num_groups=norm_num_groups,
            ),
            ConvBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_num_groups=norm_num_groups,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FPN backbone.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature tensor of shape (B, out_channels, H/ds, W/ds)
            where ds is the downsample_factor.
        """
        # ===== Bottom-up pathway =====
        x = self.stem(x)  # (B, hidden_dims[0], H, W)

        # Store stage outputs for lateral connections
        stage_outputs: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            stage_outputs.append(x)

        # ===== Top-down pathway with lateral connections =====
        fpn_outputs: List[torch.Tensor] = []

        # Start from the top (smallest spatial size)
        prev_fpn = self.lateral_convs[-1](stage_outputs[-1])
        prev_fpn = self.fpn_convs[-1](prev_fpn)
        fpn_outputs.append(prev_fpn)

        # Process remaining stages (top-down)
        for i in range(self.num_stages - 2, -1, -1):
            # Lateral connection
            lateral = self.lateral_convs[i](stage_outputs[i])

            # Upsample previous FPN output and add lateral
            upsampled = F.interpolate(
                prev_fpn,
                size=lateral.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            prev_fpn = lateral + upsampled
            prev_fpn = self.fpn_convs[i](prev_fpn)
            fpn_outputs.append(prev_fpn)

        # Reverse to get outputs from low to high resolution
        fpn_outputs = fpn_outputs[::-1]

        # ===== Fusion: downsample all to lowest resolution and concat =====
        target_size = fpn_outputs[-1].shape[2:]  # Smallest spatial size
        fused_features = []
        for fpn_out in fpn_outputs:
            if fpn_out.shape[2:] != target_size:
                fpn_out = F.interpolate(
                    fpn_out,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            fused_features.append(fpn_out)

        # Concatenate and fuse
        fused = torch.cat(fused_features, dim=1)  # (B, out_channels * num_stages, H', W')
        out = self.fusion(fused)  # (B, out_channels, H', W')

        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"hidden_dims={self.hidden_dims}, "
            f"out_channels={self.out_channels}, "
            f"downsample_factor={self.downsample_factor}"
        )


class SimpleCNNBackbone(nn.Module):
    """Simple CNN backbone as fallback alternative to FPN.

    Uses a series of strided convolutions for downsampling without
    the multi-scale feature fusion of FPN.

    Args:
        in_channels: Number of input channels.
        hidden_dims: Tuple of hidden dimensions for each stage.
        out_channels: Number of output channels.
        downsample_factor: Total downsampling factor (must be power of 2).
        norm_num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
        out_channels: int = 256,
        downsample_factor: int = 8,
        norm_num_groups: int = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.downsample_factor = downsample_factor

        import math
        num_stages = int(math.log2(downsample_factor))
        assert 2 ** num_stages == downsample_factor

        layers: List[nn.Module] = []

        # Initial conv (no downsampling)
        layers.append(
            ConvBlock(
                in_channels,
                hidden_dims[0],
                kernel_size=7,
                stride=1,
                padding=3,
                norm_num_groups=norm_num_groups,
            )
        )

        # Downsampling stages
        prev_channels = hidden_dims[0]
        for i in range(num_stages):
            stage_channels = hidden_dims[i] if i < len(hidden_dims) else hidden_dims[-1]
            layers.append(
                ResidualBlock(
                    prev_channels,
                    stage_channels,
                    stride=2,
                    norm_num_groups=norm_num_groups,
                )
            )
            prev_channels = stage_channels

        # Final projection to output channels
        layers.append(
            ConvBlock(
                prev_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_num_groups=norm_num_groups,
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through simple CNN backbone.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature tensor of shape (B, out_channels, H/ds, W/ds).
        """
        return self.layers(x)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"hidden_dims={self.hidden_dims}, "
            f"out_channels={self.out_channels}, "
            f"downsample_factor={self.downsample_factor}"
        )


def build_backbone(
    backbone_type: str,
    in_channels: int,
    hidden_dims: Tuple[int, ...],
    out_channels: int,
    downsample_factor: int = 4,
    norm_num_groups: int = 8,
) -> nn.Module:
    """Factory function for backbone networks.

    Args:
        backbone_type: Type of backbone ("fpn" or "simple").
        in_channels: Number of input channels.
        hidden_dims: Tuple of hidden dimensions.
        out_channels: Number of output channels.
        downsample_factor: Spatial downsampling factor.
        norm_num_groups: Number of groups for GroupNorm.

    Returns:
        Backbone network module.
    """
    if backbone_type == "fpn":
        return FPNBackbone(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            out_channels=out_channels,
            downsample_factor=downsample_factor,
            norm_num_groups=norm_num_groups,
        )
    elif backbone_type == "simple":
        return SimpleCNNBackbone(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            out_channels=out_channels,
            downsample_factor=downsample_factor,
            norm_num_groups=norm_num_groups,
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
