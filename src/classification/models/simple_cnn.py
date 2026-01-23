"""CNN classifiers for real vs. synthetic discrimination.

Intentionally simple architectures to measure inherent distinguishability
rather than classifier power. Lower AUC = higher synthetic quality.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNNClassifier(nn.Module):
    """Simple CNN for binary classification.

    Architecture: ConvBlocks → GAP → FC → Logit

    Args:
        in_channels: Number of input channels (1 or 2).
        channels: Channel progression for conv blocks.
        fc_dim: Hidden dimension before output.
        dropout: Dropout probability.
        use_batch_norm: Whether to use batch normalization.
    """

    def __init__(
        self,
        in_channels: int = 2,
        channels: list[int] | None = None,
        fc_dim: int = 256,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        conv_blocks: list[nn.Module] = []
        prev_ch = in_channels
        for ch in channels:
            conv_blocks.append(ConvBlock(prev_ch, ch, use_batch_norm))
            prev_ch = ch
        self.conv_layers = nn.Sequential(*conv_blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits of shape (B, 1)."""
        x = self.conv_layers(x)
        x = self.global_pool(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Basic residual block with optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=not use_batch_norm
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=not use_batch_norm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            shortcut_layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            ]
            if use_batch_norm:
                shortcut_layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResNetClassifier(nn.Module):
    """ResNet-style classifier for more discriminative evaluation.

    Args:
        in_channels: Number of input channels (1 or 2).
        channels: Channel progression per stage.
        num_blocks: Residual blocks per stage.
        fc_dim: Hidden dimension before output.
        dropout: Dropout probability.
        use_batch_norm: Whether to use batch normalization.
    """

    def __init__(
        self,
        in_channels: int = 2,
        channels: list[int] | None = None,
        num_blocks: int = 2,
        fc_dim: int = 256,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        self.conv1 = nn.Conv2d(in_channels, channels[0], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        stages: list[nn.Module] = []
        prev_ch = channels[0]
        for i, ch in enumerate(channels):
            stride = 2 if i > 0 else 1
            blocks: list[nn.Module] = [ResidualBlock(prev_ch, ch, stride=stride, use_batch_norm=use_batch_norm)]
            for _ in range(1, num_blocks):
                blocks.append(ResidualBlock(ch, ch, use_batch_norm=use_batch_norm))
            stages.append(nn.Sequential(*blocks))
            prev_ch = ch
        self.stages = nn.Sequential(*stages)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits of shape (B, 1)."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.stages(x)
        x = self.global_pool(x)
        return self.classifier(x)
