"""CNN classifier for real vs synthetic discrimination."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        use_batch_norm: Whether to use batch normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AuditionClassifier(nn.Module):
    """Simple CNN classifier for real vs synthetic discrimination.

    Architecture:
        - Sequence of ConvBlocks with increasing channels
        - Global Average Pooling
        - Fully connected layer for classification

    The model is intentionally simple to measure inherent distinguishability
    rather than classifier power.

    Args:
        in_channels: Number of input channels (2 for image + mask).
        channels: List of channel sizes for conv blocks.
        fc_dim: Dimension of FC layer before output.
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

        # Build convolutional blocks
        conv_blocks = []
        prev_channels = in_channels
        for ch in channels:
            conv_blocks.append(ConvBlock(prev_channels, ch, use_batch_norm))
            prev_channels = ch

        self.conv_layers = nn.Sequential(*conv_blocks)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_dim, 1),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits of shape (B, 1) for binary classification.
        """
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "AuditionClassifier":
        """Create classifier from configuration.

        Args:
            cfg: Configuration dictionary.

        Returns:
            Initialized classifier.
        """
        model_cfg = cfg.model
        return cls(
            in_channels=model_cfg.in_channels,
            channels=list(model_cfg.channels),
            fc_dim=model_cfg.fc_dim,
            dropout=model_cfg.dropout,
            use_batch_norm=model_cfg.use_batch_norm,
        )


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling.

    For more complex architectures if needed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not use_batch_norm
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_batch_norm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ResNetClassifier(nn.Module):
    """ResNet-style classifier for more challenging cases.

    Use this if the simple CNN is not discriminative enough.

    Args:
        in_channels: Number of input channels.
        channels: List of channel sizes for residual blocks.
        num_blocks: Number of residual blocks per stage.
        fc_dim: Dimension of FC layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 2,
        channels: list[int] | None = None,
        num_blocks: int = 2,
        fc_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        # Build residual stages
        stages = []
        prev_channels = channels[0]
        for i, ch in enumerate(channels):
            stride = 2 if i > 0 else 1
            blocks = [ResidualBlock(prev_channels, ch, stride=stride)]
            for _ in range(1, num_blocks):
                blocks.append(ResidualBlock(ch, ch))
            stages.append(nn.Sequential(*blocks))
            prev_channels = ch

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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.stages(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
