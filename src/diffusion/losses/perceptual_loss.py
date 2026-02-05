"""Perceptual loss module using LPIPS for diffusion models.

Provides LPIPS-based perceptual loss with:
- Offline VGG weights support for HPC environments
- Single-channel to 3-channel expansion for medical images
- Timestep-based gating to apply loss only at low noise levels
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VGGFeatureExtractor(nn.Module):
    """VGG16 feature extractor for LPIPS computation.

    Extracts features from multiple layers of VGG16 for perceptual loss.
    Supports loading weights from local file for offline HPC usage.
    """

    # Layer indices to extract features from (conv1_2, conv2_2, conv3_3, conv4_3, conv5_3)
    FEATURE_LAYERS = [3, 8, 15, 22, 29]

    def __init__(
        self,
        weights_path: str | Path | None = None,
        requires_grad: bool = False,
    ) -> None:
        """Initialize VGG feature extractor.

        Args:
            weights_path: Path to VGG16 weights file. If None, downloads from torchvision.
            requires_grad: Whether to compute gradients for VGG weights.
        """
        super().__init__()

        # Load VGG16 model
        if weights_path is not None:
            weights_path = Path(weights_path)
            if weights_path.exists():
                logger.info(f"Loading VGG16 weights from: {weights_path}")
                from torchvision.models import vgg16
                self.vgg = vgg16(weights=None)
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                self.vgg.load_state_dict(state_dict)
            else:
                logger.warning(f"VGG weights not found at {weights_path}, downloading...")
                from torchvision.models import vgg16, VGG16_Weights
                self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            logger.info("Downloading VGG16 weights from torchvision")
            from torchvision.models import vgg16, VGG16_Weights
            self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Only keep features (convolutional layers)
        self.features = self.vgg.features

        # Freeze weights if not requiring gradients
        if not requires_grad:
            for param in self.features.parameters():
                param.requires_grad = False

        # Register normalization buffers (ImageNet normalization)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features from multiple VGG layers.

        Args:
            x: Input tensor of shape (B, 3, H, W), values in [0, 1].

        Returns:
            List of feature tensors from each layer.
        """
        # Normalize with ImageNet statistics
        x = (x - self.mean) / self.std

        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.FEATURE_LAYERS:
                features.append(x)

        return features


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) loss.

    Based on Zhang et al. "The Unreasonable Effectiveness of Deep Features
    as a Perceptual Metric" (CVPR 2018).

    Features:
    - Offline weights support for HPC environments
    - Single-channel to 3-channel expansion for medical images
    - Timestep-based gating to apply loss only at low noise levels
    - Option to apply only to image channel (not mask)
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        loss_weight: float = 1.0,
        t_threshold: float = 0.5,
        apply_to_mask: bool = False,
        use_learned_weights: bool = True,
        lpips_weights_path: str | Path | None = None,
    ) -> None:
        """Initialize LPIPS loss.

        Args:
            weights_path: Path to VGG16 backbone weights (for offline HPC).
            loss_weight: Scaling factor for the loss.
            t_threshold: Only apply loss when t/T <= t_threshold.
                Higher values = more timesteps included. Default 0.5 means
                loss only applied in the second half of denoising.
            apply_to_mask: Whether to apply LPIPS to mask channel too.
            use_learned_weights: Whether to use learned channel weights (LPIPS v0.1).
            lpips_weights_path: Path to pre-trained LPIPS weights (linear layers).
        """
        super().__init__()

        self.loss_weight = loss_weight
        self.t_threshold = t_threshold
        self.apply_to_mask = apply_to_mask

        # VGG feature extractor
        self.vgg = VGGFeatureExtractor(weights_path=weights_path, requires_grad=False)

        # Number of channels at each feature layer
        self.feature_channels = [64, 128, 256, 512, 512]

        # Learned linear weights for each layer (LPIPS v0.1 style)
        self.use_learned_weights = use_learned_weights
        if use_learned_weights:
            self.linear_weights = nn.ModuleList([
                nn.Conv2d(c, 1, kernel_size=1, bias=False)
                for c in self.feature_channels
            ])

            # Initialize with uniform weights
            for lin in self.linear_weights:
                nn.init.ones_(lin.weight)
                lin.weight.data /= lin.weight.shape[1]  # Normalize by channels

            # Load pre-trained LPIPS weights if provided
            if lpips_weights_path is not None:
                self._load_lpips_weights(lpips_weights_path)

    def _load_lpips_weights(self, path: str | Path) -> None:
        """Load pre-trained LPIPS linear weights.

        Args:
            path: Path to LPIPS weights file.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"LPIPS weights not found at {path}, using default initialization")
            return

        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)

            # Map keys from LPIPS official format to our format
            # Official LPIPS uses lin0.model.weight, lin1.model.weight, etc.
            for i, lin in enumerate(self.linear_weights):
                key = f"lin{i}.model.1.weight"
                if key in state_dict:
                    lin.weight.data = state_dict[key]
                else:
                    # Try alternative key format
                    alt_key = f"lins.{i}.weight"
                    if alt_key in state_dict:
                        lin.weight.data = state_dict[alt_key]

            logger.info(f"Loaded LPIPS weights from {path}")
        except Exception as e:
            logger.warning(f"Failed to load LPIPS weights from {path}: {e}")

    def _expand_to_3_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Expand single-channel image to 3 channels.

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            Tensor of shape (B, 3, H, W).
        """
        return x.repeat(1, 3, 1, 1)

    def _normalize_to_01(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to [0, 1] range.

        Args:
            x: Input tensor in [-1, 1] range.

        Returns:
            Tensor in [0, 1] range.
        """
        return (x + 1.0) / 2.0

    def _compute_lpips(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LPIPS between prediction and target.

        Args:
            pred: Predicted image, shape (B, 1, H, W), range [-1, 1].
            target: Target image, shape (B, 1, H, W), range [-1, 1].

        Returns:
            LPIPS loss (scalar).
        """
        # Normalize to [0, 1] and expand to 3 channels
        pred_3ch = self._expand_to_3_channels(self._normalize_to_01(pred))
        target_3ch = self._expand_to_3_channels(self._normalize_to_01(target))

        # Extract features
        pred_features = self.vgg(pred_3ch)
        target_features = self.vgg(target_3ch)

        # Compute LPIPS
        total_loss = 0.0
        for i, (pf, tf) in enumerate(zip(pred_features, target_features)):
            # Normalize features
            pf_norm = F.normalize(pf, p=2, dim=1)
            tf_norm = F.normalize(tf, p=2, dim=1)

            # Squared difference
            diff = (pf_norm - tf_norm) ** 2

            if self.use_learned_weights:
                # Apply learned weights
                weighted = self.linear_weights[i](diff)
                total_loss = total_loss + weighted.mean()
            else:
                # Simple mean over all dimensions
                total_loss = total_loss + diff.mean()

        return total_loss

    def forward(
        self,
        x0_pred: torch.Tensor,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        num_train_timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute LPIPS loss with timestep gating.

        Args:
            x0_pred: Predicted x0, shape (B, 2, H, W) where ch0=image, ch1=mask.
            x0: Target x0, shape (B, 2, H, W).
            timesteps: Current timesteps, shape (B,).
            num_train_timesteps: Total number of training timesteps (T).

        Returns:
            Tuple of (weighted_loss, details_dict).
        """
        B = x0_pred.shape[0]
        device = x0_pred.device

        # Compute t/T ratio for each sample
        t_ratio = timesteps.float() / num_train_timesteps

        # Create mask for samples where t/T <= threshold (low noise regime)
        # Note: Low t means low noise, so we want t_ratio <= threshold
        gate_mask = (t_ratio <= self.t_threshold).float()  # (B,)

        # Check if any samples should have LPIPS applied
        if gate_mask.sum() == 0:
            # No samples in low-noise regime, return zero loss
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {
                "lpips_image": torch.tensor(0.0, device=device),
                "lpips_mask": torch.tensor(0.0, device=device),
                "lpips_gated_fraction": torch.tensor(0.0, device=device),
            }

        # Split channels
        pred_image = x0_pred[:, 0:1]  # (B, 1, H, W)
        target_image = x0[:, 0:1]

        # Compute LPIPS for image channel
        lpips_image = self._compute_lpips(pred_image, target_image)

        # Compute LPIPS for mask channel if enabled
        if self.apply_to_mask:
            pred_mask = x0_pred[:, 1:2]
            target_mask = x0[:, 1:2]
            lpips_mask = self._compute_lpips(pred_mask, target_mask)
        else:
            lpips_mask = torch.tensor(0.0, device=device)

        # Total LPIPS (weighted by gate)
        total_lpips = lpips_image + lpips_mask

        # Apply gating: weight loss by fraction of samples in low-noise regime
        gated_fraction = gate_mask.mean()
        gated_loss = total_lpips * gated_fraction

        # Apply loss weight
        weighted_loss = self.loss_weight * gated_loss

        details = {
            "lpips_image": lpips_image.detach(),
            "lpips_mask": lpips_mask.detach() if self.apply_to_mask else torch.tensor(0.0, device=device),
            "lpips_raw": total_lpips.detach(),
            "lpips_gated": gated_loss.detach(),
            "lpips_weighted": weighted_loss.detach(),
            "lpips_gated_fraction": gated_fraction.detach(),
        }

        return weighted_loss, details


def create_lpips_loss(cfg) -> LPIPSLoss | None:
    """Factory function to create LPIPS loss from config.

    Args:
        cfg: Configuration object with loss.perceptual section.

    Returns:
        LPIPSLoss instance or None if not enabled.
    """
    perceptual_cfg = cfg.loss.get("perceptual", {})

    if not perceptual_cfg.get("enabled", False):
        return None

    return LPIPSLoss(
        weights_path=perceptual_cfg.get("vgg_weights_path"),
        loss_weight=perceptual_cfg.get("loss_weight", 0.1),
        t_threshold=perceptual_cfg.get("t_threshold", 0.5),
        apply_to_mask=perceptual_cfg.get("apply_to_mask", False),
        use_learned_weights=perceptual_cfg.get("use_learned_weights", True),
        lpips_weights_path=perceptual_cfg.get("lpips_weights_path"),
    )
