"""Image quality metrics for JS-DDPM.

Provides PSNR and SSIM metrics for evaluating generated images.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted images, shape (B, C, H, W) or (B, 1, H, W).
        target: Target images, shape (B, C, H, W) or (B, 1, H, W).
        data_range: Range of valid values (default 2.0 for [-1, 1]).

    Returns:
        PSNR value (scalar tensor).
    """
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse == 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 10 * torch.log10(data_range**2 / mse)


def psnr_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Compute PSNR per sample in batch.

    Args:
        pred: Predicted images, shape (B, C, H, W).
        target: Target images, shape (B, C, H, W).
        data_range: Range of valid values.

    Returns:
        PSNR values, shape (B,).
    """
    # Compute MSE per sample
    mse = (pred - target) ** 2
    mse = mse.view(pred.shape[0], -1).mean(dim=1)

    # Handle zero MSE
    psnr_vals = torch.where(
        mse == 0,
        torch.full_like(mse, float("inf")),
        10 * torch.log10(data_range**2 / mse),
    )
    return psnr_vals


def gaussian_kernel(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 1,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a Gaussian kernel for SSIM computation.

    Args:
        window_size: Size of the Gaussian window.
        sigma: Standard deviation of the Gaussian.
        channels: Number of channels.
        device: Device to create kernel on.

    Returns:
        Gaussian kernel, shape (channels, 1, window_size, window_size).
    """
    # Create 1D Gaussian
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    # Create 2D Gaussian
    kernel = g.unsqueeze(0) * g.unsqueeze(1)  # (H, W)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    kernel = kernel.repeat(channels, 1, 1, 1)  # (C, 1, H, W)

    return kernel


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Compute Structural Similarity Index (SSIM).

    Args:
        pred: Predicted images, shape (B, C, H, W).
        target: Target images, shape (B, C, H, W).
        window_size: Size of the Gaussian window.
        sigma: Standard deviation for Gaussian.
        data_range: Range of valid values (default 2.0 for [-1, 1]).
        k1: SSIM parameter.
        k2: SSIM parameter.

    Returns:
        SSIM value (scalar tensor).
    """
    C = pred.shape[1]
    kernel = gaussian_kernel(window_size, sigma, C, pred.device)

    # Padding for same output size
    pad = window_size // 2

    # Compute means
    mu_pred = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu_target = F.conv2d(target, kernel, padding=pad, groups=C)

    mu_pred_sq = mu_pred**2
    mu_target_sq = mu_target**2
    mu_pred_target = mu_pred * mu_target

    # Compute variances and covariance
    sigma_pred_sq = F.conv2d(pred**2, kernel, padding=pad, groups=C) - mu_pred_sq
    sigma_target_sq = (
        F.conv2d(target**2, kernel, padding=pad, groups=C) - mu_target_sq
    )
    sigma_pred_target = (
        F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu_pred_target
    )

    # SSIM formula
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )

    return ssim_map.mean()


def ssim_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Compute SSIM per sample in batch.

    Args:
        pred: Predicted images, shape (B, C, H, W).
        target: Target images, shape (B, C, H, W).
        window_size: Size of the Gaussian window.
        sigma: Standard deviation for Gaussian.
        data_range: Range of valid values.
        k1: SSIM parameter.
        k2: SSIM parameter.

    Returns:
        SSIM values, shape (B,).
    """
    B, C = pred.shape[:2]
    kernel = gaussian_kernel(window_size, sigma, C, pred.device)
    pad = window_size // 2

    # Compute means
    mu_pred = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu_target = F.conv2d(target, kernel, padding=pad, groups=C)

    mu_pred_sq = mu_pred**2
    mu_target_sq = mu_target**2
    mu_pred_target = mu_pred * mu_target

    # Compute variances and covariance
    sigma_pred_sq = F.conv2d(pred**2, kernel, padding=pad, groups=C) - mu_pred_sq
    sigma_target_sq = (
        F.conv2d(target**2, kernel, padding=pad, groups=C) - mu_target_sq
    )
    sigma_pred_target = (
        F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu_pred_target
    )

    # SSIM formula
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )

    # Mean over spatial and channel dimensions
    return ssim_map.view(B, -1).mean(dim=1)


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.0,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute Dice coefficient for binary masks.

    Args:
        pred: Predicted masks in {-1, +1}, shape (B, 1, H, W).
        target: Target masks in {-1, +1}, shape (B, 1, H, W).
        threshold: Threshold for binarization.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice coefficient (scalar tensor).
    """
    # Binarize: >threshold becomes 1, else 0
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    # Flatten spatial dimensions
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)

    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def dice_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.0,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute Dice coefficient per sample.

    Args:
        pred: Predicted masks, shape (B, 1, H, W).
        target: Target masks, shape (B, 1, H, W).
        threshold: Threshold for binarization.
        smooth: Smoothing factor.

    Returns:
        Dice coefficients, shape (B,).
    """
    B = pred.shape[0]

    pred_binary = (pred > threshold).float().view(B, -1)
    target_binary = (target > threshold).float().view(B, -1)

    intersection = (pred_binary * target_binary).sum(dim=1)
    union = pred_binary.sum(dim=1) + target_binary.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


class MetricsCalculator:
    """Utility class for computing multiple metrics."""

    def __init__(
        self,
        data_range: float = 2.0,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> None:
        """Initialize the metrics calculator.

        Args:
            data_range: Range of valid values.
            window_size: SSIM window size.
            sigma: SSIM Gaussian sigma.
        """
        self.data_range = data_range
        self.window_size = window_size
        self.sigma = sigma

    def compute_all(
        self,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
        pred_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all metrics.

        Args:
            pred_image: Predicted images, shape (B, 1, H, W).
            target_image: Target images, shape (B, 1, H, W).
            pred_mask: Optional predicted masks.
            target_mask: Optional target masks.

        Returns:
            Dictionary of metric values.
        """
        metrics = {
            "psnr": psnr(pred_image, target_image, self.data_range),
            "ssim": ssim(
                pred_image, target_image,
                self.window_size, self.sigma, self.data_range
            ),
        }

        if pred_mask is not None and target_mask is not None:
            metrics["dice"] = dice_coefficient(pred_mask, target_mask)

        return metrics
