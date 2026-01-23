"""Focal Frequency Loss for diffusion model training.

Implements the focal frequency loss from:
"Focal Frequency Loss for Image Reconstruction and Synthesis"
Jiang et al., ICCV 2021
https://github.com/EndlessSora/focal-frequency-loss

This loss operates in the frequency domain, adaptively focusing on
hard-to-synthesize frequency components by down-weighting easy ones.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss for image reconstruction.

    Computes loss in frequency domain, adaptively focusing on
    hard-to-synthesize frequency components by down-weighting easy ones.

    The loss operates on x0_pred (reconstructed from noise prediction)
    vs x0 (original image) rather than on noise predictions directly.

    Reference:
        Jiang, L., Dai, B., Wu, W., & Loy, C. C. (2021).
        Focal Frequency Loss for Image Reconstruction and Synthesis.
        In ICCV 2021.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
    ) -> None:
        """Initialize FocalFrequencyLoss.

        Args:
            loss_weight: Scaling factor for the final loss value.
            alpha: Spectrum weight matrix scaling factor (focal term).
                   Higher alpha increases focus on hard frequencies.
                   Default 1.0 provides linear weighting.
            patch_factor: Factor to divide image into patches for local
                         frequency analysis. 1 = no patching (global FFT).
            ave_spectrum: If True, use minibatch average spectrum for
                         weight matrix computation.
            log_matrix: If True, apply log(1 + x) to weight matrix for
                       numerical stability with large frequency differences.
            batch_matrix: If True, compute weight matrix from batch statistics
                         instead of per-sample.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

        logger.info(
            f"FocalFrequencyLoss: loss_weight={loss_weight}, alpha={alpha}, "
            f"patch_factor={patch_factor}, ave_spectrum={ave_spectrum}, "
            f"log_matrix={log_matrix}, batch_matrix={batch_matrix}"
        )

    def _tensor_to_freq(self, x: torch.Tensor) -> torch.Tensor:
        """Convert spatial tensor to frequency domain.

        Uses real-to-complex 2D FFT with orthonormal normalization.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Frequency tensor of shape (B, C, H, W, 2) where last dim
            contains [real, imag] components.
        """
        # 2D FFT with orthonormal normalization
        # Cast to float32: cuFFT only supports power-of-2 dims in half precision,
        # and FP32 is more numerically stable for frequency domain operations
        freq = torch.fft.fft2(x.float(), norm="ortho")
        # Stack real and imaginary parts
        freq = torch.stack([freq.real, freq.imag], dim=-1)
        return freq

    def _compute_focal_weight(
        self,
        pred_freq: torch.Tensor,
        target_freq: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal weight matrix from frequency difference.

        The weight matrix adaptively emphasizes frequencies where
        prediction differs most from target.

        Args:
            pred_freq: Predicted frequency tensor (B, C, H, W, 2).
            target_freq: Target frequency tensor (B, C, H, W, 2).

        Returns:
            Weight matrix of shape (B, C, H, W).
        """
        # Compute magnitude of frequency difference
        diff = pred_freq - target_freq
        diff_magnitude = torch.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2 + 1e-8)

        if self.batch_matrix:
            # Use batch-wise statistics
            weight = diff_magnitude.mean(dim=0, keepdim=True)
            weight = weight.expand_as(diff_magnitude)
        else:
            weight = diff_magnitude

        if self.log_matrix:
            # Apply log for numerical stability
            weight = torch.log1p(weight)

        # Normalize weight matrix to [0, 1]
        weight_max = weight.amax(dim=(-2, -1), keepdim=True)
        weight = weight / (weight_max + 1e-8)

        # Apply focal term: higher alpha = more focus on hard frequencies
        weight = weight ** self.alpha

        return weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute focal frequency loss.

        Args:
            pred: Predicted x0, shape (B, C, H, W).
            target: Target x0, shape (B, C, H, W).

        Returns:
            Tuple of (loss, details_dict).
        """
        B, C, H, W = pred.shape

        # Handle patch factor (divide image into patches)
        if self.patch_factor > 1:
            if H % self.patch_factor != 0 or W % self.patch_factor != 0:
                raise ValueError(
                    f"Image dimensions ({H}, {W}) must be divisible by "
                    f"patch_factor ({self.patch_factor})"
                )

            patch_h = H // self.patch_factor
            patch_w = W // self.patch_factor

            # Reshape to patches: (B, C, pf, ph, pf, pw) -> (B*pf*pf, C, ph, pw)
            pred = pred.view(
                B, C, self.patch_factor, patch_h, self.patch_factor, patch_w
            )
            pred = pred.permute(0, 2, 4, 1, 3, 5).contiguous()
            pred = pred.view(-1, C, patch_h, patch_w)

            target = target.view(
                B, C, self.patch_factor, patch_h, self.patch_factor, patch_w
            )
            target = target.permute(0, 2, 4, 1, 3, 5).contiguous()
            target = target.view(-1, C, patch_h, patch_w)

        # Convert to frequency domain
        pred_freq = self._tensor_to_freq(pred)
        target_freq = self._tensor_to_freq(target)

        # Compute focal weight matrix
        if self.ave_spectrum:
            # Use average spectrum for weight computation
            pred_freq_avg = pred_freq.mean(dim=0, keepdim=True)
            target_freq_avg = target_freq.mean(dim=0, keepdim=True)
            weight = self._compute_focal_weight(pred_freq_avg, target_freq_avg)
            weight = weight.expand(pred_freq.shape[:-1])
        else:
            weight = self._compute_focal_weight(pred_freq, target_freq)

        # Compute frequency domain MSE (sum of squared real and imag differences)
        freq_diff = pred_freq - target_freq
        freq_mse = freq_diff[..., 0] ** 2 + freq_diff[..., 1] ** 2

        # Compute raw (unweighted) FFL for logging
        ffl_raw = freq_mse.mean()

        # Apply focal weighting
        weighted_loss = weight * freq_mse

        # Final loss with scaling
        loss = weighted_loss.mean() * self.loss_weight

        details = {
            "ffl_raw": ffl_raw.detach(),
            "ffl_weighted": loss.detach(),
        }

        return loss, details
