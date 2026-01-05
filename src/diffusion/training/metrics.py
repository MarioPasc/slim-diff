"""Image quality metrics for JS-DDPM using MONAI.

Provides PSNR, SSIM, Dice, and Hausdorff Distance metrics
for evaluating generated images and segmentation masks.
"""

from __future__ import annotations

import logging
import warnings

import torch
from monai.metrics import (
    compute_dice,
    compute_hausdorff_distance,
    PSNRMetric,
    SSIMMetric,
)

logger = logging.getLogger(__name__)

# Suppress expected MONAI warnings for empty masks (control samples)
warnings.filterwarnings(
    "ignore",
    message=".*ground truth of class.*is all 0.*",
    category=UserWarning,
    module="monai.metrics.utils"
)
warnings.filterwarnings(
    "ignore",
    message=".*prediction of class.*is all 0.*",
    category=UserWarning,
    module="monai.metrics.utils"
)
# Suppress MONAI FutureWarning about deprecated parameter (will be fixed in MONAI 1.7.0)
warnings.filterwarnings(
    "ignore",
    message=".*always_return_as_numpy.*",
    category=FutureWarning,
    module="monai.utils.deprecate_utils"
)


class MetricsCalculator:
    """Utility class for computing multiple metrics using MONAI."""

    def __init__(
        self,
        data_range: float = 2.0,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> None:
        """Initialize the metrics calculator with MONAI metrics.

        Args:
            data_range: Range of valid values (e.g., 2.0 for [-1, 1]).
            window_size: SSIM window size.
            sigma: SSIM Gaussian sigma.
        """
        # Image quality metrics
        # max_val is the dynamic range (difference between max and min allowed values)
        # For data in [-1, 1], the range is 2.0
        self.psnr_metric = PSNRMetric(
            max_val=data_range,  # Use data_range for consistency (2.0 for [-1, 1])
            reduction="mean",
        )

        self.ssim_metric = SSIMMetric(
            spatial_dims=2,
            data_range=data_range,
            win_size=window_size,
            kernel_sigma=sigma,
            k1=0.01,
            k2=0.03,
            reduction="mean",
        )

        logger.info(
            f"Initialized MONAI metrics: "
            f"PSNR(max_val=1.0), "
            f"SSIM(spatial_dims=2, data_range={data_range}, win_size={window_size}, sigma={sigma}), "
            f"Dice (functional), HD95 (functional)"
        )

    def compute_all(
        self,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
        pred_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute all metrics using MONAI.

        Uses functional forms for Dice/HD95 to avoid state accumulation issues.
        Dice and HD95 are only computed for samples that have lesions in the target
        (i.e., target_mask has positive values). Samples without lesions are excluded
        from segmentation metric computation.

        Args:
            pred_image: Predicted images, shape (B, 1, H, W).
            target_image: Target images, shape (B, 1, H, W).
            pred_mask: Optional predicted masks, shape (B, 1, H, W).
            target_mask: Optional target masks, shape (B, 1, H, W).

        Returns:
            Dictionary of metric values as Python floats.
        """
        # Compute image quality metrics (direct computation per batch)
        with torch.no_grad():
            psnr_tensor = self.psnr_metric(pred_image, target_image)
            ssim_tensor = self.ssim_metric(pred_image, target_image)

        # Extract scalar values
        metrics = {
            "psnr": psnr_tensor.mean().item() if psnr_tensor.numel() > 1 else psnr_tensor.item(),
            "ssim": ssim_tensor.mean().item() if ssim_tensor.numel() > 1 else ssim_tensor.item(),
        }

        # Compute segmentation metrics if masks provided
        if pred_mask is not None and target_mask is not None:
            # Binarize masks (threshold at 0.0 for [-1, +1] range)
            pred_mask_binary = (pred_mask > 0.0).float()
            target_mask_binary = (target_mask > 0.0).float()

            # Add channel dimension if needed (MONAI expects BCHW format)
            if pred_mask_binary.dim() == 3:
                pred_mask_binary = pred_mask_binary.unsqueeze(1)
            if target_mask_binary.dim() == 3:
                target_mask_binary = target_mask_binary.unsqueeze(1)

            # CRITICAL FIX: Filter samples to only include those with lesions in target
            # This prevents NaN from propagating when computing mean over all samples
            B = target_mask_binary.shape[0]

            # Check per-sample if target has any lesion pixels
            # Sum over C, H, W dimensions to get per-sample lesion counts
            target_has_lesion = target_mask_binary.sum(dim=(1, 2, 3)) > 0  # Shape: (B,)

            # Get indices of samples with lesions
            lesion_indices = torch.where(target_has_lesion)[0]

            if len(lesion_indices) == 0:
                # No samples with lesions in this batch - skip segmentation metrics
                metrics["dice"] = float('nan')
                metrics["hd95"] = float('nan')
            else:
                # Filter to only samples with lesions
                pred_filtered = pred_mask_binary[lesion_indices]
                target_filtered = target_mask_binary[lesion_indices]

                with torch.no_grad():
                    # Compute Dice only on filtered samples (those with lesions)
                    dice_tensor = compute_dice(
                        pred_filtered,
                        target_filtered,
                        include_background=True,
                    )
                    metrics["dice"] = dice_tensor.mean().item() if dice_tensor.numel() > 1 else dice_tensor.item()

                    # Compute HD95 on filtered samples
                    # Also check if predictions have any positive pixels
                    pred_has_lesion = pred_filtered.sum(dim=(1, 2, 3)) > 0
                    valid_for_hd95 = pred_has_lesion  # Target already guaranteed to have lesions
                    valid_indices = torch.where(valid_for_hd95)[0]

                    if len(valid_indices) == 0:
                        # No samples with lesions in both pred and target
                        metrics["hd95"] = float('nan')
                    else:
                        try:
                            hd95_tensor = compute_hausdorff_distance(
                                pred_filtered[valid_indices],
                                target_filtered[valid_indices],
                                include_background=False,
                                distance_metric="euclidean",
                                percentile=95,
                            )
                            # Filter out any remaining NaN values (edge cases)
                            valid_hd95 = hd95_tensor[~torch.isnan(hd95_tensor)]
                            if len(valid_hd95) > 0:
                                metrics["hd95"] = valid_hd95.mean().item()
                            else:
                                metrics["hd95"] = float('nan')
                        except Exception as e:
                            # MONAI HD95 can fail on edge cases
                            logger.warning(f"HD95 computation failed: {e}")
                            metrics["hd95"] = float('nan')

        return metrics

    def reset(self) -> None:
        """Reset MONAI metrics that maintain state.

        This is critical for preventing CUDA initialization errors in DataLoader workers.
        MONAI metrics cache CUDA tensors internally, which causes crashes when workers
        are forked after validation (CUDA contexts cannot be shared across processes).

        Call this method after each validation epoch to ensure clean state.

        Note: Only PSNR and SSIM metrics need resetting since we use functional forms
        for Dice and HD95 which don't maintain state.
        """
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        # No reset needed for functional forms (compute_dice, compute_hausdorff_distance)
