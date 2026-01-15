"""Lesion-specific quality metrics for evaluating generated lesion masks.

Provides detailed morphological, intensity, and shape metrics specifically
for assessing the quality of generated lesions in epilepsy FLAIR images.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import label, sobel
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image

logger = logging.getLogger(__name__)


class LesionQualityMetrics:
    """Compute lesion-specific quality metrics for generated samples.

    Focuses on evaluating the realism and quality of generated lesions
    beyond standard segmentation metrics (Dice/HD95).
    """

    def __init__(
        self,
        min_lesion_size_px: int = 5,
        intensity_percentile_bg: float = 50.0,
    ) -> None:
        """Initialize lesion quality metrics.

        Args:
            min_lesion_size_px: Minimum lesion size (pixels) to count as valid lesion.
            intensity_percentile_bg: Percentile for background intensity estimation.
        """
        self.min_lesion_size_px = min_lesion_size_px
        self.intensity_percentile_bg = intensity_percentile_bg

        logger.info(
            f"LesionQualityMetrics: min_lesion_size={min_lesion_size_px}px, "
            f"bg_percentile={intensity_percentile_bg}"
        )

    def compute_all(
        self,
        pred_image: torch.Tensor,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute all lesion quality metrics.

        Args:
            pred_image: Predicted images, shape (B, 1, H, W) in [-1, 1].
            pred_mask: Predicted masks, shape (B, 1, H, W) in {-1, +1}.
            target_mask: Optional target masks for comparison, shape (B, 1, H, W).

        Returns:
            Dictionary of metric values. Returns NaN for metrics that cannot
            be computed (e.g., no lesions in prediction).
        """
        # Convert to numpy for processing
        pred_image_np = pred_image.detach().cpu().numpy()
        pred_mask_np = pred_mask.detach().cpu().numpy()

        # Binarize mask: >0 is lesion, <=0 is background
        pred_mask_binary = (pred_mask_np > 0).astype(np.uint8)

        # Process batch
        B = pred_image_np.shape[0]
        batch_metrics = {
            'lesion_count': [],
            'lesion_total_area': [],
            'lesion_mean_size': [],
            'lesion_largest_size': [],
            'lesion_intensity_contrast': [],
            'lesion_snr': [],
            'lesion_mean_circularity': [],
            'lesion_mean_solidity': [],
            'lesion_boundary_sharpness': [],
        }

        for b in range(B):
            img = pred_image_np[b, 0]  # (H, W)
            mask = pred_mask_binary[b, 0]  # (H, W)

            # Compute metrics for this sample
            sample_metrics = self._compute_single_sample(img, mask)

            # Accumulate
            for key, value in sample_metrics.items():
                batch_metrics[key].append(value)

        # Aggregate across batch (nanmean to handle samples without lesions)
        aggregated = {}
        for key, values in batch_metrics.items():
            # Filter out NaNs and compute mean
            valid_values = [v for v in values if not np.isnan(v)]
            if len(valid_values) > 0:
                aggregated[key] = float(np.mean(valid_values))
            else:
                aggregated[key] = float('nan')

        return aggregated

    def _compute_single_sample(
        self,
        image: NDArray[np.float32],
        mask_binary: NDArray[np.uint8],
    ) -> dict[str, float]:
        """Compute lesion metrics for a single sample.

        Args:
            image: 2D image array (H, W) in [-1, 1].
            mask_binary: 2D binary mask (H, W) in {0, 1}.

        Returns:
            Dictionary of metric values for this sample.
        """
        metrics = {}

        # Label connected components
        labeled_mask, num_lesions = label(mask_binary)

        # Filter out small components (noise)
        if num_lesions > 0:
            regions = regionprops(labeled_mask)
            valid_regions = [r for r in regions if r.area >= self.min_lesion_size_px]
            num_valid_lesions = len(valid_regions)
        else:
            valid_regions = []
            num_valid_lesions = 0

        # Lesion count
        metrics['lesion_count'] = float(num_valid_lesions)

        # If no valid lesions, return NaN for all other metrics
        if num_valid_lesions == 0:
            metrics['lesion_total_area'] = float('nan')
            metrics['lesion_mean_size'] = float('nan')
            metrics['lesion_largest_size'] = float('nan')
            metrics['lesion_intensity_contrast'] = float('nan')
            metrics['lesion_snr'] = float('nan')
            metrics['lesion_mean_circularity'] = float('nan')
            metrics['lesion_mean_solidity'] = float('nan')
            metrics['lesion_boundary_sharpness'] = float('nan')
            return metrics

        # Lesion size metrics
        lesion_sizes = [r.area for r in valid_regions]
        metrics['lesion_total_area'] = float(np.sum(lesion_sizes))
        metrics['lesion_mean_size'] = float(np.mean(lesion_sizes))
        metrics['lesion_largest_size'] = float(np.max(lesion_sizes))

        # Intensity metrics
        # Get lesion pixels and background pixels
        lesion_pixels = image[mask_binary > 0]
        brain_mask = image > -0.9  # Assume brain is anything above -0.9 in [-1,1]
        background_pixels = image[brain_mask & (mask_binary == 0)]

        if len(lesion_pixels) > 0 and len(background_pixels) > 0:
            lesion_mean_intensity = np.mean(lesion_pixels)
            bg_mean_intensity = np.percentile(background_pixels, self.intensity_percentile_bg)
            bg_std_intensity = np.std(background_pixels)

            # Contrast: lesion should be hyperintense compared to background
            metrics['lesion_intensity_contrast'] = float(lesion_mean_intensity - bg_mean_intensity)

            # SNR: signal-to-noise ratio
            if bg_std_intensity > 0:
                metrics['lesion_snr'] = float(
                    (lesion_mean_intensity - bg_mean_intensity) / bg_std_intensity
                )
            else:
                metrics['lesion_snr'] = float('nan')
        else:
            metrics['lesion_intensity_contrast'] = float('nan')
            metrics['lesion_snr'] = float('nan')

        # Shape metrics: circularity and solidity
        circularities = []
        solidities = []

        for region in valid_regions:
            # Circularity: 4π × area / perimeter²
            # Perfect circle = 1.0, irregular shapes < 1.0
            if region.perimeter > 0:
                circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
                circularity = min(circularity, 1.0)  # Clamp to [0, 1]
                circularities.append(circularity)

            # Solidity: area / convex_hull_area
            # Solid shape = 1.0, fragmented/speckled < 1.0
            try:
                region_mask = labeled_mask == region.label
                convex_hull = convex_hull_image(region_mask)
                convex_area = np.sum(convex_hull)
                if convex_area > 0:
                    solidity = region.area / convex_area
                    solidities.append(solidity)
            except Exception:
                # convex_hull_image can fail on edge cases
                pass

        if len(circularities) > 0:
            metrics['lesion_mean_circularity'] = float(np.mean(circularities))
        else:
            metrics['lesion_mean_circularity'] = float('nan')

        if len(solidities) > 0:
            metrics['lesion_mean_solidity'] = float(np.mean(solidities))
        else:
            metrics['lesion_mean_solidity'] = float('nan')

        # Boundary sharpness: mean gradient magnitude at lesion boundaries
        # Compute gradient magnitude
        grad_x = sobel(image, axis=0)
        grad_y = sobel(image, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Find boundary pixels (lesion pixels adjacent to background)
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(mask_binary, iterations=1)
        boundary = dilated & (~mask_binary.astype(bool))

        if np.sum(boundary) > 0:
            boundary_gradients = grad_mag[boundary]
            metrics['lesion_boundary_sharpness'] = float(np.mean(boundary_gradients))
        else:
            metrics['lesion_boundary_sharpness'] = float('nan')

        return metrics


def compute_lesion_quality_metrics(
    pred_image: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    min_lesion_size_px: int = 5,
) -> dict[str, float]:
    """Convenience function to compute lesion quality metrics.

    Args:
        pred_image: Predicted images, shape (B, 1, H, W).
        pred_mask: Predicted masks, shape (B, 1, H, W).
        target_mask: Optional target masks, shape (B, 1, H, W).
        min_lesion_size_px: Minimum lesion size to count.

    Returns:
        Dictionary of aggregated lesion quality metrics.
    """
    calculator = LesionQualityMetrics(min_lesion_size_px=min_lesion_size_px)
    return calculator.compute_all(pred_image, pred_mask, target_mask)
