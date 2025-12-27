"""MONAI transforms for segmentation."""

from __future__ import annotations

from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandZoomd,
)
from omegaconf import DictConfig


class SegmentationTransforms:
    """Build MONAI transforms for segmentation."""

    @staticmethod
    def build_train_transforms(cfg: DictConfig):
        """Build training transforms with augmentation.

        Args:
            cfg: Configuration object

        Returns:
            MONAI Compose transform or None
        """
        if not cfg.augmentation.enabled:
            return None

        transforms = []
        aug_cfg = cfg.augmentation

        # Spatial transforms (apply to both image and mask)
        if aug_cfg.random_flip.enabled:
            transforms.append(
                RandFlipd(
                    keys=["image", "mask"],
                    prob=aug_cfg.random_flip.prob,
                    spatial_axis=aug_cfg.random_flip.spatial_axis,
                )
            )

        if aug_cfg.random_rotate.enabled:
            transforms.append(
                RandRotated(
                    keys=["image", "mask"],
                    prob=aug_cfg.random_rotate.prob,
                    range_x=aug_cfg.random_rotate.range_x,
                    range_y=aug_cfg.random_rotate.range_y,
                    mode=["bilinear", "nearest"],
                    padding_mode=aug_cfg.random_rotate.padding_mode,
                )
            )

        if aug_cfg.random_scale.enabled:
            transforms.append(
                RandZoomd(
                    keys=["image", "mask"],
                    prob=aug_cfg.random_scale.prob,
                    min_zoom=aug_cfg.random_scale.scale_range[0],
                    max_zoom=aug_cfg.random_scale.scale_range[1],
                    mode=["area", "nearest"],
                )
            )

        # Intensity transforms (image only)
        if aug_cfg.random_gaussian_noise.enabled:
            transforms.append(
                RandGaussianNoised(
                    keys=["image"],
                    prob=aug_cfg.random_gaussian_noise.prob,
                    mean=aug_cfg.random_gaussian_noise.mean,
                    std=aug_cfg.random_gaussian_noise.std,
                )
            )

        if aug_cfg.random_gaussian_smooth.enabled:
            transforms.append(
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=aug_cfg.random_gaussian_smooth.prob,
                    sigma_x=aug_cfg.random_gaussian_smooth.sigma_range,
                    sigma_y=aug_cfg.random_gaussian_smooth.sigma_range,
                )
            )

        if aug_cfg.random_gamma.enabled:
            transforms.append(
                RandAdjustContrastd(
                    keys=["image"],
                    prob=aug_cfg.random_gamma.prob,
                    gamma=aug_cfg.random_gamma.gamma_range,
                )
            )

        if len(transforms) == 0:
            return None

        return Compose(transforms)

    @staticmethod
    def build_val_transforms(cfg: DictConfig):
        """Build validation transforms (no augmentation).

        Args:
            cfg: Configuration object

        Returns:
            None (no transforms for validation)
        """
        return None
