"""MONAI transforms for segmentation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from monai.transforms import (
    Compose,
    MapTransform,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandZoomd,
    Randomizable,
)
from omegaconf import DictConfig


class RandAdjustContrastdNeg1To1(MapTransform):
    """Random gamma contrast adjustment for data in [-1, 1] range.

    MONAI's RandAdjustContrastd expects [0, 1] range. This wrapper:
    1. Converts [-1, 1] to [0, 1]
    2. Applies gamma adjustment
    3. Converts back to [-1, 1]
    """

    def __init__(self, keys, prob: float = 0.1, gamma: tuple = (0.5, 4.5)):
        """Initialize transform.

        Args:
            keys: Keys to apply transform to
            prob: Probability of applying transform
            gamma: Gamma range (min, max)
        """
        super().__init__(keys)
        self.inner_transform = RandAdjustContrastd(
            keys=keys, prob=prob, gamma=gamma
        )

    def __call__(self, data):
        """Apply transform with range conversion."""
        d = dict(data)

        # Convert from [-1, 1] to [0, 1]
        for key in self.keys:
            d[key] = (d[key] + 1.0) / 2.0

        # Apply gamma
        d = self.inner_transform(d)

        # Convert back from [0, 1] to [-1, 1]
        for key in self.keys:
            d[key] = d[key] * 2.0 - 1.0

        return d


class TrackableCompose(Compose):
    """Compose wrapper that tracks which transforms are applied.

    Tracks augmentation statistics for logging and analysis.
    Statistics can be accessed via get_stats() and reset via reset_stats().
    """

    def __init__(self, transforms: List):
        """Initialize trackable compose.

        Args:
            transforms: List of MONAI transforms
        """
        super().__init__(transforms)
        self.stats = defaultdict(int)  # Count of times each transform was applied
        self.total_calls = 0

        # Store transform names for tracking
        self.transform_names = []
        for t in self.transforms:
            # Get a readable name for the transform
            name = t.__class__.__name__
            # Remove 'd' suffix for dictionary transforms
            if name.endswith('d'):
                name = name[:-1]
            self.transform_names.append(name)

    def __call__(self, data):
        """Apply transforms and track which ones are applied."""
        self.total_calls += 1
        d = dict(data)

        for i, (transform, name) in enumerate(zip(self.transforms, self.transform_names)):
            # Apply the transform
            d = transform(d)

            # Check if this is a randomizable transform and if it was applied
            if isinstance(transform, Randomizable):
                # MONAI's randomizable transforms have _do_transform attribute
                if hasattr(transform, '_do_transform') and transform._do_transform:
                    self.stats[name] += 1
            # For non-randomizable transforms (always applied), count every call
            else:
                self.stats[name] += 1

        return d

    def get_stats(self) -> Dict[str, int]:
        """Get augmentation statistics.

        Returns:
            Dict mapping transform name to application count
        """
        return dict(self.stats)

    def get_total_calls(self) -> int:
        """Get total number of times compose was called.

        Returns:
            Total call count
        """
        return self.total_calls

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats.clear()
        self.total_calls = 0


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
            # Use custom transform that handles [-1, 1] data range
            transforms.append(
                RandAdjustContrastdNeg1To1(
                    keys=["image"],
                    prob=aug_cfg.random_gamma.prob,
                    gamma=tuple(aug_cfg.random_gamma.gamma_range),
                )
            )

        if len(transforms) == 0:
            return None

        return TrackableCompose(transforms)

    @staticmethod
    def build_val_transforms(cfg: DictConfig):
        """Build validation transforms (no augmentation).

        Args:
            cfg: Configuration object

        Returns:
            None (no transforms for validation)
        """
        return None
