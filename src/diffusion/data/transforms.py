"""MONAI transforms for 3D volume preprocessing and slice extraction.

This module provides transform compositions for:
1. Loading and orienting 3D volumes
2. Resampling to target spacing (physical resampling)
3. Cropping/padding to exact ROI size
4. Intensity normalization to [-1, 1]
5. Mask binarization and mapping to {-1, +1}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    MapTransform,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    Spacingd,
)
from numpy.typing import NDArray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class CreateZeroMaskd(MapTransform):
    """Create a zero mask for control subjects.

    This transform creates a mask of zeros with the same spatial dimensions
    as the image, used for control subjects that don't have lesion labels.
    """

    def __init__(
        self,
        image_key: str = "image",
        mask_key: str = "seg",
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize the transform.

        Args:
            image_key: Key for the image in the data dictionary.
            mask_key: Key for the output mask.
            allow_missing_keys: Whether to allow missing keys.
        """
        super().__init__([mask_key], allow_missing_keys)
        self.image_key = image_key
        self.mask_key = mask_key

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply the transform.

        Args:
            data: Dictionary containing the image.

        Returns:
            Dictionary with added zero mask.
        """
        d = dict(data)
        image = d[self.image_key]

        # Create zero mask with same shape as image
        if isinstance(image, torch.Tensor):
            mask = torch.zeros_like(image)
        else:
            mask = np.zeros_like(image)

        d[self.mask_key] = mask

        # Copy metadata if present
        if f"{self.image_key}_meta_dict" in d:
            d[f"{self.mask_key}_meta_dict"] = d[f"{self.image_key}_meta_dict"].copy()

        return d


class BinarizeMaskd(MapTransform):
    """Binarize mask and map to {-1, +1} for diffusion.

    Maps 0 -> -1 and >0 -> +1.
    """

    def __init__(
        self,
        keys: list[str] | str = "seg",
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize the transform.

        Args:
            keys: Keys to transform.
            allow_missing_keys: Whether to allow missing keys.
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply the transform.

        Args:
            data: Dictionary containing the mask.

        Returns:
            Dictionary with binarized and scaled mask.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key]
            # Binarize: >0 becomes 1, else 0
            if isinstance(mask, torch.Tensor):
                binary = (mask > 0).float()
            else:
                binary = (mask > 0).astype(np.float32)
            # Map to {-1, +1}: 0 -> -1, 1 -> +1
            d[key] = binary * 2.0 - 1.0
        return d


def get_volume_transforms(
    cfg: DictConfig,
    has_label: bool = True,
) -> Compose:
    """Build transform composition for 3D volume preprocessing.

    Args:
        cfg: Configuration containing transform parameters.
        has_label: Whether the data has labels (epilepsy) or not (control).

    Returns:
        MONAI Compose transform.
    """
    transform_cfg = cfg.data.transforms
    target_spacing = tuple(transform_cfg.target_spacing)
    roi_size = tuple(transform_cfg.roi_size)
    intensity_cfg = transform_cfg.intensity_norm

    # Define keys based on whether we have labels
    keys = ["image", "seg"] if has_label else ["image"]
    all_keys = ["image", "seg"]  # Always include seg for downstream

    transforms = []

    # 1. Load images
    if has_label:
        transforms.append(LoadImaged(keys=keys, image_only=False))
    else:
        transforms.append(LoadImaged(keys=["image"], image_only=False))

    # 2. Ensure channel first
    if has_label:
        transforms.append(EnsureChannelFirstd(keys=keys))
    else:
        transforms.append(EnsureChannelFirstd(keys=["image"]))

    # 3. For control subjects, create zero mask
    if not has_label:
        transforms.append(
            CreateZeroMaskd(image_key="image", mask_key="seg")
        )

    # 4. Orientation to RAS
    transforms.append(Orientationd(keys=all_keys, axcodes="RAS"))

    # 5. Resample to target spacing
    # Image uses bilinear, mask uses nearest
    transforms.append(
        Spacingd(
            keys=["image"],
            pixdim=target_spacing,
            mode="bilinear",
        )
    )
    transforms.append(
        Spacingd(
            keys=["seg"],
            pixdim=target_spacing,
            mode="nearest",
        )
    )

    # 6. Crop/pad to exact ROI size
    transforms.append(
        ResizeWithPadOrCropd(keys=all_keys, spatial_size=roi_size)
    )

    # 7. Intensity normalization for image
    if intensity_cfg.type == "percentile":
        transforms.append(
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=intensity_cfg.lower,
                upper=intensity_cfg.upper,
                b_min=intensity_cfg.b_min,
                b_max=intensity_cfg.b_max,
                clip=intensity_cfg.clip,
            )
        )
    else:  # minmax
        transforms.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=1.0,
                b_min=intensity_cfg.b_min,
                b_max=intensity_cfg.b_max,
                clip=intensity_cfg.clip,
            )
        )

    # 8. Binarize mask and map to {-1, +1}
    transforms.append(BinarizeMaskd(keys=["seg"]))

    return Compose(transforms)


def get_train_transforms(cfg: DictConfig) -> tuple[Compose, Compose]:
    """Get separate transforms for epilepsy and control data.

    Args:
        cfg: Configuration.

    Returns:
        Tuple of (epilepsy_transforms, control_transforms).
    """
    epilepsy_transforms = get_volume_transforms(cfg, has_label=True)
    control_transforms = get_volume_transforms(cfg, has_label=False)
    return epilepsy_transforms, control_transforms


def extract_axial_slice(
    volume: torch.Tensor | NDArray,
    z_index: int,
) -> torch.Tensor | NDArray:
    """Extract an axial slice from a 3D volume.

    Assumes volume shape is (C, H, W, D) or (C, X, Y, Z) where
    the last dimension is the axial (z) direction after RAS orientation.

    Args:
        volume: 4D tensor of shape (C, H, W, D).
        z_index: Index along the last (z) dimension.

    Returns:
        2D slice of shape (C, H, W).
    """
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D volume, got {volume.ndim}D")

    return volume[:, :, :, z_index]


def check_brain_content(
    slice_data: torch.Tensor | NDArray,
    threshold: float = -0.9,
    min_fraction: float = 0.05,
) -> bool:
    """Check if a slice has sufficient brain content.

    Args:
        slice_data: 2D slice of shape (C, H, W) or (H, W), values in [-1, 1].
        threshold: Intensity threshold for brain pixels.
        min_fraction: Minimum fraction of pixels above threshold.

    Returns:
        True if slice has sufficient brain content.
    """
    if slice_data.ndim == 3:
        slice_data = slice_data[0]  # Take first channel

    if isinstance(slice_data, torch.Tensor):
        brain_pixels = (slice_data > threshold).float().mean().item()
    else:
        brain_pixels = (slice_data > threshold).astype(float).mean()

    return brain_pixels >= min_fraction


def check_lesion_content(
    mask_slice: torch.Tensor | NDArray,
) -> bool:
    """Check if a mask slice contains lesion.

    Args:
        mask_slice: 2D mask of shape (C, H, W) or (H, W), values in {-1, +1}.

    Returns:
        True if slice contains lesion pixels (values > 0).
    """
    if mask_slice.ndim == 3:
        mask_slice = mask_slice[0]  # Take first channel

    if isinstance(mask_slice, torch.Tensor):
        has_lesion = (mask_slice > 0).any().item()
    else:
        has_lesion = (mask_slice > 0).any()

    return bool(has_lesion)


def slice_to_tensor(
    image_slice: NDArray,
    mask_slice: NDArray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert numpy slices to torch tensors.

    Args:
        image_slice: Image slice of shape (C, H, W) or (H, W).
        mask_slice: Mask slice of shape (C, H, W) or (H, W).

    Returns:
        Tuple of (image_tensor, mask_tensor) both of shape (1, H, W).
    """
    # Ensure channel dimension
    if image_slice.ndim == 2:
        image_slice = image_slice[np.newaxis]
    if mask_slice.ndim == 2:
        mask_slice = mask_slice[np.newaxis]

    image_tensor = torch.from_numpy(image_slice.astype(np.float32))
    mask_tensor = torch.from_numpy(mask_slice.astype(np.float32))

    return image_tensor, mask_tensor
