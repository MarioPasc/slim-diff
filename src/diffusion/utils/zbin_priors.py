"""Z-bin anatomical ROI priors for post-processing generated samples.

This module provides utilities for:
1. Computing z-bin occupancy priors from cached training slices (offline)
2. Loading priors and applying post-processing to generated samples (online)

The priors are atlas-style spatial maps (per z-bin) that define expected brain
regions, used to remove out-of-brain speckle noise from generated images.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    generate_binary_structure,
    label,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Otsu Threshold (NumPy implementation - no sklearn dependency)
# =============================================================================


def otsu_threshold(data: NDArray[np.float64]) -> float:
    """Compute Otsu's threshold on 1D data array.

    Implements Otsu's method to find the optimal threshold that minimizes
    intra-class variance between foreground and background.

    Args:
        data: 1D array of values to threshold.

    Returns:
        Optimal threshold value.
    """
    # Flatten and remove NaN/Inf
    data = data.ravel()
    data = data[np.isfinite(data)]

    if len(data) == 0:
        return 0.5

    # Compute histogram
    n_bins = 256
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Total pixels
    total = hist.sum()
    if total == 0:
        return 0.5

    # Precompute cumulative sums
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg

    # Cumulative mean
    sum_bg = np.cumsum(hist * bin_centers)
    sum_fg = sum_bg[-1] - sum_bg

    # Calculate means
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = sum_bg / weight_bg
        mean_fg = sum_fg / weight_fg

    # Calculate between-class variance
    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    # Find threshold that maximizes between-class variance
    idx = np.nanargmax(variance)
    threshold = bin_centers[idx]

    return float(threshold)


# =============================================================================
# Brain Foreground Mask Computation (per-slice)
# =============================================================================


def compute_brain_foreground_mask(
    image: NDArray[np.float32],
    gaussian_sigma_px: float,
    min_component_px: int,
    n_components_to_keep: int = 1,
    relaxed_threshold_factor: float = 0.1,
) -> NDArray[np.bool_] | None:
    """Compute binary brain foreground mask for a single slice.

    Algorithm:
    1. Robust percentile scaling (p1, p99)
    2. Gaussian smoothing
    3. Otsu thresholding
    4. Top N largest connected components
    5. Fill holes

    Args:
        image: 2D image array (H, W).
        gaussian_sigma_px: Sigma for Gaussian smoothing.
        min_component_px: Minimum component size to keep.
        n_components_to_keep: Number of largest components to keep (default: 1).
        relaxed_threshold_factor: Factor for relaxed threshold on smaller components.

    Returns:
        Binary mask (H, W) or None if slice is invalid.
    """
    # Ensure 2D
    if image.ndim != 2:
        image = image.squeeze()
    if image.ndim != 2:
        return None

    # Robust percentile scaling
    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)

    if p99 <= p1:
        return None

    # Scale to [0, 1]
    scaled = np.clip((image - p1) / (p99 - p1), 0, 1).astype(np.float64)

    # Gaussian smoothing
    smoothed = gaussian_filter(scaled, sigma=gaussian_sigma_px)

    # Otsu thresholding
    threshold = otsu_threshold(smoothed)
    binary = smoothed > threshold

    # Find connected components
    labeled, n_components = label(binary)

    if n_components == 0:
        return None

    # Find component sizes
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignore background

    # Get indices of top N largest components
    # argsort gives indices from smallest to largest, so we reverse it
    sorted_indices = np.argsort(component_sizes)[::-1]

    # Keep up to n_components_to_keep largest components
    n_to_keep = min(n_components_to_keep, n_components)
    keep_indices = []

    for idx in sorted_indices:
        if idx == 0:  # Skip background
            continue

        # For multi-component: apply full threshold only to largest component
        # Additional components use 10% of threshold (more lenient)
        if len(keep_indices) == 0:
            # First component: must meet full threshold
            if component_sizes[idx] >= min_component_px:
                keep_indices.append(idx)
        else:
            # Additional components: use relaxed threshold (10% of original)
            # This allows small brainstem structures in low z-bins
            relaxed_threshold = min_component_px * relaxed_threshold_factor
            if component_sizes[idx] >= relaxed_threshold:
                keep_indices.append(idx)

        if len(keep_indices) >= n_to_keep:
            break

    if len(keep_indices) == 0:
        return None

    # Create mask with all kept components
    mask = np.zeros_like(binary, dtype=bool)
    for idx in keep_indices:
        mask |= (labeled == idx)

    # Fill holes in the combined mask
    mask = binary_fill_holes(mask)

    return mask.astype(np.bool_)


# =============================================================================
# Offline Prior Computation
# =============================================================================


def compute_zbin_priors(
    cache_dir: Path,
    z_bins: int,
    z_range: tuple[int, int],
    prob_threshold: float,
    dilate_radius_px: int,
    gaussian_sigma_px: float,
    min_component_px: int,
    n_first_bins: int = 0,
    max_components_for_first_bins: int = 1,
    relaxed_threshold_factor: float = 0.1,
) -> dict[str, Any]:
    """Compute z-bin occupancy priors from cached slices.

    For each z-bin, computes occupancy probability map from all cached
    slices, then converts to binary ROI with dilation.

    Args:
        cache_dir: Path to slice cache directory.
        z_bins: Number of z-position bins.
        z_range: (min_z, max_z) for z-bin mapping.
        prob_threshold: Probability threshold for ROI.
        dilate_radius_px: Dilation radius for tolerance.
        gaussian_sigma_px: Smoothing sigma for mask computation.
        min_component_px: Minimum component size.
        n_first_bins: Number of low z-bins for multi-component handling (default: 0).
        max_components_for_first_bins: Keep top N components for first bins (default: 1).
        relaxed_threshold_factor: Factor for relaxed threshold on smaller components.

    Returns:
        Dict with 'priors' (dict[int, ndarray]) and 'metadata'.
    """
    cache_dir = Path(cache_dir)

    # Initialize accumulators for each z-bin
    occupancy: dict[int, NDArray[np.int32] | None] = {b: None for b in range(z_bins)}
    counts: dict[int, int] = {b: 0 for b in range(z_bins)}
    image_shape: tuple[int, int] | None = None

    # Read all cached slices from all splits
    for split in ["train", "val", "test"]:
        csv_path = cache_dir / f"{split}.csv"
        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}, skipping")
            continue

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                z_bin = int(row["z_bin"])
                filepath = cache_dir / row["filepath"]

                if not filepath.exists():
                    continue

                # Load slice
                try:
                    data = np.load(filepath, allow_pickle=True)
                    image = data["image"]
                except Exception as e:
                    logger.debug(f"Failed to load {filepath}: {e}")
                    continue

                # Ensure 2D
                if image.ndim == 3:
                    image = image.squeeze()

                # Track image shape
                if image_shape is None:
                    image_shape = image.shape
                    logger.info(f"Detected image shape: {image_shape}")

                # Compute foreground mask for this slice
                # Use multi-component logic for first N bins (near neck/brainstem)
                n_components = (
                    max_components_for_first_bins
                    if z_bin < n_first_bins
                    else 1
                )
                mask = compute_brain_foreground_mask(
                    image,
                    gaussian_sigma_px,
                    min_component_px,
                    n_components,
                    relaxed_threshold_factor,
                )

                if mask is None:
                    continue

                # Accumulate occupancy
                if occupancy[z_bin] is None:
                    occupancy[z_bin] = np.zeros(image_shape, dtype=np.int32)

                occupancy[z_bin] += mask.astype(np.int32)
                counts[z_bin] += 1

    if image_shape is None:
        raise ValueError("No valid slices found in cache")

    # Convert occupancy counts to probability and threshold to ROI
    priors: dict[int, NDArray[np.bool_]] = {}

    # Create dilation structuring element
    struct = generate_binary_structure(2, 1)  # Cross-shaped
    if dilate_radius_px > 1:
        struct = np.ones((2 * dilate_radius_px + 1, 2 * dilate_radius_px + 1), dtype=bool)

    for z_bin in range(z_bins):
        if counts[z_bin] == 0 or occupancy[z_bin] is None:
            # No data for this bin - use empty mask
            logger.warning(f"Z-bin {z_bin} has no data, using empty prior")
            priors[z_bin] = np.zeros(image_shape, dtype=np.bool_)
            continue

        # Compute probability
        prob = occupancy[z_bin].astype(np.float64) / counts[z_bin]

        # Threshold
        roi = prob >= prob_threshold

        # Dilate for tolerance
        roi = binary_dilation(roi, structure=struct, iterations=1)

        priors[z_bin] = roi.astype(np.bool_)

        logger.debug(
            f"Z-bin {z_bin}: {counts[z_bin]} slices, "
            f"ROI coverage: {roi.sum() / roi.size:.2%}"
        )

    # Create metadata
    metadata = {
        "z_bins": z_bins,
        "z_range": z_range,
        "prob_threshold": prob_threshold,
        "dilate_radius_px": dilate_radius_px,
        "gaussian_sigma_px": gaussian_sigma_px,
        "min_component_px": min_component_px,
        "n_first_bins": n_first_bins,
        "max_components_for_first_bins": max_components_for_first_bins,
        "relaxed_threshold_factor": relaxed_threshold_factor,
        "image_shape": image_shape,
        "slice_counts": counts,
    }

    total_slices = sum(counts.values())
    logger.info(f"Computed priors from {total_slices} slices across {z_bins} bins")

    return {"priors": priors, "metadata": metadata}


def save_zbin_priors(
    priors: dict[int, NDArray[np.bool_]],
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    """Save z-bin priors to NPZ file with metadata.

    Args:
        priors: Dict mapping z-bin to boolean ROI array.
        metadata: Dict of parameters used to compute priors.
        output_path: Path to save NPZ file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare arrays for saving
    save_dict: dict[str, Any] = {"metadata": metadata}

    for z_bin, roi in priors.items():
        save_dict[f"bin_{z_bin}"] = roi.astype(np.uint8)

    np.savez_compressed(output_path, **save_dict)
    logger.info(f"Saved z-bin priors to {output_path}")


def load_zbin_priors(
    cache_dir: Path,
    filename: str,
    z_bins: int,
) -> dict[int, NDArray[np.bool_]]:
    """Load z-bin priors from NPZ file.

    Validates metadata z_bins matches expected value.

    Args:
        cache_dir: Cache directory containing priors file.
        filename: Priors filename.
        z_bins: Expected number of z-bins (for validation).

    Returns:
        Dict mapping z-bin index to boolean ROI array.

    Raises:
        FileNotFoundError: If priors file doesn't exist.
        ValueError: If z_bins mismatch or file invalid.
    """
    cache_dir = Path(cache_dir)
    priors_path = cache_dir / filename

    if not priors_path.exists():
        raise FileNotFoundError(f"Priors file not found: {priors_path}")

    data = np.load(priors_path, allow_pickle=True)

    # Validate metadata
    if "metadata" not in data:
        raise ValueError("Invalid priors file: missing metadata")

    metadata = data["metadata"].item()
    stored_z_bins = metadata.get("z_bins")

    if stored_z_bins != z_bins:
        raise ValueError(
            f"Z-bins mismatch: priors have {stored_z_bins}, expected {z_bins}"
        )

    # Load all bin arrays
    priors: dict[int, NDArray[np.bool_]] = {}
    for z_bin in range(z_bins):
        key = f"bin_{z_bin}"
        if key not in data:
            raise ValueError(f"Missing prior for z-bin {z_bin}")
        priors[z_bin] = data[key].astype(np.bool_)

    logger.info(f"Loaded z-bin priors for {len(priors)} bins from {priors_path}")
    return priors


def get_anatomical_weights(
    z_bins_batch: torch.Tensor,
    priors: dict[int, NDArray[np.bool_]],
    in_brain_weight: float = 1.0,
    out_brain_weight: float = 0.1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate spatial weight maps from anatomical priors for training loss.

    Creates a weight map for each sample in the batch based on z-bin priors,
    allowing the training loss to downweight out-of-brain regions.

    Args:
        z_bins_batch: Z-bin indices for batch, shape (B,).
        priors: Dict mapping z-bin to boolean ROI array (H, W).
        in_brain_weight: Weight for in-brain pixels (default: 1.0).
        out_brain_weight: Weight for out-of-brain pixels (default: 0.1).
        device: Target device for output tensor. If None, uses z_bins_batch.device.

    Returns:
        Weight tensor of shape (B, 1, H, W) with values in_brain_weight or
        out_brain_weight based on prior ROI masks.

    Example:
        >>> z_bins = torch.tensor([5, 10, 15])  # Batch of 3
        >>> priors = load_zbin_priors(cache_dir, filename, z_bins=30)
        >>> weights = get_anatomical_weights(z_bins, priors, 1.0, 0.1)
        >>> weights.shape
        torch.Size([3, 1, 128, 128])
    """
    if device is None:
        device = z_bins_batch.device

    B = z_bins_batch.shape[0]
    z_bins_np = z_bins_batch.cpu().numpy()

    # Get first prior to determine spatial dimensions
    first_prior = next(iter(priors.values()))
    H, W = first_prior.shape

    # Initialize weight tensor
    weights = torch.full(
        (B, 1, H, W),
        fill_value=out_brain_weight,
        dtype=torch.float32,
        device=device,
    )

    # Fill weights based on priors
    for i, z_bin in enumerate(z_bins_np):
        z_bin_int = int(z_bin)
        if z_bin_int in priors:
            # Get ROI mask for this z-bin
            roi_mask = priors[z_bin_int]  # (H, W) boolean array

            # Convert to tensor and set in-brain weights
            roi_tensor = torch.from_numpy(roi_mask).to(device)
            weights[i, 0, :, :] = torch.where(
                roi_tensor,
                torch.tensor(in_brain_weight, device=device),
                torch.tensor(out_brain_weight, device=device),
            )
        # else: keep all weights as out_brain_weight (no prior available)

    return weights


def get_anatomical_priors_as_input(
    z_bins_batch: torch.Tensor | list[int],
    priors: dict[int, NDArray[np.bool_]],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Get anatomical priors as input channel for model conditioning.

    Converts boolean anatomical priors to float tensors suitable for
    concatenation as an input channel. The priors are converted to
    float in range [0, 1] where 1 indicates in-brain region.

    Args:
        z_bins_batch: Z-bin indices for batch, shape (B,) tensor or list of length B.
        priors: Dict mapping z-bin to boolean ROI array (H, W).
        device: Target device for output tensor. If None, uses z_bins_batch.device
                (if tensor) or defaults to 'cpu'.

    Returns:
        Prior tensor of shape (B, 1, H, W) with values in [0, 1].

    Example:
        >>> z_bins = torch.tensor([5, 10, 15])  # Batch of 3
        >>> priors = load_zbin_priors(cache_dir, filename, z_bins=30)
        >>> prior_input = get_anatomical_priors_as_input(z_bins, priors)
        >>> prior_input.shape
        torch.Size([3, 1, 128, 128])
    """
    # Handle both tensor and list inputs
    if isinstance(z_bins_batch, torch.Tensor):
        if device is None:
            device = z_bins_batch.device
        z_bins_np = z_bins_batch.cpu().numpy()
        B = z_bins_batch.shape[0]
    else:
        # List input
        if device is None:
            device = torch.device('cpu')
        z_bins_np = np.array(z_bins_batch)
        B = len(z_bins_batch)

    # Get first prior to determine spatial dimensions
    first_prior = next(iter(priors.values()))
    H, W = first_prior.shape

    # Initialize prior tensor (default: all zeros = out-of-brain)
    prior_tensor = torch.zeros(
        (B, 1, H, W),
        dtype=torch.float32,
        device=device,
    )

    # Fill priors based on z-bins
    for i, z_bin in enumerate(z_bins_np):
        z_bin_int = int(z_bin)
        if z_bin_int in priors:
            # Get ROI mask for this z-bin
            roi_mask = priors[z_bin_int]  # (H, W) boolean array

            # Convert to float tensor: True -> 1.0, False -> 0.0
            roi_tensor = torch.from_numpy(roi_mask.astype(np.float32)).to(device)
            prior_tensor[i, 0, :, :] = roi_tensor
        # else: keep all zeros (no prior available)

    return prior_tensor


# =============================================================================
# Online Post-Processing
# =============================================================================


def apply_zbin_prior_postprocess(
    img: NDArray[np.float32],
    lesion: NDArray[np.float32],
    z_bin: int,
    priors: dict[int, NDArray[np.bool_]],
    gaussian_sigma_px: float,
    min_component_px: int,
    fallback: str,
    n_first_bins: int = 0,
    max_components_for_first_bins: int = 1,
    relaxed_threshold_factor: float = 0.1,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Apply z-bin prior post-processing to a generated sample.

    Algorithm:
    1. Get prior ROI for z_bin
    2. Robust scaling within prior region
    3. Gaussian smoothing
    4. Otsu threshold within prior
    5. Top N largest connected components (N=3 for first bins, 1 for others)
    6. Enforce min_component_px (fallback if too small)
    7. Fill holes -> final brain mask B
    8. Apply: img_out = img * B, lesion_out = lesion * B

    Args:
        img: Generated image (H, W).
        lesion: Generated lesion mask (H, W).
        z_bin: Z-bin index for this slice.
        priors: Dict of z-bin -> ROI arrays.
        gaussian_sigma_px: Smoothing sigma.
        min_component_px: Minimum component size.
        fallback: "prior" or "empty".
        n_first_bins: Number of low z-bins for multi-component handling (default: 0).
        max_components_for_first_bins: Keep top N components for first bins (default: 1).
        relaxed_threshold_factor: Factor for relaxed threshold on smaller components.

    Returns:
        Tuple of (cleaned_img, cleaned_lesion).
    """
    # Ensure 2D
    img_2d = img.squeeze() if img.ndim == 3 else img
    lesion_2d = lesion.squeeze() if lesion.ndim == 3 else lesion

    # Get prior ROI
    if z_bin not in priors:
        logger.warning(f"Z-bin {z_bin} not in priors, returning unchanged")
        return img, lesion

    prior = priors[z_bin]

    # Check if prior is empty
    if not prior.any():
        if fallback == "empty":
            return np.zeros_like(img), np.zeros_like(lesion)
        else:
            return img, lesion

    # Robust scaling within prior region only
    vals = img_2d[prior]
    p1 = np.percentile(vals, 1)
    p99 = np.percentile(vals, 99)

    if p99 <= p1:
        # Cannot compute meaningful threshold - use fallback
        if fallback == "prior":
            brain_mask = prior.astype(np.bool_)
        else:
            brain_mask = np.zeros_like(prior, dtype=np.bool_)
    else:
        # Scale full image (for context), but threshold within prior
        scaled = np.clip((img_2d - p1) / (p99 - p1), 0, 1).astype(np.float64)

        # Gaussian smoothing
        smoothed = gaussian_filter(scaled, sigma=gaussian_sigma_px)

        # Compute Otsu threshold on values within prior only
        threshold = otsu_threshold(smoothed[prior])

        # Create candidate mask: threshold within prior only
        candidate = np.zeros_like(prior, dtype=np.bool_)
        candidate[prior] = smoothed[prior] > threshold

        # Find connected components
        labeled, n_components = label(candidate)

        if n_components == 0:
            if fallback == "prior":
                brain_mask = prior.astype(np.bool_)
            else:
                brain_mask = np.zeros_like(prior, dtype=np.bool_)
        else:
            # Determine number of components to keep
            n_components_to_keep = (
                max_components_for_first_bins
                if z_bin < n_first_bins
                else 1
            )

            # Find component sizes
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0  # Ignore background

            # Get indices of top N largest components
            sorted_indices = np.argsort(component_sizes)[::-1]

            # Keep up to n_components_to_keep largest components
            n_to_keep = min(n_components_to_keep, n_components)
            keep_indices = []

            for idx in sorted_indices:
                if idx == 0:  # Skip background
                    continue

                # For multi-component: apply full threshold only to largest component
                # Additional components use 10% of threshold (more lenient)
                if len(keep_indices) == 0:
                    # First component: must meet full threshold
                    if component_sizes[idx] >= min_component_px:
                        keep_indices.append(idx)
                else:
                    # Additional components: use relaxed threshold
                    relaxed_threshold = min_component_px * relaxed_threshold_factor
                    if component_sizes[idx] >= relaxed_threshold:
                        keep_indices.append(idx)

                if len(keep_indices) >= n_to_keep:
                    break

            if len(keep_indices) == 0:
                # No components large enough - use fallback
                if fallback == "prior":
                    brain_mask = prior.astype(np.bool_)
                else:
                    brain_mask = np.zeros_like(prior, dtype=np.bool_)
            else:
                # Create mask with all kept components
                brain_mask = np.zeros_like(candidate, dtype=bool)
                for idx in keep_indices:
                    brain_mask |= (labeled == idx)

                # Fill holes in the combined mask
                brain_mask = binary_fill_holes(brain_mask)

    # Apply mask to both image and lesion
    img_out = img_2d * brain_mask.astype(img_2d.dtype)
    lesion_out = lesion_2d * brain_mask.astype(lesion_2d.dtype)

    # Restore original shape if needed
    if img.ndim == 3:
        img_out = img_out[np.newaxis]
    if lesion.ndim == 3:
        lesion_out = lesion_out[np.newaxis]

    return img_out.astype(np.float32), lesion_out.astype(np.float32)


def apply_postprocess_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    z_bins: list[int],
    priors: dict[int, NDArray[np.bool_]],
    cfg: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply post-processing to a batch of samples.

    Args:
        images: Batch of images (B, 1, H, W).
        masks: Batch of masks (B, 1, H, W).
        z_bins: Z-bin indices for each sample (list of length B).
        priors: Loaded prior ROIs.
        cfg: Config with postprocessing params.

    Returns:
        Tuple of (cleaned_images, cleaned_masks), same shapes.
    """
    pp_cfg = cfg.postprocessing.zbin_priors

    device = images.device
    dtype = images.dtype

    B = images.shape[0]
    cleaned_images = []
    cleaned_masks = []

    for i in range(B):
        img = images[i].cpu().numpy()  # (1, H, W)
        mask = masks[i].cpu().numpy()  # (1, H, W)
        z_bin = z_bins[i] if isinstance(z_bins[i], int) else int(z_bins[i])

        img_clean, mask_clean = apply_zbin_prior_postprocess(
            img,
            mask,
            z_bin,
            priors,
            pp_cfg.gaussian_sigma_px,
            pp_cfg.min_component_px,
            pp_cfg.fallback,
            pp_cfg.get("n_first_bins", 0),
            pp_cfg.get("max_components_for_first_bins", 1),
            pp_cfg.get("relaxed_threshold_factor", 0.1),
        )

        cleaned_images.append(torch.from_numpy(img_clean))
        cleaned_masks.append(torch.from_numpy(mask_clean))

    # Stack back to batch
    cleaned_images_tensor = torch.stack(cleaned_images, dim=0).to(device=device, dtype=dtype)
    cleaned_masks_tensor = torch.stack(cleaned_masks, dim=0).to(device=device, dtype=dtype)

    return cleaned_images_tensor, cleaned_masks_tensor
