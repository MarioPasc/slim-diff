"""Kernel Inception Distance (KID) computation for evaluating synthetic replicas.

This module provides core functionality for computing KID metrics between
synthetic replicas and real test data, using InceptionV3 features via torchmetrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm


def load_test_slices(
    test_csv: Path, valid_zbins: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load test set images and z-bins from CSV.

    Args:
        test_csv: Path to test.csv file with columns:
            filepath, subject_id, z_index, z_bin, etc.
        valid_zbins: List of valid z-bins to include (default: 0-29).
            If None, includes all z-bins.

    Returns:
        images: (N, H, W) float32 array in [-1, 1]
        zbins: (N,) int array
    """
    if valid_zbins is None:
        valid_zbins = list(range(30))

    # Load CSV
    df = pd.read_csv(test_csv)

    # Base directory for resolving relative paths
    base_dir = test_csv.parent

    # Filter by valid z-bins
    df_filtered = df[df["z_bin"].isin(valid_zbins)]

    images_list = []
    zbins_list = []

    # Load each slice
    for _, row in tqdm(
        df_filtered.iterrows(),
        total=len(df_filtered),
        desc="Loading test slices",
        leave=False,
    ):
        filepath = base_dir / row["filepath"]

        if not filepath.exists():
            print(f"Warning: Missing file {filepath}, skipping...")
            continue

        # Load NPZ slice
        data = np.load(filepath)
        image = data["image"]  # (H, W)

        # Ensure float32
        image = image.astype(np.float32)

        images_list.append(image)
        zbins_list.append(row["z_bin"])

    # Stack into arrays
    images = np.stack(images_list, axis=0)  # (N, H, W)
    zbins = np.array(zbins_list, dtype=np.int32)

    print(f"Loaded {len(images)} test slices from {len(df_filtered)} entries")
    return images, zbins


def load_replica(
    replica_path: Path, valid_zbins: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load replica images and z-bins from NPZ file.

    Args:
        replica_path: Path to replica_{ID:03d}.npz file
        valid_zbins: List of valid z-bins to include (default: 0-29).
            If None, includes all z-bins.

    Returns:
        images: (N, H, W) float32 array in [-1, 1]
        zbins: (N,) int array
    """
    if valid_zbins is None:
        valid_zbins = list(range(30))

    # Load replica NPZ
    data = np.load(replica_path)
    images = data["images"]  # (N, H, W)
    zbins = data["zbin"]  # (N,)

    # Filter by valid z-bins
    mask = np.isin(zbins, valid_zbins)
    images = images[mask]
    zbins = zbins[mask]

    # Ensure float32
    images = images.astype(np.float32)
    zbins = zbins.astype(np.int32)

    print(f"Loaded {len(images)} samples from {replica_path.name} (filtered to bins 0-29)")
    return images, zbins


def preprocess_for_inception(
    images: np.ndarray, target_size: int = 299
) -> torch.Tensor:
    """Prepare single-channel images for InceptionV3.

    Steps:
        1. Denormalize [-1, 1] → [0, 1]
        2. Replicate channel: (N, H, W) → (N, 3, H, W)
        3. Resize to 299x299 using bilinear interpolation
        4. Convert to torch.Tensor

    Args:
        images: (N, H, W) float32 array in [-1, 1]
        target_size: Target spatial size (default: 299 for InceptionV3)

    Returns:
        images: (N, 3, 299, 299) float32 tensor in [0, 1]
    """
    # Denormalize [-1, 1] → [0, 1]
    images = (images + 1.0) / 2.0

    # Convert to torch tensor
    images = torch.from_numpy(images)  # (N, H, W)

    # Add channel dimension and replicate 3x: (N, H, W) → (N, 1, H, W) → (N, 3, H, W)
    images = images.unsqueeze(1)  # (N, 1, H, W)
    images = images.repeat(1, 3, 1, 1)  # (N, 3, H, W)

    # Resize to target_size using bilinear interpolation
    if images.shape[-1] != target_size or images.shape[-2] != target_size:
        images = F.interpolate(
            images, size=(target_size, target_size), mode="bilinear", align_corners=False
        )

    return images


def extract_features_batched(
    images: torch.Tensor,
    kid_metric: KernelInceptionDistance,
    is_real: bool,
    batch_size: int = 32,
    device: str = "cuda",
) -> None:
    """Extract InceptionV3 features in batches and update KID metric.

    Args:
        images: (N, 3, 299, 299) preprocessed images in [0, 1]
        kid_metric: TorchMetrics KID instance
        is_real: True for test set, False for synthetic
        batch_size: Batch size for feature extraction
        device: cuda or cpu
    """
    N = images.shape[0]
    kid_metric = kid_metric.to(device)

    # Process in batches
    for i in tqdm(
        range(0, N, batch_size),
        desc=f"Extracting features ({'real' if is_real else 'fake'})",
        leave=False,
    ):
        batch = images[i : i + batch_size].to(device)

        # Convert to uint8 [0, 255] as expected by torchmetrics with normalize=False
        # Actually, torchmetrics expects float [0, 1] with normalize=True
        # Let's use normalize=True mode to pass float [0, 1] directly
        kid_metric.update(batch, real=is_real)

    # Move back to CPU to free GPU memory
    kid_metric = kid_metric.cpu()
    torch.cuda.empty_cache()


def compute_global_kid(
    real_images: np.ndarray,
    synth_images: np.ndarray,
    subset_size: int = 1000,
    num_subsets: int = 100,
    degree: int = 3,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Compute global KID across all samples.

    Args:
        real_images: (N, H, W) float32 array in [-1, 1]
        synth_images: (N, H, W) float32 array in [-1, 1]
        subset_size: Number of samples per subset for KID
        num_subsets: Number of subsets for KID statistics
        degree: Polynomial kernel degree
        batch_size: Batch size for feature extraction
        device: cuda or cpu

    Returns:
        Dictionary with:
            - kid_mean: float
            - kid_std: float
            - n_real: int
            - n_synth: int
    """
    # Adjust subset_size if needed
    n_real = len(real_images)
    n_synth = len(synth_images)
    min_samples = min(n_real, n_synth)

    if min_samples < subset_size:
        adjusted_subset_size = max(min_samples // 2, 10)
        print(
            f"Warning: Too few samples ({min_samples}). "
            f"Reducing subset_size from {subset_size} to {adjusted_subset_size}"
        )
        subset_size = adjusted_subset_size

    # Preprocess images
    real_prep = preprocess_for_inception(real_images)
    synth_prep = preprocess_for_inception(synth_images)

    # Initialize KID metric with normalize=True to accept float [0, 1]
    kid_metric = KernelInceptionDistance(
        feature=2048,  # Use default InceptionV3 features
        subset_size=subset_size,
        subsets=num_subsets,
        degree=degree,
        normalize=True,  # Expect float [0, 1]
    )

    # Extract features
    extract_features_batched(real_prep, kid_metric, is_real=True, batch_size=batch_size, device=device)
    extract_features_batched(synth_prep, kid_metric, is_real=False, batch_size=batch_size, device=device)

    # Compute KID
    kid_mean, kid_std = kid_metric.compute()

    return {
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
        "n_real": n_real,
        "n_synth": n_synth,
    }


def compute_zbin_kid(
    real_images: np.ndarray,
    real_zbins: np.ndarray,
    synth_images: np.ndarray,
    synth_zbins: np.ndarray,
    target_zbin: int,
    subset_size: int = 1000,
    num_subsets: int = 100,
    degree: int = 3,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Compute KID for specific bin and rest-of-bins.

    Args:
        real_images: (N, H, W) float32 array in [-1, 1]
        real_zbins: (N,) int array
        synth_images: (N, H, W) float32 array in [-1, 1]
        synth_zbins: (N,) int array
        target_zbin: Target z-bin to isolate
        subset_size: Number of samples per subset for KID
        num_subsets: Number of subsets for KID statistics
        degree: Polynomial kernel degree
        batch_size: Batch size for feature extraction
        device: cuda or cpu

    Returns:
        Dictionary with:
            - kid_zbin_mean: float
            - kid_zbin_std: float
            - n_real_zbin: int
            - n_synth_zbin: int
            - kid_rest_mean: float
            - kid_rest_std: float
            - n_real_rest: int
            - n_synth_rest: int
    """
    # Separate bin vs rest
    real_bin_mask = real_zbins == target_zbin
    synth_bin_mask = synth_zbins == target_zbin

    real_bin = real_images[real_bin_mask]
    synth_bin = synth_images[synth_bin_mask]

    real_rest = real_images[~real_bin_mask]
    synth_rest = synth_images[~synth_bin_mask]

    # Initialize result dict
    result = {
        "n_real_zbin": len(real_bin),
        "n_synth_zbin": len(synth_bin),
        "n_real_rest": len(real_rest),
        "n_synth_rest": len(synth_rest),
    }

    # Compute KID for bin
    if len(real_bin) >= 10 and len(synth_bin) >= 10:
        kid_bin_result = compute_global_kid(
            real_bin,
            synth_bin,
            subset_size=subset_size,
            num_subsets=num_subsets,
            degree=degree,
            batch_size=batch_size,
            device=device,
        )
        result["kid_zbin_mean"] = kid_bin_result["kid_mean"]
        result["kid_zbin_std"] = kid_bin_result["kid_std"]
    else:
        print(f"Warning: Bin {target_zbin} has too few samples, skipping KID computation")
        result["kid_zbin_mean"] = float("nan")
        result["kid_zbin_std"] = float("nan")

    # Compute KID for rest
    if len(real_rest) >= 10 and len(synth_rest) >= 10:
        kid_rest_result = compute_global_kid(
            real_rest,
            synth_rest,
            subset_size=subset_size,
            num_subsets=num_subsets,
            degree=degree,
            batch_size=batch_size,
            device=device,
        )
        result["kid_rest_mean"] = kid_rest_result["kid_mean"]
        result["kid_rest_std"] = kid_rest_result["kid_std"]
    else:
        print(f"Warning: Rest (excluding bin {target_zbin}) has too few samples, skipping KID computation")
        result["kid_rest_mean"] = float("nan")
        result["kid_rest_std"] = float("nan")

    return result
