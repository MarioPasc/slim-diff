"""Nearest Neighbor Distance in Feature Space.

Computes the distance from each synthetic sample to its nearest neighbor
in the real (training) feature distribution. This metric helps detect:
- Mode collapse (many synthetic samples map to same real neighbor)
- Memorization (very small NN distances suggest copying)
- Coverage gaps (real samples with no nearby synthetic samples)

Uses InceptionV3 features (2048-D) for consistency with KID/FID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .kid import InceptionFeatureExtractor


@dataclass
class NNDistanceResult:
    """Result container for nearest neighbor distance computation."""

    # Per-synthetic sample: distance to nearest real sample
    synth_to_real_mean: float
    synth_to_real_std: float
    synth_to_real_median: float

    # Per-real sample: distance to nearest synthetic sample (coverage)
    real_to_synth_mean: float
    real_to_synth_std: float
    real_to_synth_median: float

    # Sample counts
    n_real: int
    n_synth: int

    # Optional: full distance arrays for further analysis
    synth_to_real_distances: np.ndarray | None = None
    real_to_synth_distances: np.ndarray | None = None

    # Optional: nearest neighbor indices
    synth_nn_indices: np.ndarray | None = None
    real_nn_indices: np.ndarray | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "synth_to_real_mean": self.synth_to_real_mean,
            "synth_to_real_std": self.synth_to_real_std,
            "synth_to_real_median": self.synth_to_real_median,
            "real_to_synth_mean": self.real_to_synth_mean,
            "real_to_synth_std": self.real_to_synth_std,
            "real_to_synth_median": self.real_to_synth_median,
            "n_real": self.n_real,
            "n_synth": self.n_synth,
            **self.metadata,
        }


class FeatureNNComputer:
    """Compute nearest neighbor distances in InceptionV3 feature space.

    This class extracts features from real and synthetic images using InceptionV3,
    then computes nearest neighbor distances to assess distribution similarity.

    Metrics computed:
    - synth_to_real: For each synthetic sample, distance to nearest real sample
      (low values may indicate memorization, high values indicate novelty)
    - real_to_synth: For each real sample, distance to nearest synthetic sample
      (high values indicate coverage gaps in the generative model)

    Example:
        >>> computer = FeatureNNComputer(device="cuda:0")
        >>> result = computer.compute(real_images, synth_images)
        >>> print(f"Mean NN distance: {result.synth_to_real_mean:.4f}")
    """

    def __init__(
        self,
        device: str = "cuda:0",
        batch_size: int = 32,
        chunk_size: int = 1000,
    ):
        """Initialize the NN distance computer.

        Args:
            device: Device for feature extraction.
            batch_size: Batch size for feature extraction.
            chunk_size: Chunk size for distance computation to manage memory.
                        Distance matrix is computed in chunks of this size.
        """
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.feature_extractor = InceptionFeatureExtractor(
            device=device,
            batch_size=batch_size,
        )

    def compute(
        self,
        real_images: np.ndarray | list[np.ndarray],
        synth_images: np.ndarray | list[np.ndarray],
        return_distances: bool = False,
        return_indices: bool = False,
        show_progress: bool = True,
    ) -> NNDistanceResult:
        """Compute NN distances between real and synthetic image sets.

        Args:
            real_images: Real images as (N, H, W) array or list of (H, W) arrays.
                         Values should be in [-1, 1] range.
            synth_images: Synthetic images in same format.
            return_distances: If True, include full distance arrays in result.
            return_indices: If True, include NN index arrays in result.
            show_progress: Whether to show progress bars.

        Returns:
            NNDistanceResult with distance statistics.
        """
        # Convert to numpy arrays if needed
        if isinstance(real_images, list):
            real_images = np.stack(real_images, axis=0)
        if isinstance(synth_images, list):
            synth_images = np.stack(synth_images, axis=0)

        # Extract features
        if show_progress:
            print("Extracting features from real images...")
        real_features = self.feature_extractor.extract_features(
            real_images, show_progress=show_progress
        )

        if show_progress:
            print("Extracting features from synthetic images...")
        synth_features = self.feature_extractor.extract_features(
            synth_images, show_progress=show_progress
        )

        # Compute from features
        return self.compute_from_features(
            real_features=real_features,
            synth_features=synth_features,
            return_distances=return_distances,
            return_indices=return_indices,
            show_progress=show_progress,
        )

    def compute_from_features(
        self,
        real_features: torch.Tensor | np.ndarray,
        synth_features: torch.Tensor | np.ndarray,
        return_distances: bool = False,
        return_indices: bool = False,
        show_progress: bool = True,
    ) -> NNDistanceResult:
        """Compute NN distances from pre-extracted features.

        Args:
            real_features: Real features as (N_real, D) tensor/array.
            synth_features: Synthetic features as (N_synth, D) tensor/array.
            return_distances: If True, include full distance arrays.
            return_indices: If True, include NN index arrays.
            show_progress: Whether to show progress bars.

        Returns:
            NNDistanceResult with distance statistics.
        """
        # Convert to numpy for efficient computation
        if isinstance(real_features, torch.Tensor):
            real_features = real_features.cpu().numpy()
        if isinstance(synth_features, torch.Tensor):
            synth_features = synth_features.cpu().numpy()

        n_real = len(real_features)
        n_synth = len(synth_features)

        # Compute synth-to-real NN distances (chunked for memory efficiency)
        synth_to_real_dists, synth_nn_idx = self._compute_nn_distances_chunked(
            query_features=synth_features,
            reference_features=real_features,
            description="Synth->Real NN" if show_progress else None,
        )

        # Compute real-to-synth NN distances
        real_to_synth_dists, real_nn_idx = self._compute_nn_distances_chunked(
            query_features=real_features,
            reference_features=synth_features,
            description="Real->Synth NN" if show_progress else None,
        )

        return NNDistanceResult(
            synth_to_real_mean=float(np.mean(synth_to_real_dists)),
            synth_to_real_std=float(np.std(synth_to_real_dists)),
            synth_to_real_median=float(np.median(synth_to_real_dists)),
            real_to_synth_mean=float(np.mean(real_to_synth_dists)),
            real_to_synth_std=float(np.std(real_to_synth_dists)),
            real_to_synth_median=float(np.median(real_to_synth_dists)),
            n_real=n_real,
            n_synth=n_synth,
            synth_to_real_distances=synth_to_real_dists if return_distances else None,
            real_to_synth_distances=real_to_synth_dists if return_distances else None,
            synth_nn_indices=synth_nn_idx if return_indices else None,
            real_nn_indices=real_nn_idx if return_indices else None,
        )

    def _compute_nn_distances_chunked(
        self,
        query_features: np.ndarray,
        reference_features: np.ndarray,
        description: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute NN distances in chunks to manage memory.

        Args:
            query_features: Features to find NNs for, shape (N_query, D).
            reference_features: Reference set to search in, shape (N_ref, D).
            description: Description for progress bar.

        Returns:
            Tuple of (nn_distances, nn_indices) arrays of shape (N_query,).
        """
        n_query = len(query_features)
        nn_distances = np.zeros(n_query, dtype=np.float32)
        nn_indices = np.zeros(n_query, dtype=np.int64)

        # Process in chunks
        n_chunks = (n_query + self.chunk_size - 1) // self.chunk_size
        iterator = range(0, n_query, self.chunk_size)

        if description:
            iterator = tqdm(iterator, total=n_chunks, desc=description)

        for start_idx in iterator:
            end_idx = min(start_idx + self.chunk_size, n_query)
            chunk = query_features[start_idx:end_idx]

            # Compute L2 distances: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            # This is more numerically stable than direct subtraction
            chunk_sq = np.sum(chunk ** 2, axis=1, keepdims=True)  # (chunk, 1)
            ref_sq = np.sum(reference_features ** 2, axis=1, keepdims=True).T  # (1, N_ref)
            cross = chunk @ reference_features.T  # (chunk, N_ref)

            # Squared L2 distances
            dists_sq = chunk_sq + ref_sq - 2 * cross  # (chunk, N_ref)
            dists_sq = np.maximum(dists_sq, 0)  # Numerical stability

            # Find nearest neighbors
            nn_idx_chunk = np.argmin(dists_sq, axis=1)
            nn_dist_chunk = np.sqrt(dists_sq[np.arange(len(chunk)), nn_idx_chunk])

            nn_distances[start_idx:end_idx] = nn_dist_chunk
            nn_indices[start_idx:end_idx] = nn_idx_chunk

        return nn_distances, nn_indices

    def compute_diversity_metrics(
        self,
        synth_features: torch.Tensor | np.ndarray,
        k: int = 5,
        show_progress: bool = True,
    ) -> dict[str, float]:
        """Compute intra-synthetic diversity using k-NN distances.

        This measures how spread out the synthetic samples are in feature space.
        Low values indicate mode collapse.

        Args:
            synth_features: Synthetic features as (N, D) tensor/array.
            k: Number of neighbors for k-NN diversity.
            show_progress: Whether to show progress.

        Returns:
            Dict with diversity metrics.
        """
        if isinstance(synth_features, torch.Tensor):
            synth_features = synth_features.cpu().numpy()

        n_synth = len(synth_features)

        # Compute self-NN distances (excluding self)
        all_knn_dists = []

        iterator = range(0, n_synth, self.chunk_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing diversity")

        for start_idx in iterator:
            end_idx = min(start_idx + self.chunk_size, n_synth)
            chunk = synth_features[start_idx:end_idx]

            # Compute distances to all other synthetic samples
            chunk_sq = np.sum(chunk ** 2, axis=1, keepdims=True)
            ref_sq = np.sum(synth_features ** 2, axis=1, keepdims=True).T
            cross = chunk @ synth_features.T
            dists_sq = np.maximum(chunk_sq + ref_sq - 2 * cross, 0)

            # Set self-distance to inf
            for i, global_idx in enumerate(range(start_idx, end_idx)):
                dists_sq[i, global_idx] = np.inf

            # Get k smallest distances
            dists = np.sqrt(dists_sq)
            knn_dists = np.partition(dists, k, axis=1)[:, :k]
            all_knn_dists.append(knn_dists.mean(axis=1))

        knn_mean_dists = np.concatenate(all_knn_dists)

        return {
            "diversity_mean": float(np.mean(knn_mean_dists)),
            "diversity_std": float(np.std(knn_mean_dists)),
            "diversity_median": float(np.median(knn_mean_dists)),
        }


def compute_per_zbin_nn(
    real_images: np.ndarray,
    real_zbins: np.ndarray,
    synth_images: np.ndarray,
    synth_zbins: np.ndarray,
    device: str = "cuda:0",
    batch_size: int = 32,
    min_samples: int = 10,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Compute NN distances stratified by z-bin.

    Args:
        real_images: Real images (N, H, W).
        real_zbins: Z-bin indices for real images (N,).
        synth_images: Synthetic images (M, H, W).
        synth_zbins: Z-bin indices for synthetic images (M,).
        device: Computation device.
        batch_size: Batch size for feature extraction.
        min_samples: Minimum samples per z-bin.
        show_progress: Show progress bars.

    Returns:
        List of dicts with per-zbin NN distance results.
    """
    computer = FeatureNNComputer(device=device, batch_size=batch_size)

    # Get unique z-bins
    all_zbins = np.union1d(np.unique(real_zbins), np.unique(synth_zbins))

    results = []
    for zbin in tqdm(all_zbins, desc="Per-zbin NN", disable=not show_progress):
        real_mask = real_zbins == zbin
        synth_mask = synth_zbins == zbin

        n_real = real_mask.sum()
        n_synth = synth_mask.sum()

        if n_real < min_samples or n_synth < min_samples:
            continue

        result = computer.compute(
            real_images[real_mask],
            synth_images[synth_mask],
            show_progress=False,
        )

        results.append({
            "zbin": int(zbin),
            "n_real": n_real,
            "n_synth": n_synth,
            "synth_to_real_mean": result.synth_to_real_mean,
            "synth_to_real_std": result.synth_to_real_std,
            "synth_to_real_median": result.synth_to_real_median,
            "real_to_synth_mean": result.real_to_synth_mean,
            "real_to_synth_std": result.real_to_synth_std,
            "real_to_synth_median": result.real_to_synth_median,
        })

    return results
