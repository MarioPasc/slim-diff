"""Kernel Inception Distance (KID) metric with shared feature extraction.

This module provides KID computation using InceptionV3 features, with
support for sharing extracted features with FID computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm


@dataclass
class MetricResult:
    """Container for metric computation results."""

    metric_name: str
    value: float
    std: float | None
    n_real: int
    n_synth: int
    metadata: dict[str, Any]


def preprocess_for_inception(
    images: np.ndarray,
    target_size: int = 299,
) -> torch.Tensor:
    """Prepare single-channel images for InceptionV3.

    Steps:
        1. Denormalize [-1, 1] -> [0, 1]
        2. Replicate channel: (N, H, W) -> (N, 3, H, W)
        3. Resize to 299x299 using bilinear interpolation

    Args:
        images: (N, H, W) float32 array in [-1, 1].
        target_size: Target spatial size (default: 299 for InceptionV3).

    Returns:
        Tensor: (N, 3, 299, 299) float32 tensor in [0, 1].
    """
    # Denormalize [-1, 1] -> [0, 1]
    images = (images + 1.0) / 2.0
    images = np.clip(images, 0.0, 1.0)

    # Convert to torch tensor
    images = torch.from_numpy(images).float()  # (N, H, W)

    # Add channel dimension and replicate 3x
    images = images.unsqueeze(1)  # (N, 1, H, W)
    images = images.repeat(1, 3, 1, 1)  # (N, 3, H, W)

    # Resize to target_size using bilinear interpolation
    if images.shape[-1] != target_size or images.shape[-2] != target_size:
        images = F.interpolate(
            images,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

    return images


class InceptionFeatureExtractor:
    """Shared InceptionV3 feature extraction for KID and FID.

    Extracts 2048-dimensional features from the final pooling layer
    of InceptionV3, preprocessed for the model's expected input format.

    Attributes:
        device: Device to use for computation.
        batch_size: Batch size for feature extraction.
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """Initialize the feature extractor.

        Args:
            device: Device to use (cuda or cpu).
            batch_size: Batch size for feature extraction.
        """
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _get_model(self) -> torch.nn.Module:
        """Lazy load InceptionV3 model."""
        if self._model is None:
            # Use torchmetrics' internal inception network
            from torchmetrics.image.inception import InceptionScore

            # Get the inception model from InceptionScore
            inception_score = InceptionScore(normalize=True)
            self._model = inception_score.inception
            self._model.eval()
            self._model.to(self.device)
        return self._model

    def extract_features(
        self,
        images: np.ndarray,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Extract InceptionV3 features from images.

        Args:
            images: (N, H, W) float32 array in [-1, 1].
            show_progress: Whether to show progress bar.

        Returns:
            Tensor: (N, 2048) feature tensor on CPU.
        """
        # Preprocess images
        images_prep = preprocess_for_inception(images)
        N = images_prep.shape[0]

        # Get model
        model = self._get_model()

        features_list = []
        iterator = range(0, N, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features", leave=False)

        with torch.no_grad():
            for i in iterator:
                batch = images_prep[i : i + self.batch_size].to(self.device)

                # Forward pass through inception
                # InceptionV3 returns different outputs depending on mode
                # We need the 2048-dim features from the avgpool layer
                feat = model(batch)

                # Handle different return types
                if isinstance(feat, tuple):
                    feat = feat[0]  # Main output

                # Flatten if needed (should be (B, 2048))
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)

                features_list.append(feat.cpu())

        # Clear GPU memory
        torch.cuda.empty_cache()

        return torch.cat(features_list, dim=0)

    def clear_model(self):
        """Clear the cached model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()


class KIDComputer:
    """Kernel Inception Distance computation.

    Uses polynomial kernel on InceptionV3 features to compute KID,
    which measures the squared MMD between real and synthetic distributions.

    Attributes:
        subset_size: Number of samples per subset.
        num_subsets: Number of subsets for statistics.
        degree: Polynomial kernel degree.
        device: Device to use for computation.
    """

    def __init__(
        self,
        subset_size: int = 1000,
        num_subsets: int = 100,
        degree: int = 3,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """Initialize the KID computer.

        Args:
            subset_size: Number of samples per subset for KID.
            num_subsets: Number of subsets for statistics.
            degree: Polynomial kernel degree.
            device: Device to use (cuda or cpu).
            batch_size: Batch size for feature extraction.
        """
        self.subset_size = subset_size
        self.num_subsets = num_subsets
        self.degree = degree
        self.device = device
        self.batch_size = batch_size

    def compute(
        self,
        real_images: np.ndarray,
        synth_images: np.ndarray,
        show_progress: bool = True,
    ) -> MetricResult:
        """Compute KID between real and synthetic images.

        Args:
            real_images: (N, H, W) float32 array in [-1, 1].
            synth_images: (N, H, W) float32 array in [-1, 1].
            show_progress: Whether to show progress bar.

        Returns:
            MetricResult with KID value and metadata.
        """
        n_real = len(real_images)
        n_synth = len(synth_images)

        # Adjust subset_size if needed
        min_samples = min(n_real, n_synth)
        subset_size = self.subset_size
        if min_samples < subset_size:
            subset_size = max(min_samples // 2, 10)
            print(f"Warning: Reducing subset_size to {subset_size}")

        # Preprocess images
        real_prep = preprocess_for_inception(real_images)
        synth_prep = preprocess_for_inception(synth_images)

        # Initialize KID metric
        kid_metric = KernelInceptionDistance(
            feature=2048,
            subset_size=subset_size,
            subsets=self.num_subsets,
            degree=self.degree,
            normalize=True,
        )

        # Extract features and update metric
        self._update_metric_batched(
            kid_metric, real_prep, is_real=True, show_progress=show_progress
        )
        self._update_metric_batched(
            kid_metric, synth_prep, is_real=False, show_progress=show_progress
        )

        # Compute KID
        kid_mean, kid_std = kid_metric.compute()

        return MetricResult(
            metric_name="kid",
            value=float(kid_mean.item()),
            std=float(kid_std.item()),
            n_real=n_real,
            n_synth=n_synth,
            metadata={
                "subset_size": subset_size,
                "num_subsets": self.num_subsets,
                "degree": self.degree,
                "feature_extractor": "inceptionv3",
                "feature_dim": 2048,
            },
        )

    def _update_metric_batched(
        self,
        kid_metric: KernelInceptionDistance,
        images: torch.Tensor,
        is_real: bool,
        show_progress: bool = True,
    ) -> None:
        """Update KID metric with batched feature extraction.

        Args:
            kid_metric: TorchMetrics KID instance.
            images: (N, 3, 299, 299) preprocessed images.
            is_real: True for real images, False for synthetic.
            show_progress: Whether to show progress bar.
        """
        N = images.shape[0]
        kid_metric = kid_metric.to(self.device)

        iterator = range(0, N, self.batch_size)
        if show_progress:
            desc = "Real features" if is_real else "Synth features"
            iterator = tqdm(iterator, desc=desc, leave=False)

        for i in iterator:
            batch = images[i : i + self.batch_size].to(self.device)
            kid_metric.update(batch, real=is_real)

        kid_metric.cpu()
        torch.cuda.empty_cache()

    def compute_from_features(
        self,
        real_features: torch.Tensor,
        synth_features: torch.Tensor,
    ) -> MetricResult:
        """Compute KID from pre-extracted features.

        Note: This is more complex because torchmetrics KID expects images.
        For feature-based computation, use manual MMD calculation.

        Args:
            real_features: (N, D) real feature tensor.
            synth_features: (N, D) synthetic feature tensor.

        Returns:
            MetricResult with KID value.
        """
        # Manual KID computation using polynomial kernel MMD
        n_real = real_features.shape[0]
        n_synth = synth_features.shape[0]

        subset_size = min(self.subset_size, n_real, n_synth)

        kid_values = []
        for _ in range(self.num_subsets):
            # Sample subsets
            real_idx = np.random.choice(n_real, subset_size, replace=False)
            synth_idx = np.random.choice(n_synth, subset_size, replace=False)

            real_sub = real_features[real_idx]
            synth_sub = synth_features[synth_idx]

            # Compute polynomial kernel MMD
            kid = self._polynomial_mmd(real_sub, synth_sub, degree=self.degree)
            kid_values.append(kid)

        kid_mean = np.mean(kid_values)
        kid_std = np.std(kid_values)

        return MetricResult(
            metric_name="kid",
            value=float(kid_mean),
            std=float(kid_std),
            n_real=n_real,
            n_synth=n_synth,
            metadata={
                "subset_size": subset_size,
                "num_subsets": self.num_subsets,
                "degree": self.degree,
                "computed_from": "features",
            },
        )

    def _polynomial_mmd(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        degree: int = 3,
        gamma: float | None = None,
        coef0: float = 1.0,
    ) -> float:
        """Compute MMD with polynomial kernel.

        Args:
            X: (N, D) first sample.
            Y: (M, D) second sample.
            degree: Polynomial degree.
            gamma: Kernel coefficient (default: 1/D).
            coef0: Independent term.

        Returns:
            MMD estimate.
        """
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        # Compute kernel matrices
        XX = self._polynomial_kernel(X, X, degree, gamma, coef0)
        YY = self._polynomial_kernel(Y, Y, degree, gamma, coef0)
        XY = self._polynomial_kernel(X, Y, degree, gamma, coef0)

        # MMD estimate
        n = X.shape[0]
        m = Y.shape[0]

        mmd = (
            (XX.sum() - XX.trace()) / (n * (n - 1))
            + (YY.sum() - YY.trace()) / (m * (m - 1))
            - 2 * XY.mean()
        )

        return float(mmd.item())

    def _polynomial_kernel(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        degree: int,
        gamma: float,
        coef0: float,
    ) -> torch.Tensor:
        """Compute polynomial kernel matrix.

        Args:
            X: (N, D) first sample.
            Y: (M, D) second sample.
            degree: Polynomial degree.
            gamma: Kernel coefficient.
            coef0: Independent term.

        Returns:
            (N, M) kernel matrix.
        """
        return (gamma * X @ Y.T + coef0) ** degree


def compute_per_zbin_kid(
    real_images: np.ndarray,
    real_zbins: np.ndarray,
    synth_images: np.ndarray,
    synth_zbins: np.ndarray,
    valid_zbins: list[int] | None = None,
    subset_size: int = 250,
    num_subsets: int = 100,
    degree: int = 3,
    device: str = "cuda",
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Compute KID for each z-bin separately.

    Args:
        real_images: (N, H, W) real images.
        real_zbins: (N,) real z-bins.
        synth_images: (M, H, W) synthetic images.
        synth_zbins: (M,) synthetic z-bins.
        valid_zbins: List of z-bins to compute (default: all unique).
        subset_size: KID subset size.
        num_subsets: Number of KID subsets.
        degree: Polynomial kernel degree.
        device: Device to use.
        batch_size: Batch size for feature extraction.

    Returns:
        List of dicts with per-zbin KID results.
    """
    if valid_zbins is None:
        valid_zbins = sorted(set(real_zbins) | set(synth_zbins))

    kid_computer = KIDComputer(
        subset_size=subset_size,
        num_subsets=num_subsets,
        degree=degree,
        device=device,
        batch_size=batch_size,
    )

    results = []
    for zbin in tqdm(valid_zbins, desc="Per-zbin KID"):
        # Filter images for this z-bin
        real_mask = real_zbins == zbin
        synth_mask = synth_zbins == zbin

        real_zbin = real_images[real_mask]
        synth_zbin = synth_images[synth_mask]

        n_real = len(real_zbin)
        n_synth = len(synth_zbin)

        # Skip if insufficient samples
        if n_real < 10 or n_synth < 10:
            results.append({
                "zbin": zbin,
                "kid": float("nan"),
                "kid_std": float("nan"),
                "n_real": n_real,
                "n_synth": n_synth,
            })
            continue

        # Compute KID
        result = kid_computer.compute(real_zbin, synth_zbin, show_progress=False)
        results.append({
            "zbin": zbin,
            "kid": result.value,
            "kid_std": result.std,
            "n_real": n_real,
            "n_synth": n_synth,
        })

    return results
