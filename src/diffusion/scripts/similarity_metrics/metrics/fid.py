"""Fréchet Inception Distance (FID) metric.

FID computes the Fréchet distance between Gaussian approximations
of the real and synthetic feature distributions from InceptionV3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import linalg
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from .kid import MetricResult, preprocess_for_inception


class FIDComputer:
    """Fréchet Inception Distance computation.

    FID = ||mu_r - mu_s||^2 + Tr(Sigma_r + Sigma_s - 2*sqrt(Sigma_r*Sigma_s))

    Where mu and Sigma are the mean and covariance of InceptionV3 features.

    Attributes:
        device: Device to use for computation.
        batch_size: Batch size for feature extraction.
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """Initialize the FID computer.

        Args:
            device: Device to use (cuda or cpu).
            batch_size: Batch size for feature extraction.
        """
        self.device = device
        self.batch_size = batch_size

    def compute(
        self,
        real_images: np.ndarray,
        synth_images: np.ndarray,
        show_progress: bool = True,
    ) -> MetricResult:
        """Compute FID between real and synthetic images.

        Args:
            real_images: (N, H, W) float32 array in [-1, 1].
            synth_images: (M, H, W) float32 array in [-1, 1].
            show_progress: Whether to show progress bar.

        Returns:
            MetricResult with FID value.
        """
        n_real = len(real_images)
        n_synth = len(synth_images)

        # Minimum samples for stable covariance estimation
        if n_real < 2048 or n_synth < 2048:
            print(f"Warning: FID typically requires >2048 samples for stability. "
                  f"Got {n_real} real, {n_synth} synth.")

        # Preprocess images
        real_prep = preprocess_for_inception(real_images)
        synth_prep = preprocess_for_inception(synth_images)

        # Initialize FID metric
        fid_metric = FrechetInceptionDistance(
            feature=2048,
            normalize=True,
        ).to(self.device)

        # Update with real images
        self._update_metric_batched(
            fid_metric, real_prep, is_real=True, show_progress=show_progress
        )

        # Update with synthetic images
        self._update_metric_batched(
            fid_metric, synth_prep, is_real=False, show_progress=show_progress
        )

        # Compute FID
        fid_value = fid_metric.compute()

        # Clean up
        fid_metric.cpu()
        torch.cuda.empty_cache()

        return MetricResult(
            metric_name="fid",
            value=float(fid_value.item()),
            std=None,  # FID doesn't have built-in std
            n_real=n_real,
            n_synth=n_synth,
            metadata={
                "feature_extractor": "inceptionv3",
                "feature_dim": 2048,
            },
        )

    def _update_metric_batched(
        self,
        fid_metric: FrechetInceptionDistance,
        images: torch.Tensor,
        is_real: bool,
        show_progress: bool = True,
    ) -> None:
        """Update FID metric with batched processing.

        Args:
            fid_metric: TorchMetrics FID instance.
            images: (N, 3, 299, 299) preprocessed images.
            is_real: True for real images, False for synthetic.
            show_progress: Whether to show progress bar.
        """
        N = images.shape[0]

        iterator = range(0, N, self.batch_size)
        if show_progress:
            desc = "FID real" if is_real else "FID synth"
            iterator = tqdm(iterator, desc=desc, leave=False)

        for i in iterator:
            batch = images[i : i + self.batch_size].to(self.device)
            fid_metric.update(batch, real=is_real)

    def compute_from_features(
        self,
        real_features: torch.Tensor,
        synth_features: torch.Tensor,
    ) -> MetricResult:
        """Compute FID from pre-extracted features.

        Args:
            real_features: (N, D) real feature tensor.
            synth_features: (M, D) synthetic feature tensor.

        Returns:
            MetricResult with FID value.
        """
        # Convert to numpy
        if isinstance(real_features, torch.Tensor):
            real_features = real_features.cpu().numpy()
        if isinstance(synth_features, torch.Tensor):
            synth_features = synth_features.cpu().numpy()

        n_real = real_features.shape[0]
        n_synth = synth_features.shape[0]

        # Compute statistics
        mu_real = np.mean(real_features, axis=0)
        mu_synth = np.mean(synth_features, axis=0)

        sigma_real = np.cov(real_features, rowvar=False)
        sigma_synth = np.cov(synth_features, rowvar=False)

        # Compute FID
        fid = self._calculate_frechet_distance(
            mu_real, sigma_real, mu_synth, sigma_synth
        )

        return MetricResult(
            metric_name="fid",
            value=float(fid),
            std=None,
            n_real=n_real,
            n_synth=n_synth,
            metadata={
                "computed_from": "features",
            },
        )

    @staticmethod
    def _calculate_frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """Calculate Fréchet distance between two Gaussians.

        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

        Args:
            mu1: Mean of first Gaussian.
            sigma1: Covariance of first Gaussian.
            mu2: Mean of second Gaussian.
            sigma2: Covariance of second Gaussian.
            eps: Small constant for numerical stability.

        Returns:
            Fréchet distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Means have different shapes"
        assert sigma1.shape == sigma2.shape, "Covariances have different shapes"

        # Mean difference term
        diff = mu1 - mu2
        mean_term = diff.dot(diff)

        # Covariance term: Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        # Use sqrtm for matrix square root
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # Handle numerical instability
        if not np.isfinite(covmean).all():
            print("Warning: Matrix square root had numerical issues, adding regularization")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Handle complex eigenvalues
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                print(f"Warning: Imaginary component in covmean: {np.max(np.abs(covmean.imag))}")
            covmean = covmean.real

        cov_term = np.trace(sigma1 + sigma2 - 2 * covmean)

        return float(mean_term + cov_term)


def compute_fid_with_bootstrapping(
    real_images: np.ndarray,
    synth_images: np.ndarray,
    n_bootstrap: int = 10,
    sample_fraction: float = 0.8,
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute FID with bootstrap confidence intervals.

    Since FID doesn't have built-in variance estimation, we use
    bootstrap resampling to estimate uncertainty.

    Args:
        real_images: (N, H, W) real images.
        synth_images: (M, H, W) synthetic images.
        n_bootstrap: Number of bootstrap iterations.
        sample_fraction: Fraction of samples to use per iteration.
        device: Device to use.
        batch_size: Batch size for feature extraction.
        seed: Random seed for reproducibility.

    Returns:
        Dict with FID mean, std, and bootstrap values.
    """
    np.random.seed(seed)

    n_real = len(real_images)
    n_synth = len(synth_images)

    n_real_sample = int(n_real * sample_fraction)
    n_synth_sample = int(n_synth * sample_fraction)

    fid_computer = FIDComputer(device=device, batch_size=batch_size)
    fid_values = []

    for i in tqdm(range(n_bootstrap), desc="FID bootstrap"):
        # Sample with replacement
        real_idx = np.random.choice(n_real, n_real_sample, replace=True)
        synth_idx = np.random.choice(n_synth, n_synth_sample, replace=True)

        real_sample = real_images[real_idx]
        synth_sample = synth_images[synth_idx]

        result = fid_computer.compute(real_sample, synth_sample, show_progress=False)
        fid_values.append(result.value)

    return {
        "fid_mean": float(np.mean(fid_values)),
        "fid_std": float(np.std(fid_values)),
        "fid_values": fid_values,
        "n_bootstrap": n_bootstrap,
        "sample_fraction": sample_fraction,
        "n_real": n_real,
        "n_synth": n_synth,
    }
