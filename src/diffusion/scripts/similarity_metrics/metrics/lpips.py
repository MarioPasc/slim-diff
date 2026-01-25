"""Learned Perceptual Image Patch Similarity (LPIPS) metric.

LPIPS measures perceptual similarity using deep features from VGG or AlexNet,
comparing images in a way that correlates with human perception.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .kid import MetricResult

# Try to import lpips package (preferred), fall back to torchmetrics
try:
    import lpips
    LPIPS_BACKEND = "lpips"
except ImportError:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    LPIPS_BACKEND = "torchmetrics"


def preprocess_for_lpips(
    images: np.ndarray,
    target_size: int | None = None,
) -> torch.Tensor:
    """Prepare single-channel images for LPIPS.

    LPIPS expects images in [-1, 1] range with 3 channels.

    Args:
        images: (N, H, W) float32 array in [-1, 1].
        target_size: Optional resize (default: keep original size).

    Returns:
        Tensor: (N, 3, H, W) float32 tensor in [-1, 1].
    """
    # Convert to torch tensor
    images = torch.from_numpy(images).float()  # (N, H, W)

    # Add channel dimension and replicate 3x
    images = images.unsqueeze(1)  # (N, 1, H, W)
    images = images.repeat(1, 3, 1, 1)  # (N, 3, H, W)

    # Resize if specified
    if target_size is not None:
        images = F.interpolate(
            images,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

    # LPIPS expects [-1, 1] range (already in this range)
    images = torch.clamp(images, -1.0, 1.0)

    return images


class LPIPSComputer:
    """Learned Perceptual Image Patch Similarity computation.

    Computes pairwise LPIPS between random pairs of real and synthetic images.
    Lower LPIPS indicates higher perceptual similarity.

    Attributes:
        net: Network backbone ("vgg" or "alex").
        device: Device to use for computation.
        batch_size: Batch size for computation.
    """

    def __init__(
        self,
        net: str = "vgg",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """Initialize the LPIPS computer.

        Args:
            net: Network backbone ("vgg" or "alex").
            device: Device to use (cuda or cpu).
            batch_size: Batch size for computation.
        """
        self.net = net
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        """Lazy load LPIPS model."""
        if self._model is None:
            if LPIPS_BACKEND == "lpips":
                self._model = lpips.LPIPS(net=self.net, verbose=False)
            else:
                self._model = LearnedPerceptualImagePatchSimilarity(
                    net_type=self.net,
                    normalize=True,
                )
            self._model = self._model.to(self.device)
            self._model.eval()
        return self._model

    def compute_pairwise(
        self,
        real_images: np.ndarray,
        synth_images: np.ndarray,
        n_pairs: int = 1000,
        seed: int = 42,
        show_progress: bool = True,
    ) -> MetricResult:
        """Compute LPIPS between random pairs of real and synthetic images.

        Samples n_pairs random (real, synth) pairs and computes mean LPIPS.
        This gives a measure of how perceptually different synthetic images
        are from real images.

        Args:
            real_images: (N, H, W) float32 array in [-1, 1].
            synth_images: (M, H, W) float32 array in [-1, 1].
            n_pairs: Number of random pairs to sample.
            seed: Random seed for reproducibility.
            show_progress: Whether to show progress bar.

        Returns:
            MetricResult with mean LPIPS and std.
        """
        np.random.seed(seed)

        n_real = len(real_images)
        n_synth = len(synth_images)

        # Adjust n_pairs if needed
        n_pairs = min(n_pairs, n_real * n_synth)

        # Sample random pairs
        real_idx = np.random.choice(n_real, n_pairs, replace=True)
        synth_idx = np.random.choice(n_synth, n_pairs, replace=True)

        # Preprocess images
        real_prep = preprocess_for_lpips(real_images)
        synth_prep = preprocess_for_lpips(synth_images)

        # Get model
        model = self._get_model()

        # Compute LPIPS in batches
        lpips_values = []
        iterator = range(0, n_pairs, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="LPIPS pairs", leave=False)

        with torch.no_grad():
            for i in iterator:
                batch_real_idx = real_idx[i : i + self.batch_size]
                batch_synth_idx = synth_idx[i : i + self.batch_size]

                batch_real = real_prep[batch_real_idx].to(self.device)
                batch_synth = synth_prep[batch_synth_idx].to(self.device)

                # Compute LPIPS
                if LPIPS_BACKEND == "lpips":
                    lpips_batch = model(batch_real, batch_synth)
                    lpips_values.extend(lpips_batch.cpu().numpy().flatten().tolist())
                else:
                    # torchmetrics computes mean over batch
                    lpips_batch = model(batch_real, batch_synth)
                    lpips_values.append(float(lpips_batch.cpu().item()))

        # Clean up
        torch.cuda.empty_cache()

        # Compute statistics
        lpips_mean = float(np.mean(lpips_values))
        lpips_std = float(np.std(lpips_values))

        return MetricResult(
            metric_name="lpips",
            value=lpips_mean,
            std=lpips_std,
            n_real=n_real,
            n_synth=n_synth,
            metadata={
                "net": self.net,
                "n_pairs": n_pairs,
                "seed": seed,
                "backend": LPIPS_BACKEND,
            },
        )

    def compute_matched_pairs(
        self,
        images_a: np.ndarray,
        images_b: np.ndarray,
        show_progress: bool = True,
    ) -> MetricResult:
        """Compute LPIPS between matched pairs of images.

        Use this when you have corresponding pairs (e.g., same z-bin, same seed).

        Args:
            images_a: (N, H, W) first set of images.
            images_b: (N, H, W) second set of images (same N).
            show_progress: Whether to show progress bar.

        Returns:
            MetricResult with mean LPIPS and std across pairs.
        """
        assert len(images_a) == len(images_b), "Must have same number of images"

        n_pairs = len(images_a)

        # Preprocess
        prep_a = preprocess_for_lpips(images_a)
        prep_b = preprocess_for_lpips(images_b)

        # Get model
        model = self._get_model()

        # Compute LPIPS in batches
        lpips_values = []
        iterator = range(0, n_pairs, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="LPIPS matched", leave=False)

        with torch.no_grad():
            for i in iterator:
                batch_a = prep_a[i : i + self.batch_size].to(self.device)
                batch_b = prep_b[i : i + self.batch_size].to(self.device)

                if LPIPS_BACKEND == "lpips":
                    lpips_batch = model(batch_a, batch_b)
                    lpips_values.extend(lpips_batch.cpu().numpy().flatten().tolist())
                else:
                    lpips_batch = model(batch_a, batch_b)
                    lpips_values.append(float(lpips_batch.cpu().item()))

        torch.cuda.empty_cache()

        lpips_mean = float(np.mean(lpips_values))
        lpips_std = float(np.std(lpips_values))

        return MetricResult(
            metric_name="lpips",
            value=lpips_mean,
            std=lpips_std,
            n_real=n_pairs,
            n_synth=n_pairs,
            metadata={
                "net": self.net,
                "n_pairs": n_pairs,
                "pair_type": "matched",
                "backend": LPIPS_BACKEND,
            },
        )

    def clear_model(self):
        """Clear the cached model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()


def compute_per_zbin_lpips(
    real_images: np.ndarray,
    real_zbins: np.ndarray,
    synth_images: np.ndarray,
    synth_zbins: np.ndarray,
    valid_zbins: list[int] | None = None,
    n_pairs_per_zbin: int = 100,
    net: str = "vgg",
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Compute LPIPS for each z-bin separately.

    Args:
        real_images: (N, H, W) real images.
        real_zbins: (N,) real z-bins.
        synth_images: (M, H, W) synthetic images.
        synth_zbins: (M,) synthetic z-bins.
        valid_zbins: List of z-bins to compute (default: all unique).
        n_pairs_per_zbin: Number of pairs to sample per z-bin.
        net: Network backbone.
        device: Device to use.
        batch_size: Batch size.
        seed: Random seed.

    Returns:
        List of dicts with per-zbin LPIPS results.
    """
    if valid_zbins is None:
        valid_zbins = sorted(set(real_zbins) | set(synth_zbins))

    lpips_computer = LPIPSComputer(net=net, device=device, batch_size=batch_size)

    results = []
    np.random.seed(seed)

    for zbin in tqdm(valid_zbins, desc="Per-zbin LPIPS"):
        # Filter images for this z-bin
        real_mask = real_zbins == zbin
        synth_mask = synth_zbins == zbin

        real_zbin = real_images[real_mask]
        synth_zbin = synth_images[synth_mask]

        n_real = len(real_zbin)
        n_synth = len(synth_zbin)

        # Skip if insufficient samples
        if n_real < 5 or n_synth < 5:
            results.append({
                "zbin": zbin,
                "lpips": float("nan"),
                "lpips_std": float("nan"),
                "n_real": n_real,
                "n_synth": n_synth,
            })
            continue

        # Adjust n_pairs for available samples
        n_pairs = min(n_pairs_per_zbin, n_real * n_synth)

        # Compute LPIPS
        result = lpips_computer.compute_pairwise(
            real_zbin,
            synth_zbin,
            n_pairs=n_pairs,
            seed=seed + zbin,  # Different seed per zbin for diversity
            show_progress=False,
        )

        results.append({
            "zbin": zbin,
            "lpips": result.value,
            "lpips_std": result.std,
            "n_real": n_real,
            "n_synth": n_synth,
            "n_pairs": n_pairs,
        })

    # Clean up
    lpips_computer.clear_model()

    return results
