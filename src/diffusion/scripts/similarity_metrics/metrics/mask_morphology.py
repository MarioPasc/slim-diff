"""Mask morphology distance metrics for evaluating generated lesion masks.

Computes Maximum Mean Discrepancy (MMD) on morphological features extracted
from binary lesion masks, analogous to KID for image quality evaluation.

The approach:
1. Extract morphological features (area, circularity, solidity, etc.) per lesion
2. Aggregate into feature distributions across all lesions
3. Compute polynomial kernel MMD between real and synthetic distributions
4. Provides per-feature Wasserstein distances for diagnostic breakdown

References:
    - Gretton et al. (2012) "A Kernel Two-Sample Test" - MMD methodology
    - Taha & Hanbury (2015) "Metrics for evaluating 3D medical image segmentation"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label
from scipy.stats import wasserstein_distance
from skimage.measure import regionprops
from tqdm import tqdm

from .kid import MetricResult


@dataclass
class MorphologicalFeatures:
    """Container for extracted morphological features per mask.

    Attributes:
        n_lesions: Number of valid lesions in the mask.
        features: (N_lesions, D) array of per-lesion features.
        feature_names: List of feature names in order.
        aggregated: Dict of sample-level aggregated statistics (optional).
    """

    n_lesions: int
    features: NDArray[np.float64]  # (N_lesions, D)
    feature_names: list[str]
    aggregated: dict[str, float] = field(default_factory=dict)


class MorphologicalFeatureExtractor:
    """Extract morphological features from binary lesion masks.

    Features extracted per connected component (lesion):
    - area: Lesion area in pixels
    - perimeter: Boundary length
    - circularity: 4*pi*area/perimeter^2 (shape regularity, 1=circle)
    - solidity: area/convex_hull_area (compactness)
    - extent: area/bbox_area (fill factor)
    - eccentricity: From fitted ellipse (0=circle, 1=line)
    - major_axis_length: Major axis of fitted ellipse
    - minor_axis_length: Minor axis of fitted ellipse
    - equivalent_diameter: sqrt(4*area/pi)

    Attributes:
        min_lesion_size_px: Minimum lesion size to include.
        feature_names: List of feature names extracted.
    """

    DEFAULT_FEATURES: list[str] = [
        "area",
        "perimeter",
        "circularity",
        "solidity",
        "extent",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "equivalent_diameter",
    ]

    def __init__(
        self,
        min_lesion_size_px: int = 5,
        features: list[str] | None = None,
    ) -> None:
        """Initialize the feature extractor.

        Args:
            min_lesion_size_px: Minimum lesion size to include (pixels).
            features: Subset of features to extract (default: all).
        """
        self.min_lesion_size_px = min_lesion_size_px
        self.feature_names = features if features is not None else self.DEFAULT_FEATURES.copy()

    def extract(self, mask: NDArray[np.float32]) -> MorphologicalFeatures:
        """Extract features from a single 2D mask.

        Args:
            mask: 2D mask in {-1, +1} (JSDDPM format) or {0, 1}.

        Returns:
            MorphologicalFeatures container with per-lesion features.
        """
        # Convert {-1, +1} -> {0, 1}
        binary_mask = self._preprocess_mask(mask)

        # Label connected components
        labeled_mask, num_lesions = label(binary_mask)

        if num_lesions == 0:
            return MorphologicalFeatures(
                n_lesions=0,
                features=np.empty((0, len(self.feature_names)), dtype=np.float64),
                feature_names=self.feature_names,
            )

        # Get region properties
        regions = regionprops(labeled_mask)

        # Filter by minimum size
        valid_regions = [r for r in regions if r.area >= self.min_lesion_size_px]

        if not valid_regions:
            return MorphologicalFeatures(
                n_lesions=0,
                features=np.empty((0, len(self.feature_names)), dtype=np.float64),
                feature_names=self.feature_names,
            )

        # Extract features per lesion
        features_list = []
        for region in valid_regions:
            feat_dict = self._extract_region_features(region)
            features_list.append([feat_dict[name] for name in self.feature_names])

        features = np.array(features_list, dtype=np.float64)

        return MorphologicalFeatures(
            n_lesions=len(valid_regions),
            features=features,
            feature_names=self.feature_names,
            aggregated=self._compute_aggregated_stats(features),
        )

    def extract_batch(
        self,
        masks: NDArray[np.float32],
        show_progress: bool = True,
        desc: str = "Extracting mask features",
    ) -> list[MorphologicalFeatures]:
        """Extract features from a batch of masks.

        Args:
            masks: (N, H, W) array of masks in {-1, +1}.
            show_progress: Whether to show progress bar.
            desc: Description for progress bar.

        Returns:
            List of MorphologicalFeatures, one per mask.
        """
        iterator = range(masks.shape[0])
        if show_progress:
            iterator = tqdm(iterator, desc=desc, leave=False)

        features_list = []
        for i in iterator:
            features = self.extract(masks[i])
            features_list.append(features)

        return features_list

    def aggregate_to_distribution(
        self,
        features_list: list[MorphologicalFeatures],
    ) -> NDArray[np.float64]:
        """Aggregate features from multiple masks into a single distribution.

        Concatenates all per-lesion features across masks to form the
        overall distribution for MMD computation.

        Args:
            features_list: List of MorphologicalFeatures from multiple masks.

        Returns:
            (N_total_lesions, D) array of features.
        """
        all_features = []
        for feat in features_list:
            if feat.n_lesions > 0:
                all_features.append(feat.features)

        if not all_features:
            return np.empty((0, len(self.feature_names)), dtype=np.float64)

        return np.vstack(all_features)

    def _preprocess_mask(self, mask: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Convert mask from {-1, +1} to binary {0, 1}.

        Args:
            mask: 2D mask in {-1, +1} (JSDDPM format).

        Returns:
            Binary mask in {0, 1}.
        """
        # Threshold at 0: >0 is lesion, <=0 is background
        binary_mask = (mask > 0).astype(np.uint8)
        return binary_mask

    def _extract_region_features(self, region) -> dict[str, float]:
        """Extract all features from a single region.

        Args:
            region: skimage regionprops region object.

        Returns:
            Dict mapping feature name to value.
        """
        feat = {}

        # Direct from regionprops - use newer attribute names (skimage >= 0.26)
        feat["area"] = float(region.area)
        feat["perimeter"] = float(region.perimeter) if region.perimeter > 0 else 0.0

        # Use hasattr to avoid deprecation warnings entirely
        if hasattr(region, "axis_major_length"):
            feat["major_axis_length"] = float(region.axis_major_length)
            feat["minor_axis_length"] = float(region.axis_minor_length)
        else:
            feat["major_axis_length"] = float(region.major_axis_length)
            feat["minor_axis_length"] = float(region.minor_axis_length)

        if hasattr(region, "equivalent_diameter_area"):
            feat["equivalent_diameter"] = float(region.equivalent_diameter_area)
        else:
            feat["equivalent_diameter"] = float(region.equivalent_diameter)

        feat["eccentricity"] = float(region.eccentricity)
        feat["extent"] = float(region.extent)  # area / bbox_area
        feat["solidity"] = float(region.solidity)  # area / convex_hull_area

        # Computed features
        # Circularity: 4*pi*area/perimeter^2 (1.0 = perfect circle)
        if region.perimeter > 0:
            circularity = 4 * np.pi * region.area / (region.perimeter**2)
            feat["circularity"] = min(float(circularity), 1.0)  # Clamp to [0, 1]
        else:
            feat["circularity"] = 0.0

        return feat

    def _compute_aggregated_stats(
        self, features: NDArray[np.float64]
    ) -> dict[str, float]:
        """Compute aggregated statistics from features.

        Args:
            features: (N_lesions, D) feature array.

        Returns:
            Dict of aggregated statistics.
        """
        if features.shape[0] == 0:
            return {}

        stats = {}
        for i, name in enumerate(self.feature_names):
            col = features[:, i]
            stats[f"{name}_mean"] = float(np.mean(col))
            stats[f"{name}_std"] = float(np.std(col))

        return stats


class MaskMorphologyDistanceComputer:
    """Compute MMD on morphological features between real and synthetic masks.

    Uses polynomial kernel MMD (same as KID) on morphological feature vectors:
    - Extract features from all masks
    - Concatenate per-lesion features into distributions
    - Compute MMD with subsampling for variance estimation

    This metric answers: "Does the distribution of lesion shapes in synthetic
    masks match the real distribution?"

    Attributes:
        feature_extractor: MorphologicalFeatureExtractor instance.
        subset_size: Number of lesions per subset for MMD.
        num_subsets: Number of subsets for variance estimation.
        degree: Polynomial kernel degree.
        normalize_features: Whether to z-score normalize features.
    """

    def __init__(
        self,
        min_lesion_size_px: int = 5,
        features: list[str] | None = None,
        subset_size: int = 500,
        num_subsets: int = 100,
        degree: int = 3,
        normalize_features: bool = True,
    ) -> None:
        """Initialize the MMD-MF computer.

        Args:
            min_lesion_size_px: Minimum lesion size for feature extraction.
            features: Subset of features to use (default: all 9).
            subset_size: Number of lesions per subset for MMD.
            num_subsets: Number of subsets for variance estimation.
            degree: Polynomial kernel degree.
            normalize_features: Whether to z-score normalize features.
        """
        self.feature_extractor = MorphologicalFeatureExtractor(
            min_lesion_size_px=min_lesion_size_px,
            features=features,
        )
        self.subset_size = subset_size
        self.num_subsets = num_subsets
        self.degree = degree
        self.normalize_features = normalize_features

        # Cache for real features (expensive to compute repeatedly)
        self._real_features_cache: NDArray[np.float64] | None = None
        self._real_features_cache_key: int | None = None

    def extract_features(
        self,
        masks: NDArray[np.float32],
        show_progress: bool = True,
        desc: str = "Extracting features",
    ) -> NDArray[np.float64]:
        """Extract and aggregate features from masks.

        Args:
            masks: (N, H, W) masks in {-1, +1}.
            show_progress: Whether to show progress bar.
            desc: Description for progress bar.

        Returns:
            (N_lesions, D) aggregated feature array.
        """
        features_list = self.feature_extractor.extract_batch(masks, show_progress=show_progress, desc=desc)
        return self.feature_extractor.aggregate_to_distribution(features_list)

    def get_real_features(
        self,
        real_masks: NDArray[np.float32],
        show_progress: bool = True,
    ) -> NDArray[np.float64]:
        """Get real features with caching.

        Args:
            real_masks: (N, H, W) real masks.
            show_progress: Whether to show progress bar.

        Returns:
            Cached or freshly computed features.
        """
        # Use id() as cache key (same array object = same features)
        cache_key = id(real_masks)
        if self._real_features_cache is not None and self._real_features_cache_key == cache_key:
            return self._real_features_cache

        # Extract and cache
        self._real_features_cache = self.extract_features(
            real_masks, show_progress=show_progress, desc="Extracting real features"
        )
        self._real_features_cache_key = cache_key
        return self._real_features_cache

    def compute(
        self,
        real_masks: NDArray[np.float32],
        synth_masks: NDArray[np.float32],
        show_progress: bool = True,
        real_features: NDArray[np.float64] | None = None,
        synth_features: NDArray[np.float64] | None = None,
    ) -> tuple[MetricResult, NDArray[np.float64], NDArray[np.float64]]:
        """Compute MMD-MF between real and synthetic masks.

        Args:
            real_masks: (N, H, W) real masks in {-1, +1}.
            synth_masks: (M, H, W) synthetic masks in {-1, +1}.
            show_progress: Whether to show progress bar.
            real_features: Pre-computed real features (optional, for caching).
            synth_features: Pre-computed synth features (optional).

        Returns:
            Tuple of (MetricResult, real_features, synth_features).
            The features are returned for reuse in Wasserstein computation.
        """
        # Extract features (with caching for real)
        if real_features is None:
            real_features = self.get_real_features(real_masks, show_progress=show_progress)

        if synth_features is None:
            synth_features = self.extract_features(
                synth_masks, show_progress=show_progress, desc="Extracting synth features"
            )

        n_real_lesions = real_features.shape[0]
        n_synth_lesions = synth_features.shape[0]

        # Check for sufficient lesions
        min_lesions = min(n_real_lesions, n_synth_lesions)
        if min_lesions < 10:
            return (
                MetricResult(
                    metric_name="mmd_mf",
                    value=float("nan"),
                    std=float("nan"),
                    n_real=len(real_masks),
                    n_synth=len(synth_masks),
                    metadata={
                        "error": f"Insufficient lesions for MMD ({min_lesions} < 10)",
                        "n_real_lesions": n_real_lesions,
                        "n_synth_lesions": n_synth_lesions,
                    },
                ),
                real_features,
                synth_features,
            )

        # Normalize features (using real statistics only)
        real_norm, synth_norm = real_features, synth_features
        if self.normalize_features:
            real_norm, synth_norm = self._normalize_features(real_features, synth_features)

        # Adjust subset_size if needed
        subset_size = min(self.subset_size, n_real_lesions, n_synth_lesions)

        # Compute MMD with subsampling for variance
        mmd_values = []
        for _ in range(self.num_subsets):
            # Random subsample
            real_idx = np.random.choice(n_real_lesions, subset_size, replace=False)
            synth_idx = np.random.choice(n_synth_lesions, subset_size, replace=False)

            real_sub = real_norm[real_idx]
            synth_sub = synth_norm[synth_idx]

            mmd = self._polynomial_mmd(real_sub, synth_sub)
            mmd_values.append(mmd)

        mmd_mean = float(np.mean(mmd_values))
        mmd_std = float(np.std(mmd_values))

        return (
            MetricResult(
                metric_name="mmd_mf",
                value=mmd_mean,
                std=mmd_std,
                n_real=len(real_masks),
                n_synth=len(synth_masks),
                metadata={
                    "n_real_lesions": n_real_lesions,
                    "n_synth_lesions": n_synth_lesions,
                    "subset_size": subset_size,
                    "num_subsets": self.num_subsets,
                    "degree": self.degree,
                    "feature_names": self.feature_extractor.feature_names,
                    "normalized": self.normalize_features,
                },
            ),
            real_features,
            synth_features,
        )

    def compute_per_feature_wasserstein(
        self,
        real_masks: NDArray[np.float32] | None = None,
        synth_masks: NDArray[np.float32] | None = None,
        real_features: NDArray[np.float64] | None = None,
        synth_features: NDArray[np.float64] | None = None,
        show_progress: bool = True,
    ) -> dict[str, float]:
        """Compute per-feature 1D Wasserstein distances.

        Provides diagnostic breakdown of which morphological features
        differ most between real and synthetic masks.

        Args:
            real_masks: (N, H, W) real masks in {-1, +1}. Not needed if real_features provided.
            synth_masks: (M, H, W) synthetic masks in {-1, +1}. Not needed if synth_features provided.
            real_features: Pre-computed real features (from compute()).
            synth_features: Pre-computed synth features (from compute()).
            show_progress: Whether to show progress bar.

        Returns:
            Dict mapping feature_name -> Wasserstein distance.
            Includes "geometric_mean" as aggregate measure.
        """
        # Use pre-computed features if available
        if real_features is None:
            if real_masks is None:
                raise ValueError("Either real_masks or real_features must be provided")
            real_features = self.get_real_features(real_masks, show_progress=show_progress)

        if synth_features is None:
            if synth_masks is None:
                raise ValueError("Either synth_masks or synth_features must be provided")
            synth_features = self.extract_features(
                synth_masks, show_progress=show_progress, desc="Extracting synth features"
            )

        feature_names = self.feature_extractor.feature_names

        # Check for sufficient data
        if real_features.shape[0] < 10 or synth_features.shape[0] < 10:
            return {name: float("nan") for name in feature_names}

        # Normalize (using real statistics)
        if self.normalize_features:
            real_norm, synth_norm = self._normalize_features(real_features, synth_features)
        else:
            real_norm, synth_norm = real_features, synth_features

        # Compute 1D Wasserstein per feature
        wasserstein_distances = {}
        for i, name in enumerate(feature_names):
            real_1d = real_norm[:, i]
            synth_1d = synth_norm[:, i]

            w_dist = wasserstein_distance(real_1d, synth_1d)
            wasserstein_distances[name] = float(w_dist)

        # Add geometric mean as aggregate
        valid_dists = [d for d in wasserstein_distances.values() if not np.isnan(d)]
        if valid_dists:
            # Add small epsilon to avoid log(0)
            wasserstein_distances["geometric_mean"] = float(
                np.exp(np.mean(np.log(np.array(valid_dists) + 1e-8)))
            )
        else:
            wasserstein_distances["geometric_mean"] = float("nan")

        return wasserstein_distances

    def _normalize_features(
        self,
        real_features: NDArray[np.float64],
        synth_features: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Z-score normalize features using real data statistics.

        Normalization is critical for MMD since kernel depends on distances.
        Uses real data statistics only to avoid leaking synthetic info.

        Args:
            real_features: (N, D) real feature array.
            synth_features: (M, D) synthetic feature array.

        Returns:
            Tuple of normalized (real_features, synth_features).
        """
        # Compute statistics from real data only
        mean = np.mean(real_features, axis=0, keepdims=True)
        std = np.std(real_features, axis=0, keepdims=True)

        # Avoid division by zero
        std = np.maximum(std, 1e-8)

        real_norm = (real_features - mean) / std
        synth_norm = (synth_features - mean) / std

        return real_norm, synth_norm

    def _polynomial_mmd(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
    ) -> float:
        """Compute polynomial kernel MMD (unbiased estimator).

        Same implementation as KID in kid.py.

        k(x, y) = (gamma * x.T @ y + coef0)^degree

        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

        Args:
            X: (N, D) first sample (real).
            Y: (M, D) second sample (synthetic).

        Returns:
            MMD estimate (squared).
        """
        gamma = 1.0 / X.shape[1]  # 1/D
        coef0 = 1.0

        # Kernel matrices
        XX = self._polynomial_kernel(X, X, gamma, coef0)
        YY = self._polynomial_kernel(Y, Y, gamma, coef0)
        XY = self._polynomial_kernel(X, Y, gamma, coef0)

        n = X.shape[0]
        m = Y.shape[0]

        # Unbiased estimator (excludes diagonal)
        mmd = (
            (XX.sum() - np.trace(XX)) / (n * (n - 1))
            + (YY.sum() - np.trace(YY)) / (m * (m - 1))
            - 2 * XY.mean()
        )

        return float(mmd)

    def _polynomial_kernel(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        gamma: float,
        coef0: float,
    ) -> NDArray[np.float64]:
        """Compute polynomial kernel matrix.

        Args:
            X: (N, D) first sample.
            Y: (M, D) second sample.
            gamma: Kernel coefficient (1/D).
            coef0: Independent term.

        Returns:
            (N, M) kernel matrix.
        """
        return (gamma * X @ Y.T + coef0) ** self.degree


def compute_per_zbin_mmd_mf(
    real_masks: NDArray[np.float32],
    real_zbins: NDArray[np.int32],
    synth_masks: NDArray[np.float32],
    synth_zbins: NDArray[np.int32],
    valid_zbins: list[int] | None = None,
    min_lesion_size_px: int = 5,
    features: list[str] | None = None,
    subset_size: int = 200,
    num_subsets: int = 50,
    degree: int = 3,
    normalize_features: bool = True,
) -> list[dict[str, Any]]:
    """Compute MMD-MF for each z-bin separately.

    Args:
        real_masks: (N, H, W) real masks in {-1, +1}.
        real_zbins: (N,) real z-bin indices.
        synth_masks: (M, H, W) synthetic masks in {-1, +1}.
        synth_zbins: (M,) synthetic z-bin indices.
        valid_zbins: List of z-bins to compute (default: all unique).
        min_lesion_size_px: Minimum lesion size.
        features: Subset of features to use.
        subset_size: Lesions per subset (smaller for per-zbin).
        num_subsets: Number of subsets.
        degree: Polynomial kernel degree.
        normalize_features: Whether to normalize features.

    Returns:
        List of dicts with per-zbin results.
    """
    if valid_zbins is None:
        valid_zbins = sorted(set(real_zbins) | set(synth_zbins))

    mmd_computer = MaskMorphologyDistanceComputer(
        min_lesion_size_px=min_lesion_size_px,
        features=features,
        subset_size=subset_size,
        num_subsets=num_subsets,
        degree=degree,
        normalize_features=normalize_features,
    )

    results = []
    for zbin in tqdm(valid_zbins, desc="Per-zbin MMD-MF"):
        # Filter masks for this z-bin
        real_mask = real_zbins == zbin
        synth_mask = synth_zbins == zbin

        real_zbin_masks = real_masks[real_mask]
        synth_zbin_masks = synth_masks[synth_mask]

        n_real = len(real_zbin_masks)
        n_synth = len(synth_zbin_masks)

        # Skip if insufficient masks
        if n_real < 5 or n_synth < 5:
            results.append(
                {
                    "zbin": zbin,
                    "mmd_mf": float("nan"),
                    "mmd_mf_std": float("nan"),
                    "n_real_masks": n_real,
                    "n_synth_masks": n_synth,
                }
            )
            continue

        # Compute MMD-MF
        result = mmd_computer.compute(
            real_zbin_masks, synth_zbin_masks, show_progress=False
        )

        results.append(
            {
                "zbin": zbin,
                "mmd_mf": result.value,
                "mmd_mf_std": result.std,
                "n_real_masks": n_real,
                "n_synth_masks": n_synth,
                "n_real_lesions": result.metadata.get("n_real_lesions", 0),
                "n_synth_lesions": result.metadata.get("n_synth_lesions", 0),
            }
        )

    return results
