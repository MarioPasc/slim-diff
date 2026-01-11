#!/usr/bin/env python3
"""Quality check for synthetic replicas against real data distribution.

This script validates generated synthetic replicas by:
1. Statistical comparison: Sample counts, brain fraction, intensity, lesion areas
2. Outlier detection: Identifies samples > 2 std from reference distribution
3. Visual inspection: Image grids comparing real vs synthetic samples

Supports two modes:
1. Single-split mode (original): Compare replicas against test set only
2. Multi-split mode: Compare replicas against union of train/val/test splits
   (useful for uniformly-sampled synthetic replicas)

Output includes detailed plots, CSV reports, and JSON summary.

Usage (single-split, backwards compatible):
    python src/diffusion/scripts/check_replica_quality.py \\
        --replicas-dir outputs/replicas/replicas \\
        --test-dist-csv docs/test_analysis/test_zbin_distribution.csv \\
        --output-dir outputs/quality_check

Usage (multi-split, for uniform replicas):
    python src/diffusion/scripts/check_replica_quality.py \\
        --replicas-dir outputs/replicas/replicas \\
        --train-dist-csv docs/train_analysis/train_zbin_distribution.csv \\
        --train-slices-csv /path/to/slice_cache/train.csv \\
        --val-dist-csv docs/val_analysis/val_zbin_distribution.csv \\
        --val-slices-csv /path/to/slice_cache/val.csv \\
        --test-dist-csv docs/test_analysis/test_zbin_distribution.csv \\
        --test-slices-csv /path/to/slice_cache/test.csv \\
        --output-dir outputs/quality_check

    Plots will show each split as separate series for comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from numpy.typing import NDArray
from omegaconf import OmegaConf
from tqdm import tqdm

# Import utilities from codebase
from src.diffusion.utils.zbin_priors import compute_brain_foreground_mask

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Domain mapping (from generate_replicas.py)
DOMAIN_MAP = {"control": 0, "epilepsy": 1}
DOMAIN_MAP_INV = {0: "control", 1: "epilepsy"}

# Color scheme (from analyze_test_distribution.py)
COLORS = {
    ('control', 0): ('steelblue', '-', 'Control (No Lesion)'),
    ('control', 1): ('darkblue', '--', 'Control (Lesion)'),
    ('epilepsy', 0): ('lightcoral', '-', 'Epilepsy (No Lesion)'),
    ('epilepsy', 1): ('darkred', '--', 'Epilepsy (Lesion)'),
}


# =============================================================================
# Utility Functions
# =============================================================================

def to_display_range(x: np.ndarray) -> np.ndarray:
    """Convert from [-1, 1] to [0, 1] for display.

    Args:
        x: Array in [-1, 1].

    Returns:
        Array in [0, 1].
    """
    return np.clip((x + 1) / 2, 0, 1)


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
    threshold: float = 0.0,
) -> np.ndarray:
    """Create image with mask overlay.

    Args:
        image: Grayscale image in [0, 1], shape (H, W).
        mask: Mask in [-1, 1] or [0, 1], shape (H, W).
        alpha: Overlay transparency.
        color: RGB color for overlay (0-255).
        threshold: Threshold for binarizing mask.

    Returns:
        RGB image with overlay, shape (H, W, 3).
    """
    # Ensure 2D
    if image.ndim == 3:
        image = image.squeeze()
    if mask.ndim == 3:
        mask = mask.squeeze()

    # Convert to RGB
    rgb = np.stack([image, image, image], axis=-1)

    # Binarize mask
    if mask.min() < 0:
        mask = to_display_range(mask)
    binary_mask = mask > (threshold + 1) / 2

    # Normalize color
    color_norm = np.array(color, dtype=np.float32) / 255.0

    # Apply overlay
    if binary_mask.any():
        for c in range(3):
            rgb[:, :, c] = np.where(
                binary_mask,
                (1 - alpha) * rgb[:, :, c] + alpha * color_norm[c],
                rgb[:, :, c],
            )

    return rgb


# =============================================================================
# Data Loading Module
# =============================================================================

def load_split_distribution(
    csv_path: Path,
    split_name: str = "test"
) -> pd.DataFrame:
    """Load distribution CSV for a specific split.

    Args:
        csv_path: Path to {split}_zbin_distribution.csv.
        split_name: Name of the split (train, val, test). Used for filtering
            and as a label in the returned DataFrame.

    Returns:
        DataFrame with columns: split, zbin, lesion_present, domain, n_slices,
        mean_brain_frac, mean_intensity, std_intensity, mean_lesion_area_px.

    Raises:
        FileNotFoundError: If CSV doesn't exist.
        ValueError: If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Distribution CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter to specified split if 'split' column exists
    if 'split' in df.columns:
        df = df[df['split'] == split_name].copy()
    else:
        # Add split column if not present
        df = df.copy()
        df['split'] = split_name

    # Validate required columns
    required_cols = ['zbin', 'lesion_present', 'domain', 'n_slices',
                     'mean_brain_frac', 'mean_intensity', 'std_intensity']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Loaded {split_name} distribution: {len(df)} conditions, "
                f"{df['n_slices'].sum()} total samples")

    return df


def load_test_distribution(csv_path: Path) -> pd.DataFrame:
    """Load test distribution CSV (backwards compatible wrapper).

    Args:
        csv_path: Path to test_zbin_distribution.csv.

    Returns:
        DataFrame with test distribution.
    """
    return load_split_distribution(csv_path, split_name="test")


def load_multi_split_distribution(
    split_configs: List[Tuple[Path, str]]
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and merge distribution CSVs from multiple splits.

    Args:
        split_configs: List of (csv_path, split_name) tuples.

    Returns:
        Tuple of:
            - combined_df: DataFrame with all splits, 'split' column preserved
            - splits_used: List of split names that were loaded
    """
    all_dfs = []
    splits_used = []

    for csv_path, split_name in split_configs:
        df = load_split_distribution(csv_path, split_name)
        df['split'] = split_name  # Ensure split column is set
        all_dfs.append(df)
        splits_used.append(split_name)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    total_samples = combined_df['n_slices'].sum()
    logger.info(f"Combined {len(splits_used)} splits: {len(combined_df)} total conditions, "
                f"{total_samples} total samples")

    return combined_df, splits_used


def load_replica(npz_path: Path) -> Optional[Dict[str, Any]]:
    """Load single replica NPZ file.

    Args:
        npz_path: Path to replica_XXX.npz file.

    Returns:
        Dict with keys: replica_id, images, masks, zbin, lesion_present, domain, seed.
        Returns None if loading fails.
    """
    try:
        # Parse replica ID from filename (e.g., replica_001.npz -> 1)
        match = re.search(r'replica_(\d+)\.npz$', npz_path.name)
        if not match:
            logger.warning(f"Could not parse replica ID from {npz_path.name}")
            return None
        replica_id = int(match.group(1))

        # Load NPZ
        data = np.load(npz_path, allow_pickle=True)

        # Extract arrays
        replica = {
            'replica_id': replica_id,
            'images': data['images'],
            'masks': data['masks'],
            'zbin': data['zbin'],
            'lesion_present': data['lesion_present'],
            'domain': data['domain'],
            'seed': data['seed'],
        }

        # Validate shapes
        n_samples = len(replica['images'])
        for key in ['masks', 'zbin', 'lesion_present', 'domain']:
            if len(replica[key]) != n_samples:
                raise ValueError(f"Shape mismatch: {key} has {len(replica[key])} samples, expected {n_samples}")

        logger.debug(f"Loaded replica {replica_id}: {n_samples} samples")
        return replica

    except Exception as e:
        logger.warning(f"Failed to load {npz_path.name}: {e}")
        return None


def load_all_replicas(
    replicas_dir: Path,
    max_replicas: Optional[int] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Load all replicas from directory.

    Args:
        replicas_dir: Directory containing replica_*.npz files.
        max_replicas: Maximum number of replicas to load (None = all).
        verbose: Show progress bar.

    Returns:
        List of replica dicts.
    """
    # Find all replica files
    replica_files = sorted(replicas_dir.glob('replica_*.npz'))

    if not replica_files:
        raise FileNotFoundError(f"No replica_*.npz files found in {replicas_dir}")

    if max_replicas is not None:
        replica_files = replica_files[:max_replicas]

    logger.info(f"Loading {len(replica_files)} replicas from {replicas_dir}")

    # Load each replica
    replicas = []
    iterator = tqdm(replica_files, desc="Loading replicas", disable=not verbose)

    for npz_path in iterator:
        replica = load_replica(npz_path)
        if replica is not None:
            replicas.append(replica)

    if not replicas:
        raise ValueError("No replicas could be loaded successfully")

    logger.info(f"Successfully loaded {len(replicas)} replicas")
    return replicas


def load_split_slices(
    csv_path: Path,
    split_name: str = "test",
    verbose: bool = True
) -> Dict[Tuple[int, int, int], List[Dict[str, Any]]]:
    """Load slices from CSV for a specific split.

    Args:
        csv_path: Path to {split}.csv with columns:
            subject_id, z_index, z_bin, pathology_class, token, source, split,
            has_lesion, filepath
        split_name: Name of the split (train, val, test).
        verbose: Show progress.

    Returns:
        Dict mapping (zbin, lesion_present, domain) -> list of slice dicts with
        keys: image, mask, subject_id, z_index, split.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Slices CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter to specified split if column exists
    if 'split' in df.columns:
        df = df[df['split'] == split_name].copy()

    # Map source to domain int
    source_to_domain = {'control': 0, 'epilepsy': 1}

    # Base directory for slice files (same directory as CSV)
    base_dir = csv_path.parent

    # Group slices by condition
    slices_by_condition: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}

    iterator = tqdm(df.iterrows(), total=len(df), desc=f"Loading {split_name} slices", disable=not verbose)

    for _, row in iterator:
        zbin = int(row['z_bin'])
        lesion_present = 1 if row['has_lesion'] else 0
        domain = source_to_domain.get(row['source'], 0)

        key = (zbin, lesion_present, domain)

        # Load the NPZ file
        npz_path = base_dir / row['filepath']
        if not npz_path.exists():
            continue

        try:
            data = np.load(npz_path)
            slice_dict = {
                'image': data['image'],
                'mask': data['mask'] if 'mask' in data else np.zeros_like(data['image']),
                'subject_id': row['subject_id'],
                'z_index': row['z_index'],
                'split': split_name,
            }

            if key not in slices_by_condition:
                slices_by_condition[key] = []
            slices_by_condition[key].append(slice_dict)

        except Exception as e:
            logger.debug(f"Failed to load {npz_path}: {e}")
            continue

    total_slices = sum(len(v) for v in slices_by_condition.values())
    logger.info(f"Loaded {total_slices} {split_name} slices across {len(slices_by_condition)} conditions")

    return slices_by_condition


def load_test_slices(
    csv_path: Path,
    verbose: bool = True
) -> Dict[Tuple[int, int, int], List[Dict[str, Any]]]:
    """Load test slices from CSV (backwards compatible wrapper).

    Args:
        csv_path: Path to test.csv.
        verbose: Show progress.

    Returns:
        Dict mapping (zbin, lesion_present, domain) -> list of slice dicts.
    """
    return load_split_slices(csv_path, split_name="test", verbose=verbose)


def load_multi_split_slices(
    split_configs: List[Tuple[Path, str]],
    verbose: bool = True
) -> Tuple[Dict[Tuple[int, int, int], List[Dict[str, Any]]], List[str]]:
    """Load and merge slices from multiple splits.

    Args:
        split_configs: List of (csv_path, split_name) tuples.
        verbose: Show progress.

    Returns:
        Tuple of:
            - combined_slices: Dict mapping (zbin, lesion_present, domain) -> list of slice dicts
            - splits_used: List of split names that were loaded
    """
    combined_slices: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    splits_used = []

    for csv_path, split_name in split_configs:
        slices = load_split_slices(csv_path, split_name, verbose=verbose)
        splits_used.append(split_name)

        # Merge into combined dict
        for key, slice_list in slices.items():
            if key not in combined_slices:
                combined_slices[key] = []
            combined_slices[key].extend(slice_list)

    total_slices = sum(len(v) for v in combined_slices.values())
    logger.info(f"Combined {len(splits_used)} splits: {total_slices} total slices")

    return combined_slices, splits_used


# =============================================================================
# Statistical Computation Module
# =============================================================================

def compute_replica_statistics(
    replica: Dict[str, Any],
    gaussian_sigma_px: float,
    min_component_px: int
) -> Dict[str, Any]:
    """Compute per-slice statistics for a replica.

    Args:
        replica: Replica dict with images, masks arrays.
        gaussian_sigma_px: Sigma for brain mask computation.
        min_component_px: Min component size for brain mask.

    Returns:
        Updated replica dict with computed statistics arrays:
        - brain_fracs: (N,) brain fraction per slice
        - mean_intensities: (N,) mean intensity within brain
        - std_intensities: (N,) std intensity within brain
        - lesion_areas: (N,) lesion area in pixels (NaN if no lesion)
    """
    n_samples = len(replica['images'])

    # Preallocate arrays
    brain_fracs = np.zeros(n_samples, dtype=np.float32)
    mean_intensities = np.zeros(n_samples, dtype=np.float32)
    std_intensities = np.zeros(n_samples, dtype=np.float32)
    lesion_areas = np.full(n_samples, np.nan, dtype=np.float32)

    # Compute statistics per slice
    for i in tqdm(range(n_samples), desc=f"Computing stats (replica {replica['replica_id']})", leave=False):
        image = replica['images'][i]
        mask = replica['masks'][i]
        lesion_present = replica['lesion_present'][i]

        # Ensure 2D
        if image.ndim == 3:
            image = image.squeeze()
        if mask.ndim == 3:
            mask = mask.squeeze()

        # Compute brain mask
        brain_mask = compute_brain_foreground_mask(
            image,
            gaussian_sigma_px,
            min_component_px,
            n_components_to_keep=1
        )

        if brain_mask is None or not brain_mask.any():
            # No brain detected - use full image
            brain_mask = np.ones_like(image, dtype=bool)

        # Brain fraction
        brain_fracs[i] = brain_mask.sum() / brain_mask.size

        # Intensity statistics within brain
        brain_pixels = image[brain_mask]
        if len(brain_pixels) > 0:
            mean_intensities[i] = brain_pixels.mean()
            std_intensities[i] = brain_pixels.std()

        # Lesion area (if applicable)
        if lesion_present == 1:
            # Convert mask to [0, 1] and threshold
            mask_norm = to_display_range(mask) if mask.min() < 0 else mask
            lesion_mask = mask_norm > 0.5
            lesion_areas[i] = lesion_mask.sum()

    # Add to replica dict
    replica['brain_fracs'] = brain_fracs
    replica['mean_intensities'] = mean_intensities
    replica['std_intensities'] = std_intensities
    replica['lesion_areas'] = lesion_areas

    return replica


def aggregate_by_condition(replica: Dict[str, Any]) -> pd.DataFrame:
    """Aggregate statistics by condition within one replica.

    Args:
        replica: Replica dict with computed statistics.

    Returns:
        DataFrame with one row per condition, columns matching test_zbin_distribution.csv.
    """
    # Create DataFrame from replica arrays
    df = pd.DataFrame({
        'zbin': replica['zbin'],
        'lesion_present': replica['lesion_present'],
        'domain': replica['domain'],
        'brain_frac': replica['brain_fracs'],
        'mean_intensity': replica['mean_intensities'],
        'std_intensity': replica['std_intensities'],
        'lesion_area': replica['lesion_areas'],
    })

    # Group by condition
    grouped = df.groupby(['zbin', 'lesion_present', 'domain'])

    # Aggregate using named aggregation tuples
    result = grouped.agg(
        n_slices=('zbin', 'size'),
        mean_brain_frac=('brain_frac', 'mean'),
        mean_intensity=('mean_intensity', 'mean'),
        std_intensity=('std_intensity', 'mean'),
        mean_lesion_area_px=('lesion_area', 'mean'),
    ).reset_index()

    # Convert domain int to string
    result['domain_str'] = result['domain'].map(DOMAIN_MAP_INV)

    return result


def compute_mean_std_across_replicas(
    replicas: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean and std statistics across all replicas.

    Args:
        replicas: List of replica dicts with computed statistics.

    Returns:
        Tuple of (mean_df, std_df) with aggregated statistics.
    """
    # Aggregate each replica
    replica_dfs = []
    for replica in replicas:
        df = aggregate_by_condition(replica)
        df['replica_id'] = replica['replica_id']
        replica_dfs.append(df)

    # Combine all replicas
    combined = pd.concat(replica_dfs, ignore_index=True)

    # Group by condition and compute mean/std across replicas
    group_cols = ['zbin', 'lesion_present', 'domain', 'domain_str']
    metric_cols = ['n_slices', 'mean_brain_frac', 'mean_intensity', 'std_intensity', 'mean_lesion_area_px']

    mean_df = combined.groupby(group_cols)[metric_cols].mean().reset_index()
    std_df = combined.groupby(group_cols)[metric_cols].std().reset_index()

    return mean_df, std_df


def compute_deviation_metrics(
    test_df: pd.DataFrame,
    replica_mean_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute deviation metrics between test and replica means.

    Args:
        test_df: Test distribution DataFrame.
        replica_mean_df: Mean statistics across replicas.

    Returns:
        DataFrame with deviation metrics per condition and metric.
    """
    # Convert test_df domain from string to int for merging
    test_df_copy = test_df.copy()
    test_df_copy['domain'] = test_df_copy['domain'].map(DOMAIN_MAP)

    # Merge on condition keys
    merged = test_df_copy.merge(
        replica_mean_df,
        on=['zbin', 'lesion_present', 'domain'],
        suffixes=('_test', '_replica')
    )

    # Compute deviations for each metric
    metrics = ['n_slices', 'mean_brain_frac', 'mean_intensity', 'std_intensity', 'mean_lesion_area_px']

    deviations = []
    for metric in metrics:
        test_col = f'{metric}_test' if f'{metric}_test' in merged.columns else metric
        replica_col = f'{metric}_replica' if f'{metric}_replica' in merged.columns else metric

        if test_col not in merged.columns or replica_col not in merged.columns:
            continue

        test_vals = merged[test_col]
        replica_vals = merged[replica_col]

        # Absolute deviation
        abs_dev = np.abs(replica_vals - test_vals)

        # Relative deviation (%)
        rel_dev = np.where(test_vals != 0, 100 * abs_dev / np.abs(test_vals), 0)

        # Create rows for this metric
        for i, (idx, row) in enumerate(merged.iterrows()):
            deviations.append({
                'zbin': row['zbin'],
                'lesion_present': row['lesion_present'],
                'domain': row['domain'],
                'metric_name': metric,
                'test_value': test_vals.iloc[i],
                'replica_mean': replica_vals.iloc[i],
                'absolute_deviation': abs_dev[i],
                'relative_deviation_pct': rel_dev[i],
            })

    deviation_df = pd.DataFrame(deviations)

    return deviation_df


# =============================================================================
# Outlier Detection Module
# =============================================================================

def detect_outliers(
    replicas: List[Dict[str, Any]],
    test_df: pd.DataFrame,
    threshold_std: float = 2.0,
    metrics: List[str] = ['brain_fracs', 'mean_intensities', 'std_intensities', 'lesion_areas']
) -> pd.DataFrame:
    """Detect outlier samples across all replicas.

    Args:
        replicas: List of replica dicts with computed statistics.
        test_df: Test distribution for expected values.
        threshold_std: Outlier threshold in standard deviations.
        metrics: List of metric array names to check.

    Returns:
        DataFrame with outlier details.
    """
    outliers = []

    # Map metric array names to test_df column names
    metric_map = {
        'brain_fracs': 'mean_brain_frac',
        'mean_intensities': 'mean_intensity',
        'std_intensities': 'std_intensity',
        'lesion_areas': 'mean_lesion_area_px',
    }

    for metric_array_name in metrics:
        if metric_array_name not in metric_map:
            continue

        test_col = metric_map[metric_array_name]

        # For each condition in test set
        for _, test_row in test_df.iterrows():
            zbin = test_row['zbin']
            lesion_present = test_row['lesion_present']
            domain_str = test_row['domain']
            domain = DOMAIN_MAP[domain_str]

            expected_mean = test_row[test_col]

            # Skip if NaN (e.g., no lesions in this condition)
            if pd.isna(expected_mean):
                continue

            # Collect all replica samples for this condition
            all_values = []
            sample_info = []  # (replica_id, sample_idx, value)

            for replica in replicas:
                # Find samples matching this condition
                mask = (replica['zbin'] == zbin) & \
                       (replica['lesion_present'] == lesion_present) & \
                       (replica['domain'] == domain)

                indices = np.where(mask)[0]
                values = replica[metric_array_name][indices]

                for idx, val in zip(indices, values):
                    if not np.isnan(val):
                        all_values.append(val)
                        sample_info.append((replica['replica_id'], idx, val))

            if len(all_values) < 2:
                continue

            # Compute population std
            population_std = np.std(all_values)

            if population_std == 0:
                continue

            # Check each sample for outliers
            for replica_id, sample_idx, value in sample_info:
                z_score = (value - expected_mean) / population_std

                if np.abs(z_score) > threshold_std:
                    outliers.append({
                        'replica_id': replica_id,
                        'sample_idx': sample_idx,
                        'zbin': zbin,
                        'lesion_present': lesion_present,
                        'domain': domain,
                        'domain_str': domain_str,
                        'metric_name': metric_array_name,
                        'value': value,
                        'expected_mean': expected_mean,
                        'population_std': population_std,
                        'z_score': z_score,
                    })

    outliers_df = pd.DataFrame(outliers)

    if len(outliers_df) > 0:
        outliers_df = outliers_df.sort_values('z_score', key=lambda x: np.abs(x), ascending=False)

    logger.info(f"Detected {len(outliers_df)} outliers (threshold: {threshold_std} std)")

    return outliers_df


# =============================================================================
# Visualization Module
# =============================================================================

def plot_sample_count_comparison(
    test_df: pd.DataFrame,
    replica_mean_df: pd.DataFrame,
    replica_std_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300
) -> None:
    """Compare sample counts between test and replicas.

    Args:
        test_df: Test distribution.
        replica_mean_df: Mean statistics across replicas.
        replica_std_df: Std statistics across replicas.
        output_path: Path to save PNG.
        dpi: DPI for saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sample Count Comparison: Test vs Replicas', fontsize=16, fontweight='bold')

    # Panel 1: Stacked bar chart - n_slices per z-bin
    ax = axes[0, 0]

    # Group by z-bin and condition
    test_grouped = test_df.groupby(['zbin', 'domain', 'lesion_present'])['n_slices'].sum().reset_index()
    replica_grouped = replica_mean_df.groupby(['zbin', 'domain', 'lesion_present'])['n_slices'].sum().reset_index()

    zbins = sorted(test_df['zbin'].unique())
    width = 0.35
    x = np.arange(len(zbins))

    # Stacked bars for test (control/epilepsy)
    test_control = []
    test_epilepsy = []
    for zb in zbins:
        control_sum = test_grouped[(test_grouped['zbin'] == zb) & (test_grouped['domain'] == 'control')]['n_slices'].sum()
        epilepsy_sum = test_grouped[(test_grouped['zbin'] == zb) & (test_grouped['domain'] == 'epilepsy')]['n_slices'].sum()
        test_control.append(control_sum)
        test_epilepsy.append(epilepsy_sum)

    ax.bar(x - width/2, test_control, width, label='Test Control', color='steelblue', alpha=0.8)
    ax.bar(x - width/2, test_epilepsy, width, bottom=test_control, label='Test Epilepsy', color='lightcoral', alpha=0.8)

    # Stacked bars for replicas
    replica_control = []
    replica_epilepsy = []
    for zb in zbins:
        control_sum = replica_grouped[(replica_grouped['zbin'] == zb) & (replica_grouped['domain'] == 0)]['n_slices'].sum()
        epilepsy_sum = replica_grouped[(replica_grouped['zbin'] == zb) & (replica_grouped['domain'] == 1)]['n_slices'].sum()
        replica_control.append(control_sum)
        replica_epilepsy.append(epilepsy_sum)

    ax.bar(x + width/2, replica_control, width, label='Replica Control', color='steelblue', alpha=0.4, hatch='//')
    ax.bar(x + width/2, replica_epilepsy, width, bottom=replica_control, label='Replica Epilepsy', color='lightcoral', alpha=0.4, hatch='//')

    ax.set_xlabel('Z-bin', fontweight='bold')
    ax.set_ylabel('Number of Slices', fontweight='bold')
    ax.set_title('Sample Count per Z-bin')
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([zbins[i] for i in range(0, len(zbins), max(1, len(zbins)//10))])
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Total counts per domain
    ax = axes[0, 1]

    test_total_control = test_df[test_df['domain'] == 'control']['n_slices'].sum()
    test_total_epilepsy = test_df[test_df['domain'] == 'epilepsy']['n_slices'].sum()
    replica_total_control = replica_mean_df[replica_mean_df['domain'] == 0]['n_slices'].sum()
    replica_total_epilepsy = replica_mean_df[replica_mean_df['domain'] == 1]['n_slices'].sum()

    labels = ['Control', 'Epilepsy']
    test_totals = [test_total_control, test_total_epilepsy]
    replica_totals = [replica_total_control, replica_total_epilepsy]

    x = np.arange(len(labels))
    ax.bar(x - width/2, test_totals, width, label='Test', color=['steelblue', 'lightcoral'], alpha=0.8)
    ax.bar(x + width/2, replica_totals, width, label='Replicas', color=['steelblue', 'lightcoral'], alpha=0.4, hatch='//')

    ax.set_ylabel('Total Slices', fontweight='bold')
    ax.set_title('Total Sample Count by Domain')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: Deviation per condition
    ax = axes[1, 0]

    # Convert test_df domain from string to int for merging
    test_df_for_merge = test_df.copy()
    test_df_for_merge['domain'] = test_df_for_merge['domain'].map(DOMAIN_MAP)
    merged = test_df_for_merge.merge(replica_mean_df, on=['zbin', 'lesion_present', 'domain'], suffixes=('_test', '_replica'))
    deviations = (merged['n_slices_replica'] - merged['n_slices_test']) / merged['n_slices_test'] * 100

    conditions = [f"z{row['zbin']}_l{row['lesion_present']}_{'c' if row['domain']==0 else 'e'}"
                  for _, row in merged.iterrows()]

    colors_per_condition = ['steelblue' if d == 0 else 'lightcoral' for d in merged['domain']]

    ax.barh(range(len(deviations)), deviations, color=colors_per_condition, alpha=0.6)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Deviation (%)', fontweight='bold')
    ax.set_ylabel('Condition', fontweight='bold')
    ax.set_title('Sample Count Deviation from Test')
    ax.set_yticks([])  # Too many to show
    ax.grid(alpha=0.3)

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    test_total = test_df['n_slices'].sum()
    replica_total = replica_mean_df['n_slices'].sum()
    mae = np.abs(deviations).mean()
    max_dev = np.abs(deviations).max()

    summary_text = f"""Sample Count Summary:

Test Total: {test_total:.0f}
Replica Total: {replica_total:.0f}
Difference: {replica_total - test_total:.0f} ({(replica_total - test_total) / test_total * 100:.1f}%)

Mean Absolute Deviation: {mae:.1f}%
Max Deviation: {max_dev:.1f}%

Conditions: {len(test_df)}
"""

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved sample count comparison to {output_path}")


def plot_brain_fraction_comparison(
    test_df: pd.DataFrame,
    replicas: List[Dict[str, Any]],
    outliers_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300
) -> None:
    """Compare brain fraction across z-bins.

    Args:
        test_df: Test distribution.
        replicas: List of replica dicts with statistics.
        outliers_df: Outlier DataFrame.
        output_path: Path to save PNG.
        dpi: DPI for saved figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('Brain Fraction Comparison: Test vs Replicas', fontsize=16, fontweight='bold')

    # For each condition (lesion_present, domain)
    for (domain_str, lesion_present), (color, linestyle, label) in COLORS.items():
        domain = DOMAIN_MAP[domain_str]

        # Test data
        test_cond = test_df[(test_df['domain'] == domain_str) & (test_df['lesion_present'] == lesion_present)]
        if len(test_cond) > 0:
            test_cond = test_cond.sort_values('zbin')
            ax.plot(test_cond['zbin'], test_cond['mean_brain_frac'],
                   color=color, linestyle=linestyle, linewidth=2, marker='o',
                   label=f'{label} (Test)', alpha=1.0, zorder=10)

        # Individual replicas (grey, thin, low alpha)
        for replica in replicas:
            mask = (replica['lesion_present'] == lesion_present) & (replica['domain'] == domain)
            if not mask.any():
                continue

            # Aggregate by zbin
            zbins_rep = []
            brain_fracs_rep = []
            for zb in np.unique(replica['zbin'][mask]):
                zb_mask = mask & (replica['zbin'] == zb)
                zbins_rep.append(zb)
                brain_fracs_rep.append(replica['brain_fracs'][zb_mask].mean())

            ax.plot(zbins_rep, brain_fracs_rep, color='grey', linestyle='-',
                   linewidth=0.5, alpha=0.3, zorder=1)

        # Mean across replicas
        all_zbins = test_cond['zbin'].values
        means = []
        stds = []

        for zb in all_zbins:
            values = []
            for replica in replicas:
                mask = (replica['zbin'] == zb) & \
                       (replica['lesion_present'] == lesion_present) & \
                       (replica['domain'] == domain)
                if mask.any():
                    values.append(replica['brain_fracs'][mask].mean())

            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        means = np.array(means)
        stds = np.array(stds)

        # Plot mean with marker
        valid = ~np.isnan(means)
        ax.plot(all_zbins[valid], means[valid], color=color, linestyle=linestyle,
               linewidth=2, marker='s', markersize=6, label=f'{label} (Replica Mean)', alpha=1.0, zorder=9)

        # Std bands
        ax.fill_between(all_zbins[valid], means[valid] - stds[valid], means[valid] + stds[valid],
                       color=color, alpha=0.2, zorder=2)

    # Outliers
    if len(outliers_df) > 0:
        outliers_brain = outliers_df[outliers_df['metric_name'] == 'brain_fracs']
        if len(outliers_brain) > 0:
            ax.scatter(outliers_brain['zbin'], outliers_brain['value'],
                      color='red', marker='x', s=100, linewidths=2,
                      label='Outliers', alpha=1.0, zorder=11)

    ax.set_xlabel('Z-bin', fontweight='bold')
    ax.set_ylabel('Brain Fraction', fontweight='bold')
    ax.set_title('Mean Brain Fraction per Z-bin')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved brain fraction comparison to {output_path}")


def plot_intensity_comparison(
    test_df: pd.DataFrame,
    replicas: List[Dict[str, Any]],
    outliers_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300
) -> None:
    """Compare intensity statistics.

    Args:
        test_df: Test distribution.
        replicas: List of replica dicts.
        outliers_df: Outlier DataFrame.
        output_path: Path to save PNG.
        dpi: DPI for saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Intensity Comparison: Test vs Replicas', fontsize=16, fontweight='bold')

    # Top row: mean_intensity (control, epilepsy)
    for col_idx, domain_str in enumerate(['control', 'epilepsy']):
        ax = axes[0, col_idx]
        domain = DOMAIN_MAP[domain_str]

        for lesion_present in [0, 1]:
            color, linestyle, label = COLORS[(domain_str, lesion_present)]

            # Test data
            test_cond = test_df[(test_df['domain'] == domain_str) & (test_df['lesion_present'] == lesion_present)]
            if len(test_cond) > 0:
                test_cond = test_cond.sort_values('zbin')
                ax.plot(test_cond['zbin'], test_cond['mean_intensity'],
                       color=color, linestyle=linestyle, linewidth=2, marker='o',
                       label=f'{label} (Test)', alpha=1.0)

            # Replica mean
            all_zbins = test_cond['zbin'].values
            means = []
            for zb in all_zbins:
                values = []
                for replica in replicas:
                    mask = (replica['zbin'] == zb) & \
                           (replica['lesion_present'] == lesion_present) & \
                           (replica['domain'] == domain)
                    if mask.any():
                        values.append(replica['mean_intensities'][mask].mean())
                means.append(np.mean(values) if values else np.nan)

            valid = ~np.isnan(means)
            ax.plot(all_zbins[valid], np.array(means)[valid], color=color, linestyle=linestyle,
                   linewidth=2, marker='s', label=f'{label} (Replica)', alpha=1.0)

        ax.set_xlabel('Z-bin', fontweight='bold')
        ax.set_ylabel('Mean Intensity', fontweight='bold')
        ax.set_title(f'Mean Intensity - {domain_str.capitalize()}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Bottom row: std_intensity (control, epilepsy)
    for col_idx, domain_str in enumerate(['control', 'epilepsy']):
        ax = axes[1, col_idx]
        domain = DOMAIN_MAP[domain_str]

        for lesion_present in [0, 1]:
            color, linestyle, label = COLORS[(domain_str, lesion_present)]

            # Test data
            test_cond = test_df[(test_df['domain'] == domain_str) & (test_df['lesion_present'] == lesion_present)]
            if len(test_cond) > 0:
                test_cond = test_cond.sort_values('zbin')
                ax.plot(test_cond['zbin'], test_cond['std_intensity'],
                       color=color, linestyle=linestyle, linewidth=2, marker='o',
                       label=f'{label} (Test)', alpha=1.0)

            # Replica mean
            all_zbins = test_cond['zbin'].values
            means = []
            for zb in all_zbins:
                values = []
                for replica in replicas:
                    mask = (replica['zbin'] == zb) & \
                           (replica['lesion_present'] == lesion_present) & \
                           (replica['domain'] == domain)
                    if mask.any():
                        values.append(replica['std_intensities'][mask].mean())
                means.append(np.mean(values) if values else np.nan)

            valid = ~np.isnan(means)
            ax.plot(all_zbins[valid], np.array(means)[valid], color=color, linestyle=linestyle,
                   linewidth=2, marker='s', label=f'{label} (Replica)', alpha=1.0)

        ax.set_xlabel('Z-bin', fontweight='bold')
        ax.set_ylabel('Std Intensity', fontweight='bold')
        ax.set_title(f'Std Intensity - {domain_str.capitalize()}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved intensity comparison to {output_path}")


def plot_lesion_area_comparison(
    test_df: pd.DataFrame,
    replicas: List[Dict[str, Any]],
    outliers_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300
) -> None:
    """Compare lesion area statistics.

    Args:
        test_df: Test distribution.
        replicas: List of replica dicts.
        outliers_df: Outlier DataFrame.
        output_path: Path to save PNG.
        dpi: DPI for saved figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Lesion Area Comparison: Test vs Replicas', fontsize=16, fontweight='bold')

    # Panel 1: Mean lesion area per z-bin
    ax = axes[0]

    # Filter to lesion_present=1
    test_lesion = test_df[test_df['lesion_present'] == 1]

    for domain_str in ['control', 'epilepsy']:
        domain = DOMAIN_MAP[domain_str]
        color, _, label = COLORS[(domain_str, 1)]

        # Test data
        test_cond = test_lesion[test_lesion['domain'] == domain_str]
        if len(test_cond) > 0:
            test_cond = test_cond.sort_values('zbin')
            valid = ~test_cond['mean_lesion_area_px'].isna()
            ax.plot(test_cond[valid]['zbin'], test_cond[valid]['mean_lesion_area_px'],
                   color=color, linestyle='-', linewidth=2, marker='o',
                   label=f'{label} (Test)', alpha=1.0)

        # Replica mean
        all_zbins = test_cond['zbin'].values
        means = []
        for zb in all_zbins:
            values = []
            for replica in replicas:
                mask = (replica['zbin'] == zb) & \
                       (replica['lesion_present'] == 1) & \
                       (replica['domain'] == domain)
                if mask.any():
                    areas = replica['lesion_areas'][mask]
                    areas = areas[~np.isnan(areas)]
                    if len(areas) > 0:
                        values.append(areas.mean())
            means.append(np.mean(values) if values else np.nan)

        valid = ~np.isnan(means)
        ax.plot(all_zbins[valid], np.array(means)[valid], color=color, linestyle='--',
               linewidth=2, marker='s', label=f'{label} (Replica)', alpha=1.0)

    ax.set_xlabel('Z-bin', fontweight='bold')
    ax.set_ylabel('Mean Lesion Area (pixels)', fontweight='bold')
    ax.set_title('Mean Lesion Area per Z-bin')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Histogram of all lesion areas
    ax = axes[1]

    # Collect all lesion areas from test (simulated from replica data since we don't have test images)
    test_lesion_areas = []
    replica_lesion_areas = []

    for replica in replicas:
        mask = (replica['lesion_present'] == 1)
        areas = replica['lesion_areas'][mask]
        areas = areas[~np.isnan(areas)]
        replica_lesion_areas.extend(areas)

    # Approximate test distribution from test_df
    for _, row in test_lesion[test_lesion['mean_lesion_area_px'].notna()].iterrows():
        # Use mean as proxy
        test_lesion_areas.extend([row['mean_lesion_area_px']] * int(row['n_slices']))

    bins = np.linspace(0, max(max(test_lesion_areas or [0]), max(replica_lesion_areas or [0])), 50)

    ax.hist(test_lesion_areas, bins=bins, alpha=0.6, color='darkred', label='Test', edgecolor='black')
    ax.hist(replica_lesion_areas, bins=bins, alpha=0.4, color='lightcoral', label='Replicas', edgecolor='black')

    ax.set_xlabel('Lesion Area (pixels)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of Lesion Areas')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved lesion area comparison to {output_path}")


def select_representative_zbins(all_zbins: List[int], n_bins: int = 5) -> List[int]:
    """Select evenly-spaced z-bins.

    Args:
        all_zbins: List of all z-bin values.
        n_bins: Number of bins to select.

    Returns:
        List of selected z-bin values.
    """
    sorted_zbins = sorted(all_zbins)
    if len(sorted_zbins) <= n_bins:
        return sorted_zbins

    indices = np.linspace(0, len(sorted_zbins) - 1, n_bins, dtype=int)
    return [sorted_zbins[i] for i in indices]


def select_sample_images(
    replica: Dict[str, Any],
    n_per_condition: int = 3,
    seed: int = 42
) -> Dict[Tuple[int, int, int], List[int]]:
    """Select example images from replica.

    Args:
        replica: Replica dict.
        n_per_condition: Number of samples per condition.
        seed: Random seed.

    Returns:
        Dict mapping (zbin, lesion_present, domain) -> list of sample indices.
    """
    np.random.seed(seed)

    selected = {}

    # Group by condition
    conditions = {}
    for idx in range(len(replica['zbin'])):
        key = (replica['zbin'][idx], replica['lesion_present'][idx], replica['domain'][idx])
        if key not in conditions:
            conditions[key] = []
        conditions[key].append(idx)

    # Select random samples per condition
    for key, indices in conditions.items():
        n_select = min(n_per_condition, len(indices))
        selected[key] = list(np.random.choice(indices, size=n_select, replace=False))

    return selected


def plot_image_grid(
    replicas: List[Dict[str, Any]],
    zbins_to_show: List[int],
    n_images_per_condition: int,
    output_path: Path,
    test_slices: Optional[Dict[Tuple[int, int, int], List[Dict[str, Any]]]] = None,
    dpi: int = 150,
    title_suffix: str = ""
) -> None:
    """Create image grid visualization comparing test vs synthetic images.

    Args:
        replicas: List of replica dicts.
        zbins_to_show: Z-bins to visualize.
        n_images_per_condition: Images per condition.
        output_path: Path to save PNG.
        test_slices: Optional dict mapping (zbin, lesion, domain) -> list of test slice dicts.
        dpi: DPI for saved figure.
        title_suffix: Suffix for figure title.
    """
    # Define conditions to display: (lesion_present, domain_str, domain_int, label, color)
    all_conditions = [
        (0, 'control', 0, 'Control (No Lesion)', 'steelblue'),
        (1, 'control', 0, 'Control (Lesion)', 'darkblue'),
        (0, 'epilepsy', 1, 'Epilepsy (No Lesion)', 'lightcoral'),
        (1, 'epilepsy', 1, 'Epilepsy (Lesion)', 'darkred'),
    ]

    # Check which conditions actually have synthetic data
    valid_conditions = []
    for lesion_present, domain_str, domain, label, color in all_conditions:
        has_data = False
        for replica in replicas:
            for zbin in zbins_to_show:
                mask = (replica['zbin'] == zbin) & \
                       (replica['lesion_present'] == lesion_present) & \
                       (replica['domain'] == domain)
                if mask.any():
                    has_data = True
                    break
            if has_data:
                break
        if has_data:
            valid_conditions.append((lesion_present, domain_str, domain, label, color))

    if not valid_conditions:
        logger.warning("No valid conditions found for image grid")
        return

    n_rows = len(zbins_to_show)
    show_test = test_slices is not None

    # Columns: for each condition, show test images then synthetic images
    if show_test:
        n_cols = len(valid_conditions) * n_images_per_condition * 2  # test + synthetic
    else:
        n_cols = len(valid_conditions) * n_images_per_condition

    # Create figure with extra space for headers
    fig = plt.figure(figsize=(n_cols * 1.5 + 1, n_rows * 1.5 + 2))

    # Create gridspec for better control (2 header rows if showing test)
    n_header_rows = 2 if show_test else 1
    gs = fig.add_gridspec(
        n_rows + n_header_rows, n_cols,
        height_ratios=[0.12] * n_header_rows + [1] * n_rows,
        hspace=0.03, wspace=0.02,
        left=0.06, right=0.99, top=0.93, bottom=0.02
    )

    fig.suptitle(f'Test vs Synthetic Comparison{title_suffix}' if show_test else f'Synthetic Replicas{title_suffix}',
                 fontsize=14, fontweight='bold', y=0.98)

    # Add headers
    if show_test:
        # Top-level headers: Test | Synthetic for each condition group
        for cond_idx, (lesion_present, domain_str, domain, label, color) in enumerate(valid_conditions):
            base_col = cond_idx * n_images_per_condition * 2

            # Condition header spanning test + synthetic
            ax_cond = fig.add_subplot(gs[0, base_col:base_col + n_images_per_condition * 2])
            ax_cond.set_facecolor(color)
            ax_cond.text(0.5, 0.5, label, ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white',
                        transform=ax_cond.transAxes)
            ax_cond.set_xticks([])
            ax_cond.set_yticks([])
            for spine in ax_cond.spines.values():
                spine.set_visible(False)

            # Test sub-header
            ax_test = fig.add_subplot(gs[1, base_col:base_col + n_images_per_condition])
            ax_test.set_facecolor('#e8e8e8')
            ax_test.text(0.5, 0.5, 'Test', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='#333',
                        transform=ax_test.transAxes)
            ax_test.set_xticks([])
            ax_test.set_yticks([])
            for spine in ax_test.spines.values():
                spine.set_visible(False)

            # Synthetic sub-header
            ax_synth = fig.add_subplot(gs[1, base_col + n_images_per_condition:base_col + n_images_per_condition * 2])
            ax_synth.set_facecolor('#d0d0d0')
            ax_synth.text(0.5, 0.5, 'Synthetic', ha='center', va='center',
                         fontsize=8, fontweight='bold', color='#333',
                         transform=ax_synth.transAxes)
            ax_synth.set_xticks([])
            ax_synth.set_yticks([])
            for spine in ax_synth.spines.values():
                spine.set_visible(False)
    else:
        # Simple headers for synthetic only
        for cond_idx, (lesion_present, domain_str, domain, label, color) in enumerate(valid_conditions):
            start_col = cond_idx * n_images_per_condition
            end_col = start_col + n_images_per_condition

            ax_header = fig.add_subplot(gs[0, start_col:end_col])
            ax_header.set_facecolor(color)
            ax_header.text(0.5, 0.5, label, ha='center', va='center',
                          fontsize=9, fontweight='bold', color='white',
                          transform=ax_header.transAxes)
            ax_header.set_xticks([])
            ax_header.set_yticks([])
            for spine in ax_header.spines.values():
                spine.set_visible(False)

    # Helper function to plot a single image
    def plot_single_image(ax, image, mask_arr, lesion_present):
        """Plot a single image with optional lesion overlay."""
        # Convert to display range and ensure float32 for matplotlib
        image_disp = to_display_range(image).astype(np.float32)

        # Ensure 2D for stacking
        if image_disp.ndim == 3:
            image_disp = image_disp.squeeze()

        # Create overlay if lesion
        if lesion_present == 1 and mask_arr is not None:
            rgb = create_overlay(image_disp, mask_arr, alpha=0.5, color=(255, 0, 0))
        else:
            rgb = np.stack([image_disp, image_disp, image_disp], axis=-1).astype(np.float32)

        ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot images
    for row_idx, zbin in enumerate(zbins_to_show):
        col_idx = 0

        for lesion_present, domain_str, domain, label, color in valid_conditions:
            condition_key = (zbin, lesion_present, domain)

            if show_test:
                # Plot test images first
                test_samples = test_slices.get(condition_key, [])
                for i in range(n_images_per_condition):
                    ax = fig.add_subplot(gs[row_idx + n_header_rows, col_idx])
                    if i < len(test_samples):
                        slice_data = test_samples[i]
                        plot_single_image(ax, slice_data['image'], slice_data.get('mask'), lesion_present)
                    else:
                        ax.set_facecolor('#f5f5f5')
                        ax.text(0.5, 0.5, '—', ha='center', va='center',
                               fontsize=12, color='#999', transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Add z-bin label on leftmost column
                    if col_idx == 0:
                        ax.set_ylabel(f'z={zbin}', fontsize=8, fontweight='bold', rotation=0,
                                     ha='right', va='center', labelpad=8)
                    col_idx += 1

            # Plot synthetic images
            samples_found = 0
            for replica in replicas:
                if samples_found >= n_images_per_condition:
                    break

                mask = (replica['zbin'] == zbin) & \
                       (replica['lesion_present'] == lesion_present) & \
                       (replica['domain'] == domain)

                indices = np.where(mask)[0]

                for idx in indices[:n_images_per_condition - samples_found]:
                    ax = fig.add_subplot(gs[row_idx + n_header_rows, col_idx])
                    plot_single_image(ax, replica['images'][idx], replica['masks'][idx], lesion_present)

                    # Add z-bin label on leftmost column (only if not showing test)
                    if col_idx == 0 and not show_test:
                        ax.set_ylabel(f'z={zbin}', fontsize=8, fontweight='bold', rotation=0,
                                     ha='right', va='center', labelpad=8)

                    col_idx += 1
                    samples_found += 1

                if samples_found >= n_images_per_condition:
                    break

            # Fill empty synthetic cells if not enough samples
            while samples_found < n_images_per_condition:
                ax = fig.add_subplot(gs[row_idx + n_header_rows, col_idx])
                ax.set_facecolor('#f0f0f0')
                ax.text(0.5, 0.5, '—', ha='center', va='center',
                       fontsize=12, color='#999', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                col_idx += 1
                samples_found += 1

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved image grid to {output_path}")


# =============================================================================
# Reporting Module
# =============================================================================

def generate_summary_report(
    test_df: pd.DataFrame,
    replicas: List[Dict[str, Any]],
    outliers_df: pd.DataFrame,
    deviation_df: pd.DataFrame
) -> str:
    """Generate text summary report.

    Args:
        test_df: Test distribution.
        replicas: List of replicas.
        outliers_df: Outlier DataFrame.
        deviation_df: Deviation metrics DataFrame.

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("REPLICA QUALITY CHECK SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Dataset overview
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append(f"Number of replicas: {len(replicas)}")
    lines.append(f"Samples per replica: {len(replicas[0]['images'])}")
    lines.append(f"Total replica samples: {sum(len(r['images']) for r in replicas)}")
    lines.append(f"Test samples: {test_df['n_slices'].sum():.0f}")
    lines.append(f"Unique conditions: {len(test_df)}")
    lines.append("")

    # Sample count validation
    lines.append("2. SAMPLE COUNT VALIDATION")
    lines.append("-" * 80)
    replica_mean_counts = {}
    for replica in replicas:
        for zbin, lesion, domain in zip(replica['zbin'], replica['lesion_present'], replica['domain']):
            key = (int(zbin), int(lesion), int(domain))
            replica_mean_counts[key] = replica_mean_counts.get(key, 0) + 1

    # Average across replicas
    for key in replica_mean_counts:
        replica_mean_counts[key] /= len(replicas)

    total_diff = 0
    for _, row in test_df.iterrows():
        key = (int(row['zbin']), int(row['lesion_present']), DOMAIN_MAP[row['domain']])
        test_count = row['n_slices']
        replica_count = replica_mean_counts.get(key, 0)
        total_diff += abs(replica_count - test_count)

    lines.append(f"Total absolute difference: {total_diff:.1f}")
    lines.append(f"Mean absolute difference per condition: {total_diff / len(test_df):.2f}")
    lines.append("")

    # Statistical comparison
    lines.append("3. STATISTICAL COMPARISON")
    lines.append("-" * 80)

    for metric in ['mean_brain_frac', 'mean_intensity', 'std_intensity', 'mean_lesion_area_px']:
        metric_devs = deviation_df[deviation_df['metric_name'] == metric]
        if len(metric_devs) > 0:
            mae = metric_devs['absolute_deviation'].mean()
            rmse = np.sqrt((metric_devs['absolute_deviation'] ** 2).mean())
            max_dev = metric_devs['absolute_deviation'].max()

            lines.append(f"{metric}:")
            lines.append(f"  MAE: {mae:.4f}")
            lines.append(f"  RMSE: {rmse:.4f}")
            lines.append(f"  Max deviation: {max_dev:.4f}")
    lines.append("")

    # Outlier summary
    lines.append("4. OUTLIER SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total outliers: {len(outliers_df)}")

    if len(outliers_df) > 0:
        total_samples = sum(len(r['images']) for r in replicas)
        outlier_rate = len(outliers_df) / total_samples * 100
        lines.append(f"Outlier rate: {outlier_rate:.2f}%")
        lines.append("")

        lines.append("Outliers by metric:")
        for metric in outliers_df['metric_name'].unique():
            count = len(outliers_df[outliers_df['metric_name'] == metric])
            lines.append(f"  {metric}: {count}")
        lines.append("")

        lines.append("Top 10 outliers by |z-score|:")
        for idx, row in outliers_df.head(10).iterrows():
            lines.append(f"  Replica {row['replica_id']}, sample {row['sample_idx']}: "
                        f"{row['metric_name']} = {row['value']:.4f} "
                        f"(expected {row['expected_mean']:.4f}, z={row['z_score']:.2f})")
    else:
        lines.append("No outliers detected.")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_summary_json(
    data: Dict[str, Any],
    output_path: Path
) -> None:
    """Export structured JSON summary.

    Args:
        data: Summary data dict.
        output_path: Path to save JSON.
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved JSON summary to {output_path}")


def save_comparison_csv(
    comparison_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Export per-condition comparison CSV.

    Args:
        comparison_df: Comparison DataFrame.
        output_path: Path to save CSV.
    """
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"Saved comparison CSV to {output_path}")


def save_outliers_csv(
    outliers_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Export outliers CSV.

    Args:
        outliers_df: Outliers DataFrame.
        output_path: Path to save CSV.
    """
    if len(outliers_df) > 0:
        outliers_df.to_csv(output_path, index=False)
        logger.info(f"Saved outliers CSV to {output_path}")
    else:
        # Save empty CSV with headers
        pd.DataFrame(columns=['replica_id', 'sample_idx', 'zbin', 'lesion_present', 'domain',
                              'metric_name', 'value', 'expected_mean', 'population_std', 'z_score']
                    ).to_csv(output_path, index=False)
        logger.info(f"Saved empty outliers CSV to {output_path}")


# =============================================================================
# Main Workflow
# =============================================================================

def run_quality_check(
    replicas_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    dist_configs: Optional[List[Tuple[Path, str]]] = None,
    slices_configs: Optional[List[Tuple[Path, str]]] = None,
    # Legacy single-split arguments for backwards compatibility
    test_dist_csv: Optional[Path] = None,
    test_slices_csv: Optional[Path] = None,
) -> None:
    """Execute complete quality check pipeline.

    Supports two modes:
    1. Legacy mode: Single test distribution (test_dist_csv, test_slices_csv)
    2. Multi-split mode: Multiple splits via dist_configs and slices_configs

    Args:
        replicas_dir: Directory with replica NPZ files.
        output_dir: Output directory.
        config: Configuration dict with parameters.
        dist_configs: List of (csv_path, split_name) for distribution CSVs.
        slices_configs: List of (csv_path, split_name) for slice CSVs.
        test_dist_csv: Legacy - Path to test distribution CSV.
        test_slices_csv: Legacy - Path to test slices CSV.
    """
    # Setup output directories
    plots_dir = output_dir / 'plots'
    reports_dir = output_dir / 'reports'
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING REPLICA QUALITY CHECK")
    logger.info("=" * 80)

    # Handle backwards compatibility: convert legacy args to configs
    if dist_configs is None and test_dist_csv is not None:
        dist_configs = [(test_dist_csv, 'test')]
    if slices_configs is None and test_slices_csv is not None:
        slices_configs = [(test_slices_csv, 'test')]

    if dist_configs is None or len(dist_configs) == 0:
        raise ValueError("At least one distribution CSV must be provided")

    # Load distribution data
    logger.info("Loading distribution data...")
    if len(dist_configs) == 1:
        # Single split mode
        ref_df = load_split_distribution(dist_configs[0][0], dist_configs[0][1])
        splits_used = [dist_configs[0][1]]
    else:
        # Multi-split mode
        ref_df, splits_used = load_multi_split_distribution(dist_configs)

    logger.info(f"Using splits: {', '.join(splits_used)}")

    # Load replicas
    replicas = load_all_replicas(replicas_dir, max_replicas=config.get('max_replicas'))

    # Load slices for image comparison if provided
    real_slices = None
    if slices_configs is not None and len(slices_configs) > 0:
        logger.info("Loading real slices for image comparison...")
        if len(slices_configs) == 1:
            real_slices = load_split_slices(slices_configs[0][0], slices_configs[0][1])
        else:
            real_slices, _ = load_multi_split_slices(slices_configs)

    # Compute statistics
    logger.info("Computing statistics...")
    for replica in replicas:
        compute_replica_statistics(
            replica,
            gaussian_sigma_px=config.get('gaussian_sigma_px', 1.0),
            min_component_px=config.get('min_component_px', 100)
        )

    replica_mean_df, replica_std_df = compute_mean_std_across_replicas(replicas)

    # Detect outliers (use combined reference distribution)
    logger.info("Detecting outliers...")
    outliers_df = detect_outliers(
        replicas,
        ref_df,
        threshold_std=config.get('outlier_threshold', 2.0)
    )

    # Compute deviations
    logger.info("Computing deviation metrics...")
    deviation_df = compute_deviation_metrics(ref_df, replica_mean_df)

    # Generate visualizations
    if not config.get('skip_images', False):
        logger.info("Generating visualizations...")

        plot_sample_count_comparison(
            ref_df, replica_mean_df, replica_std_df,
            plots_dir / 'sample_count_comparison.png',
            dpi=config.get('dpi', 300)
        )

        plot_brain_fraction_comparison(
            ref_df, replicas, outliers_df,
            plots_dir / 'brain_fraction_comparison.png',
            dpi=config.get('dpi', 300)
        )

        plot_intensity_comparison(
            ref_df, replicas, outliers_df,
            plots_dir / 'intensity_comparison.png',
            dpi=config.get('dpi', 300)
        )

        plot_lesion_area_comparison(
            ref_df, replicas, outliers_df,
            plots_dir / 'lesion_area_comparison.png',
            dpi=config.get('dpi', 300)
        )

        # Image grids
        all_zbins = sorted(ref_df['zbin'].unique())
        representative_zbins = select_representative_zbins(
            all_zbins,
            n_bins=config.get('n_representative_zbins', 5)
        )

        plot_image_grid(
            replicas,
            all_zbins,
            n_images_per_condition=config.get('n_images_per_condition', 3),
            output_path=plots_dir / 'image_grid_detailed.png',
            test_slices=real_slices,
            dpi=config.get('dpi', 150),
            title_suffix=" (All Z-bins)"
        )

        plot_image_grid(
            replicas,
            representative_zbins,
            n_images_per_condition=config.get('n_images_per_condition', 3),
            output_path=plots_dir / 'image_grid_representative.png',
            test_slices=real_slices,
            dpi=config.get('dpi', 150),
            title_suffix=" (Representative Z-bins)"
        )

    # Generate reports
    logger.info("Generating reports...")

    summary_text = generate_summary_report(ref_df, replicas, outliers_df, deviation_df)
    with open(reports_dir / 'summary.txt', 'w') as f:
        f.write(summary_text)
    print(summary_text)

    summary_json = {
        'metadata': {
            'n_replicas': len(replicas),
            'n_ref_samples': int(ref_df['n_slices'].sum()),
            'n_conditions': len(ref_df),
            'splits_used': splits_used,
            'outlier_threshold': config.get('outlier_threshold', 2.0),
            'timestamp': datetime.now().isoformat(),
        },
        'outliers': {
            'total': len(outliers_df),
            'by_metric': outliers_df['metric_name'].value_counts().to_dict() if len(outliers_df) > 0 else {},
        },
        'deviations': {
            metric: {
                'mae': float(deviation_df[deviation_df['metric_name'] == metric]['absolute_deviation'].mean()),
                'rmse': float(np.sqrt((deviation_df[deviation_df['metric_name'] == metric]['absolute_deviation'] ** 2).mean())),
            }
            for metric in deviation_df['metric_name'].unique()
        },
    }
    save_summary_json(summary_json, reports_dir / 'summary.json')

    save_comparison_csv(deviation_df, reports_dir / 'comparison.csv')
    save_outliers_csv(outliers_df, reports_dir / 'outliers.csv')

    logger.info("=" * 80)
    logger.info("QUALITY CHECK COMPLETE")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)


# =============================================================================
# CLI Interface
# =============================================================================

def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='Quality check for synthetic replicas against real data distribution. '
                    'Supports comparing against one or more data splits (train/val/test).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-split (backwards compatible):
  python -m src.diffusion.scripts.check_replica_quality \\
      --replicas-dir outputs/replicas/replicas \\
      --test-dist-csv docs/test_analysis/test_zbin_distribution.csv \\
      --output-dir outputs/quality_check

  # Multi-split (for uniform replicas):
python -m src.diffusion.scripts.check_replica_quality \
--replicas-dir /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/replicas \
--train-dist-csv docs/train_analysis/train_zbin_distribution.csv \
--train-slices-csv /media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache/train.csv \
--val-dist-csv docs/val_analysis/val_zbin_distribution.csv \
--val-slices-csv /media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache/val.csv \
--test-dist-csv docs/test_analysis/test_zbin_distribution.csv \
--test-slices-csv /media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache/test.csv \
--output-dir /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/quality_check
        """
    )

    # Required arguments
    parser.add_argument('--replicas-dir', type=str, required=True,
                        help='Directory containing replica_*.npz files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for reports and plots')

    # Split arguments (at least one dist-csv required)
    split_group = parser.add_argument_group(
        'Data splits',
        'Provide at least one distribution CSV. Multiple splits will be combined.'
    )
    split_group.add_argument('--train-dist-csv', type=str, default=None,
                             help='Path to train_zbin_distribution.csv (optional)')
    split_group.add_argument('--train-slices-csv', type=str, default=None,
                             help='Path to train.csv with slice filepaths (optional)')
    split_group.add_argument('--val-dist-csv', type=str, default=None,
                             help='Path to val_zbin_distribution.csv (optional)')
    split_group.add_argument('--val-slices-csv', type=str, default=None,
                             help='Path to val.csv with slice filepaths (optional)')
    split_group.add_argument('--test-dist-csv', type=str, default=None,
                             help='Path to test_zbin_distribution.csv (optional, but at least one dist-csv required)')
    split_group.add_argument('--test-slices-csv', type=str, default=None,
                             help='Path to test.csv with slice filepaths (optional)')

    # Optional configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML (for brain mask params)')
    parser.add_argument('--outlier-threshold', type=float, default=2.0,
                        help='Outlier detection threshold in std units (default: 2.0)')
    parser.add_argument('--max-replicas', type=int, default=None,
                        help='Limit number of replicas to load (for testing)')

    # Image visualization options
    parser.add_argument('--n-images-per-condition', type=int, default=3,
                        help='Number of example images per condition (default: 3)')
    parser.add_argument('--n-representative-zbins', type=int, default=5,
                        help='Number of z-bins for representative grid (default: 5)')
    parser.add_argument('--skip-images', action='store_true',
                        help='Skip image visualization (faster)')

    # Brain mask computation parameters
    parser.add_argument('--gaussian-sigma-px', type=float, default=1.0,
                        help='Gaussian sigma for brain mask (default: 1.0)')
    parser.add_argument('--min-component-px', type=int, default=100,
                        help='Minimum component size for brain mask (default: 100)')

    # Output options
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved plots (default: 300)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Update logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Convert to Path objects
    replicas_dir = Path(args.replicas_dir)
    output_dir = Path(args.output_dir)

    # Build split configs
    dist_configs: List[Tuple[Path, str]] = []
    slices_configs: List[Tuple[Path, str]] = []

    if args.train_dist_csv:
        train_dist_path = Path(args.train_dist_csv)
        if not train_dist_path.exists():
            raise FileNotFoundError(f"Train distribution CSV not found: {train_dist_path}")
        dist_configs.append((train_dist_path, 'train'))
        if args.train_slices_csv:
            train_slices_path = Path(args.train_slices_csv)
            if not train_slices_path.exists():
                raise FileNotFoundError(f"Train slices CSV not found: {train_slices_path}")
            slices_configs.append((train_slices_path, 'train'))

    if args.val_dist_csv:
        val_dist_path = Path(args.val_dist_csv)
        if not val_dist_path.exists():
            raise FileNotFoundError(f"Val distribution CSV not found: {val_dist_path}")
        dist_configs.append((val_dist_path, 'val'))
        if args.val_slices_csv:
            val_slices_path = Path(args.val_slices_csv)
            if not val_slices_path.exists():
                raise FileNotFoundError(f"Val slices CSV not found: {val_slices_path}")
            slices_configs.append((val_slices_path, 'val'))

    if args.test_dist_csv:
        test_dist_path = Path(args.test_dist_csv)
        if not test_dist_path.exists():
            raise FileNotFoundError(f"Test distribution CSV not found: {test_dist_path}")
        dist_configs.append((test_dist_path, 'test'))
        if args.test_slices_csv:
            test_slices_path = Path(args.test_slices_csv)
            if not test_slices_path.exists():
                raise FileNotFoundError(f"Test slices CSV not found: {test_slices_path}")
            slices_configs.append((test_slices_path, 'test'))

    # Validate that at least one distribution CSV is provided
    if not dist_configs:
        raise ValueError(
            "At least one distribution CSV must be provided. "
            "Use --train-dist-csv, --val-dist-csv, or --test-dist-csv."
        )

    # Validate replicas directory
    if not replicas_dir.exists():
        raise FileNotFoundError(f"Replicas directory not found: {replicas_dir}")

    # Setup logging to file
    log_file = output_dir / 'check_replica_quality.log'
    output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # Build config dict
    config = {
        'outlier_threshold': args.outlier_threshold,
        'max_replicas': args.max_replicas,
        'n_images_per_condition': args.n_images_per_condition,
        'n_representative_zbins': args.n_representative_zbins,
        'skip_images': args.skip_images,
        'gaussian_sigma_px': args.gaussian_sigma_px,
        'min_component_px': args.min_component_px,
        'dpi': args.dpi,
    }

    # Load config YAML if provided
    if args.config:
        yaml_cfg = OmegaConf.load(args.config)
        if 'postprocessing' in yaml_cfg and 'zbin_priors' in yaml_cfg.postprocessing:
            pp_cfg = yaml_cfg.postprocessing.zbin_priors
            config['gaussian_sigma_px'] = pp_cfg.get('gaussian_sigma_px', config['gaussian_sigma_px'])
            config['min_component_px'] = pp_cfg.get('min_component_px', config['min_component_px'])

    # Run quality check with split configs
    run_quality_check(
        replicas_dir=replicas_dir,
        output_dir=output_dir,
        config=config,
        dist_configs=dist_configs if dist_configs else None,
        slices_configs=slices_configs if slices_configs else None,
    )


if __name__ == "__main__":
    main()
