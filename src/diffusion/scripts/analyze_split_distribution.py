#!/usr/bin/env python3
"""Analyze dataset split distribution and extract empirical statistics.

This script analyzes a specific dataset split (train, val, or test) from the slice cache
and computes statistics for synthetic data generation or dataset analysis.

Output CSV contains one row per (zbin, lesion_present, domain) combination with:
- Slice counts
- Brain fraction statistics
- Intensity statistics
- Lesion area statistics (when applicable)

Usage:
    python src/diffusion/scripts/analyze_split_distribution.py \
        --config src/diffusion/config/jsddpm.yaml \
        --test \
        --output-dir outputs/test_analysis
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Import brain mask computation from existing utilities
from src.diffusion.utils.zbin_priors import compute_brain_foreground_mask

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> DictConfig:
    """Load and resolve configuration from YAML file.

    Args:
        config_path: Path to jsddpm.yaml configuration file.

    Returns:
        Resolved OmegaConf DictConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)  # Resolve interpolations like ${data.root_dir}

    logger.info(f"Loaded configuration from {config_path}")
    return cfg


def load_split_csv(cache_dir: Path, split: str) -> pd.DataFrame:
    """Load split CSV (train.csv, val.csv, or test.csv) from cache directory.

    Args:
        cache_dir: Path to slice cache directory.
        split: Split name ('train', 'val', or 'test').

    Returns:
        DataFrame with columns: subject_id, z_index, z_bin, pathology_class,
        token, source, split, has_lesion, filepath.

    Raises:
        FileNotFoundError: If CSV doesn't exist.
        ValueError: If required columns are missing.
    """
    csv_path = cache_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{split.capitalize()} CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ['z_bin', 'has_lesion', 'source', 'filepath']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {split}.csv: {missing}")

    logger.info(f"Loaded {len(df)} {split} slices from {csv_path}")
    return df


def compute_slice_statistics(
    npz_path: Path,
    gaussian_sigma_px: float,
    min_component_px: int,
    has_lesion: bool,
) -> Optional[Dict[str, float]]:
    """Compute statistics for a single slice.

    Algorithm:
    1. Load NPZ file (image, mask)
    2. Compute brain mask using compute_brain_foreground_mask()
    3. Calculate brain_frac = brain_mask.sum() / brain_mask.size
    4. Extract brain pixels: brain_pixels = image[brain_mask]
    5. Compute mean_intensity = brain_pixels.mean()
    6. Compute std_intensity = brain_pixels.std()
    7. If has_lesion, compute lesion_area_px = (mask > 0).sum()

    Args:
        npz_path: Path to .npz file.
        gaussian_sigma_px: Sigma for brain mask computation.
        min_component_px: Minimum component size for brain mask.
        has_lesion: Whether this slice has a lesion (from CSV metadata).

    Returns:
        Dict with keys:
        - 'brain_frac': float (fraction of pixels that are brain)
        - 'mean_intensity': float (mean intensity within brain)
        - 'std_intensity': float (std intensity within brain)
        - 'lesion_area_px': float (lesion area if has_lesion, else None)

        Returns None if file cannot be loaded or brain mask computation fails.
    """
    # Load NPZ file
    try:
        data = np.load(npz_path, allow_pickle=True)
        image = data['image']
        mask = data['mask']
    except Exception as e:
        logger.warning(f"Failed to load {npz_path.name}: {e}")
        return None

    # Ensure 2D
    if image.ndim == 3:
        image = image.squeeze()
    if mask.ndim == 3:
        mask = mask.squeeze()

    if image.ndim != 2:
        logger.warning(f"Invalid image dimensions for {npz_path.name}: {image.shape}")
        return None

    # Compute brain foreground mask using Otsu + connected components
    brain_mask = compute_brain_foreground_mask(
        image,
        gaussian_sigma_px=gaussian_sigma_px,
        min_component_px=min_component_px,
        n_components_to_keep=1,
    )

    if brain_mask is None:
        logger.warning(f"Brain mask computation failed for {npz_path.name}")
        return None

    # Compute brain fraction
    brain_frac = brain_mask.sum() / brain_mask.size

    # Extract brain pixels
    brain_pixels = image[brain_mask]

    if brain_pixels.size == 0:
        logger.warning(f"No brain pixels found for {npz_path.name}")
        return None

    # Compute intensity statistics (only within brain)
    mean_intensity = float(brain_pixels.mean())
    std_intensity = float(brain_pixels.std())

    # Compute lesion area if applicable
    lesion_area_px = None
    if has_lesion:
        # Binarize mask (threshold at 0 for [-1, 1] range)
        lesion_binary = mask > 0
        lesion_area_px = float(lesion_binary.sum())

    return {
        'brain_frac': brain_frac,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'lesion_area_px': lesion_area_px,
    }


def aggregate_group_statistics(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate statistics across multiple slices in a group.

    Args:
        stats_list: List of per-slice statistics dicts.

    Returns:
        Dict with aggregated statistics:
        - 'n_slices': int (count)
        - 'mean_brain_frac': float
        - 'mean_intensity': float
        - 'std_intensity': float
        - 'mean_lesion_area_px': float (NaN if no lesions)
    """
    n_slices = len(stats_list)

    # Extract arrays for each statistic
    brain_fracs = np.array([s['brain_frac'] for s in stats_list])
    mean_intensities = np.array([s['mean_intensity'] for s in stats_list])
    std_intensities = np.array([s['std_intensity'] for s in stats_list])

    # Lesion areas (only non-None values)
    lesion_areas = [s['lesion_area_px'] for s in stats_list if s['lesion_area_px'] is not None]

    # Compute means
    aggregated = {
        'n_slices': n_slices,
        'mean_brain_frac': float(brain_fracs.mean()),
        'mean_intensity': float(mean_intensities.mean()),
        'std_intensity': float(std_intensities.mean()),  # Mean of per-slice stds
        'mean_lesion_area_px': float(np.mean(lesion_areas)) if lesion_areas else np.nan,
    }

    return aggregated


def analyze_distribution(
    df: pd.DataFrame,
    cache_dir: Path,
    cfg: DictConfig,
    split: str,
) -> pd.DataFrame:
    """Analyze dataset split distribution and compute statistics.

    Algorithm:
    1. Create grouping columns from CSV data
    2. GroupBy: ['z_bin', 'lesion_present', 'domain']
    3. For each group:
        a. Get list of filepaths
        b. For each filepath, compute slice statistics
        c. Filter out None results (failed slices)
        d. Aggregate statistics across group
        e. Create row with all stats
    4. Combine all rows into output DataFrame

    Args:
        df: Split CSV DataFrame.
        cache_dir: Path to cache directory.
        cfg: Configuration object.
        split: Split name.

    Returns:
        DataFrame with columns: split, zbin, lesion_present, domain, n_slices,
        mean_brain_frac, mean_intensity, std_intensity, mean_lesion_area_px.
    """
    # Extract parameters from config
    gaussian_sigma_px = cfg.postprocessing.zbin_priors.gaussian_sigma_px
    min_component_px = cfg.postprocessing.zbin_priors.min_component_px

    # Create grouping columns
    df = df.copy()
    df['domain'] = df['source']  # Already 'control' or 'epilepsy'
    df['lesion_present'] = df['has_lesion'].astype(int)

    # GroupBy
    grouped = df.groupby(['z_bin', 'lesion_present', 'domain'])

    logger.info(f"Found {len(grouped)} unique (z_bin, lesion_present, domain) groups")

    # Process each group
    results = []
    for (z_bin, lesion_present, domain), group_df in tqdm(
        grouped,
        desc="Processing groups",
        unit="group"
    ):
        # Get filepaths for this group
        filepaths = group_df['filepath'].tolist()
        has_lesion_flags = group_df['has_lesion'].tolist()

        # Compute statistics for each slice
        stats_list = []
        for filepath, has_lesion in zip(filepaths, has_lesion_flags):
            npz_path = cache_dir / filepath

            if not npz_path.exists():
                logger.warning(f"NPZ file not found: {npz_path}")
                continue

            stats = compute_slice_statistics(
                npz_path,
                gaussian_sigma_px,
                min_component_px,
                has_lesion,
            )

            if stats is not None:
                stats_list.append(stats)

        # Skip group if no valid slices
        if len(stats_list) == 0:
            logger.warning(
                f"No valid slices for group (z_bin={z_bin}, "
                f"lesion_present={lesion_present}, domain={domain})"
            )
            continue

        # Aggregate statistics
        aggregated = aggregate_group_statistics(stats_list)

        # Create result row
        row = {
            'split': split,
            'zbin': int(z_bin),
            'lesion_present': int(lesion_present),
            'domain': domain,
            **aggregated
        }
        results.append(row)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by (zbin, lesion_present, domain)
    results_df = results_df.sort_values(['zbin', 'lesion_present', 'domain'])
    results_df = results_df.reset_index(drop=True)

    logger.info(f"Computed statistics for {len(results_df)} groups")

    return results_df


def save_distribution_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save distribution analysis to CSV.

    Args:
        df: Results DataFrame.
        output_path: Output CSV path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure column order
    columns = [
        'split', 'zbin', 'lesion_present', 'domain', 'n_slices',
        'mean_brain_frac', 'mean_intensity', 'std_intensity', 'mean_lesion_area_px'
    ]

    df[columns].to_csv(output_path, index=False)
    logger.info(f"Saved distribution CSV to {output_path}")


def print_summary_statistics(df: pd.DataFrame, split: str) -> None:
    """Print console summary of distribution.

    Prints:
    1. Total N (sum of n_slices)
    2. Per-zbin totals (grouped sum)
    3. Mixture proportions:
        - Control vs Epilepsy
        - Lesion vs No-lesion
        - Per-domain lesion rates
    4. Statistical summaries (brain_frac, intensity ranges)

    Args:
        df: Results DataFrame.
        split: Split name.
    """
    print("\n" + "=" * 70)
    print(f"{split.upper()} SET DISTRIBUTION SUMMARY")
    print("=" * 70)

    # Total N
    total_n = df['n_slices'].sum()
    print(f"\nTotal {split} slices: {total_n}")

    # Per-zbin totals
    print(f"\nSlices per z-bin:")
    zbin_totals = df.groupby('zbin')['n_slices'].sum().sort_index()
    for zbin, count in zbin_totals.items():
        print(f"  z-bin {zbin:2d}: {count:4d} slices")

    # Mixture proportions
    print(f"\nDomain distribution:")
    domain_counts = df.groupby('domain')['n_slices'].sum()
    for domain, count in domain_counts.items():
        proportion = count / total_n * 100
        print(f"  {domain:9s}: {count:4d} slices ({proportion:5.1f}%)")

    print(f"\nLesion presence:")
    lesion_counts = df.groupby('lesion_present')['n_slices'].sum()
    for lesion_flag, count in lesion_counts.items():
        label = "With lesion" if lesion_flag == 1 else "No lesion"
        proportion = count / total_n * 100
        print(f"  {label:12s}: {count:4d} slices ({proportion:5.1f}%)")

    print(f"\nPer-domain lesion rates:")
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        domain_total = domain_df['n_slices'].sum()
        domain_lesion = domain_df[domain_df['lesion_present'] == 1]['n_slices'].sum()
        rate = domain_lesion / domain_total * 100 if domain_total > 0 else 0
        print(f"  {domain:9s}: {domain_lesion:4d}/{domain_total:4d} ({rate:5.1f}%)")

    # Statistical summaries
    print(f"\nBrain fraction statistics:")
    print(f"  Mean: {df['mean_brain_frac'].mean():.3f}")
    print(f"  Min:  {df['mean_brain_frac'].min():.3f}")
    print(f"  Max:  {df['mean_brain_frac'].max():.3f}")

    print(f"\nIntensity statistics (within brain):")
    print(f"  Mean intensity: {df['mean_intensity'].mean():.3f} ± {df['mean_intensity'].std():.3f}")
    print(f"  Range: [{df['mean_intensity'].min():.3f}, {df['mean_intensity'].max():.3f}]")

    # Lesion area (only for lesion_present=1)
    lesion_df = df[df['lesion_present'] == 1]
    if len(lesion_df) > 0:
        lesion_areas = lesion_df['mean_lesion_area_px'].dropna()
        if len(lesion_areas) > 0:
            print(f"\nLesion area statistics (pixels):")
            print(f"  Mean: {lesion_areas.mean():.1f}")
            print(f"  Std:  {lesion_areas.std():.1f}")
            print(f"  Min:  {lesion_areas.min():.1f}")
            print(f"  Max:  {lesion_areas.max():.1f}")

    print("=" * 70 + "\n")


def create_visualizations(
    df: pd.DataFrame,
    output_dir: Path,
    z_bins: int,
    split: str,
) -> None:
    """Create distribution visualization plots.

    Creates 2x2 panel figure:
    - Panel 1: Stacked bar chart - n_slices per z-bin (4 groups)
    - Panel 2: Line plot - mean_brain_frac per z-bin (4 lines)
    - Panel 3: Line plot - mean_intensity per z-bin (4 lines)
    - Panel 4: Histogram - distribution of mean_lesion_area_px

    Args:
        df: Results DataFrame.
        output_dir: Output directory for plots.
        z_bins: Total number of z-bins.
        split: Split name.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{split.capitalize()} Set Distribution Analysis', fontsize=16, fontweight='bold')

    # Color scheme
    colors = {
        ('control', 0): ('steelblue', '-', 'Control (No Lesion)'),
        ('control', 1): ('darkblue', '--', 'Control (Lesion)'),
        ('epilepsy', 0): ('lightcoral', '-', 'Epilepsy (No Lesion)'),
        ('epilepsy', 1): ('darkred', '--', 'Epilepsy (Lesion)'),
    }

    # Panel 1: Stacked bar chart - n_slices per z-bin
    ax = axes[0, 0]

    # Prepare data for stacking
    all_zbins = np.arange(z_bins)
    stack_data = {}
    for (domain, lesion_present), (color, _, label) in colors.items():
        group_df = df[(df['domain'] == domain) & (df['lesion_present'] == lesion_present)]
        # Create full array with zeros, fill in values
        counts = np.zeros(z_bins)
        for _, row in group_df.iterrows():
            counts[int(row['zbin'])] = row['n_slices']
        stack_data[(domain, lesion_present)] = (counts, color, label)

    # Create stacked bars
    bottom = np.zeros(z_bins)
    for (domain, lesion_present), (counts, color, label) in stack_data.items():
        ax.bar(all_zbins, counts, bottom=bottom, color=color, label=label, width=0.8)
        bottom += counts

    ax.set_xlabel('Z-bin', fontweight='bold')
    ax.set_ylabel('Number of Slices', fontweight='bold')
    ax.set_title('Slice Count Distribution per Z-bin')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Line plot - mean_brain_frac per z-bin
    ax = axes[0, 1]
    for (domain, lesion_present), (color, linestyle, label) in colors.items():
        group_df = df[(df['domain'] == domain) & (df['lesion_present'] == lesion_present)]
        if len(group_df) > 0:
            group_df = group_df.sort_values('zbin')
            ax.plot(
                group_df['zbin'],
                group_df['mean_brain_frac'],
                color=color,
                linestyle=linestyle,
                marker='o',
                markersize=3,
                label=label,
                linewidth=1.5,
            )

    ax.set_xlabel('Z-bin', fontweight='bold')
    ax.set_ylabel('Mean Brain Fraction', fontweight='bold')
    ax.set_title('Brain Fraction per Z-bin')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])

    # Panel 3: Line plot - mean_intensity per z-bin
    ax = axes[1, 0]
    for (domain, lesion_present), (color, linestyle, label) in colors.items():
        group_df = df[(df['domain'] == domain) & (df['lesion_present'] == lesion_present)]
        if len(group_df) > 0:
            group_df = group_df.sort_values('zbin')
            ax.plot(
                group_df['zbin'],
                group_df['mean_intensity'],
                color=color,
                linestyle=linestyle,
                marker='o',
                markersize=3,
                label=label,
                linewidth=1.5,
            )

    ax.set_xlabel('Z-bin', fontweight='bold')
    ax.set_ylabel('Mean Intensity (within brain)', fontweight='bold')
    ax.set_title('Intensity Distribution per Z-bin')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 4: Histogram - lesion area distribution
    ax = axes[1, 1]
    lesion_df = df[df['lesion_present'] == 1]
    lesion_areas = lesion_df['mean_lesion_area_px'].dropna()

    if len(lesion_areas) > 0:
        # Separate by domain
        for domain in lesion_df['domain'].unique():
            domain_lesion_df = lesion_df[lesion_df['domain'] == domain]
            domain_areas = domain_lesion_df['mean_lesion_area_px'].dropna()

            if len(domain_areas) > 0:
                color = 'darkblue' if domain == 'control' else 'darkred'
                ax.hist(
                    domain_areas,
                    bins=20,
                    color=color,
                    alpha=0.6,
                    label=domain.capitalize(),
                    edgecolor='black',
                )

        ax.set_xlabel('Mean Lesion Area (pixels)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Lesion Area Distribution')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No lesion data available', ha='center', va='center')
        ax.set_xlabel('Mean Lesion Area (pixels)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Lesion Area Distribution')

    # Adjust layout and save
    plt.tight_layout()
    output_path = output_dir / f'{split}_distribution_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved visualization to {output_path}")


def validate_output(df: pd.DataFrame, z_bins: int) -> None:
    """Validate output DataFrame before saving.

    Args:
        df: Results DataFrame.
        z_bins: Expected number of z-bins.

    Raises:
        ValueError: If validation fails.
    """
    # Check columns
    expected_cols = [
        'split', 'zbin', 'lesion_present', 'domain', 'n_slices',
        'mean_brain_frac', 'mean_intensity', 'std_intensity', 'mean_lesion_area_px'
    ]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing output columns: {missing}")

    # Check z-bin range
    if df['zbin'].min() < 0 or df['zbin'].max() >= z_bins:
        raise ValueError(
            f"Invalid z-bin values: range [{df['zbin'].min()}, {df['zbin'].max()}], "
            f"expected [0, {z_bins-1}]"
        )

    # Check n_slices > 0
    if (df['n_slices'] <= 0).any():
        raise ValueError("Found groups with n_slices <= 0")

    # Sanity checks
    if (df['mean_brain_frac'] < 0).any() or (df['mean_brain_frac'] > 1).any():
        logger.warning("mean_brain_frac outside [0, 1] range detected")

    if (df['mean_intensity'] < -1.5).any() or (df['mean_intensity'] > 1.5).any():
        logger.warning("mean_intensity outside expected [-1, 1] range detected")

    lesion_areas = df[df['lesion_present'] == 1]['mean_lesion_area_px'].dropna()
    if len(lesion_areas) > 0 and (lesion_areas < 0).any():
        raise ValueError("Negative lesion areas found")

    logger.info("Output validation passed")


def main() -> None:
    """Main entry point for split distribution analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze dataset split distribution and extract empirical statistics'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/diffusion/config/jsddpm.yaml',
        help='Path to configuration YAML file'
    )
    
    # Split selection group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Analyze train split')
    group.add_argument('--val', action='store_true', help='Analyze validation split')
    group.add_argument('--test', action='store_true', help='Analyze test split')

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (optional, defaults to {split}_zbin_distribution.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for visualizations (optional, defaults to outputs/{split}_analysis)'
    )
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip visualization generation'
    )

    args = parser.parse_args()

    # Determine split
    if args.train:
        split = 'train'
    elif args.val:
        split = 'val'
    else:
        split = 'test'

    # Set default output paths if not provided
    if args.output is None:
        args.output = f'{split}_zbin_distribution.csv'
    
    if args.output_dir is None:
        args.output_dir = f'outputs/{split}_analysis'

    try:
        # Load configuration
        logger.info(f"Starting {split} distribution analysis...")
        cfg = load_config(args.config)

        # Extract paths and parameters
        cache_dir = Path(cfg.data.cache_dir)
        z_bins = cfg.conditioning.z_bins

        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"Z-bins: {z_bins}")

        # Load split CSV
        df = load_split_csv(cache_dir, split)

        # Analyze distribution
        results_df = analyze_distribution(df, cache_dir, cfg, split)

        # Validate output
        validate_output(results_df, z_bins)

        # Save CSV
        save_distribution_csv(results_df, Path(args.output_dir) / args.output)

        # Print summary
        print_summary_statistics(results_df, split)

        # Create visualizations (if not disabled)
        if not args.no_visualizations:
            create_visualizations(results_df, args.output_dir, z_bins, split)

        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == '__main__':
    main()