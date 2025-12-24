#!/usr/bin/env python3
"""
Analyze and plot the distribution of epilepsy lesions across slice depths.

This script reads the cached slice data and visualizes how lesion frequency
varies with slice index (z-position), helping identify at which depths
epilepsy lesions are most commonly found.

Usage:
    python src/diffusion/scripts/analyze_lesion_distribution.py
    python src/diffusion/scripts/analyze_lesion_distribution.py --config src/diffusion/config/jsddpm.yaml
    python src/diffusion/scripts/analyze_lesion_distribution.py --output lesion_distribution.png
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    cfg = OmegaConf.load(config_path)
    # Resolve interpolations (e.g., ${data.root_dir})
    OmegaConf.resolve(cfg)
    return cfg


def load_cache_data(cache_dir: Path, splits: List[str] = ["train", "val", "test"]) -> pd.DataFrame:
    """
    Load cached slice data from CSV files.

    Args:
        cache_dir: Path to the slice cache directory
        splits: List of splits to load (default: ["train", "val", "test"])

    Returns:
        Combined DataFrame with all splits
    """
    dfs = []
    for split in splits:
        csv_path = cache_dir / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} slices from {split}.csv")
            dfs.append(df)
        else:
            print(f"Warning: {csv_path} not found, skipping...")

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {cache_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal slices loaded: {len(combined)}")
    return combined


def analyze_lesion_distribution(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Analyze the distribution of lesions across slice indices.

    Args:
        df: DataFrame with slice data

    Returns:
        Tuple of (lesion_counts_by_z, statistics)
    """
    # Filter for lesion slices only
    lesion_slices = df[df['has_lesion'] == True].copy()

    # Group by z_index and count occurrences
    lesion_counts = lesion_slices.groupby('z_index').size()

    # Compute statistics
    stats = {
        'total_slices': len(df),
        'total_lesion_slices': len(lesion_slices),
        'lesion_percentage': (len(lesion_slices) / len(df)) * 100,
        'unique_subjects_with_lesions': lesion_slices['subject_id'].nunique(),
        'min_z': lesion_counts.index.min() if len(lesion_counts) > 0 else None,
        'max_z': lesion_counts.index.max() if len(lesion_counts) > 0 else None,
        'peak_z': lesion_counts.idxmax() if len(lesion_counts) > 0 else None,
        'peak_count': lesion_counts.max() if len(lesion_counts) > 0 else None,
    }

    return lesion_counts, stats


def plot_lesion_distribution(
    lesion_counts: pd.Series,
    stats: Dict[str, int],
    output_path: str = None,
    show: bool = True
) -> None:
    """
    Create a visualization of lesion distribution by slice index.

    Args:
        lesion_counts: Series with z_index as index and counts as values
        stats: Dictionary of statistics
        output_path: Optional path to save the figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot bar chart
    ax.bar(lesion_counts.index, lesion_counts.values, color='#E74C3C', alpha=0.7, edgecolor='black')

    # Add peak line
    if stats['peak_z'] is not None:
        ax.axvline(stats['peak_z'], color='blue', linestyle='--', linewidth=2,
                   label=f"Peak at z={stats['peak_z']} ({stats['peak_count']} lesions)")

    # Add mean line
    mean_z = (lesion_counts.index.values * lesion_counts.values).sum() / lesion_counts.values.sum()
    ax.axvline(mean_z, color='green', linestyle='--', linewidth=2,
               label=f"Mean z={mean_z:.1f}")

    # Labels and title
    ax.set_xlabel('Slice Index (z)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Lesion Occurrences', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Epilepsy Lesions Across Slice Depths', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Legend
    ax.legend(loc='upper right', fontsize=10)

    # Add statistics text box
    stats_text = (
        f"Statistics:\n"
        f"Total slices: {stats['total_slices']:,}\n"
        f"Lesion slices: {stats['total_lesion_slices']:,} ({stats['lesion_percentage']:.1f}%)\n"
        f"Unique subjects: {stats['unique_subjects_with_lesions']}\n"
        f"Z range: [{stats['min_z']}, {stats['max_z']}]"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save if output path specified
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

    # Show if requested
    if show:
        plt.show()

    plt.close()


def plot_lesion_distribution_by_split(
    df: pd.DataFrame,
    output_path: str = None,
    show: bool = True
) -> None:
    """
    Create a visualization showing lesion distribution for each split separately.

    Args:
        df: DataFrame with slice data
        output_path: Optional path to save the figure
        show: Whether to display the plot
    """
    lesion_df = df[df['has_lesion'] == True].copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    splits = ['train', 'val', 'test']
    colors = ['#3498DB', '#E74C3C', '#2ECC71']

    for i, (split, color) in enumerate(zip(splits, colors)):
        split_lesions = lesion_df[lesion_df['split'] == split]
        lesion_counts = split_lesions.groupby('z_index').size()

        if len(lesion_counts) > 0:
            axes[i].bar(lesion_counts.index, lesion_counts.values,
                       color=color, alpha=0.7, edgecolor='black')

            # Add statistics
            n_lesions = len(split_lesions)
            n_subjects = split_lesions['subject_id'].nunique()
            peak_z = lesion_counts.idxmax()
            peak_count = lesion_counts.max()

            axes[i].set_ylabel('Count', fontsize=11, fontweight='bold')
            axes[i].set_title(f'{split.upper()} - {n_lesions} lesions from {n_subjects} subjects | '
                             f'Peak: z={peak_z} ({peak_count} lesions)',
                             fontsize=12, fontweight='bold')
            axes[i].grid(axis='y', alpha=0.3, linestyle='--')
            axes[i].axvline(peak_z, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

    axes[2].set_xlabel('Slice Index (z)', fontsize=12, fontweight='bold')
    fig.suptitle('Lesion Distribution by Dataset Split', fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    if output_path:
        split_path = Path(output_path)
        split_output = split_path.parent / (split_path.stem + '_by_split' + split_path.suffix)
        plt.savefig(split_output, dpi=300, bbox_inches='tight')
        print(f"Split-wise plot saved to: {split_output}")

    if show:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and plot epilepsy lesion distribution by slice depth',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/diffusion/config/jsddpm.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='lesion_distribution.png',
        help='Output path for the plot (set to empty string to skip saving)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot (only save)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Which splits to include in analysis'
    )
    parser.add_argument(
        '--split-wise',
        action='store_true',
        help='Also create split-wise visualization'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    cfg = load_config(args.config)

    # Get cache directory
    cache_dir = Path(cfg.data.cache_dir)
    print(f"Cache directory: {cache_dir}\n")

    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    # Load data
    df = load_cache_data(cache_dir, splits=args.splits)

    # Analyze lesion distribution
    print("\nAnalyzing lesion distribution...")
    lesion_counts, stats = analyze_lesion_distribution(df)

    # Print statistics
    print("\n" + "="*60)
    print("LESION DISTRIBUTION STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key:.<40} {value}")
    print("="*60 + "\n")

    # Plot combined distribution
    output_path = args.output if args.output else None
    plot_lesion_distribution(
        lesion_counts,
        stats,
        output_path=output_path,
        show=not args.no_show
    )

    # Plot split-wise distribution if requested
    if args.split_wise:
        plot_lesion_distribution_by_split(
            df,
            output_path=output_path,
            show=not args.no_show
        )

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
