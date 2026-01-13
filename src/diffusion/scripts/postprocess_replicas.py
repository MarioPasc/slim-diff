#!/usr/bin/env python3
"""Post-process synthetic replicas with brain extraction.

This script applies brain extraction to remove background noise from synthetic
replica files. It processes .npz replica files and applies the brain extraction
algorithm from zbin_priors.py to both images and lesion masks.

Usage:
    # Process replicas with output to new directory
    python -m src.diffusion.scripts.postprocess_replicas \
        --replicas-dir /path/to/replicas \
        --replicas 0,1,2,3 \
        --out-dir /path/to/processed \
        --demo

    # In-place processing (overwrites originals - make backup first!)
    python -m src.diffusion.scripts.postprocess_replicas \
        --replicas-dir /path/to/replicas \
        --replicas 0 \
        --overwrite \
        --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Use non-interactive backend for headless environments
matplotlib.use('Agg')

# Import brain extraction from zbin_priors
from src.diffusion.utils.zbin_priors import compute_brain_foreground_mask

# Import visualization utilities from check_replica_quality
from src.diffusion.scripts.check_replica_quality import (
    create_overlay,
    to_display_range,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DOMAIN_MAP = {"control": 0, "epilepsy": 1}
DOMAIN_MAP_INV = {0: "control", 1: "epilepsy"}

# Condition labels for visualization
CONDITIONS = [
    (0, "control", 0, "Control (No Lesion)", "steelblue"),     # lesion_present, domain_str, domain, label, color
    #(1, "control", 0, "Control (Lesion)", "steelblue"),
    #(0, "epilepsy", 1, "Epilepsy (No Lesion)", "lightcoral"),
    (1, "epilepsy", 1, "Epilepsy (Lesion)", "lightcoral"),
]


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-process synthetic replicas with brain extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific replicas to new directory
  python -m src.diffusion.scripts.postprocess_replicas \\
      --replicas-dir /path/to/replicas \\
      --replicas 0,1,2,3,4,5 \\
      --out-dir /path/to/processed \\
      --demo

  # In-place processing (CAUTION: overwrites originals!)
  python -m src.diffusion.scripts.postprocess_replicas \\
      --replicas-dir /path/to/replicas \\
      --replicas 0 \\
      --overwrite \\
      --demo \\
      --demo-output /path/to/before_after.png

  # Custom parameters
  python -m src.diffusion.scripts.postprocess_replicas \\
      --replicas-dir /path/to/replicas \\
      --replicas 0,1,2 \\
      --out-dir /path/to/processed \\
      --gaussian-sigma-px 2.5 \\
      --min-component-px 600 \\
      --demo
        """,
    )

    # Required arguments
    parser.add_argument(
        "--replicas-dir",
        type=Path,
        required=True,
        help="Directory containing replica .npz files",
    )
    parser.add_argument(
        "--replicas",
        type=str,
        required=True,
        help="Comma-separated replica IDs to process (e.g., '0,1,2,3')",
    )

    # Output control (mutually exclusive)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original replica files (CAUTION: make backup first!)",
    )
    output_group.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for processed replicas",
    )

    # Brain extraction parameters
    parser.add_argument(
        "--gaussian-sigma-px",
        type=float,
        default=0.7,
        help="Gaussian smoothing sigma in pixels (default: 0.7)",
    )
    parser.add_argument(
        "--min-component-px",
        type=int,
        default=500,
        help="Minimum component size in pixels (default: 500)",
    )
    parser.add_argument(
        "--n-components-to-keep",
        type=int,
        default=1,
        help="Number of largest components to keep (default: 1)",
    )
    parser.add_argument(
        "--relaxed-threshold-factor",
        type=float,
        default=0.1,
        help="Relaxed threshold factor for secondary components (default: 0.1)",
    )
    parser.add_argument(
        "--n-first-bins",
        type=int,
        default=7,
        help="Number of low z-bins for multi-component handling (default: 7)",
    )
    parser.add_argument(
        "--max-components-for-first-bins",
        type=int,
        default=3,
        help="Max components to keep for first bins (default: 3)",
    )
    parser.add_argument(
        "--background-value",
        type=float,
        default=-1.0,
        help="Value to use for out-of-brain regions (default: -1.0)",
    )

    # Visualization options
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate before/after comparison visualization",
    )
    parser.add_argument(
        "--demo-output",
        type=Path,
        default=None,
        help="Output path for demo visualization (default: replicas_dir/postprocess_demo.png)",
    )
    parser.add_argument(
        "--n-images-per-condition",
        type=int,
        default=3,
        help="Number of images per condition in demo (default: 3)",
    )
    parser.add_argument(
        "--demo-seed",
        type=int,
        default=42,
        help="Random seed for selecting demo samples (default: 42)",
    )

    return parser.parse_args()


# ============================================================================
# Core Processing Functions
# ============================================================================

def load_replica(replica_path: Path) -> dict:
    """Load .npz replica file and return dict with all arrays.

    Args:
        replica_path: Path to replica .npz file

    Returns:
        Dictionary containing all replica arrays

    Raises:
        FileNotFoundError: If replica file doesn't exist
        ValueError: If required keys are missing
    """
    if not replica_path.exists():
        raise FileNotFoundError(f"Replica file not found: {replica_path}")

    data = np.load(replica_path)

    # Required keys
    required_keys = [
        'images', 'masks', 'zbin', 'lesion_present', 'domain',
        'condition_token', 'seed', 'k_index', 'replica_id'
    ]

    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(f"Missing required keys in {replica_path}: {missing_keys}")

    return {key: data[key] for key in data.keys()}


def process_replica(
    images: NDArray[np.float32],
    masks: NDArray[np.float32],
    zbins: NDArray[np.int32],
    gaussian_sigma_px: float,
    min_component_px: int,
    n_components_to_keep: int,
    relaxed_threshold_factor: float,
    n_first_bins: int,
    max_components_for_first_bins: int,
    background_value: float = 0.0,
    batch_size: int = 1000,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict]:
    """Apply brain extraction to all samples in replica.

    Args:
        images: (N, H, W) image array
        masks: (N, H, W) mask array
        zbins: (N,) z-bin indices
        gaussian_sigma_px: Gaussian smoothing sigma
        min_component_px: Minimum component size
        n_components_to_keep: Number of components to keep (base)
        relaxed_threshold_factor: Relaxed threshold for secondary components
        n_first_bins: Number of low z-bins for special handling
        max_components_for_first_bins: Max components for first bins
        background_value: Value to use for out-of-brain regions
        batch_size: Process in batches to save memory

    Returns:
        Tuple of (processed_images, processed_masks, stats_dict)
    """
    N = images.shape[0]
    processed_images = np.zeros_like(images)
    processed_masks = np.zeros_like(masks)

    failures = 0
    total_processed = 0

    # Process in batches
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_indices = range(batch_start, batch_end)

        for i in batch_indices:
            # Determine n_components based on z-bin
            zbin = int(zbins[i])
            if zbin < n_first_bins:
                n_components = max_components_for_first_bins
            else:
                n_components = n_components_to_keep

            # Compute brain mask
            brain_mask = compute_brain_foreground_mask(
                image=images[i],
                gaussian_sigma_px=gaussian_sigma_px,
                min_component_px=min_component_px,
                n_components_to_keep=n_components,
                relaxed_threshold_factor=relaxed_threshold_factor,
            )

            if brain_mask is None:
                # Fallback: keep original
                processed_images[i] = images[i]
                processed_masks[i] = masks[i]
                failures += 1
            else:
                # Apply mask to both image and lesion mask
                processed_images[i] = np.where(brain_mask, images[i], background_value)
                processed_masks[i] = np.where(brain_mask, masks[i], background_value)

            total_processed += 1

        # Progress update
        if (batch_end % 5000 == 0) or (batch_end == N):
            progress = (batch_end / N) * 100
            bar_length = 40
            filled = int(bar_length * batch_end // N)
            bar = '█' * filled + '░' * (bar_length - filled)
            logger.info(f"  Processing: [{bar}] {progress:5.1f}% ({batch_end}/{N})")

    stats = {
        'total_samples': total_processed,
        'failures': failures,
        'failure_rate': failures / total_processed if total_processed > 0 else 0.0,
    }

    return processed_images, processed_masks, stats


def save_replica(
    output_path: Path,
    replica_data: dict,
    processed_images: NDArray[np.float32],
    processed_masks: NDArray[np.float32],
) -> None:
    """Save processed replica maintaining original structure.

    Args:
        output_path: Output path for processed .npz file
        replica_data: Original replica data dict
        processed_images: Processed images array
        processed_masks: Processed masks array
    """
    # Determine original dtype
    original_dtype = replica_data['images'].dtype

    # Save with all original keys
    np.savez_compressed(
        output_path,
        images=processed_images.astype(original_dtype),
        masks=processed_masks.astype(original_dtype),
        zbin=replica_data['zbin'],
        lesion_present=replica_data['lesion_present'],
        domain=replica_data['domain'],
        condition_token=replica_data['condition_token'],
        seed=replica_data['seed'],
        k_index=replica_data['k_index'],
        replica_id=replica_data['replica_id'],
    )


def update_metadata(
    json_path: Path,
    output_json_path: Path,
    processing_params: dict,
) -> None:
    """Update metadata JSON with postprocessing information.

    Args:
        json_path: Original metadata JSON path
        output_json_path: Output metadata JSON path
        processing_params: Dictionary of processing parameters
    """
    if not json_path.exists():
        logger.warning(f"  Metadata file not found: {json_path}")
        return

    try:
        with open(json_path, 'r') as f:
            meta = json.load(f)

        # Add postprocessing section
        meta['postprocessing'] = {
            'applied': True,
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'brain_extraction',
            'parameters': processing_params,
        }

        # Save updated metadata
        with open(output_json_path, 'w') as f:
            json.dump(meta, f, indent=2)

    except Exception as e:
        logger.warning(f"  Failed to update metadata: {e}")


# ============================================================================
# Demo Visualization Functions
# ============================================================================

def select_representative_zbins(
    all_zbins: NDArray[np.int32],
    n_representative: int = 5,
) -> list[int]:
    """Select evenly-spaced representative z-bins.

    Args:
        all_zbins: Array of all z-bin values
        n_representative: Number of z-bins to select

    Returns:
        List of representative z-bin indices
    """
    unique_zbins = np.unique(all_zbins)
    n_zbins = len(unique_zbins)

    if n_zbins <= n_representative:
        return sorted(unique_zbins.tolist())

    # Select evenly-spaced indices
    indices = np.linspace(0, n_zbins - 1, n_representative, dtype=int)
    return sorted(unique_zbins[indices].tolist())


def select_demo_samples(
    replica_data: dict,
    zbins_to_show: list[int],
    n_images_per_condition: int,
    seed: int,
) -> dict:
    """Select random samples for each (zbin, condition) pair.

    Args:
        replica_data: Replica data dictionary
        zbins_to_show: List of z-bins to visualize
        n_images_per_condition: Number of images per condition
        seed: Random seed

    Returns:
        Dictionary mapping (zbin, lesion_present, domain) -> list of sample indices
    """
    np.random.seed(seed)
    selected = {}

    for zbin in zbins_to_show:
        for lesion_present, _, domain, _, _ in CONDITIONS:
            # Find matching samples
            mask = (
                (replica_data['zbin'] == zbin) &
                (replica_data['lesion_present'] == lesion_present) &
                (replica_data['domain'] == domain)
            )
            indices = np.where(mask)[0]

            # Randomly select n images
            if len(indices) > 0:
                n_select = min(n_images_per_condition, len(indices))
                selected_indices = np.random.choice(indices, size=n_select, replace=False)
                key = (zbin, lesion_present, domain)
                selected[key] = selected_indices.tolist()

    return selected


def create_demo_visualization(
    original_replica: dict,
    processed_replica: dict,
    output_path: Path,
    n_representative_zbins: int = 5,
    n_images_per_condition: int = 3,
    seed: int = 42,
) -> None:
    """Create grid showing pre-processed vs post-processed images.

    Args:
        original_replica: Original replica data dict
        processed_replica: Processed replica data dict
        output_path: Output path for visualization
        n_representative_zbins: Number of z-bins to show
        n_images_per_condition: Number of images per condition
        seed: Random seed for sample selection
    """
    logger.info("\nGenerating demo visualization...")

    # Select representative z-bins
    zbins_to_show = select_representative_zbins(
        original_replica['zbin'],
        n_representative_zbins,
    )
    logger.info(f"  Selected z-bins: {zbins_to_show}")
    logger.info(f"  Samples per condition: {n_images_per_condition}")

    # Select samples for each condition
    selected_samples = select_demo_samples(
        original_replica,
        zbins_to_show,
        n_images_per_condition,
        seed,
    )

    # Create figure
    n_rows = len(zbins_to_show)
    n_cols = len(CONDITIONS) * n_images_per_condition * 2  # pre + post for each condition

    fig = plt.figure(figsize=(4 * len(CONDITIONS) * n_images_per_condition, 2.5 * n_rows))

    # Add gridspec with header rows
    n_header_rows = 2
    gs = fig.add_gridspec(
        n_rows + n_header_rows, n_cols,
        height_ratios=[0.12] * n_header_rows + [1] * n_rows,
        hspace=0.03, wspace=0.02,
        left=0.02, right=0.98, top=0.95, bottom=0.02,
    )

    # Create headers
    for cond_idx, (lesion_present, domain_str, domain, label, color) in enumerate(CONDITIONS):
        base_col = cond_idx * n_images_per_condition * 2

        # Condition header (top row) spanning pre + post
        ax_cond = fig.add_subplot(gs[0, base_col:base_col + n_images_per_condition * 2])
        ax_cond.set_facecolor(color)
        ax_cond.text(
            0.5, 0.5, label,
            ha='center', va='center',
            fontsize=11, fontweight='bold', color='white',
        )
        ax_cond.set_xticks([])
        ax_cond.set_yticks([])

        # Pre-processed sub-header (second row)
        ax_pre = fig.add_subplot(gs[1, base_col:base_col + n_images_per_condition])
        ax_pre.set_facecolor('#e8e8e8')
        ax_pre.text(
            0.5, 0.5, 'Pre-processed',
            ha='center', va='center',
            fontsize=10, fontweight='bold', color='#333',
        )
        ax_pre.set_xticks([])
        ax_pre.set_yticks([])

        # Post-processed sub-header (second row)
        ax_post = fig.add_subplot(gs[1, base_col + n_images_per_condition:base_col + n_images_per_condition * 2])
        ax_post.set_facecolor('#d0d0d0')
        ax_post.text(
            0.5, 0.5, 'Post-processed',
            ha='center', va='center',
            fontsize=10, fontweight='bold', color='#333',
        )
        ax_post.set_xticks([])
        ax_post.set_yticks([])

    # Plot images
    for row_idx, zbin in enumerate(zbins_to_show):
        col_idx = 0

        for lesion_present, domain_str, domain, label, color in CONDITIONS:
            condition_key = (zbin, lesion_present, domain)
            sample_indices = selected_samples.get(condition_key, [])

            # Pre-processed images
            for img_idx in range(n_images_per_condition):
                ax = fig.add_subplot(gs[row_idx + n_header_rows, col_idx])

                if img_idx < len(sample_indices):
                    idx = sample_indices[img_idx]
                    image = original_replica['images'][idx]
                    mask = original_replica['masks'][idx]

                    # Convert to display range and create overlay
                    image_disp = to_display_range(image).astype(np.float32)

                    if lesion_present == 1 and mask is not None:
                        mask_disp = to_display_range(mask).astype(np.float32)
                        rgb = create_overlay(
                            image_disp, mask_disp,
                            alpha=0.5, color=(255, 0, 0), threshold=0.0
                        )
                    else:
                        rgb = np.stack([image_disp, image_disp, image_disp], axis=-1)

                    ax.imshow(rgb)
                else:
                    # Empty placeholder
                    ax.set_facecolor('#f5f5f5')
                    ax.text(0.5, 0.5, '—', ha='center', va='center', fontsize=20, color='#ccc')

                ax.set_xticks([])
                ax.set_yticks([])

                # Add z-bin label on first column
                if col_idx == 0:
                    ax.set_ylabel(f'Z-bin {zbin}', fontsize=9, fontweight='bold')

                col_idx += 1

            # Post-processed images
            for img_idx in range(n_images_per_condition):
                ax = fig.add_subplot(gs[row_idx + n_header_rows, col_idx])

                if img_idx < len(sample_indices):
                    idx = sample_indices[img_idx]
                    image = processed_replica['images'][idx]
                    mask = processed_replica['masks'][idx]

                    # Convert to display range and create overlay
                    image_disp = to_display_range(image).astype(np.float32)

                    if lesion_present == 1 and mask is not None:
                        mask_disp = to_display_range(mask).astype(np.float32)
                        rgb = create_overlay(
                            image_disp, mask_disp,
                            alpha=0.5, color=(255, 0, 0), threshold=0.0
                        )
                    else:
                        rgb = np.stack([image_disp, image_disp, image_disp], axis=-1)

                    ax.imshow(rgb)
                else:
                    # Empty placeholder
                    ax.set_facecolor('#f5f5f5')
                    ax.text(0.5, 0.5, '—', ha='center', va='center', fontsize=20, color='#ccc')

                ax.set_xticks([])
                ax.set_yticks([])
                col_idx += 1

    # Add title
    fig.suptitle(
        'Brain Extraction Post-Processing: Before vs After',
        fontsize=14, fontweight='bold', y=0.98,
    )

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info(f"  ✓ Saved: {output_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Print banner
    logger.info("=" * 80)
    logger.info("Brain Extraction Post-Processing for Synthetic Replicas")
    logger.info("=" * 80)

    # Validate replicas directory
    if not args.replicas_dir.exists():
        logger.error(f"Error: Replicas directory not found: {args.replicas_dir}")
        sys.exit(1)

    # Parse replica IDs
    try:
        replica_ids = [int(x.strip()) for x in args.replicas.split(',')]
    except ValueError:
        logger.error(f"Error: Invalid replica IDs: {args.replicas}")
        logger.error("       Expected comma-separated integers (e.g., '0,1,2,3')")
        sys.exit(1)

    # Determine output directory
    if args.overwrite:
        output_dir = args.replicas_dir
        logger.warning("\n⚠️  WARNING: --overwrite mode enabled!")
        logger.warning("    Original replica files will be REPLACED.")
        logger.warning("    Make sure you have a backup before proceeding.")
        response = input("\nContinue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Aborted.")
            sys.exit(0)
    else:
        output_dir = args.out_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set demo output path
    if args.demo:
        if args.demo_output is None:
            args.demo_output = output_dir / "postprocess_demo.png"

    # Collect algorithm parameters
    algorithm_params = {
        'gaussian_sigma_px': args.gaussian_sigma_px,
        'min_component_px': args.min_component_px,
        'n_components_to_keep': args.n_components_to_keep,
        'relaxed_threshold_factor': args.relaxed_threshold_factor,
        'n_first_bins': args.n_first_bins,
        'max_components_for_first_bins': args.max_components_for_first_bins,
        'background_value': args.background_value,
    }

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Replicas directory: {args.replicas_dir}")
    logger.info(f"  Replica IDs: {replica_ids}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("\nBrain Extraction Parameters:")
    for key, value in algorithm_params.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"\nProcessing {len(replica_ids)} replica(s)...\n")

    # Storage for demo visualization
    demo_original = None
    demo_processed = None

    # Process each replica
    total_samples = 0
    total_failures = 0

    for idx, replica_id in enumerate(replica_ids, 1):
        replica_filename = f"replica_{replica_id:03d}.npz"
        replica_path = args.replicas_dir / replica_filename

        logger.info(f"[{idx}/{len(replica_ids)}] Processing {replica_filename}...")

        try:
            # Load replica
            replica_data = load_replica(replica_path)
            n_samples = len(replica_data['images'])
            logger.info(f"  Samples: {n_samples}")

            # Process
            proc_images, proc_masks, stats = process_replica(
                images=replica_data['images'],
                masks=replica_data['masks'],
                zbins=replica_data['zbin'],
                **algorithm_params,
            )

            # Track statistics
            total_samples += stats['total_samples']
            total_failures += stats['failures']

            logger.info(f"  Brain extraction failures: {stats['failures']} ({stats['failure_rate']*100:.2f}%)")

            # Save processed replica
            output_path = output_dir / replica_filename
            save_replica(output_path, replica_data, proc_images, proc_masks)
            logger.info(f"  ✓ Saved: {output_path}")

            # Update metadata
            json_path = args.replicas_dir / f"replica_{replica_id:03d}_meta.json"
            output_json_path = output_dir / f"replica_{replica_id:03d}_meta.json"
            update_metadata(json_path, output_json_path, algorithm_params)
            if json_path.exists():
                logger.info(f"  ✓ Updated: {output_json_path}")

            # Store for demo (use first replica only)
            if args.demo and demo_original is None:
                demo_original = replica_data
                demo_processed = {
                    **replica_data,
                    'images': proc_images,
                    'masks': proc_masks,
                }

        except Exception as e:
            logger.error(f"  ✗ Failed to process {replica_filename}: {e}")
            continue

        logger.info("")  # Blank line between replicas

    # Summary
    logger.info("=" * 80)
    logger.info("All replicas processed!")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Brain extraction failures: {total_failures} ({total_failures/total_samples*100:.2f}%)")

    # Generate demo visualization
    if args.demo and demo_original is not None:
        create_demo_visualization(
            demo_original,
            demo_processed,
            args.demo_output,
            n_representative_zbins=5,
            n_images_per_condition=args.n_images_per_condition,
            seed=args.demo_seed,
        )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
