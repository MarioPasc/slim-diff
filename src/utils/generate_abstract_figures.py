#!/usr/bin/env python3
"""
generate_abstract_figures.py - Generate visualizations for ICIP 2026 abstract.

Creates publication-ready figures from the slice cache:
    1. Multi-bin axial slices with lesion overlay (for healthy and epilepsy patients)
    2. Single slice image and mask (separately)
    3. Diffusion noise simulation at t=100, t=300, t=1000
    4. Z-bin distribution bar plot (lesion vs non-lesion slices)

Usage:
    python -m src.utils.generate_abstract_figures \
        --cache-dir /media/mpascual/Sandisk2TB/research/jsddpm/data/epilepsy/slice_cache \
        --output-dir/media/mpascual/Sandisk2TB/research/jsddpm/results/epilepsy/icip2026/abstract \
        --epilepsy-patient MRIe_063 \
        --healthy-patient MRIe_072

References:
    - Paul Tol colorblind-safe palettes
    - IEEE ICIP 2026 publication guidelines
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Import plot settings
try:
    from src.diffusion.scripts.similarity_metrics.plotting.settings import (
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
    )
except ImportError:
    # Fallback if not in path
    PAUL_TOL_BRIGHT = {
        "blue": "#4477AA",
        "red": "#EE6677",
        "green": "#228833",
        "yellow": "#CCBB44",
        "cyan": "#66CCEE",
        "purple": "#AA3377",
        "grey": "#BBBBBB",
    }
    PLOT_SETTINGS = {
        "font_size": 10,
        "dpi_print": 300,
        "legend_fontsize": 9,
    }
    def apply_ieee_style():
        pass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FigureConfig:
    """Configuration for figure generation."""
    # Lesion overlay color (Paul Tol red)
    lesion_color: Tuple[float, float, float] = (0.93, 0.40, 0.47)  # #EE6677
    lesion_alpha: float = 0.5

    # Image display
    cmap: str = "gray"
    dpi: int = 300

    # Diffusion schedule (cosine)
    num_timesteps: int = 1000
    beta_start: float = 0.0015
    beta_end: float = 0.0195

    # Bar plot colors
    lesion_bar_color: str = "#EE6677"  # Paul Tol red
    nolesion_bar_color: str = "#4477AA"  # Paul Tol blue


# =============================================================================
# Diffusion Utilities
# =============================================================================

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    """Cosine beta schedule from Nichol & Dhariwal."""
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999)


def get_alpha_cumprod(timesteps: int) -> np.ndarray:
    """Get cumulative product of alphas."""
    betas = cosine_beta_schedule(timesteps)
    alphas = 1.0 - betas
    return np.cumprod(alphas)


def add_noise(x0: np.ndarray, t: int, num_timesteps: int = 1000) -> np.ndarray:
    """Add noise to image at timestep t (0 to num_timesteps-1)."""
    alphas_cumprod = get_alpha_cumprod(num_timesteps)
    # Clamp t to valid range (t=1000 means full noise, use t=999)
    t_idx = min(t, num_timesteps - 1)
    alpha_t = alphas_cumprod[t_idx]

    noise = np.random.randn(*x0.shape).astype(np.float32)
    noisy = np.sqrt(alpha_t) * x0 + np.sqrt(1 - alpha_t) * noise
    return noisy


# =============================================================================
# Data Loading
# =============================================================================

def load_slice(cache_dir: pathlib.Path, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load image and mask from npz file."""
    full_path = cache_dir / filepath
    data = np.load(full_path)
    image = data['image']
    mask = data['mask']
    return image, mask


def load_csv_data(cache_dir: pathlib.Path) -> pd.DataFrame:
    """Load and combine train/val CSV files."""
    train_csv = cache_dir / "train.csv"
    val_csv = cache_dir / "val.csv"

    dfs = []
    if train_csv.exists():
        dfs.append(pd.read_csv(train_csv))
    if val_csv.exists():
        dfs.append(pd.read_csv(val_csv))

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {cache_dir}")

    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Visualization Functions
# =============================================================================

def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    lesion_color: Tuple[float, float, float] = (0.93, 0.40, 0.47),
    alpha: float = 0.5
) -> np.ndarray:
    """Create RGB image with lesion overlay."""
    # Normalize image to [0, 1]
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Convert to RGB
    rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)

    # Create lesion mask (mask is in [-1, 1], lesion is +1)
    lesion_mask = mask > 0

    # Overlay lesion color
    for c in range(3):
        rgb[..., c] = np.where(
            lesion_mask,
            (1 - alpha) * rgb[..., c] + alpha * lesion_color[c],
            rgb[..., c]
        )

    return np.clip(rgb, 0, 1)


def save_image(
    image: np.ndarray,
    output_path: pathlib.Path,
    cmap: str = "gray",
    dpi: int = 300,
    title: Optional[str] = None
) -> None:
    """Save a single image without axes."""
    fig, ax = plt.subplots(figsize=(4, 4))

    if image.ndim == 2:
        ax.imshow(image, cmap=cmap, aspect='equal')
    else:
        ax.imshow(image, aspect='equal')

    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10)

    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_individual_slices(
    slices: List[Tuple[np.ndarray, np.ndarray, int]],  # (image, mask, z_bin)
    output_dir: pathlib.Path,
    prefix: str,
    cfg: FigureConfig,
    with_overlay: bool = True
) -> None:
    """Save each slice as a separate image file."""
    for image, mask, z_bin in slices:
        if with_overlay:
            output = create_overlay(image, mask, cfg.lesion_color, cfg.lesion_alpha)
            save_image(output, output_dir / f"{prefix}_zbin{z_bin:02d}.png", dpi=cfg.dpi)
        else:
            # Save image only (grayscale)
            save_image(image, output_dir / f"{prefix}_zbin{z_bin:02d}.png", cfg.cmap, cfg.dpi)


def save_diffusion_sequence(
    image: np.ndarray,
    mask: np.ndarray,
    timesteps: List[Tuple[int, str]],  # (t_index, label)
    output_dir: pathlib.Path,
    prefix: str,
    cfg: FigureConfig
) -> None:
    """Save diffusion noising sequence for image and mask."""
    np.random.seed(42)  # Reproducibility

    # Save original
    save_image(image, output_dir / f"{prefix}_image_t0.png", cfg.cmap, cfg.dpi)

    # Mask: convert from [-1, 1] to binary [0, 1] for visualization
    mask_binary = (mask > 0).astype(np.float32)
    save_image(mask_binary, output_dir / f"{prefix}_mask_t0.png", "gray", cfg.dpi)

    # Save noised versions
    for t_idx, t_label in timesteps:
        # Image
        noisy_image = add_noise(image, t_idx, cfg.num_timesteps)
        save_image(noisy_image, output_dir / f"{prefix}_image_t{t_label}.png", cfg.cmap, cfg.dpi)

        # Mask (noise in [-1, 1] space, then threshold for display)
        noisy_mask = add_noise(mask, t_idx, cfg.num_timesteps)
        # Show the actual noisy mask (continuous values)
        save_image(noisy_mask, output_dir / f"{prefix}_mask_t{t_label}.png", "gray", cfg.dpi)


def create_zbin_barplot(
    df: pd.DataFrame,
    output_path: pathlib.Path,
    cfg: FigureConfig
) -> None:
    """Create bar plot showing per-zbin distribution of lesion/non-lesion slices."""
    # Apply IEEE style
    try:
        apply_ieee_style()
    except Exception:
        pass

    # Aggregate by z_bin and has_lesion
    grouped = df.groupby(['z_bin', 'has_lesion']).size().unstack(fill_value=0)
    grouped.columns = ['No Lesion', 'Lesion']

    # Ensure all z_bins are present
    all_zbins = range(30)
    grouped = grouped.reindex(all_zbins, fill_value=0)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 3))

    x = np.arange(len(grouped))
    width = 0.4

    # Plot bars
    bars1 = ax.bar(
        x - width/2,
        grouped['No Lesion'],
        width,
        label='Control',
        color=cfg.nolesion_bar_color,
        edgecolor='black',
        linewidth=0.5
    )
    bars2 = ax.bar(
        x + width/2,
        grouped['Lesion'],
        width,
        label='Lesion',
        color=cfg.lesion_bar_color,
        edgecolor='black',
        linewidth=0.5
    )

    # Styling
    ax.set_xlabel('Z-bin', fontsize=11)
    ax.set_ylabel('Number of slices', fontsize=11)
    ax.set_xticks(x[::2])  # Show every other tick
    ax.set_xticklabels([str(i) for i in range(0, 30, 2)], fontsize=9)
    ax.tick_params(axis='y', labelsize=9)

    # Legend in upper right
    ax.legend(loc='upper right', fontsize=10, frameon=False)

    # Grid (y-axis only)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=cfg.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# Main Generation Functions
# =============================================================================

def select_target_zbins(n_bins: int, max_zbin: int = 29) -> List[int]:
    """Select evenly spaced z-bins across the full range."""
    indices = np.linspace(0, max_zbin, n_bins, dtype=int)
    return list(indices)


def generate_epilepsy_slices(
    df: pd.DataFrame,
    cache_dir: pathlib.Path,
    output_dir: pathlib.Path,
    target_zbins: List[int],
    cfg: FigureConfig
) -> None:
    """Generate epilepsy slices - one per target z-bin, searching across ALL patients."""
    lesion_df = df[df['has_lesion'] == True]

    slices = []
    for z_bin in target_zbins:
        bin_df = lesion_df[lesion_df['z_bin'] == z_bin]
        if len(bin_df) == 0:
            print(f"Warning: No lesion slice found for z-bin {z_bin}")
            continue
        # Pick the one with largest lesion area (best visibility)
        row = bin_df.sort_values('lesion_area_px', ascending=False).iloc[0]
        image, mask = load_slice(cache_dir, row['filepath'])
        slices.append((image, mask, z_bin))
        print(f"  z-bin {z_bin}: {row['subject_id']} (area={row['lesion_area_px']})")

    # Save individual slices
    save_individual_slices(slices, output_dir, "epilepsy", cfg, with_overlay=True)


def generate_healthy_slices(
    df: pd.DataFrame,
    cache_dir: pathlib.Path,
    output_dir: pathlib.Path,
    patient_id: str,
    target_zbins: List[int],
    cfg: FigureConfig
) -> None:
    """Generate healthy/control slices from a single patient for target z-bins."""
    patient_df = df[df['subject_id'] == patient_id]
    nolesion_df = patient_df[patient_df['has_lesion'] == False]

    slices = []
    for z_bin in target_zbins:
        bin_df = nolesion_df[nolesion_df['z_bin'] == z_bin]
        if len(bin_df) == 0:
            print(f"Warning: No control slice found for z-bin {z_bin} in {patient_id}")
            continue
        # Pick first available slice
        row = bin_df.iloc[0]
        image, mask = load_slice(cache_dir, row['filepath'])
        slices.append((image, mask, z_bin))
        print(f"  z-bin {z_bin}: {row['subject_id']}")

    # Save individual slices (no overlay since no lesion)
    save_individual_slices(slices, output_dir, "healthy", cfg, with_overlay=False)


def generate_single_slice_and_diffusion(
    df: pd.DataFrame,
    cache_dir: pathlib.Path,
    output_dir: pathlib.Path,
    patient_id: str,
    cfg: FigureConfig
) -> None:
    """Generate single slice (image, mask, overlay) and diffusion sequence."""
    patient_df = df[df['subject_id'] == patient_id]

    # Find middle slice with good lesion
    lesion_df = patient_df[patient_df['has_lesion'] == True]
    lesion_df = lesion_df.sort_values('lesion_area_px', ascending=False)

    # Pick slice with maximum lesion area
    best_slice = lesion_df.iloc[0]
    z_bin = best_slice['z_bin']

    print(f"Selected slice: {best_slice['filepath']} (z_bin={z_bin}, area={best_slice['lesion_area_px']})")

    # Load slice
    image, mask = load_slice(cache_dir, best_slice['filepath'])

    # (2) Save image and mask separately
    save_image(image, output_dir / "single_image.png", cfg.cmap, cfg.dpi)
    mask_binary = (mask > 0).astype(np.float32)
    save_image(mask_binary, output_dir / "single_mask.png", "gray", cfg.dpi)

    # Save overlay
    overlay = create_overlay(image, mask, cfg.lesion_color, cfg.lesion_alpha)
    save_image(overlay, output_dir / "single_overlay.png", dpi=cfg.dpi)

    # (3) Diffusion sequence
    # Format: (t_index, label) - use 999 internally for t=1000 since indices are 0-999
    timesteps = [(100, "100"), (300, "300"), (999, "1000")]
    save_diffusion_sequence(image, mask, timesteps, output_dir, "diffusion", cfg)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate abstract figures from slice cache."
    )
    parser.add_argument(
        "--cache-dir", type=pathlib.Path, required=True,
        help="Path to slice_cache directory"
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, required=True,
        help="Output directory for figures"
    )
    parser.add_argument(
        "--epilepsy-patient", type=str, default="MRIe_063",
        help="Subject ID for epilepsy patient (default: MRIe_063)"
    )
    parser.add_argument(
        "--healthy-patient", type=str, default=None,
        help="Subject ID for healthy/control slices (default: use non-lesion slices from any patient)"
    )
    parser.add_argument(
        "--n-bins", type=int, default=5,
        help="Number of z-bins to show per patient (default: 5)"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output DPI (default: 300)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup
    cfg = FigureConfig(dpi=args.dpi)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.cache_dir}")
    df = load_csv_data(args.cache_dir)
    print(f"Loaded {len(df)} slices from {df['subject_id'].nunique()} subjects")

    # Select target z-bins (same for both epilepsy and healthy)
    target_zbins = select_target_zbins(args.n_bins)
    print(f"Target z-bins: {target_zbins}")

    # (1) Individual slices for epilepsy (search across all patients for best lesion per z-bin)
    print(f"\n=== Generating epilepsy slices (best lesion per z-bin) ===")
    generate_epilepsy_slices(df, args.cache_dir, args.output_dir, target_zbins, cfg)

    # (1) Individual slices for healthy/control from a single patient
    if args.healthy_patient:
        healthy_id = args.healthy_patient
    else:
        # Find patient with most non-lesion slices covering target z-bins
        nolesion = df[df['has_lesion'] == False]
        healthy_id = nolesion.groupby('subject_id').size().idxmax()

    print(f"\n=== Generating control slices ({healthy_id}) ===")
    generate_healthy_slices(df, args.cache_dir, args.output_dir, healthy_id, target_zbins, cfg)

    # (2) & (3) Single slice and diffusion sequence
    print(f"\n=== Generating single slice and diffusion sequence ===")
    generate_single_slice_and_diffusion(
        df, args.cache_dir, args.output_dir,
        args.epilepsy_patient, cfg
    )

    # (4) Z-bin distribution bar plot
    print(f"\n=== Generating z-bin distribution bar plot ===")
    create_zbin_barplot(df, args.output_dir / "zbin_distribution.pdf", cfg)

    print(f"\n=== All figures saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()
