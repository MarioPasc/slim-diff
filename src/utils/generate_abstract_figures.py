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
        PREDICTION_TYPE_COLORS,
        PREDICTION_TYPE_LABELS,
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
    PREDICTION_TYPE_COLORS = {
        "epsilon": "#EE7733",  # orange
        "velocity": "#009988",  # green
        "x0": "#882255",  # wine
    }
    PREDICTION_TYPE_LABELS = {
        "epsilon": r"$\epsilon$-prediction",
        "velocity": r"$\mathbf{v}$-prediction",
        "x0": r"$\mathbf{x}_0$-prediction",
    }
    PLOT_SETTINGS = {
        "font_size": 10,
        "dpi_print": 300,
        "legend_fontsize": 9,
    }
    def apply_ieee_style():
        pass

import matplotlib.colors as mcolors


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
# Assessment Visualization Functions
# =============================================================================

def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color to RGB tuple (0-1 range)."""
    return mcolors.to_rgb(hex_color)


def add_frame_to_image(
    image: np.ndarray,
    frame_color: Tuple[float, float, float],
    frame_width: int = 8
) -> np.ndarray:
    """Add a colored frame around an image."""
    h, w = image.shape[:2]

    # Ensure image is RGB
    if image.ndim == 2:
        img_rgb = np.stack([image, image, image], axis=-1)
    else:
        img_rgb = image.copy()

    # Normalize to [0, 1] if needed
    if img_rgb.max() > 1.0:
        img_rgb = img_rgb / 255.0

    # Create framed image (larger canvas)
    framed_h = h + 2 * frame_width
    framed_w = w + 2 * frame_width
    framed = np.zeros((framed_h, framed_w, 3), dtype=np.float32)

    # Fill with frame color
    framed[:, :] = frame_color

    # Place image in center
    framed[frame_width:frame_width+h, frame_width:frame_width+w] = img_rgb

    return framed


def save_framed_image(
    image: np.ndarray,
    output_path: pathlib.Path,
    frame_color: Tuple[float, float, float],
    frame_width: int = 8,
    cmap: str = "gray",
    dpi: int = 300
) -> None:
    """Save image with colored frame."""
    # Normalize grayscale to [0, 1]
    if image.ndim == 2:
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    else:
        img_norm = image

    framed = add_frame_to_image(img_norm, frame_color, frame_width)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(framed, aspect='equal')
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def load_synthetic_samples(
    results_dir: pathlib.Path,
    prediction_type: str,
    p_cond: float,
    lp_norm: float,
    n_samples: int = 5,
    lesion_only: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load synthetic samples from experiment replicas."""
    # Build experiment path
    p_cond_str = f"self_cond_p_{p_cond}"
    lp_str = f"lp_{lp_norm}"

    # Handle different naming for prediction types
    pred_name = prediction_type if prediction_type != "x0" else "x0"
    exp_name = f"{pred_name}_{lp_str}"

    exp_path = results_dir / p_cond_str / exp_name / "replicas"

    if not exp_path.exists():
        print(f"Warning: Experiment path not found: {exp_path}")
        return []

    # Find replica files
    replica_files = sorted(exp_path.glob("replica_*.npz"))
    replica_files = [f for f in replica_files if "meta" not in f.name]

    if not replica_files:
        print(f"Warning: No replica files found in {exp_path}")
        return []

    # Load samples from first replica
    data = np.load(replica_files[0])
    images = data['images']
    masks = data['masks']
    lesion_present = data['lesion_present']

    # Filter for lesion samples if requested
    if lesion_only:
        lesion_idx = np.where(lesion_present == 1)[0]
        if len(lesion_idx) == 0:
            print(f"Warning: No lesion samples found for {prediction_type}")
            return []
        # Select random samples
        np.random.seed(42)
        selected_idx = np.random.choice(lesion_idx, min(n_samples, len(lesion_idx)), replace=False)
    else:
        np.random.seed(42)
        selected_idx = np.random.choice(len(images), min(n_samples, len(images)), replace=False)

    samples = []
    for idx in selected_idx:
        img = images[idx].astype(np.float32)
        mask = masks[idx].astype(np.float32)
        samples.append((img, mask))

    return samples


def load_real_samples(
    cache_dir: pathlib.Path,
    n_samples: int = 5,
    lesion_only: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load real samples from slice cache."""
    df = load_csv_data(cache_dir)

    if lesion_only:
        df = df[df['has_lesion'] == True]

    # Select samples with good lesion area
    df = df.sort_values('lesion_area_px', ascending=False)

    samples = []
    for _, row in df.head(n_samples).iterrows():
        image, mask = load_slice(cache_dir, row['filepath'])
        samples.append((image, mask))

    return samples


def generate_assessment_samples(
    results_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    output_dir: pathlib.Path,
    p_cond: float,
    lp_norm: float,
    n_samples: int,
    cfg: FigureConfig
) -> None:
    """Generate framed sample images for each prediction type and real data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_types = ["epsilon", "velocity", "x0"]

    # Generate synthetic samples for each prediction type
    for pred_type in prediction_types:
        print(f"\n  Loading {pred_type} samples...")
        samples = load_synthetic_samples(
            results_dir, pred_type, p_cond, lp_norm, n_samples, lesion_only=True
        )

        if not samples:
            continue

        frame_color = hex_to_rgb(PREDICTION_TYPE_COLORS[pred_type])

        for i, (image, mask) in enumerate(samples):
            # Save with overlay and frame
            overlay = create_overlay(image, mask, cfg.lesion_color, cfg.lesion_alpha)
            save_framed_image(
                overlay,
                output_dir / f"{pred_type}_sample{i:02d}_overlay.png",
                frame_color, frame_width=8, dpi=cfg.dpi
            )

            # Save image only with frame
            img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
            save_framed_image(
                img_norm,
                output_dir / f"{pred_type}_sample{i:02d}_image.png",
                frame_color, frame_width=8, dpi=cfg.dpi
            )

            # Save mask only with frame (binary)
            mask_binary = (mask > 0).astype(np.float32)
            save_framed_image(
                mask_binary,
                output_dir / f"{pred_type}_sample{i:02d}_mask.png",
                frame_color, frame_width=8, dpi=cfg.dpi
            )

        print(f"    Saved {len(samples)} samples for {pred_type}")

    # Generate real samples (no frame or neutral frame)
    print(f"\n  Loading real samples...")
    real_samples = load_real_samples(cache_dir, n_samples, lesion_only=True)

    real_frame_color = (0.3, 0.3, 0.3)  # Dark gray for real

    for i, (image, mask) in enumerate(real_samples):
        # Save with overlay
        overlay = create_overlay(image, mask, cfg.lesion_color, cfg.lesion_alpha)
        save_framed_image(
            overlay,
            output_dir / f"real_sample{i:02d}_overlay.png",
            real_frame_color, frame_width=8, dpi=cfg.dpi
        )

        # Save image only
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        save_framed_image(
            img_norm,
            output_dir / f"real_sample{i:02d}_image.png",
            real_frame_color, frame_width=8, dpi=cfg.dpi
        )

        # Save mask only
        mask_binary = (mask > 0).astype(np.float32)
        save_framed_image(
            mask_binary,
            output_dir / f"real_sample{i:02d}_mask.png",
            real_frame_color, frame_width=8, dpi=cfg.dpi
        )

    print(f"    Saved {len(real_samples)} real samples")


def compute_simple_features(images: np.ndarray) -> np.ndarray:
    """Compute simple feature vectors for embedding visualization.

    Uses a combination of statistics to create discriminative features:
    - Mean intensity in different regions
    - Variance/texture measures
    - Edge density
    """
    features = []
    for img in images:
        img = img.astype(np.float32)

        # Global stats
        mean_val = np.mean(img)
        std_val = np.std(img)

        # Quadrant means
        h, w = img.shape
        q1 = np.mean(img[:h//2, :w//2])
        q2 = np.mean(img[:h//2, w//2:])
        q3 = np.mean(img[h//2:, :w//2])
        q4 = np.mean(img[h//2:, w//2:])

        # Edge features (gradient magnitude)
        gy, gx = np.gradient(img)
        grad_mag = np.sqrt(gx**2 + gy**2)
        edge_mean = np.mean(grad_mag)
        edge_std = np.std(grad_mag)

        # High-frequency content (difference from smoothed)
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(img, sigma=3)
        hf_content = np.mean(np.abs(img - smoothed))

        # Histogram features
        hist, _ = np.histogram(img.flatten(), bins=16, range=(-1, 1))
        hist = hist / hist.sum()

        feat = [mean_val, std_val, q1, q2, q3, q4, edge_mean, edge_std, hf_content]
        feat.extend(hist.tolist())
        features.append(feat)

    return np.array(features)


def create_embedding_plot(
    results_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    output_dir: pathlib.Path,
    p_cond: float,
    lp_norm: float,
    n_samples: int,
    cfg: FigureConfig
) -> None:
    """Create 2D embedding plots comparing real vs synthetic distributions."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_types = ["x0", "velocity", "epsilon"]  # Order: closest to farthest

    # Load real samples
    print("  Loading real samples for embedding...")
    real_samples = load_real_samples(cache_dir, n_samples=100, lesion_only=True)
    real_images = np.array([s[0] for s in real_samples])
    real_features = compute_simple_features(real_images)

    # Load synthetic samples for each prediction type
    all_synthetic = {}
    for pred_type in prediction_types:
        print(f"  Loading {pred_type} samples for embedding...")
        samples = load_synthetic_samples(
            results_dir, pred_type, p_cond, lp_norm, n_samples=100, lesion_only=True
        )
        if samples:
            images = np.array([s[0] for s in samples])
            features = compute_simple_features(images)
            all_synthetic[pred_type] = features

    if not all_synthetic:
        print("  Warning: No synthetic samples found for embedding plot")
        return

    # Combine all features for dimensionality reduction
    all_features = [real_features]
    labels = ['Real'] * len(real_features)

    for pred_type in prediction_types:
        if pred_type in all_synthetic:
            all_features.append(all_synthetic[pred_type])
            labels.extend([pred_type] * len(all_synthetic[pred_type]))

    combined = np.vstack(all_features)

    # Apply PCA first, then t-SNE for better visualization
    print("  Computing embeddings (PCA + t-SNE)...")
    pca = PCA(n_components=min(20, combined.shape[1]))
    pca_features = pca.fit_transform(combined)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings = tsne.fit_transform(pca_features)

    # Create the plot
    try:
        apply_ieee_style()
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot each group
    labels = np.array(labels)

    # Real data
    real_mask = labels == 'Real'
    ax.scatter(
        embeddings[real_mask, 0], embeddings[real_mask, 1],
        c='#333333', marker='o', s=30, alpha=0.6, label='Real', zorder=5
    )

    # Synthetic data by prediction type (in order: x0, velocity, epsilon)
    for pred_type in prediction_types:
        if pred_type in all_synthetic:
            mask = labels == pred_type
            ax.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                c=PREDICTION_TYPE_COLORS[pred_type],
                marker='s', s=25, alpha=0.5,
                label=PREDICTION_TYPE_LABELS[pred_type],
                zorder=4
            )

    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, frameon=False)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', labelsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "embedding_comparison.pdf", dpi=cfg.dpi, bbox_inches='tight')
    fig.savefig(output_dir / "embedding_comparison.png", dpi=cfg.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'embedding_comparison.pdf'}")


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

    # Assessment arguments
    parser.add_argument(
        "--assessment", action="store_true",
        help="Generate assessment figures (framed samples + embedding plots)"
    )
    parser.add_argument(
        "--results-dir", type=pathlib.Path, default=None,
        help="Path to results directory (for assessment mode)"
    )
    parser.add_argument(
        "--p-cond", type=float, default=0.5,
        help="Self-conditioning probability for assessment (default: 0.5)"
    )
    parser.add_argument(
        "--lp-norm", type=float, default=2.0,
        help="Lp norm value for assessment (default: 2.0)"
    )
    parser.add_argument(
        "--n-assessment-samples", type=int, default=5,
        help="Number of samples per prediction type for assessment (default: 5)"
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

    # (5) Assessment figures (if requested)
    if args.assessment:
        if args.results_dir is None:
            print("\nWarning: --results-dir required for assessment mode")
        else:
            assessment_dir = args.output_dir / "Assessment"
            print(f"\n=== Generating assessment figures ===")
            print(f"  Results dir: {args.results_dir}")
            print(f"  p_cond: {args.p_cond}, lp_norm: {args.lp_norm}")

            # (5.1) Framed samples per prediction type
            print(f"\n--- Generating framed samples ---")
            generate_assessment_samples(
                args.results_dir, args.cache_dir, assessment_dir,
                args.p_cond, args.lp_norm, args.n_assessment_samples, cfg
            )

            # (5.2) Embedding visualization
            print(f"\n--- Generating embedding plots ---")
            create_embedding_plot(
                args.results_dir, args.cache_dir, assessment_dir,
                args.p_cond, args.lp_norm, args.n_assessment_samples, cfg
            )

            print(f"\n=== Assessment figures saved to {assessment_dir} ===")

    print(f"\n=== All figures saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()
