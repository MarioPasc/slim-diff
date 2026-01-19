"""
Visualization utility for z-bin priors overlay.

This module provides functions to visualize slice data with overlayed z-bin priors
to understand how anatomical conditioning affects the model.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


def visualize_patient_with_zbin_priors(
    volume: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    zbin_priors: np.ndarray | dict,
    z_range: tuple[int, int],
    z_bins: int = 30,
    alpha: float = 0.3,
    prior_color: tuple[int, int, int] = (0, 255, 255),  # Cyan
    lesion_color: tuple[int, int, int] = (255, 0, 0),  # Red
    figsize: tuple[int, int] = (20, 12),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualize a patient's slices with overlayed z-bin priors.

    Creates a grid visualization showing:
    - One row per z-bin in the range
    - For each row: original slice, mask, prior overlay, and combined view

    Args:
        volume: 3D volume (H, W, D) or (C, H, W, D)
        mask: 3D mask (H, W, D) or (C, H, W, D)
        zbin_priors: Either a dict with 'priors' key or 3D array (z_bins, H, W)
        z_range: Tuple of (min_z, max_z) for slice range
        z_bins: Number of z-bins (default: 30)
        alpha: Alpha value for overlays (default: 0.3)
        prior_color: RGB color for prior overlay (default: cyan)
        lesion_color: RGB color for lesion overlay (default: red)
        figsize: Figure size (default: (20, 12))
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Handle channel dimension
    if volume.ndim == 4:
        volume = volume[0]  # Take first channel
    if mask.ndim == 4:
        mask = mask[0]  # Take first channel

    # Extract priors array
    if isinstance(zbin_priors, dict):
        priors = zbin_priors["priors"]
    else:
        priors = zbin_priors

    # Compute z-bin assignments for the range
    min_z, max_z = z_range
    z_indices = np.arange(min_z, max_z + 1)
    bin_edges = np.linspace(0, len(z_indices), z_bins + 1)
    bin_assignments = np.digitize(np.arange(len(z_indices)), bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, z_bins - 1)

    # Select representative slices (one per bin)
    selected_slices = []
    selected_bins = []
    for bin_idx in range(z_bins):
        slice_indices_in_bin = np.where(bin_assignments == bin_idx)[0]
        if len(slice_indices_in_bin) > 0:
            # Take middle slice of bin
            middle_idx = slice_indices_in_bin[len(slice_indices_in_bin) // 2]
            global_z = min_z + middle_idx
            if global_z < volume.shape[2]:
                selected_slices.append(global_z)
                selected_bins.append(bin_idx)

    n_slices = len(selected_slices)
    if n_slices == 0:
        logger.warning("No valid slices found in z-range")
        return None

    # Create figure with 4 columns: [image, mask, prior, combined]
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_slices, 4, figure=fig, hspace=0.1, wspace=0.05)

    # Column titles
    titles = ["Image", "Mask", "Prior", "Combined"]
    for col_idx, title in enumerate(titles):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")

    # Plot each slice
    for row_idx, (z_idx, bin_idx) in enumerate(zip(selected_slices, selected_bins)):
        # Extract slices
        image_slice = volume[:, :, z_idx]
        mask_slice = mask[:, :, z_idx]
        prior_slice = priors[bin_idx]

        # Normalize image to [0, 1] for display
        image_norm = (image_slice - image_slice.min()) / (
            image_slice.max() - image_slice.min() + 1e-8
        )

        # Column 1: Image
        ax1 = fig.add_subplot(gs[row_idx, 0])
        ax1.imshow(image_norm, cmap="gray")
        ax1.axis("off")
        ax1.text(
            -10,
            image_norm.shape[0] // 2,
            f"z={z_idx}\nbin={bin_idx}",
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

        # Column 2: Mask (with lesion overlay)
        ax2 = fig.add_subplot(gs[row_idx, 1])
        rgb_mask = _create_overlay(
            image_norm,
            mask_slice > 0,
            alpha=0.5,
            color=lesion_color,
        )
        ax2.imshow(rgb_mask)
        ax2.axis("off")

        # Column 3: Prior (with prior overlay)
        ax3 = fig.add_subplot(gs[row_idx, 2])
        rgb_prior = _create_overlay(
            image_norm,
            prior_slice > 0,
            alpha=alpha,
            color=prior_color,
        )
        ax3.imshow(rgb_prior)
        ax3.axis("off")

        # Column 4: Combined (both overlays)
        ax4 = fig.add_subplot(gs[row_idx, 3])
        # Start with prior overlay
        rgb_combined = _create_overlay(
            image_norm,
            prior_slice > 0,
            alpha=alpha,
            color=prior_color,
        )
        # Add lesion overlay on top
        lesion_mask = mask_slice > 0
        if lesion_mask.any():
            rgb_combined = _blend_overlay(
                rgb_combined,
                lesion_mask,
                alpha=0.4,
                color=lesion_color,
            )
        ax4.imshow(rgb_combined)
        ax4.axis("off")

    # Add legend
    fig.text(
        0.5,
        0.02,
        f"Z-range: [{min_z}, {max_z}] | Z-bins: {z_bins} | "
        f"Prior alpha: {alpha} | Colors: Prior=Cyan, Lesion=Red",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved z-bin prior visualization to {save_path}")

    return fig


def visualize_single_slice_with_prior(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    prior_slice: np.ndarray,
    z_idx: int,
    bin_idx: int,
    alpha: float = 0.3,
    prior_color: tuple[int, int, int] = (0, 255, 255),
    lesion_color: tuple[int, int, int] = (255, 0, 0),
    figsize: tuple[int, int] = (15, 4),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualize a single slice with overlayed z-bin prior.

    Creates a 4-panel visualization: [image, mask, prior, combined]

    Args:
        image_slice: 2D image slice (H, W)
        mask_slice: 2D mask slice (H, W)
        prior_slice: 2D prior slice (H, W)
        z_idx: Z-index of the slice
        bin_idx: Z-bin index
        alpha: Alpha value for overlays (default: 0.3)
        prior_color: RGB color for prior overlay (default: cyan)
        lesion_color: RGB color for lesion overlay (default: red)
        figsize: Figure size (default: (15, 4))
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    # Normalize image to [0, 1]
    image_norm = (image_slice - image_slice.min()) / (
        image_slice.max() - image_slice.min() + 1e-8
    )

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Column 1: Image
    axes[0].imshow(image_norm, cmap="gray")
    axes[0].set_title(f"Image (z={z_idx}, bin={bin_idx})")
    axes[0].axis("off")

    # Column 2: Mask overlay
    rgb_mask = _create_overlay(
        image_norm,
        mask_slice > 0,
        alpha=0.5,
        color=lesion_color,
    )
    axes[1].imshow(rgb_mask)
    axes[1].set_title("Lesion Mask")
    axes[1].axis("off")

    # Column 3: Prior overlay
    rgb_prior = _create_overlay(
        image_norm,
        prior_slice > 0,
        alpha=alpha,
        color=prior_color,
    )
    axes[2].imshow(rgb_prior)
    axes[2].set_title("Z-Bin Prior")
    axes[2].axis("off")

    # Column 4: Combined
    rgb_combined = _create_overlay(
        image_norm,
        prior_slice > 0,
        alpha=alpha,
        color=prior_color,
    )
    lesion_mask = mask_slice > 0
    if lesion_mask.any():
        rgb_combined = _blend_overlay(
            rgb_combined,
            lesion_mask,
            alpha=0.4,
            color=lesion_color,
        )
    axes[3].imshow(rgb_combined)
    axes[3].set_title("Combined")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved single-slice visualization to {save_path}")

    return fig


def _create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Create RGB overlay of mask on grayscale image.

    Args:
        image: Grayscale image in [0, 1]
        mask: Binary mask
        alpha: Overlay transparency
        color: RGB color tuple (0-255)

    Returns:
        RGB image with overlay (H, W, 3) in [0, 1]
    """
    # Convert grayscale to RGB
    rgb = np.stack([image, image, image], axis=-1)

    # Create colored overlay
    color_norm = np.array(color) / 255.0
    overlay = np.zeros_like(rgb)
    overlay[mask] = color_norm

    # Blend
    result = rgb * (1 - alpha * mask[..., None]) + overlay * alpha

    return np.clip(result, 0, 1)


def _blend_overlay(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Blend an additional overlay onto an existing RGB image.

    Args:
        base_rgb: Existing RGB image in [0, 1]
        mask: Binary mask for new overlay
        alpha: Overlay transparency
        color: RGB color tuple (0-255)

    Returns:
        RGB image with blended overlay (H, W, 3) in [0, 1]
    """
    # Create colored overlay
    color_norm = np.array(color) / 255.0
    overlay = np.zeros_like(base_rgb)
    overlay[mask] = color_norm

    # Blend
    result = base_rgb * (1 - alpha * mask[..., None]) + overlay * alpha

    return np.clip(result, 0, 1)


# Command-line interface for standalone usage
def main():
    """CLI for z-bin prior visualization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize patient slices with z-bin priors overlay"
    )
    parser.add_argument(
        "--volume",
        type=str,
        required=True,
        help="Path to patient volume (.nii.gz or .npz)",
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to patient mask (.nii.gz or .npz)",
    )
    parser.add_argument(
        "--priors",
        type=str,
        required=True,
        help="Path to z-bin priors (.npz file)",
    )
    parser.add_argument(
        "--z-range",
        type=int,
        nargs=2,
        default=[30, 90],
        help="Z-range [min max] (default: 30 90)",
    )
    parser.add_argument(
        "--z-bins",
        type=int,
        default=30,
        help="Number of z-bins (default: 30)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Alpha value for prior overlay (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for visualization",
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading volume from {args.volume}")
    if args.volume.endswith(".npz"):
        volume = np.load(args.volume)["data"]
    else:
        import nibabel as nib

        volume = nib.load(args.volume).get_fdata()

    logger.info(f"Loading mask from {args.mask}")
    if args.mask.endswith(".npz"):
        mask = np.load(args.mask)["data"]
    else:
        import nibabel as nib

        mask = nib.load(args.mask).get_fdata()

    logger.info(f"Loading priors from {args.priors}")
    priors = np.load(args.priors)

    # Create visualization
    fig = visualize_patient_with_zbin_priors(
        volume=volume,
        mask=mask,
        zbin_priors=priors,
        z_range=tuple(args.z_range),
        z_bins=args.z_bins,
        alpha=args.alpha,
        save_path=args.output,
    )

    if fig:
        plt.show()


if __name__ == "__main__":
    main()
