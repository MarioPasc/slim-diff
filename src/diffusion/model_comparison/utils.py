"""Shared utilities for model comparison.

This module provides plotting settings, color palettes, and helper functions
used across all visualization modules.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Plot Settings (consistent with plot_kid_results.py)
# =============================================================================

PLOT_SETTINGS = {
    "font_family": "serif",
    "font_serif": ["Times New Roman", "DejaVu Serif"],
    "font_size": 9,
    "axes_labelsize": 8,
    "axes_titlesize": 9,
    "axes_spine_width": 0.8,
    "axes_spine_color": "0.2",
    "tick_labelsize": 12,
    "tick_major_width": 0.6,
    "tick_minor_width": 0.4,
    "tick_direction": "in",
    "tick_length_major": 3.5,
    "tick_length_minor": 2.0,
    "legend_fontsize": 10,
    "legend_framealpha": 0.9,
    "legend_frameon": False,
    "legend_edgecolor": "0.8",
    "grid_linestyle": ":",
    "grid_alpha": 0.7,
    "grid_linewidth": 0.6,
    "line_width": 2.0,
    "axis_labelsize": 14,
    "xtick_fontsize": 12,
    "ytick_fontsize": 12,
    "xlabel_fontsize": 14,
    "ylabel_fontsize": 14,
    "title_fontsize": 16,
}

# Model color palette for consistent coloring across plots
MODEL_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-green
    "#17becf",  # Cyan
]

# Line styles for distinguishing models when colors are similar
MODEL_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


def apply_plot_settings() -> None:
    """Apply global matplotlib settings for consistent styling."""
    plt.rcParams.update(
        {
            "font.family": PLOT_SETTINGS["font_family"],
            "font.serif": PLOT_SETTINGS["font_serif"],
            "font.size": PLOT_SETTINGS["font_size"],
            "axes.labelsize": PLOT_SETTINGS["axes_labelsize"],
            "axes.titlesize": PLOT_SETTINGS["axes_titlesize"],
            "xtick.labelsize": PLOT_SETTINGS["tick_labelsize"],
            "ytick.labelsize": PLOT_SETTINGS["tick_labelsize"],
            "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
            "xtick.minor.width": PLOT_SETTINGS["tick_minor_width"],
            "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
            "ytick.minor.width": PLOT_SETTINGS["tick_minor_width"],
            "xtick.direction": PLOT_SETTINGS["tick_direction"],
            "ytick.direction": PLOT_SETTINGS["tick_direction"],
            "legend.fontsize": PLOT_SETTINGS["legend_fontsize"],
            "legend.framealpha": PLOT_SETTINGS["legend_framealpha"],
            "legend.frameon": PLOT_SETTINGS["legend_frameon"],
            "legend.edgecolor": PLOT_SETTINGS["legend_edgecolor"],
            "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
            "grid.alpha": PLOT_SETTINGS["grid_alpha"],
            "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
            "axes.grid": True,
        }
    )


def get_model_color(model_idx: int) -> str:
    """Get color for a model by index.

    Args:
        model_idx: Model index (0-based).

    Returns:
        Hex color string.
    """
    return MODEL_COLORS[model_idx % len(MODEL_COLORS)]


def get_model_linestyle(model_idx: int) -> str:
    """Get line style for a model by index.

    Args:
        model_idx: Model index (0-based).

    Returns:
        Matplotlib line style string or tuple.
    """
    return MODEL_LINESTYLES[model_idx % len(MODEL_LINESTYLES)]


# =============================================================================
# Display Helpers
# =============================================================================


def to_display_range(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert from [-1, 1] to [0, 1] for display.

    Args:
        x: Array with values in [-1, 1].

    Returns:
        Array clipped to [0, 1].
    """
    return np.clip((x + 1) / 2, 0, 1)


def create_overlay(
    image: NDArray[np.floating],
    mask: NDArray[np.floating],
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
    threshold: float = 0.0,
) -> NDArray[np.floating]:
    """Create image with mask overlay.

    Args:
        image: Grayscale image in [0, 1], shape (H, W).
        mask: Mask in [-1, 1], shape (H, W).
        alpha: Overlay transparency (0=transparent, 1=opaque).
        color: RGB color for overlay (0-255).
        threshold: Threshold for binarizing mask.

    Returns:
        RGB image with overlay, shape (H, W, 3), values in [0, 1].
    """
    # Convert grayscale to RGB
    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1)
    else:
        rgb = image.copy()

    # Binarize mask
    binary_mask = mask > threshold

    # Normalize color to [0, 1]
    color_norm = np.array(color) / 255.0

    # Apply overlay where mask is positive
    for c in range(3):
        rgb[:, :, c] = np.where(
            binary_mask, (1 - alpha) * rgb[:, :, c] + alpha * color_norm[c], rgb[:, :, c]
        )

    return np.clip(rgb, 0, 1)


def create_boundary_overlay(
    image: NDArray[np.floating],
    mask: NDArray[np.floating],
    color: tuple[int, int, int] = (0, 255, 0),
    threshold: float = 0.0,
    linewidth: int = 2,
) -> NDArray[np.floating]:
    """Create image with mask boundary overlay.

    Args:
        image: Grayscale image in [0, 1], shape (H, W).
        mask: Mask in [-1, 1], shape (H, W).
        color: RGB color for boundary (0-255).
        threshold: Threshold for binarizing mask.
        linewidth: Width of boundary line in pixels.

    Returns:
        RGB image with boundary overlay, shape (H, W, 3), values in [0, 1].
    """
    from scipy import ndimage

    # Convert grayscale to RGB
    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1)
    else:
        rgb = image.copy()

    # Binarize mask
    binary_mask = mask > threshold

    # Find boundary using erosion
    eroded = ndimage.binary_erosion(binary_mask, iterations=linewidth)
    boundary = binary_mask & ~eroded

    # Normalize color to [0, 1]
    color_norm = np.array(color) / 255.0

    # Apply boundary color
    for c in range(3):
        rgb[:, :, c] = np.where(boundary, color_norm[c], rgb[:, :, c])

    return np.clip(rgb, 0, 1)


# =============================================================================
# File/Path Helpers
# =============================================================================


def find_latest_checkpoint(model_dir: Path) -> Path:
    """Find the latest checkpoint in a model directory.

    Args:
        model_dir: Path to model results directory.

    Returns:
        Path to latest checkpoint.

    Raises:
        FileNotFoundError: If no checkpoints found.
    """
    checkpoint_dir = model_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found at {checkpoint_dir}")

    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")

    # Sort by modification time (latest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return checkpoints[0]


def find_best_checkpoint(model_dir: Path, metric: str = "val_loss") -> Path:
    """Find the best checkpoint based on metric in filename.

    Args:
        model_dir: Path to model results directory.
        metric: Metric name to search for (e.g., 'val_loss').

    Returns:
        Path to best checkpoint.

    Raises:
        FileNotFoundError: If no checkpoints found.
    """
    checkpoint_dir = model_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found at {checkpoint_dir}")

    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")

    # Try to extract metric value from filename
    # Checkpoint filenames look like: jsddpm-epoch=0280-val_loss=0.0000.ckpt
    best_ckpt = None
    best_value = float("inf")

    # Match metric=value pattern, being careful not to capture trailing period before .ckpt
    pattern = re.compile(rf"{metric}=([0-9]+(?:\.[0-9]+)?)")
    for ckpt in checkpoints:
        match = pattern.search(ckpt.stem)  # Use stem to exclude .ckpt extension
        if match:
            value_str = match.group(1)
            try:
                value = float(value_str)
                if value < best_value:
                    best_value = value
                    best_ckpt = ckpt
            except ValueError:
                continue

    if best_ckpt is None:
        # Fallback to latest
        return find_latest_checkpoint(model_dir)

    return best_ckpt


def extract_timestep_from_column(col_name: str) -> int | None:
    """Extract timestep number from column name.

    Handles patterns like 'diagnostics/pred_mse_image_t0500'.

    Args:
        col_name: Column name with timestep suffix.

    Returns:
        Timestep integer or None if not found.
    """
    match = re.search(r"_t(\d+)$", col_name)
    if match:
        return int(match.group(1))
    return None


def extract_channel_from_column(col_name: str) -> str | None:
    """Extract channel name from column name.

    Handles patterns like 'diagnostics/pred_mse_image_t0500'.

    Args:
        col_name: Column name with channel identifier.

    Returns:
        Channel name ('image' or 'mask') or None if not found.
    """
    if "_image_" in col_name:
        return "image"
    elif "_mask_" in col_name:
        return "mask"
    return None


# =============================================================================
# Data Processing Helpers
# =============================================================================


def smooth_series(
    values: NDArray[np.floating], window: int = 5
) -> NDArray[np.floating]:
    """Apply rolling average smoothing to a series.

    Args:
        values: Input array.
        window: Rolling window size.

    Returns:
        Smoothed array (same length, NaN at edges).
    """
    if window <= 1:
        return values

    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")

    # Handle edges by using smaller windows
    for i in range(window // 2):
        if i < len(values):
            smoothed[i] = np.mean(values[: i + window // 2 + 1])
        if len(values) - 1 - i >= 0:
            smoothed[len(values) - 1 - i] = np.mean(
                values[len(values) - 1 - i - window // 2 :]
            )

    return smoothed


def compute_ranking(values: dict[str, float], lower_is_better: bool = True) -> dict[str, int]:
    """Compute ranking of models based on metric values.

    Args:
        values: Dictionary mapping model_name -> metric_value.
        lower_is_better: If True, lower values rank higher.

    Returns:
        Dictionary mapping model_name -> rank (1 = best).
    """
    sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=not lower_is_better)
    return {name: rank + 1 for rank, (name, _) in enumerate(sorted_items)}
