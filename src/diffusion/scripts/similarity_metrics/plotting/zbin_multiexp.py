"""Per-zbin multi-experiment plotting for similarity metrics.

Creates plots comparing metrics across z-bins for multiple experiments,
with visual encoding for prediction type and Lp norm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plot settings for publication quality
PLOT_SETTINGS = {
    "font_family": "serif",
    "font_serif": ["Times New Roman", "DejaVu Serif"],
    "font_size": 10,
    "axes_labelsize": 12,
    "tick_labelsize": 10,
    "legend_fontsize": 9,
    "title_fontsize": 14,
    "line_width": 1.5,
    "marker_size": 4,
    "grid_alpha": 0.3,
    "grid_linestyle": ":",
}

# Color scheme for prediction types
PREDICTION_TYPE_COLORS = {
    "epsilon": "#E74C3C",  # Red
    "velocity": "#27AE60",  # Green
    "x0": "#3498DB",       # Blue
}

# Line styles for Lp norms
LP_NORM_STYLES = {
    1.5: "-",    # Solid
    2.0: "--",   # Dashed
    2.5: ":",    # Dotted
}

# Alpha values for Lp norms (higher alpha = more prominent)
LP_NORM_ALPHAS = {
    1.5: 1.0,
    2.0: 0.75,
    2.5: 0.5,
}


def apply_plot_settings():
    """Apply global matplotlib settings."""
    plt.rcParams.update({
        "font.family": PLOT_SETTINGS["font_family"],
        "font.serif": PLOT_SETTINGS["font_serif"],
        "font.size": PLOT_SETTINGS["font_size"],
        "axes.labelsize": PLOT_SETTINGS["axes_labelsize"],
        "xtick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "ytick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "legend.fontsize": PLOT_SETTINGS["legend_fontsize"],
        "axes.grid": True,
        "grid.alpha": PLOT_SETTINGS["grid_alpha"],
        "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
    })


def plot_zbin_multiexperiment(
    df_zbin: pd.DataFrame,
    metric_col: str,
    output_dir: Path,
    metric_name: str | None = None,
    baseline_real: float | None = None,
    baseline_std: float | None = None,
    figsize: tuple[float, float] = (14, 8),
    formats: list[str] = ["png", "pdf"],
    show_error_bands: bool = True,
    show_legend: bool = True,
    title: str | None = None,
) -> None:
    """Create per-zbin metric plot for all experiments.

    Visual encoding:
    - Color by prediction type: epsilon=red, velocity=green, x0=blue
    - Line style by Lp: 1.5=solid, 2.0=dashed, 2.5=dotted
    - Alpha gradient by Lp: 1.5=1.0, 2.0=0.75, 2.5=0.5
    - Error bands: ±1 std (shaded area)
    - Baseline: horizontal dashed gray line

    Args:
        df_zbin: DataFrame with columns: experiment, prediction_type, lp_norm,
                 zbin, {metric_col}, {metric_col}_std.
        metric_col: Column name for metric values (e.g., "kid_zbin").
        output_dir: Output directory for plots.
        metric_name: Display name for metric (default: derived from col name).
        baseline_real: Optional baseline value (real vs real).
        baseline_std: Optional baseline std.
        figsize: Figure size.
        formats: Output formats.
        show_error_bands: Whether to show error bands.
        show_legend: Whether to show legend.
        title: Optional title (default: auto-generated).
    """
    apply_plot_settings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive metric name from column if not provided
    if metric_name is None:
        metric_name = metric_col.replace("_zbin", "").upper()

    # Check required columns
    std_col = f"{metric_col}_std"
    required_cols = ["prediction_type", "lp_norm", "zbin", metric_col]
    for col in required_cols:
        if col not in df_zbin.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    has_std = std_col in df_zbin.columns

    # Get unique values
    prediction_types = sorted(df_zbin["prediction_type"].unique())
    lp_norms = sorted(df_zbin["lp_norm"].unique())
    zbins = sorted(df_zbin["zbin"].unique())

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot baseline if provided
    if baseline_real is not None:
        ax.axhline(
            baseline_real,
            color="gray",
            linestyle="--",
            linewidth=PLOT_SETTINGS["line_width"],
            label=f"Real baseline: {baseline_real:.4f}",
            zorder=1,
        )
        if baseline_std is not None:
            ax.fill_between(
                [zbins[0] - 0.5, zbins[-1] + 0.5],
                baseline_real - baseline_std,
                baseline_real + baseline_std,
                color="gray",
                alpha=0.15,
                zorder=1,
            )

    # Plot each experiment
    for pred_type in prediction_types:
        color = PREDICTION_TYPE_COLORS.get(pred_type, "black")

        for lp_norm in lp_norms:
            # Filter data for this experiment
            mask = (df_zbin["prediction_type"] == pred_type) & (df_zbin["lp_norm"] == lp_norm)
            exp_data = df_zbin[mask].sort_values("zbin")

            if len(exp_data) == 0:
                continue

            x = exp_data["zbin"].values
            y = exp_data[metric_col].values

            linestyle = LP_NORM_STYLES.get(lp_norm, "-")
            alpha = LP_NORM_ALPHAS.get(lp_norm, 1.0)
            label = f"{pred_type} (Lp={lp_norm})"

            # Plot line
            ax.plot(
                x, y,
                color=color,
                linestyle=linestyle,
                linewidth=PLOT_SETTINGS["line_width"],
                alpha=alpha,
                marker="o",
                markersize=PLOT_SETTINGS["marker_size"],
                label=label,
                zorder=2,
            )

            # Plot error bands if available
            if show_error_bands and has_std:
                y_std = exp_data[std_col].values
                ax.fill_between(
                    x,
                    y - y_std,
                    y + y_std,
                    color=color,
                    alpha=alpha * 0.15,
                    zorder=1,
                )

    # Configure axes
    ax.set_xlabel("Z-bin", fontsize=PLOT_SETTINGS["axes_labelsize"], fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=PLOT_SETTINGS["axes_labelsize"], fontweight="bold")
    ax.set_xlim(zbins[0] - 0.5, zbins[-1] + 0.5)

    # Set title
    if title:
        ax.set_title(title, fontsize=PLOT_SETTINGS["title_fontsize"], fontweight="bold")

    # Configure legend
    if show_legend:
        # Create custom legend with better organization
        ax.legend(
            loc="upper right",
            ncol=3,
            framealpha=0.9,
            edgecolor="0.8",
        )

    # Tight layout
    plt.tight_layout()

    # Save in requested formats
    base_name = f"{metric_col}_zbin_comparison"
    for fmt in formats:
        output_path = output_dir / f"{base_name}.{fmt}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def plot_zbin_by_prediction_type(
    df_zbin: pd.DataFrame,
    metric_col: str,
    output_dir: Path,
    metric_name: str | None = None,
    baseline_real: float | None = None,
    figsize: tuple[float, float] = (16, 5),
    formats: list[str] = ["png", "pdf"],
) -> None:
    """Create separate per-zbin plots for each prediction type.

    Creates a 1x3 subplot grid with one panel per prediction type,
    showing all Lp norms within each panel.

    Args:
        df_zbin: DataFrame with columns: prediction_type, lp_norm, zbin, metric.
        metric_col: Column name for metric values.
        output_dir: Output directory.
        metric_name: Display name for metric.
        baseline_real: Optional baseline value.
        figsize: Figure size for full figure.
        formats: Output formats.
    """
    apply_plot_settings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metric_name is None:
        metric_name = metric_col.replace("_zbin", "").upper()

    prediction_types = sorted(df_zbin["prediction_type"].unique())
    lp_norms = sorted(df_zbin["lp_norm"].unique())
    zbins = sorted(df_zbin["zbin"].unique())

    n_types = len(prediction_types)
    fig, axes = plt.subplots(1, n_types, figsize=figsize, sharey=True)

    if n_types == 1:
        axes = [axes]

    for ax, pred_type in zip(axes, prediction_types):
        color = PREDICTION_TYPE_COLORS.get(pred_type, "black")

        # Plot baseline
        if baseline_real is not None:
            ax.axhline(
                baseline_real,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                zorder=1,
            )

        # Plot each Lp norm
        for lp_norm in lp_norms:
            mask = (df_zbin["prediction_type"] == pred_type) & (df_zbin["lp_norm"] == lp_norm)
            exp_data = df_zbin[mask].sort_values("zbin")

            if len(exp_data) == 0:
                continue

            x = exp_data["zbin"].values
            y = exp_data[metric_col].values

            linestyle = LP_NORM_STYLES.get(lp_norm, "-")
            alpha = LP_NORM_ALPHAS.get(lp_norm, 1.0)

            ax.plot(
                x, y,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                alpha=alpha,
                marker="o",
                markersize=3,
                label=f"Lp={lp_norm}",
            )

        ax.set_title(pred_type.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Z-bin", fontsize=10)
        ax.set_xlim(zbins[0] - 0.5, zbins[-1] + 0.5)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel(metric_name, fontsize=10, fontweight="bold")

    plt.tight_layout()

    # Save
    base_name = f"{metric_col}_by_prediction_type"
    for fmt in formats:
        output_path = output_dir / f"{base_name}.{fmt}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def create_legend_figure(
    output_dir: Path,
    formats: list[str] = ["png", "pdf"],
) -> None:
    """Create a standalone legend figure explaining the visual encoding.

    Args:
        output_dir: Output directory.
        formats: Output formats.
    """
    apply_plot_settings()
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.axis("off")

    # Create dummy lines for legend
    lines = []
    labels = []

    # Prediction types
    for pred_type, color in PREDICTION_TYPE_COLORS.items():
        line, = ax.plot([], [], color=color, linewidth=2, linestyle="-")
        lines.append(line)
        labels.append(f"Prediction: {pred_type}")

    # Lp norms
    for lp_norm, linestyle in LP_NORM_STYLES.items():
        line, = ax.plot([], [], color="black", linewidth=2, linestyle=linestyle)
        lines.append(line)
        labels.append(f"Lp norm: {lp_norm}")

    ax.legend(lines, labels, loc="center", ncol=3, fontsize=10, frameon=True)

    plt.tight_layout()

    for fmt in formats:
        output_path = output_dir / f"legend.{fmt}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()
