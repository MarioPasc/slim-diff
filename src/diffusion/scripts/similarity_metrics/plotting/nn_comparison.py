"""Nearest Neighbor Distance visualization for ICIP 2026.

Creates plots for analyzing feature-space nearest neighbor distances:
1. Global NN distance boxplots by prediction type and Lp norm
2. Per-zbin NN distance line plots
3. NN distance distribution histograms
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .settings import (
    PLOT_SETTINGS,
    PREDICTION_TYPE_COLORS,
    PREDICTION_TYPE_LABELS,
    PREDICTION_TYPE_LABELS_SHORT,
    LP_NORM_LABELS,
    LP_NORM_HATCHES,
    LP_NORM_MARKERS,
    LP_NORM_LINESTYLES,
    apply_ieee_style,
    get_significance_stars,
)


def plot_nn_boxplots(
    df: pd.DataFrame,
    metric_col: str = "synth_to_real_mean",
    output_dir: Path | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    formats: list[str] = ["pdf", "png"],
    highlight_best: bool = True,
    show_points: bool = True,
) -> plt.Axes | None:
    """Create boxplots for NN distance comparison.

    Args:
        df: DataFrame with columns: prediction_type, lp_norm, replica_id, {metric_col}.
        metric_col: Column name for NN distance metric.
        output_dir: Output directory for saving.
        ax: Optional axes to plot on.
        title: Plot title.
        ylabel: Y-axis label.
        formats: Output formats.
        highlight_best: Highlight best (lowest) configuration.
        show_points: Show individual data points.

    Returns:
        Axes if ax was provided, None otherwise.
    """
    apply_ieee_style()

    # Create figure if needed
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(
            figsize=(
                PLOT_SETTINGS["figure_width_double"] * 0.5,
                PLOT_SETTINGS["figure_width_double"] * 0.4,
            )
        )

    # Get unique values
    prediction_types = sorted(df["prediction_type"].unique())
    lp_norms = sorted(df["lp_norm"].unique())

    # Position calculations
    group_spacing = 1.5
    box_width = PLOT_SETTINGS.get("boxplot_width_factor", 0.25)

    positions = []
    colors = []
    hatches = []
    data_to_plot = []
    lp_norms_for_boxes = []

    # Track best configuration
    best_value = float("inf")
    best_box_idx = None

    box_idx = 0
    for i, pred_type in enumerate(prediction_types):
        group_center = i * group_spacing
        base_color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")

        for j, lp_norm in enumerate(lp_norms):
            mask = (df["prediction_type"] == pred_type) & (df["lp_norm"] == lp_norm)
            values = df.loc[mask, metric_col].dropna().values

            if len(values) == 0:
                continue

            pos = group_center + (j - 1) * box_width * 1.5
            positions.append(pos)
            colors.append(base_color)
            hatches.append(LP_NORM_HATCHES.get(lp_norm, None))
            data_to_plot.append(values)
            lp_norms_for_boxes.append(lp_norm)

            # Track best (lowest) configuration
            mean_val = np.mean(values)
            if mean_val < best_value:
                best_value = mean_val
                best_box_idx = box_idx
            box_idx += 1

    if not data_to_plot:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return ax if not created_figure else None

    # Create boxplots
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        whis=PLOT_SETTINGS.get("boxplot_whis", (5, 95)),
    )

    # Style boxplots
    for patch, color, hatch in zip(bp["boxes"], colors, hatches):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_linewidth(PLOT_SETTINGS["boxplot_linewidth"])
        patch.set_edgecolor("black")
        if hatch:
            patch.set_hatch(hatch)

    for element in ["whiskers", "caps"]:
        plt.setp(bp[element], color="black", linewidth=PLOT_SETTINGS["boxplot_linewidth"])
    plt.setp(bp["medians"], color="black", linewidth=PLOT_SETTINGS["boxplot_linewidth"] * 1.2)

    # Overlay single large marker at median
    if show_points:
        for pos, values, color, lp_norm in zip(positions, data_to_plot, colors, lp_norms_for_boxes):
            median_val = np.median(values)
            marker = LP_NORM_MARKERS.get(lp_norm, "o")
            ax.scatter(
                pos,
                median_val,
                c=color,
                marker=marker,
                s=PLOT_SETTINGS["marker_size"] ** 2 * 4,
                alpha=0.9,
                edgecolors="black",
                linewidths=1.0,
                zorder=3,
            )

    # Highlight best with circle
    if highlight_best and best_box_idx is not None:
        best_pos = positions[best_box_idx]
        best_data = data_to_plot[best_box_idx]
        median_val = np.median(best_data)

        ax.scatter(
            best_pos,
            median_val,
            s=PLOT_SETTINGS["marker_size"] ** 2 * 16,
            facecolors="none",
            edgecolors="gold",
            linewidths=1.5,
            zorder=5,
        )

    # Configure axes
    n_types = len(prediction_types)
    tick_positions = [i * group_spacing for i in range(n_types)]
    tick_labels = [PREDICTION_TYPE_LABELS_SHORT.get(t, t) for t in prediction_types]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=PLOT_SETTINGS["tick_labelsize"])

    if ylabel is None:
        ylabel = "NN Distance" if "synth_to_real" in metric_col else "Coverage Distance"
    ax.set_ylabel(ylabel, fontsize=PLOT_SETTINGS["axes_labelsize"])

    if title:
        ax.set_title(title, fontsize=PLOT_SETTINGS["axes_titlesize"])

    ax.yaxis.grid(True, alpha=PLOT_SETTINGS["grid_alpha"], linestyle=PLOT_SETTINGS["grid_linestyle"])
    ax.set_axisbelow(True)

    # Save if standalone
    if created_figure and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        base_name = f"nn_{metric_col}_boxplot"
        for fmt in formats:
            output_path = output_dir / f"{base_name}.{fmt}"
            plt.savefig(output_path, dpi=PLOT_SETTINGS["dpi_print"], bbox_inches="tight")
            print(f"Saved: {output_path}")
        plt.close()
        return None

    return ax


def plot_nn_zbin_lines(
    df: pd.DataFrame,
    metric_col: str = "synth_to_real_mean",
    output_dir: Path | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    formats: list[str] = ["pdf", "png"],
    show_error_bands: bool = True,
) -> plt.Axes | None:
    """Create per-zbin line plot for NN distances.

    Args:
        df: DataFrame with columns: prediction_type, lp_norm, zbin, {metric_col}.
        metric_col: Column name for NN distance metric.
        output_dir: Output directory.
        ax: Optional axes.
        title: Plot title.
        ylabel: Y-axis label.
        formats: Output formats.
        show_error_bands: Show std bands.

    Returns:
        Axes if provided, None otherwise.
    """
    apply_ieee_style()

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(
            figsize=(
                PLOT_SETTINGS["figure_width_double"] * 0.6,
                PLOT_SETTINGS["figure_width_double"] * 0.35,
            )
        )

    prediction_types = sorted(df["prediction_type"].unique())
    lp_norms = sorted(df["lp_norm"].unique())

    std_col = metric_col.replace("_mean", "_std") if "_mean" in metric_col else None

    for pred_type in prediction_types:
        color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")

        for lp_norm in lp_norms:
            mask = (df["prediction_type"] == pred_type) & (df["lp_norm"] == lp_norm)
            subset = df.loc[mask].sort_values("zbin")

            if len(subset) == 0:
                continue

            zbins = subset["zbin"].values
            values = subset[metric_col].values

            linestyle = LP_NORM_LINESTYLES.get(lp_norm, "-")
            marker = LP_NORM_MARKERS.get(lp_norm, "o")

            ax.plot(
                zbins,
                values,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=PLOT_SETTINGS["marker_size"],
                linewidth=PLOT_SETTINGS["line_width"],
                alpha=0.8,
            )

            # Error bands
            if show_error_bands and std_col and std_col in subset.columns:
                std_values = subset[std_col].values
                ax.fill_between(
                    zbins,
                    values - std_values,
                    values + std_values,
                    color=color,
                    alpha=0.15,
                )

    ax.set_xlabel("Z-bin", fontsize=PLOT_SETTINGS["axes_labelsize"])
    if ylabel is None:
        ylabel = "NN Distance"
    ax.set_ylabel(ylabel, fontsize=PLOT_SETTINGS["axes_labelsize"])

    if title:
        ax.set_title(title, fontsize=PLOT_SETTINGS["axes_titlesize"])

    ax.yaxis.grid(True, alpha=PLOT_SETTINGS["grid_alpha"], linestyle=PLOT_SETTINGS["grid_linestyle"])
    ax.set_axisbelow(True)

    if created_figure and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        base_name = f"nn_{metric_col}_zbin"
        for fmt in formats:
            output_path = output_dir / f"{base_name}.{fmt}"
            plt.savefig(output_path, dpi=PLOT_SETTINGS["dpi_print"], bbox_inches="tight")
            print(f"Saved: {output_path}")
        plt.close()
        return None

    return ax


def plot_nn_histogram(
    distances: np.ndarray,
    output_dir: Path | None = None,
    title: str = "NN Distance Distribution",
    xlabel: str = "Distance",
    n_bins: int = 50,
    formats: list[str] = ["pdf", "png"],
    color: str = "#0077BB",
) -> None:
    """Plot histogram of NN distances.

    Args:
        distances: Array of NN distances.
        output_dir: Output directory.
        title: Plot title.
        xlabel: X-axis label.
        n_bins: Number of histogram bins.
        formats: Output formats.
        color: Histogram color.
    """
    apply_ieee_style()

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.hist(distances, bins=n_bins, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Add statistics
    mean_val = np.mean(distances)
    median_val = np.median(distances)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="orange", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.2f}")

    ax.set_xlabel(xlabel, fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_ylabel("Count", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_title(title, fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            output_path = output_dir / f"nn_distance_histogram.{fmt}"
            plt.savefig(output_path, dpi=PLOT_SETTINGS["dpi_print"], bbox_inches="tight")
            print(f"Saved: {output_path}")

    plt.close()


def create_nn_summary_figure(
    df_global: pd.DataFrame,
    df_zbin: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    formats: list[str] = ["pdf", "png"],
) -> None:
    """Create a summary figure with NN distance analysis.

    Layout:
    - Top row: Synth->Real boxplot | Real->Synth boxplot
    - Bottom row: Per-zbin Synth->Real (if available)

    Args:
        df_global: Global NN metrics DataFrame.
        df_zbin: Per-zbin NN metrics DataFrame (optional).
        output_dir: Output directory.
        formats: Output formats.
    """
    apply_ieee_style()

    n_rows = 2 if df_zbin is not None else 1
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(PLOT_SETTINGS["figure_width_double"], PLOT_SETTINGS["figure_width_double"] * 0.4 * n_rows),
        squeeze=False,
    )

    # Top left: Synth->Real NN distance
    plot_nn_boxplots(
        df_global,
        metric_col="synth_to_real_mean",
        ax=axes[0, 0],
        title="(a) Synth$\\rightarrow$Real NN",
        ylabel="NN Distance",
    )

    # Top right: Real->Synth NN distance (coverage)
    plot_nn_boxplots(
        df_global,
        metric_col="real_to_synth_mean",
        ax=axes[0, 1],
        title="(b) Real$\\rightarrow$Synth NN",
        ylabel="Coverage Distance",
    )

    # Bottom: Per-zbin (if available)
    if df_zbin is not None and len(axes) > 1:
        plot_nn_zbin_lines(
            df_zbin,
            metric_col="synth_to_real_mean",
            ax=axes[1, 0],
            title="(c) Per-Zbin Synth$\\rightarrow$Real",
        )
        plot_nn_zbin_lines(
            df_zbin,
            metric_col="real_to_synth_mean",
            ax=axes[1, 1],
            title="(d) Per-Zbin Real$\\rightarrow$Synth",
        )

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            output_path = output_dir / f"nn_summary_figure.{fmt}"
            plt.savefig(output_path, dpi=PLOT_SETTINGS["dpi_print"], bbox_inches="tight")
            print(f"Saved: {output_path}")

    plt.close()
