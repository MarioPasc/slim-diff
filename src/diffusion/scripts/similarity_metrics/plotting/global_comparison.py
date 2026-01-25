"""Global metric comparison plotting with statistical significance.

Creates grouped bar charts comparing metrics across experiments,
with significance brackets and best configuration highlighting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .zbin_multiexp import (
    PLOT_SETTINGS,
    PREDICTION_TYPE_COLORS,
    apply_plot_settings,
)


# Darker shades for Lp norm color gradient (within prediction type)
def get_lp_color(base_color: str, lp_norm: float) -> str:
    """Get color shade for Lp norm within a prediction type.

    Higher Lp = lighter shade.

    Args:
        base_color: Base hex color.
        lp_norm: Lp norm value.

    Returns:
        Adjusted hex color.
    """
    import matplotlib.colors as mcolors

    # Convert hex to RGB
    rgb = mcolors.hex2color(base_color)

    # Adjust lightness based on Lp norm
    # 1.5 -> darkest, 2.5 -> lightest
    lightness_factor = 0.7 + (lp_norm - 1.5) * 0.3  # 0.7 to 1.0

    # Blend with white
    rgb_adjusted = tuple(
        min(1.0, c * lightness_factor + (1 - lightness_factor) * 0.3)
        for c in rgb
    )

    return mcolors.rgb2hex(rgb_adjusted)


def plot_global_comparison(
    df_global: pd.DataFrame,
    metric_col: str,
    output_dir: Path,
    comparison_results: dict[str, Any] | None = None,
    metric_name: str | None = None,
    baseline_real: float | None = None,
    baseline_std: float | None = None,
    figsize: tuple[float, float] = (12, 6),
    formats: list[str] = ["png", "pdf"],
    show_significance: bool = True,
    highlight_best: bool = True,
    title: str | None = None,
) -> None:
    """Create grouped bar chart for global metric comparison.

    Layout:
    - 3 groups (epsilon, velocity, x0) on x-axis
    - 3 bars per group (Lp 1.5, 2.0, 2.5) with color gradient
    - Error bars: ±1 std across replicas
    - Significance brackets between groups (if p < 0.05)
    - Star marker on best configuration
    - Horizontal dashed line for baseline

    Args:
        df_global: DataFrame with columns: experiment, prediction_type, lp_norm,
                   replica_id, {metric_col}.
        metric_col: Column name for metric values (e.g., "kid_global").
        output_dir: Output directory.
        comparison_results: Optional dict from between_group_comparison().
        metric_name: Display name for metric.
        baseline_real: Optional baseline value.
        baseline_std: Optional baseline std.
        figsize: Figure size.
        formats: Output formats.
        show_significance: Whether to show significance brackets.
        highlight_best: Whether to highlight best configuration.
        title: Optional title.
    """
    apply_plot_settings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metric_name is None:
        metric_name = metric_col.replace("_global", "").upper()

    # Aggregate statistics by experiment
    stats = df_global.groupby(["prediction_type", "lp_norm"]).agg({
        metric_col: ["mean", "std", "count"],
    }).reset_index()
    stats.columns = ["prediction_type", "lp_norm", "mean", "std", "count"]

    # Get unique values
    prediction_types = sorted(df_global["prediction_type"].unique())
    lp_norms = sorted(df_global["lp_norm"].unique())

    n_types = len(prediction_types)
    n_lp = len(lp_norms)

    # Bar positions
    bar_width = 0.25
    group_width = n_lp * bar_width + 0.1
    group_positions = np.arange(n_types) * group_width

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot baseline
    if baseline_real is not None:
        ax.axhline(
            baseline_real,
            color="gray",
            linestyle="--",
            linewidth=2,
            label=f"Real baseline: {baseline_real:.4f}",
            zorder=1,
        )
        if baseline_std is not None:
            ax.axhspan(
                baseline_real - baseline_std,
                baseline_real + baseline_std,
                color="gray",
                alpha=0.1,
                zorder=1,
            )

    # Find best configuration
    best_config = None
    best_value = float("inf")
    if highlight_best:
        for _, row in stats.iterrows():
            if row["mean"] < best_value:
                best_value = row["mean"]
                best_config = (row["prediction_type"], row["lp_norm"])

    # Plot bars for each group
    bars_plotted = []
    for i, pred_type in enumerate(prediction_types):
        base_color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")

        for j, lp_norm in enumerate(lp_norms):
            # Get stats for this experiment
            mask = (stats["prediction_type"] == pred_type) & (stats["lp_norm"] == lp_norm)
            exp_stats = stats[mask]

            if len(exp_stats) == 0:
                continue

            mean_val = exp_stats["mean"].values[0]
            std_val = exp_stats["std"].values[0]

            # Bar position
            x_pos = group_positions[i] + j * bar_width

            # Color (gradient by Lp)
            color = get_lp_color(base_color, lp_norm)

            # Plot bar
            bar = ax.bar(
                x_pos,
                mean_val,
                width=bar_width * 0.9,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )

            # Error bar
            ax.errorbar(
                x_pos,
                mean_val,
                yerr=std_val,
                fmt="none",
                color="black",
                capsize=4,
                capthick=1,
                zorder=3,
            )

            bars_plotted.append({
                "pred_type": pred_type,
                "lp_norm": lp_norm,
                "x_pos": x_pos,
                "mean": mean_val,
                "std": std_val,
            })

            # Highlight best with star
            if highlight_best and best_config == (pred_type, lp_norm):
                ax.scatter(
                    x_pos,
                    mean_val + std_val + 0.02 * ax.get_ylim()[1],
                    marker="*",
                    s=200,
                    color="gold",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=4,
                )

    # Add significance brackets
    if show_significance and comparison_results is not None:
        _add_significance_brackets(ax, comparison_results, prediction_types, group_positions, bar_width, n_lp)

    # Configure x-axis
    ax.set_xticks(group_positions + (n_lp - 1) * bar_width / 2)
    ax.set_xticklabels([t.capitalize() for t in prediction_types], fontsize=12)
    ax.set_xlabel("Prediction Type", fontsize=PLOT_SETTINGS["axes_labelsize"], fontweight="bold")

    # Configure y-axis
    ax.set_ylabel(metric_name, fontsize=PLOT_SETTINGS["axes_labelsize"], fontweight="bold")

    # Title
    if title:
        ax.set_title(title, fontsize=PLOT_SETTINGS["title_fontsize"], fontweight="bold")

    # Create custom legend
    _create_bar_legend(ax, prediction_types, lp_norms, baseline_real is not None, highlight_best)

    # Grid
    ax.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    base_name = f"{metric_col}_comparison"
    for fmt in formats:
        output_path = output_dir / f"{base_name}.{fmt}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def _add_significance_brackets(
    ax: plt.Axes,
    comparison_results: dict[str, Any],
    prediction_types: list[str],
    group_positions: np.ndarray,
    bar_width: float,
    n_lp: int,
) -> None:
    """Add significance brackets between groups.

    Args:
        ax: Matplotlib axes.
        comparison_results: Dict from between_group_comparison().
        prediction_types: List of prediction types.
        group_positions: X positions of groups.
        bar_width: Width of individual bars.
        n_lp: Number of Lp norms.
    """
    posthoc_pvalues = comparison_results.get("posthoc_pvalues", {})
    posthoc_significant = comparison_results.get("posthoc_significant", {})

    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Track bracket height to avoid overlap
    bracket_y = y_max + y_range * 0.05
    bracket_increment = y_range * 0.08

    for comp_key, p_val in posthoc_pvalues.items():
        if not posthoc_significant.get(comp_key, False):
            continue

        # Parse comparison key (e.g., "epsilon_vs_velocity")
        parts = comp_key.split("_vs_")
        if len(parts) != 2:
            continue

        type1, type2 = parts

        # Find indices
        try:
            idx1 = prediction_types.index(type1)
            idx2 = prediction_types.index(type2)
        except ValueError:
            continue

        # Get center positions for each group
        x1 = group_positions[idx1] + (n_lp - 1) * bar_width / 2
        x2 = group_positions[idx2] + (n_lp - 1) * bar_width / 2

        # Draw bracket
        bracket_height = 0.02 * y_range

        # Horizontal line at top
        ax.plot(
            [x1, x1, x2, x2],
            [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
            color="black",
            linewidth=1,
        )

        # Significance stars
        stars = _get_significance_stars(p_val)
        ax.text(
            (x1 + x2) / 2,
            bracket_y + bracket_height + 0.01 * y_range,
            stars,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        # Move to next bracket level
        bracket_y += bracket_increment

    # Adjust y-axis to fit brackets
    ax.set_ylim(y_min, bracket_y + y_range * 0.05)


def _get_significance_stars(p_val: float) -> str:
    """Get significance stars for p-value.

    Args:
        p_val: P-value.

    Returns:
        String with stars.
    """
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "n.s."


def _create_bar_legend(
    ax: plt.Axes,
    prediction_types: list[str],
    lp_norms: list[float],
    has_baseline: bool,
    has_best: bool,
) -> None:
    """Create custom legend for bar plot.

    Args:
        ax: Matplotlib axes.
        prediction_types: List of prediction types.
        lp_norms: List of Lp norms.
        has_baseline: Whether baseline is shown.
        has_best: Whether best is highlighted.
    """
    handles = []
    labels = []

    # Prediction type colors (use medium Lp for representative color)
    for pred_type in prediction_types:
        base_color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")
        patch = mpatches.Patch(color=base_color, label=pred_type.capitalize())
        handles.append(patch)
        labels.append(pred_type.capitalize())

    # Lp norm legend (shown as text annotation)
    ax.text(
        0.02, 0.98,
        f"Bar shading: Lp {min(lp_norms)} (dark) to {max(lp_norms)} (light)",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        style="italic",
        color="gray",
    )

    # Baseline
    if has_baseline:
        line = plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=2)
        handles.append(line)
        labels.append("Real baseline")

    # Best marker
    if has_best:
        star = plt.Line2D(
            [0], [0],
            marker="*",
            color="gold",
            markersize=12,
            markeredgecolor="black",
            linestyle="None",
        )
        handles.append(star)
        labels.append("Best config")

    ax.legend(
        handles, labels,
        loc="upper right",
        framealpha=0.9,
        edgecolor="0.8",
    )


def plot_metric_summary_table(
    df_global: pd.DataFrame,
    metrics: list[str],
    output_dir: Path,
    formats: list[str] = ["png", "pdf"],
) -> None:
    """Create a summary table figure showing all metrics.

    Args:
        df_global: DataFrame with metric columns.
        metrics: List of metric column names.
        output_dir: Output directory.
        formats: Output formats.
    """
    apply_plot_settings()
    output_dir = Path(output_dir)

    # Aggregate statistics
    agg_dict = {m: ["mean", "std"] for m in metrics if m in df_global.columns}
    if not agg_dict:
        print("Warning: No valid metrics found for summary table")
        return

    stats = df_global.groupby(["prediction_type", "lp_norm"]).agg(agg_dict)
    stats = stats.round(4)

    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    # Format table data
    table_data = []
    for (pred_type, lp_norm), row in stats.iterrows():
        row_data = [f"{pred_type}_lp_{lp_norm}"]
        for metric in metrics:
            if metric in df_global.columns:
                mean = row[(metric, "mean")]
                std = row[(metric, "std")]
                row_data.append(f"{mean:.4f} ± {std:.4f}")
        table_data.append(row_data)

    # Column headers
    col_labels = ["Experiment"] + [m.replace("_global", "").upper() for m in metrics if m in df_global.columns]

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for j, label in enumerate(col_labels):
        table[(0, j)].set_facecolor("#E8E8E8")
        table[(0, j)].set_text_props(fontweight="bold")

    plt.title("Similarity Metrics Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    # Save
    for fmt in formats:
        output_path = output_dir / f"metrics_summary_table.{fmt}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()
