"""Global metric comparison plotting with statistical significance.

Creates boxplots comparing metrics across experiments,
with significance brackets, effect sizes, and jittered data points.

ICIP 2026 version with:
- Boxplots instead of bar charts
- Hatch patterns for Lp norm encoding
- Paul Tol colorblind-friendly palettes
- Effect sizes (Cliff's delta) on significance brackets
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .settings import (
    PLOT_SETTINGS,
    PREDICTION_TYPE_COLORS,
    PREDICTION_TYPE_LABELS,
    LP_NORM_LABELS,
    LP_NORM_HATCHES,
    apply_ieee_style,
    format_metric_label,
    get_significance_stars,
)


def plot_global_boxplots(
    df_global: pd.DataFrame,
    metric_col: str,
    output_dir: Path | None = None,
    comparison_results: dict[str, Any] | None = None,
    metric_name: str | None = None,
    baseline_real: float | None = None,
    baseline_std: float | None = None,
    figsize: tuple[float, float] | None = None,
    formats: list[str] = ["png", "pdf"],
    show_significance: bool = True,
    show_effect_sizes: bool = True,
    show_points: bool = True,
    highlight_best: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes | None:
    """Create boxplots for global metric comparison.

    Visual encoding:
    - X-axis: Prediction type (epsilon, velocity, x0)
    - Within each type: 3 boxes for Lp norms (1.5, 2.0, 2.5)
    - Color by prediction type (Paul Tol palette)
    - Hatch pattern by Lp norm (none, //, \\\\)
    - Overlay: Individual replica points (jittered)
    - Significance brackets with stars + effect sizes

    Args:
        df_global: DataFrame with columns: experiment, prediction_type, lp_norm,
                   replica_id, {metric_col}.
        metric_col: Column name for metric values (e.g., "kid_global").
        output_dir: Output directory (optional if ax provided).
        comparison_results: Optional dict from between_group_comparison().
        metric_name: Display name for metric.
        baseline_real: Optional baseline value.
        baseline_std: Optional baseline std.
        figsize: Figure size.
        formats: Output formats.
        show_significance: Whether to show significance brackets.
        show_effect_sizes: Whether to show effect sizes on brackets.
        show_points: Whether to show individual data points.
        highlight_best: Whether to highlight best configuration.
        title: Optional title.
        ax: Optional existing axes to plot on (for subplots).

    Returns:
        The axes object if ax was provided, None otherwise.
    """
    apply_ieee_style()

    if figsize is None:
        figsize = (
            PLOT_SETTINGS["figure_width_double"] * 0.5,
            PLOT_SETTINGS["figure_width_double"] * 0.4,
        )

    if metric_name is None:
        metric_name = format_metric_label(metric_col)

    # Get unique values
    prediction_types = sorted(df_global["prediction_type"].unique())
    lp_norms = sorted(df_global["lp_norm"].unique())

    n_types = len(prediction_types)
    n_lp = len(lp_norms)

    # Create figure if no axes provided
    created_figure = ax is None
    if created_figure:
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Plot baseline
    if baseline_real is not None:
        ax.axhline(
            baseline_real,
            color="gray",
            linestyle="--",
            linewidth=PLOT_SETTINGS["line_width"],
            label="Real baseline",
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

    # Position calculations
    group_spacing = 1.0
    box_width = 0.22
    positions = []
    colors = []
    hatches = []
    data_to_plot = []

    # Find best configuration
    best_config = None
    best_value = float("inf")

    # Prepare data for boxplot
    for i, pred_type in enumerate(prediction_types):
        group_center = i * group_spacing
        base_color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")

        for j, lp_norm in enumerate(lp_norms):
            mask = (
                (df_global["prediction_type"] == pred_type) &
                (df_global["lp_norm"] == lp_norm)
            )
            values = df_global.loc[mask, metric_col].values

            if len(values) == 0:
                continue

            pos = group_center + (j - 1) * box_width * 1.3
            positions.append(pos)
            colors.append(base_color)
            hatches.append(LP_NORM_HATCHES.get(lp_norm, None))
            data_to_plot.append(values)

            # Track best configuration
            mean_val = np.mean(values)
            if highlight_best and mean_val < best_value:
                best_value = mean_val
                best_config = (pred_type, lp_norm, pos)

    # Create boxplots
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,  # We'll show points separately
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

    # Overlay individual points with jitter
    if show_points:
        for pos, values, color in zip(positions, data_to_plot, colors):
            jitter = np.random.uniform(-box_width * 0.3, box_width * 0.3, len(values))
            ax.scatter(
                pos + jitter,
                values,
                c=color,
                s=PLOT_SETTINGS["marker_size"] ** 2,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )

    # Highlight best with star
    if highlight_best and best_config is not None:
        pred_type, lp_norm, best_pos = best_config
        y_max = max(np.max(d) for d in data_to_plot)
        ax.scatter(
            best_pos,
            y_max * 1.05,
            marker="*",
            s=150,
            color="gold",
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )

    # Add significance brackets
    if show_significance and comparison_results is not None:
        _add_significance_brackets_boxplot(
            ax,
            comparison_results,
            prediction_types,
            group_spacing,
            PLOT_SETTINGS,
            show_effect_sizes=show_effect_sizes,
        )

    # Configure x-axis
    tick_positions = [i * group_spacing for i in range(n_types)]
    tick_labels = [PREDICTION_TYPE_LABELS.get(t, t.capitalize()) for t in prediction_types]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=PLOT_SETTINGS["tick_labelsize"])

    # Configure y-axis
    ax.set_ylabel(metric_name, fontsize=PLOT_SETTINGS["axes_labelsize"])

    # Title
    if title:
        ax.set_title(title, fontsize=PLOT_SETTINGS["axes_titlesize"])

    # Grid
    ax.yaxis.grid(True, alpha=PLOT_SETTINGS["grid_alpha"], linestyle=PLOT_SETTINGS["grid_linestyle"])
    ax.set_axisbelow(True)

    # Save if we created the figure
    if created_figure and output_dir is not None:
        # Create custom legend
        _create_boxplot_legend(ax, prediction_types, lp_norms, baseline_real is not None, highlight_best)

        plt.tight_layout()

        base_name = f"{metric_col}_boxplot_comparison"
        for fmt in formats:
            output_path = output_dir / f"{base_name}.{fmt}"
            plt.savefig(
                output_path,
                dpi=PLOT_SETTINGS["dpi_print"],
                bbox_inches="tight",
            )
            print(f"Saved: {output_path}")

        plt.close()
        return None
    else:
        return ax


def _add_significance_brackets_boxplot(
    ax: plt.Axes,
    comparison_results: dict[str, Any],
    groups: list[str],
    group_spacing: float,
    settings: dict,
    show_effect_sizes: bool = True,
) -> None:
    """Add significance brackets between prediction types for boxplots.

    Args:
        ax: Matplotlib axes.
        comparison_results: Dict from between_group_comparison().
        groups: List of prediction types.
        group_spacing: Spacing between group centers.
        settings: Plot settings dict.
        show_effect_sizes: Whether to show Cliff's delta below stars.
    """
    posthoc_pvalues = comparison_results.get("posthoc_pvalues", {})
    posthoc_significant = comparison_results.get("posthoc_significant", {})
    effect_sizes = comparison_results.get("effect_sizes", {})

    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    bracket_y = y_max + y_range * 0.05
    bracket_increment = y_range * 0.15 if show_effect_sizes else y_range * 0.10

    for comp_key, p_val in posthoc_pvalues.items():
        if not posthoc_significant.get(comp_key, False):
            continue

        # Parse comparison (e.g., "epsilon_vs_x0")
        parts = comp_key.split("_vs_")
        if len(parts) != 2:
            continue

        g1, g2 = parts
        try:
            idx1 = groups.index(g1)
            idx2 = groups.index(g2)
        except ValueError:
            continue

        x1 = idx1 * group_spacing
        x2 = idx2 * group_spacing

        # Draw bracket
        bracket_height = y_range * 0.02

        ax.plot(
            [x1, x1, x2, x2],
            [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
            color="black",
            linewidth=settings["significance_bracket_linewidth"],
            clip_on=False,
        )

        # Significance stars
        stars = get_significance_stars(p_val)
        ax.text(
            (x1 + x2) / 2,
            bracket_y + bracket_height + y_range * 0.01,
            stars,
            ha="center",
            va="bottom",
            fontsize=settings["significance_text_fontsize"],
            fontweight="bold",
        )

        # Effect size (Cliff's delta)
        if show_effect_sizes:
            effect = effect_sizes.get(comp_key, {})
            d_value = effect.get("cliffs_delta")
            if d_value is not None:
                ax.text(
                    (x1 + x2) / 2,
                    bracket_y + bracket_height + y_range * 0.05,
                    f"d={d_value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=settings["effect_size_fontsize"],
                    style="italic",
                )

        bracket_y += bracket_increment

    # Adjust y-limits to fit brackets
    ax.set_ylim(y_min, bracket_y + y_range * 0.02)


def _create_boxplot_legend(
    ax: plt.Axes,
    prediction_types: list[str],
    lp_norms: list[float],
    has_baseline: bool,
    has_best: bool,
) -> None:
    """Create custom legend for boxplot.

    Args:
        ax: Matplotlib axes.
        prediction_types: List of prediction types.
        lp_norms: List of Lp norms.
        has_baseline: Whether baseline is shown.
        has_best: Whether best is highlighted.
    """
    handles = []
    labels = []

    # Prediction type colors
    for pred_type in prediction_types:
        color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")
        pred_label = PREDICTION_TYPE_LABELS.get(pred_type, pred_type.capitalize())
        patch = mpatches.Patch(facecolor=color, edgecolor="black", alpha=0.6, label=pred_label)
        handles.append(patch)
        labels.append(pred_label)

    # Lp norm hatches
    for lp_norm in lp_norms:
        hatch = LP_NORM_HATCHES.get(lp_norm, None)
        lp_label = LP_NORM_LABELS.get(lp_norm, f"Lp={lp_norm}")
        patch = mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            hatch=hatch if hatch else "",
            label=lp_label,
        )
        handles.append(patch)
        labels.append(lp_label)

    # Baseline
    if has_baseline:
        line = plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5)
        handles.append(line)
        labels.append("Real baseline")

    # Best marker
    if has_best:
        star = plt.Line2D(
            [0], [0],
            marker="*",
            color="gold",
            markersize=10,
            markeredgecolor="black",
            linestyle="None",
        )
        handles.append(star)
        labels.append("Best config")

    ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=PLOT_SETTINGS["legend_frameon"],
    )


# Keep legacy function for backward compatibility
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
    """Create global metric comparison plot (now uses boxplots).

    This function is kept for backward compatibility.
    It now calls plot_global_boxplots() internally.

    Args:
        df_global: DataFrame with metric data.
        metric_col: Column name for metric values.
        output_dir: Output directory.
        comparison_results: Statistical comparison results.
        metric_name: Display name for metric.
        baseline_real: Optional baseline value.
        baseline_std: Optional baseline std.
        figsize: Figure size.
        formats: Output formats.
        show_significance: Whether to show significance brackets.
        highlight_best: Whether to highlight best configuration.
        title: Optional title.
    """
    plot_global_boxplots(
        df_global=df_global,
        metric_col=metric_col,
        output_dir=output_dir,
        comparison_results=comparison_results,
        metric_name=metric_name,
        baseline_real=baseline_real,
        baseline_std=baseline_std,
        figsize=figsize,
        formats=formats,
        show_significance=show_significance,
        show_effect_sizes=True,
        show_points=True,
        highlight_best=highlight_best,
        title=title,
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
    apply_ieee_style()
    output_dir = Path(output_dir)

    # Aggregate statistics
    agg_dict = {m: ["mean", "std"] for m in metrics if m in df_global.columns}
    if not agg_dict:
        print("Warning: No valid metrics found for summary table")
        return

    stats = df_global.groupby(["prediction_type", "lp_norm"]).agg(agg_dict)
    stats = stats.round(4)

    # Create figure with table
    fig, ax = plt.subplots(figsize=(PLOT_SETTINGS["figure_width_double"], 4))
    ax.axis("off")

    # Format table data
    table_data = []
    for (pred_type, lp_norm), row in stats.iterrows():
        pred_label = PREDICTION_TYPE_LABELS.get(pred_type, pred_type)
        lp_label = LP_NORM_LABELS.get(lp_norm, f"Lp={lp_norm}")
        row_data = [f"{pred_label}, {lp_label}"]
        for metric in metrics:
            if metric in df_global.columns:
                mean = row[(metric, "mean")]
                std = row[(metric, "std")]
                row_data.append(f"{mean:.4f} +/- {std:.4f}")
        table_data.append(row_data)

    # Column headers
    col_labels = ["Experiment"] + [format_metric_label(m) for m in metrics if m in df_global.columns]

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(PLOT_SETTINGS["font_size"])
    table.scale(1.2, 1.5)

    # Color header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#E8E8E8")
        table[(0, j)].set_text_props(fontweight="bold")

    plt.title(
        "Similarity Metrics Summary",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    # Save
    for fmt in formats:
        output_path = output_dir / f"metrics_summary_table.{fmt}"
        plt.savefig(output_path, dpi=PLOT_SETTINGS["dpi_print"], bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()
