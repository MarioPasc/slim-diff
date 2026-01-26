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
    PREDICTION_TYPE_LABELS_SHORT,
    LP_NORM_LABELS,
    LP_NORM_HATCHES,
    LP_NORM_MARKERS,
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
    group_spacing = 1.5
    box_width = PLOT_SETTINGS.get("boxplot_width_factor", 0.25)
    box_whis = PLOT_SETTINGS.get("boxplot_whis", (5, 95))
    positions = []
    colors = []
    hatches = []
    data_to_plot = []
    lp_norms_for_boxes = []  # Track Lp norm for each box

    # Find best configuration
    best_config = None
    best_value = float("inf")
    best_box_idx = None

    # Prepare data for boxplot
    box_idx = 0
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

            pos = group_center + (j - 1) * box_width * 1.5
            positions.append(pos)
            colors.append(base_color)
            hatches.append(LP_NORM_HATCHES.get(lp_norm, None))
            data_to_plot.append(values)
            lp_norms_for_boxes.append(lp_norm)

            # Track best configuration
            mean_val = np.mean(values)
            if highlight_best and mean_val < best_value:
                best_value = mean_val
                best_config = (pred_type, lp_norm, pos)
                best_box_idx = box_idx
            box_idx += 1

    # Create boxplots with percentile-based whiskers
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,  # We'll show points separately
        whis=box_whis,  # Percentile whiskers (5th-95th by default)
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
        for pos, values, color, lp_norm in zip(positions, data_to_plot, colors, lp_norms_for_boxes):
            jitter = np.random.uniform(-box_width * 0.3, box_width * 0.3, len(values))
            marker = LP_NORM_MARKERS.get(lp_norm, "o")
            ax.scatter(
                pos + jitter,
                values,
                c=color,
                marker=marker,
                s=PLOT_SETTINGS["marker_size"] ** 2,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )

    # Highlight best with star (placed just above the box, not the whisker)
    if highlight_best and best_config is not None and best_box_idx is not None:
        pred_type, lp_norm, best_pos = best_config
        # Get the Q3 (top of box) for the best boxplot
        best_data = data_to_plot[best_box_idx]
        q3 = np.percentile(best_data, 75)

        # Place star just above the box (Q3) - smaller offset
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        star_y = q3 + y_range * 0.12

        ax.scatter(
            best_pos,
            star_y,
            marker="*",
            s=80,  # Smaller star
            color="gold",
            edgecolors="black",
            linewidths=0.4,
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

    # Configure x-axis - use short labels to prevent overlap
    tick_positions = [i * group_spacing for i in range(n_types)]
    # Use short labels for boxplots (full labels go in unified legend)
    tick_labels = [PREDICTION_TYPE_LABELS_SHORT.get(t, t) for t in prediction_types]
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

    Brackets are sorted by span width (narrower first) to avoid overlaps.
    Effect sizes are shown inline with stars: "*** (d=0.85)"

    Args:
        ax: Matplotlib axes.
        comparison_results: Dict from between_group_comparison().
        groups: List of prediction types.
        group_spacing: Spacing between group centers.
        settings: Plot settings dict.
        show_effect_sizes: Whether to show Cliff's delta with stars.
    """
    posthoc_pvalues = comparison_results.get("posthoc_pvalues", {})
    posthoc_significant = comparison_results.get("posthoc_significant", {})
    effect_sizes = comparison_results.get("effect_sizes", {})

    # Collect significant comparisons
    significant_comps = []
    for comp_key, p_val in posthoc_pvalues.items():
        if not posthoc_significant.get(comp_key, False):
            continue

        parts = comp_key.split("_vs_")
        if len(parts) != 2:
            continue

        g1, g2 = parts
        try:
            idx1 = groups.index(g1)
            idx2 = groups.index(g2)
        except ValueError:
            continue

        # Sort indices so smaller is first
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        span = idx2 - idx1
        significant_comps.append((span, idx1, idx2, comp_key, p_val))

    # Sort by span (narrower brackets first, then by position)
    significant_comps.sort(key=lambda x: (x[0], x[1]))

    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Starting y position and increment
    bracket_y = y_max + y_range * 0.1
    bracket_increment = y_range * 0.35
    bracket_height = y_range * 0.02

    for span, idx1, idx2, comp_key, p_val in significant_comps:
        x1 = idx1 * group_spacing
        x2 = idx2 * group_spacing

        # Draw bracket
        ax.plot(
            [x1, x1, x2, x2],
            [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
            color="black",
            linewidth=settings["significance_bracket_linewidth"],
            clip_on=False,
        )

        # Build annotation text: stars + effect size
        stars = get_significance_stars(p_val)
        if show_effect_sizes:
            effect = effect_sizes.get(comp_key, {})
            d_value = effect.get("cliffs_delta")
            if d_value is not None:
                annotation = f"{stars} (d={d_value:.2f})"
            else:
                annotation = stars
        else:
            annotation = stars

        # Place annotation above bracket
        ax.text(
            (x1 + x2) / 2,
            bracket_y + bracket_height + y_range * 0.015,
            annotation,
            ha="center",
            va="bottom",
            fontsize=settings["effect_size_fontsize"],
            fontweight="bold",
        )

        # Move to next bracket level
        bracket_y += bracket_increment

    # Adjust y-limits to fit all brackets
    if significant_comps:
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
