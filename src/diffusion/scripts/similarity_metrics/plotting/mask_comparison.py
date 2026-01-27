"""Mask quality visualization with IEEE ICIP 2026 publication style.

Creates a single-column figure with two subplots for mask quality evaluation:
1. Global MMD-MF boxplot (analogous to KID boxplot for images)
2. Per-feature Wasserstein heatmap (diagnostic breakdown)

The figure answers:
- Which prediction type yields masks most similar to real masks?
- Within each prediction type, which Lp norm maximizes this similarity?
- Which morphological features are well/poorly captured?
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
    apply_ieee_style,
    get_significance_stars,
    IEEE_COLUMN_WIDTH_INCHES,
)


# Feature display labels (shorter for heatmap columns)
FEATURE_DISPLAY_LABELS = {
    "area": "Area",
    "perimeter": "Perim.",
    "circularity": "Circ.",
    "solidity": "Solid.",
    "extent": "Extent",
    "eccentricity": "Eccen.",
    "major_axis_length": "Major",
    "minor_axis_length": "Minor",
    "equivalent_diameter": "Eq.Dia.",
}


def create_mask_quality_figure(
    df_global: pd.DataFrame,
    wasserstein_df: pd.DataFrame,
    output_dir: Path,
    comparison_results: dict[str, Any] | None = None,
    baseline_mmd: float | None = None,
    baseline_mmd_std: float | None = None,
    formats: list[str] = ["pdf", "png"],
    figsize: tuple[float, float] | None = None,
) -> None:
    """Create 2-panel mask quality figure for ICIP publication.

    Layout (single column, 3.39 inches wide):
    +------------------------+
    |  (a) MMD-MF Boxplots   |
    |  [grouped by pred_type]|
    +------------------------+
    |  (b) Wasserstein Heat  |
    |  [features x configs]  |
    +------------------------+

    Args:
        df_global: DataFrame with columns: experiment, prediction_type, lp_norm,
                   replica_id, mmd_mf_global, mmd_mf_global_std.
        wasserstein_df: DataFrame with per-feature Wasserstein distances.
                        Columns: experiment, prediction_type, lp_norm, replica_id,
                        plus feature columns (area, circularity, etc.).
        output_dir: Output directory for saving figures.
        comparison_results: Statistical comparison results for significance brackets.
        baseline_mmd: Baseline MMD value (real vs real) for reference line.
        baseline_mmd_std: Baseline MMD std for shaded region.
        formats: Output formats (default: ["pdf", "png"]).
        figsize: Override figure size (default: single column width).
    """
    apply_ieee_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default figsize: single column width, 1.6x height for stacked plots
    if figsize is None:
        figsize = (IEEE_COLUMN_WIDTH_INCHES, IEEE_COLUMN_WIDTH_INCHES * 1.6)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1, 0.8], "hspace": 0.35},
    )

    # Panel (a): MMD-MF Boxplots
    _plot_mmd_boxplots(
        ax=axes[0],
        df=df_global,
        metric_col="mmd_mf_global",
        comparison_results=comparison_results,
        baseline=baseline_mmd,
        baseline_std=baseline_mmd_std,
    )
    axes[0].set_title(
        "(a) Mask Morphology Distance",
        loc="left",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        fontweight="bold",
    )

    # Panel (b): Per-feature Wasserstein Heatmap
    _plot_wasserstein_heatmap(
        ax=axes[1],
        df=wasserstein_df,
    )
    axes[1].set_title(
        "(b) Per-Feature Quality",
        loc="left",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        fontweight="bold",
    )

    plt.tight_layout()

    # Save
    for fmt in formats:
        output_path = output_dir / f"mask_quality_figure.{fmt}"
        plt.savefig(
            output_path,
            dpi=PLOT_SETTINGS["dpi_print"],
            bbox_inches="tight",
        )
        print(f"Saved: {output_path}")

    plt.close()


def _plot_mmd_boxplots(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    comparison_results: dict[str, Any] | None = None,
    baseline: float | None = None,
    baseline_std: float | None = None,
) -> None:
    """Plot MMD-MF boxplots grouped by prediction type.

    Visual encoding:
    - X-axis: Prediction type (epsilon, velocity, x0)
    - Within each type: 3 boxes for Lp norms (1.5, 2.0, 2.5)
    - Color by prediction type (Paul Tol palette)
    - Hatch pattern by Lp norm

    Args:
        ax: Matplotlib axes to plot on.
        df: DataFrame with mmd_mf_global column.
        metric_col: Column name for metric values.
        comparison_results: Statistical comparison results.
        baseline: Baseline value for reference line.
        baseline_std: Baseline std for shaded region.
    """
    # Get unique values
    prediction_types = sorted(df["prediction_type"].unique())
    lp_norms = sorted(df["lp_norm"].unique())

    # Plot baseline if provided
    if baseline is not None:
        ax.axhline(
            baseline,
            color="gray",
            linestyle="--",
            linewidth=PLOT_SETTINGS["line_width"],
            label="Real baseline",
            zorder=1,
        )
        if baseline_std is not None:
            ax.axhspan(
                baseline - baseline_std,
                baseline + baseline_std,
                color="gray",
                alpha=0.1,
                zorder=1,
            )

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

    # Prepare data for boxplot
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

            # Track best configuration
            mean_val = np.mean(values)
            if mean_val < best_value:
                best_value = mean_val
                best_box_idx = box_idx
            box_idx += 1

    if not data_to_plot:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

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
        plt.setp(
            bp[element], color="black", linewidth=PLOT_SETTINGS["boxplot_linewidth"]
        )
    plt.setp(
        bp["medians"],
        color="black",
        linewidth=PLOT_SETTINGS["boxplot_linewidth"] * 1.2,
    )

    # Overlay individual points with jitter
    for pos, values, color, lp_norm in zip(
        positions, data_to_plot, colors, lp_norms_for_boxes
    ):
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

    # Highlight best with star
    if best_box_idx is not None:
        best_pos = positions[best_box_idx]
        best_data = data_to_plot[best_box_idx]
        q3 = np.percentile(best_data, 75)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        star_y = q3 + y_range * 0.12

        ax.scatter(
            best_pos,
            star_y,
            marker="*",
            s=80,
            color="gold",
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
        )

    # Add significance brackets if provided
    if comparison_results is not None:
        _add_significance_brackets(
            ax,
            comparison_results,
            prediction_types,
            group_spacing,
        )

    # Configure x-axis
    n_types = len(prediction_types)
    tick_positions = [i * group_spacing for i in range(n_types)]
    tick_labels = [PREDICTION_TYPE_LABELS_SHORT.get(t, t) for t in prediction_types]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=PLOT_SETTINGS["tick_labelsize"])

    # Configure y-axis
    ax.set_ylabel("MMD-MF", fontsize=PLOT_SETTINGS["axes_labelsize"])

    # Grid
    ax.yaxis.grid(
        True,
        alpha=PLOT_SETTINGS["grid_alpha"],
        linestyle=PLOT_SETTINGS["grid_linestyle"],
    )
    ax.set_axisbelow(True)


def _add_significance_brackets(
    ax: plt.Axes,
    comparison_results: dict[str, Any],
    groups: list[str],
    group_spacing: float,
) -> None:
    """Add significance brackets between prediction types.

    Args:
        ax: Matplotlib axes.
        comparison_results: Dict from between_group_comparison().
        groups: List of prediction types.
        group_spacing: Spacing between group centers.
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

        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        span = idx2 - idx1
        significant_comps.append((span, idx1, idx2, comp_key, p_val))

    # Sort by span
    significant_comps.sort(key=lambda x: (x[0], x[1]))

    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

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
            linewidth=PLOT_SETTINGS["significance_bracket_linewidth"],
            clip_on=False,
        )

        # Build annotation
        stars = get_significance_stars(p_val)
        effect = effect_sizes.get(comp_key, {})
        d_value = effect.get("cliffs_delta")
        if d_value is not None:
            annotation = f"{stars} (d={d_value:.2f})"
        else:
            annotation = stars

        # Place annotation
        ax.text(
            (x1 + x2) / 2,
            bracket_y + bracket_height + y_range * 0.015,
            annotation,
            ha="center",
            va="bottom",
            fontsize=PLOT_SETTINGS["effect_size_fontsize"],
            fontweight="bold",
        )

        bracket_y += bracket_increment

    # Adjust y-limits
    if significant_comps:
        ax.set_ylim(y_min, bracket_y + y_range * 0.02)


def _plot_wasserstein_heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    cmap: str = "RdYlGn_r",
) -> None:
    """Plot per-feature Wasserstein distance heatmap.

    Args:
        ax: Matplotlib axes.
        df: DataFrame with columns: experiment, prediction_type, lp_norm,
            and feature columns (area, circularity, etc.).
        cmap: Colormap (red=high distance=bad, green=low=good).
    """
    # Identify feature columns (exclude metadata)
    exclude_cols = {
        "experiment",
        "prediction_type",
        "lp_norm",
        "self_cond_p",
        "replica_id",
        "geometric_mean",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if not feature_cols:
        ax.text(
            0.5,
            0.5,
            "No feature data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Order feature columns consistently
    ordered_features = [
        f for f in FEATURE_DISPLAY_LABELS.keys() if f in feature_cols
    ]
    # Add any remaining features not in display labels
    ordered_features += [f for f in feature_cols if f not in ordered_features]

    # Average across replicas and pivot
    pivot_df = df.groupby(["prediction_type", "lp_norm"])[ordered_features].mean()

    # Sort by prediction type then lp_norm for consistent ordering
    pred_order = ["epsilon", "velocity", "x0"]
    lp_order = [1.5, 2.0, 2.5]

    # Create ordered index
    ordered_index = []
    for pred in pred_order:
        for lp in lp_order:
            try:
                if (pred, lp) in pivot_df.index:
                    ordered_index.append((pred, lp))
            except Exception:
                pass

    if ordered_index:
        pivot_df = pivot_df.loc[ordered_index]

    # Create row labels
    row_labels = []
    for pred_type, lp_norm in pivot_df.index:
        pred_label = PREDICTION_TYPE_LABELS_SHORT.get(pred_type, pred_type)
        lp_label = LP_NORM_LABELS.get(lp_norm, f"L{lp_norm}")
        row_labels.append(f"{pred_label}, {lp_label}")

    # Create column labels
    col_labels = [FEATURE_DISPLAY_LABELS.get(f, f) for f in ordered_features]

    # Plot heatmap manually (avoid seaborn dependency issues)
    data = pivot_df.values

    # Create heatmap
    im = ax.imshow(data, aspect="auto", cmap=cmap)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Wasserstein Distance", fontsize=PLOT_SETTINGS["legend_fontsize"])
    cbar.ax.tick_params(labelsize=PLOT_SETTINGS["tick_labelsize"] - 1)

    # Set ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(
        col_labels,
        rotation=45,
        ha="right",
        fontsize=PLOT_SETTINGS["tick_labelsize"] - 1,
    )
    ax.set_yticklabels(row_labels, fontsize=PLOT_SETTINGS["tick_labelsize"] - 1)

    # Add cell annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            if not np.isnan(val):
                # Choose text color based on value
                text_color = "white" if val > np.nanmedian(data) else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=PLOT_SETTINGS["annotation_fontsize"] - 1,
                    color=text_color,
                )

    # Add grid lines between cells
    ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add horizontal lines to separate prediction types (every 3 rows)
    for i in range(3, len(row_labels), 3):
        ax.axhline(y=i - 0.5, color="black", linewidth=1.5)


def plot_mask_metrics_comparison(
    df_global: pd.DataFrame,
    output_dir: Path,
    comparison_results: dict[str, Any] | None = None,
    baseline_mmd: float | None = None,
    formats: list[str] = ["pdf", "png"],
) -> None:
    """Standalone MMD-MF boxplot comparison (without heatmap).

    Args:
        df_global: DataFrame with mmd_mf_global column.
        output_dir: Output directory.
        comparison_results: Statistical comparison results.
        baseline_mmd: Baseline value for reference line.
        formats: Output formats.
    """
    apply_ieee_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figsize = (
        PLOT_SETTINGS["figure_width_double"] * 0.5,
        PLOT_SETTINGS["figure_width_double"] * 0.4,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    _plot_mmd_boxplots(
        ax=ax,
        df=df_global,
        metric_col="mmd_mf_global",
        comparison_results=comparison_results,
        baseline=baseline_mmd,
    )

    # Create legend
    _create_boxplot_legend(ax, df_global, baseline_mmd is not None)

    plt.tight_layout()

    for fmt in formats:
        output_path = output_dir / f"mmd_mf_boxplot_comparison.{fmt}"
        plt.savefig(output_path, dpi=PLOT_SETTINGS["dpi_print"], bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def _create_boxplot_legend(
    ax: plt.Axes,
    df: pd.DataFrame,
    has_baseline: bool,
) -> None:
    """Create custom legend for boxplot.

    Args:
        ax: Matplotlib axes.
        df: DataFrame with prediction_type and lp_norm columns.
        has_baseline: Whether baseline is shown.
    """
    handles = []
    labels = []

    # Prediction type colors
    prediction_types = sorted(df["prediction_type"].unique())
    for pred_type in prediction_types:
        color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")
        pred_label = PREDICTION_TYPE_LABELS.get(pred_type, pred_type.capitalize())
        patch = mpatches.Patch(
            facecolor=color, edgecolor="black", alpha=0.6, label=pred_label
        )
        handles.append(patch)
        labels.append(pred_label)

    # Lp norm hatches
    lp_norms = sorted(df["lp_norm"].unique())
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
    star = plt.Line2D(
        [0],
        [0],
        marker="*",
        color="gold",
        markersize=10,
        markeredgecolor="black",
        linestyle="None",
    )
    handles.append(star)
    labels.append("Best config")

    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=PLOT_SETTINGS["legend_frameon"],
    )
