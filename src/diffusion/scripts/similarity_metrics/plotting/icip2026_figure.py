"""Main ICIP 2026 publication figure generation.

Creates a 2x2 subplot layout with:
- Top row: Per-zbin KID and LPIPS with representative MRI images
- Bottom row: Global KID and LPIPS boxplots with significance
- Unified legend outside all subplots

Optionally creates a 2x3 layout with mask metrics:
- Columns 0-1: Image metrics (KID, LPIPS per-zbin and global)
- Column 2: Mask metrics (MMD-MF boxplot and per-feature heatmap)
- Vertical separator between image and mask metric panels

IEEE publication-ready with Paul Tol colorblind-friendly palettes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .settings import (
    PLOT_SETTINGS,
    PREDICTION_TYPE_COLORS,
    PREDICTION_TYPE_LABELS,
    PREDICTION_TYPE_LABELS_SHORT,
    LP_NORM_STYLES,
    LP_NORM_MARKERS,
    LP_NORM_LABELS,
    LP_NORM_HATCHES,
    apply_ieee_style,
)
from .zbin_multiexp import (
    plot_zbin_multiexperiment,
    add_representative_images,
)
from .global_comparison import plot_global_boxplots
from .mask_comparison import _plot_mmd_boxplots, _plot_wasserstein_heatmap, _plot_wasserstein_heatmap_transposed


def create_unified_legend(
    fig: plt.Figure,
    prediction_types: list[str] | None = None,
    lp_norms: list[float] | None = None,
    include_baseline: bool = True,
    include_lp_norms: bool = False,
    include_best: bool = False,
    ncol: int | None = None,
    bbox_y: float = -0.04,
) -> None:
    """Create unified legend below all subplots.

    Args:
        fig: Matplotlib figure.
        prediction_types: List of prediction types (default: epsilon, velocity, x0).
        lp_norms: List of Lp norms (default: 1.5, 2.0, 2.5).
        include_baseline: Whether to include baseline in legend.
        include_lp_norms: Whether to include Lp norm line styles in legend.
        include_best: Whether to include best star marker in legend.
        ncol: Number of columns for legend. If None, auto-determined.
        bbox_y: Y position of legend bbox_to_anchor.
    """
    if prediction_types is None:
        prediction_types = ["epsilon", "velocity", "x0"]
    if lp_norms is None:
        lp_norms = [1.5, 2.0, 2.5]

    handles = []
    labels = []

    # Prediction type legend entries (colored patches)
    for pred_type in prediction_types:
        color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")
        pred_label = PREDICTION_TYPE_LABELS_SHORT.get(pred_type, pred_type)
        patch = mpatches.Patch(
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )
        handles.append(patch)
        labels.append(pred_label)

    # Baseline
    if include_baseline:
        line = Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5)
        handles.append(line)
        labels.append("Baseline")

    # Lp norm line styles (when subplot legends are disabled)
    if include_lp_norms:
        for lp_norm in lp_norms:
            linestyle = LP_NORM_STYLES.get(lp_norm, "-")
            marker = LP_NORM_MARKERS.get(lp_norm, "o")
            lp_label = LP_NORM_LABELS.get(lp_norm, f"Lp={lp_norm}")
            line = Line2D(
                [0], [0],
                color="black",
                linestyle=linestyle,
                linewidth=PLOT_SETTINGS["line_width"],
                marker=marker,
                markersize=PLOT_SETTINGS["marker_size"],
            )
            handles.append(line)
            labels.append(lp_label)

    # Best marker (circle)
    if include_best:
        circle = Line2D(
            [0], [0],
            marker="o",
            color="none",
            markersize=10,
            markeredgecolor="gold",
            markeredgewidth=1.5,
            linestyle="None",
        )
        handles.append(circle)
        labels.append("Best")

    # Determine layout
    n_items = len(handles)
    if ncol is None:
        # Auto-determine: single row if fits, otherwise 4 columns
        if n_items > 5:
            ncol = 4
        else:
            ncol = n_items

    # Place legend below figure
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=False,
        bbox_to_anchor=(0.5, bbox_y),
        columnspacing=PLOT_SETTINGS["legend_columnspacing"],
        handletextpad=PLOT_SETTINGS["legend_handletextpad"],
    )


def create_subplot_legend_lines(
    ax: plt.Axes,
    lp_norms: list[float],
    loc: str = "upper right",
) -> None:
    """Create per-subplot legend for line plots (Lp norm styles).

    Args:
        ax: Target axes.
        lp_norms: List of Lp norms to include.
        loc: Legend location.
    """
    handles = []
    labels = []

    for lp_norm in lp_norms:
        linestyle = LP_NORM_STYLES.get(lp_norm, "-")
        marker = LP_NORM_MARKERS.get(lp_norm, "o")
        lp_label = LP_NORM_LABELS.get(lp_norm, f"Lp={lp_norm}")
        line = Line2D(
            [0], [0],
            color="black",
            linestyle=linestyle,
            linewidth=PLOT_SETTINGS["line_width"],
            marker=marker,
            markersize=PLOT_SETTINGS["marker_size"],
        )
        handles.append(line)
        labels.append(lp_label)

    ax.legend(
        handles, labels,
        loc=loc,
        fontsize=PLOT_SETTINGS["legend_fontsize"] - 1,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        framealpha=0.9,
    )


def create_subplot_legend_boxes(
    ax: plt.Axes,
    lp_norms: list[float],
    include_best: bool = True,
    loc: str = "upper right",
) -> None:
    """Create per-subplot legend for boxplots (Lp norm hatches + best star).

    Args:
        ax: Target axes.
        lp_norms: List of Lp norms to include.
        include_best: Whether to include best marker.
        loc: Legend location.
    """
    handles = []
    labels = []

    # Hatch pattern legend
    for lp_norm in lp_norms:
        hatch = LP_NORM_HATCHES.get(lp_norm, None)
        lp_label = LP_NORM_LABELS.get(lp_norm, f"Lp={lp_norm}")
        patch = mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            hatch=hatch if hatch else "",
            linewidth=0.5,
        )
        handles.append(patch)
        labels.append(lp_label)

    # Best marker (circle)
    if include_best:
        circle = Line2D(
            [0], [0],
            marker="o",
            color="none",
            markersize=10,
            markeredgecolor="gold",
            markeredgewidth=1.5,
            linestyle="None",
        )
        handles.append(circle)
        labels.append("Best")

    ax.legend(
        handles, labels,
        loc=loc,
        fontsize=PLOT_SETTINGS["legend_fontsize"] - 1,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        framealpha=0.9,
    )


def create_icip2026_figure(
    df_global: pd.DataFrame,
    df_zbin: pd.DataFrame,
    output_dir: Path,
    test_csv: Path | None = None,
    comparison_results: dict[str, Any] | None = None,
    baseline_kid: float | None = None,
    baseline_kid_std: float | None = None,
    baseline_lpips: float | None = None,
    baseline_lpips_std: float | None = None,
    formats: list[str] = ["pdf", "png"],
    show_images: bool = True,
    show_images_on_lpips: bool = False,
    image_x_offset: float | None = None,
    image_y_offset: float | None = None,
    show_subplot_legends: bool = True,
    show_effect_sizes: bool = False,
    legend_ncol: int = 8,
    aspect_ratio: float = 0.55,
) -> None:
    """Create the main 2x2 publication figure for ICIP 2026.

    Designed for LaTeX \\begin{figure*} environment (two-column, wider than tall).

    Layout (columns = per-zbin, rows share y-axis):
        +--------------------------------------------------+
        |        Representative MRI Images (6 total)       |
        +--------------------------------------------------+
        | (A) Per-zbin KID        | (B) Global KID (sharey)|
        +-------------------------+------------------------+
        |        (Optional MRI Images for LPIPS)           |
        +--------------------------------------------------+
        | (C) Per-zbin LPIPS      | (D) Global LPIPS       |
        |     (sharex with A)     |     (sharey with C)    |
        +-------------------------+------------------------+
        |      Unified Legend (8-column, single row)       |
        +--------------------------------------------------+

    Args:
        df_global: DataFrame with global metric values per replica.
        df_zbin: DataFrame with per-zbin metric values.
        output_dir: Output directory for figure files.
        test_csv: Path to test.csv for representative images.
        comparison_results: Statistical comparison results dict.
        baseline_kid: Real baseline KID value.
        baseline_kid_std: Real baseline KID std.
        baseline_lpips: Real baseline LPIPS value.
        baseline_lpips_std: Real baseline LPIPS std.
        formats: Output formats (pdf, png).
        show_images: Whether to show representative MRI images on KID plot.
        show_images_on_lpips: Whether to also show images above LPIPS plot.
        image_x_offset: Horizontal offset for images (fraction).
        image_y_offset: Vertical offset for images (fraction).
        show_subplot_legends: Whether to show per-subplot legends (Lp norms).
            If False, all legend items are shown in unified bottom legend.
        show_effect_sizes: Whether to show effect sizes (Cliff's delta) on
            significance brackets.
        legend_ncol: Number of columns for unified legend (default: 8 for figure*).
        aspect_ratio: Height/width ratio (default: 0.55 for wider than tall).
    """
    apply_ieee_style()

    # Figure dimensions (IEEE double column, wider than tall for figure*)
    fig_width = PLOT_SETTINGS["figure_width_double"]
    fig_height = fig_width * aspect_ratio

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create GridSpec with space for images and legend
    # Column widths: per-zbin wider than global boxplots
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.4, 1],  # Per-zbin wider
        height_ratios=[1, 1],
        hspace=0.35,
        wspace=0.08,  # Tight for shared y-axis
        left=0.07,
        right=0.96,
        top=0.88 if show_images else 0.95,
        bottom=0.10,  # Space for legend
    )

    # Get z-bins for image positioning
    zbins = np.array(sorted(df_zbin["zbin"].unique()))

    # Get unique values for legends
    prediction_types = sorted(df_global["prediction_type"].unique())
    lp_norms = sorted(df_global["lp_norm"].unique())

    # Extract comparison results for each metric
    kid_comparison = None
    lpips_comparison = None
    if comparison_results is not None:
        kid_comparison = comparison_results.get("kid_global", {}).get("between_group")
        lpips_comparison = comparison_results.get("lpips_global", {}).get("between_group")

    # =========================================================================
    # Row 0: KID metrics
    # =========================================================================

    # Panel A: Per-zbin KID (row 0, col 0)
    ax_kid_zbin = fig.add_subplot(gs[0, 0])
    plot_zbin_multiexperiment(
        df_zbin,
        "kid_zbin",
        output_dir=None,
        baseline_real=baseline_kid,
        baseline_std=baseline_kid_std,
        show_legend=False,
        legend_outside=False,
        ax=ax_kid_zbin,
    )
    ax_kid_zbin.text(
        -0.08, 1.05, "(A)",
        transform=ax_kid_zbin.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )
    # Add Lp norm legend to this subplot (if enabled)
    if show_subplot_legends:
        create_subplot_legend_lines(ax_kid_zbin, lp_norms, loc="upper right")

    # Add representative images above panel A
    if show_images and test_csv is not None:
        add_representative_images(
            ax_kid_zbin,
            test_csv,
            zbins,
            image_step=PLOT_SETTINGS["image_step"],
            image_x_offset=image_x_offset,
            image_y_offset=image_y_offset,
        )

    # Panel B: Global KID boxplots (row 0, col 1) - shares y with A
    ax_kid_global = fig.add_subplot(gs[0, 1], sharey=ax_kid_zbin)
    plot_global_boxplots(
        df_global,
        "kid_global",
        output_dir=None,
        comparison_results=kid_comparison,
        baseline_real=baseline_kid,
        baseline_std=baseline_kid_std,
        show_significance=True,
        show_effect_sizes=show_effect_sizes,
        show_points=True,
        highlight_best=True,
        ax=ax_kid_global,
    )
    ax_kid_global.text(
        -0.08, 1.05, "(B)",
        transform=ax_kid_global.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )
    # Remove y-axis label (shared with A)
    ax_kid_global.set_ylabel("")
    plt.setp(ax_kid_global.get_yticklabels(), visible=False)
    # Add Lp hatch + best legend to this subplot (if enabled)
    if show_subplot_legends:
        create_subplot_legend_boxes(ax_kid_global, lp_norms, include_best=True, loc="upper right")

    # =========================================================================
    # Row 1: LPIPS metrics
    # =========================================================================

    # Panel C: Per-zbin LPIPS (row 1, col 0) - shares x with A
    ax_lpips_zbin = fig.add_subplot(gs[1, 0], sharex=ax_kid_zbin)
    if "lpips_zbin" in df_zbin.columns:
        plot_zbin_multiexperiment(
            df_zbin,
            "lpips_zbin",
            output_dir=None,
            baseline_real=baseline_lpips,
            baseline_std=baseline_lpips_std,
            show_legend=False,
            legend_outside=False,
            ax=ax_lpips_zbin,
        )
    else:
        ax_lpips_zbin.text(
            0.5, 0.5, "LPIPS data\nnot available",
            transform=ax_lpips_zbin.transAxes,
            ha="center", va="center",
            fontsize=PLOT_SETTINGS["font_size"],
            style="italic",
        )
        ax_lpips_zbin.set_xlabel("Z-bin", fontsize=PLOT_SETTINGS["axes_labelsize"])
        ax_lpips_zbin.set_ylabel("LPIPS", fontsize=PLOT_SETTINGS["axes_labelsize"])

    ax_lpips_zbin.text(
        -0.08, 1.05, "(C)",
        transform=ax_lpips_zbin.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )
    # Add Lp norm legend to this subplot (if enabled)
    if show_subplot_legends:
        create_subplot_legend_lines(ax_lpips_zbin, lp_norms, loc="upper right")

    # Add representative images above LPIPS panel (if enabled)
    if show_images_on_lpips and test_csv is not None:
        add_representative_images(
            ax_lpips_zbin,
            test_csv,
            zbins,
            image_step=PLOT_SETTINGS["image_step"],
            image_x_offset=image_x_offset,
            image_y_offset=image_y_offset,
        )

    # Panel D: Global LPIPS boxplots (row 1, col 1) - shares y with C
    ax_lpips_global = fig.add_subplot(gs[1, 1], sharey=ax_lpips_zbin)
    if "lpips_global" in df_global.columns:
        plot_global_boxplots(
            df_global,
            "lpips_global",
            output_dir=None,
            comparison_results=lpips_comparison,
            baseline_real=baseline_lpips,
            baseline_std=baseline_lpips_std,
            show_significance=True,
            show_effect_sizes=show_effect_sizes,
            show_points=True,
            highlight_best=True,
            ax=ax_lpips_global,
        )
    else:
        ax_lpips_global.text(
            0.5, 0.5, "LPIPS data\nnot available",
            transform=ax_lpips_global.transAxes,
            ha="center", va="center",
            fontsize=PLOT_SETTINGS["font_size"],
            style="italic",
        )
        ax_lpips_global.set_ylabel("LPIPS", fontsize=PLOT_SETTINGS["axes_labelsize"])

    ax_lpips_global.text(
        -0.08, 1.05, "(D)",
        transform=ax_lpips_global.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )
    # Remove y-axis label (shared with C)
    ax_lpips_global.set_ylabel("")
    plt.setp(ax_lpips_global.get_yticklabels(), visible=False)
    # Add Lp hatch + best legend to this subplot (if enabled)
    if show_subplot_legends:
        create_subplot_legend_boxes(ax_lpips_global, lp_norms, include_best=True, loc="upper right")

    # =========================================================================
    # Unified legend at bottom
    # Uses configurable ncol for figure* (default: 8 columns, single row)
    # =========================================================================
    create_unified_legend(
        fig,
        prediction_types=prediction_types,
        lp_norms=lp_norms,
        include_baseline=baseline_kid is not None or baseline_lpips is not None,
        include_lp_norms=not show_subplot_legends,  # Include if no subplot legends
        include_best=not show_subplot_legends,       # Include if no subplot legends
        ncol=legend_ncol,
        bbox_y=-0.05,
    )

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = output_dir / f"icip2026_similarity_metrics.{fmt}"
        fig.savefig(
            output_path,
            dpi=PLOT_SETTINGS["dpi_print"],
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print(f"Saved: {output_path}")

    plt.close(fig)


def create_icip2026_figure_with_masks(
    df_global: pd.DataFrame,
    df_zbin: pd.DataFrame,
    df_mask_global: pd.DataFrame,
    df_mask_wasserstein: pd.DataFrame,
    output_dir: Path,
    test_csv: Path | None = None,
    comparison_results: dict[str, Any] | None = None,
    mask_comparison_results: dict[str, Any] | None = None,
    baseline_kid: float | None = None,
    baseline_kid_std: float | None = None,
    baseline_lpips: float | None = None,
    baseline_lpips_std: float | None = None,
    baseline_mmd: float | None = None,
    baseline_mmd_std: float | None = None,
    formats: list[str] = ["pdf", "png"],
    show_images: bool = True,
    show_images_on_lpips: bool = False,
    image_x_offset: float | None = None,
    image_y_offset: float | None = None,
    show_subplot_legends: bool = True,
    show_effect_sizes: bool = False,
    legend_ncol: int = 8,
    aspect_ratio: float = 0.55,
) -> None:
    """Create a 2x3 publication figure combining image and mask metrics.

    Layout:
        +--------------------------------------------------+-------------------+
        |        Representative MRI Images (6 total)       |                   |
        +--------------------------------------------------+                   |
        | (A.1) Per-zbin KID  | (A.2) Global KID (sharey)  || (B.1) MMD-MF     |
        +---------------------+----------------------------+|     Boxplots     |
        |        (Optional MRI Images for LPIPS)           ||                   |
        +--------------------------------------------------+-------------------+
        | (A.3) Per-zbin LPIPS| (A.4) Global LPIPS         || (B.2) Per-Feature|
        |     (sharex with A) |     (sharey with C)        ||     Heatmap      |
        +---------------------+----------------------------+-------------------+
        |           Unified Legend (single row)                                |
        +----------------------------------------------------------------------+

    The vertical line separator visually distinguishes image metrics (A panels)
    from mask morphology metrics (B panels).

    Args:
        df_global: DataFrame with global image metric values per replica.
        df_zbin: DataFrame with per-zbin image metric values.
        df_mask_global: DataFrame with mask MMD-MF values per replica.
        df_mask_wasserstein: DataFrame with per-feature Wasserstein distances.
        output_dir: Output directory for figure files.
        test_csv: Path to test.csv for representative images.
        comparison_results: Statistical comparison results for image metrics.
        mask_comparison_results: Statistical comparison results for mask metrics.
        baseline_kid: Real baseline KID value.
        baseline_kid_std: Real baseline KID std.
        baseline_lpips: Real baseline LPIPS value.
        baseline_lpips_std: Real baseline LPIPS std.
        baseline_mmd: Real baseline MMD-MF value.
        baseline_mmd_std: Real baseline MMD-MF std.
        formats: Output formats (pdf, png).
        show_images: Whether to show representative MRI images on KID plot.
        show_images_on_lpips: Whether to also show images above LPIPS plot.
        image_x_offset: Horizontal offset for images (fraction).
        image_y_offset: Vertical offset for images (fraction).
        show_subplot_legends: Whether to show per-subplot legends (Lp norms).
        show_effect_sizes: Whether to show effect sizes on significance brackets.
        legend_ncol: Number of columns for unified legend.
        aspect_ratio: Height/width ratio.
    """
    apply_ieee_style()

    # Figure dimensions (wider for 3 columns + separator space)
    fig_width = PLOT_SETTINGS["figure_width_double"] * 1.45
    fig_height = fig_width * aspect_ratio

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create two separate GridSpecs: one for image metrics (A panels), one for mask metrics (B panels)
    # This allows us to have clear visual separation between the two groups

    # Image metrics: columns 0-1 (A.1-A.4)
    gs_image = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.3, 0.9],
        height_ratios=[1, 1],
        hspace=0.35,
        wspace=0.08,
        left=0.06,
        right=0.555,  # More space for separation
        top=0.88 if show_images else 0.95,
        bottom=0.12,
    )

    # Mask metrics: column 2 (B.1-B.2)
    gs_mask = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1, 1],
        hspace=0.35,
        left=0.68,  # More separation from image metrics
        right=0.98,
        top=0.88 if show_images else 0.95,
        bottom=0.12,
    )

    # Get z-bins for image positioning
    zbins = np.array(sorted(df_zbin["zbin"].unique()))

    # Get unique values for legends
    prediction_types = sorted(df_global["prediction_type"].unique())
    lp_norms = sorted(df_global["lp_norm"].unique())

    # Extract comparison results for each metric
    kid_comparison = None
    lpips_comparison = None
    if comparison_results is not None:
        kid_comparison = comparison_results.get("kid_global", {}).get("between_group")
        lpips_comparison = comparison_results.get("lpips_global", {}).get("between_group")

    # =========================================================================
    # Row 0, Col 0: Per-zbin KID (A.1)
    # =========================================================================
    ax_kid_zbin = fig.add_subplot(gs_image[0, 0])
    plot_zbin_multiexperiment(
        df_zbin,
        "kid_zbin",
        output_dir=None,
        baseline_real=baseline_kid,
        baseline_std=baseline_kid_std,
        show_legend=False,
        legend_outside=False,
        ax=ax_kid_zbin,
    )
    ax_kid_zbin.text(
        -0.08, 1.05, "(A.1)",
        transform=ax_kid_zbin.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )
    ax_kid_zbin.set_ylim(0.0, 1.0)  # Fixed range for KID
    if show_subplot_legends:
        create_subplot_legend_lines(ax_kid_zbin, lp_norms, loc="upper right")

    # Add representative images above panel A.1
    if show_images and test_csv is not None:
        add_representative_images(
            ax_kid_zbin,
            test_csv,
            zbins,
            image_step=PLOT_SETTINGS["image_step"],
            image_x_offset=image_x_offset,
            image_y_offset=image_y_offset,
        )

    # =========================================================================
    # Row 0, Col 1: Global KID (A.2) - shares y with A.1
    # =========================================================================
    ax_kid_global = fig.add_subplot(gs_image[0, 1], sharey=ax_kid_zbin)
    plot_global_boxplots(
        df_global,
        "kid_global",
        output_dir=None,
        comparison_results=kid_comparison,
        baseline_real=baseline_kid,
        baseline_std=baseline_kid_std,
        show_significance=True,
        show_effect_sizes=show_effect_sizes,
        show_points=True,
        highlight_best=True,
        ax=ax_kid_global,
    )
    ax_kid_global.text(
        -0.08, 1.05, "(A.2)",
        transform=ax_kid_global.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )
    ax_kid_global.set_ylabel("")
    plt.setp(ax_kid_global.get_yticklabels(), visible=False)
    if show_subplot_legends:
        create_subplot_legend_boxes(ax_kid_global, lp_norms, include_best=True, loc="upper right")

    # =========================================================================
    # Row 0, Col 2: MMD-MF Boxplots (B.1)
    # =========================================================================
    ax_mmd = fig.add_subplot(gs_mask[0, 0])
    _plot_mmd_boxplots(
        ax=ax_mmd,
        df=df_mask_global,
        metric_col="mmd_mf_global",
        comparison_results=mask_comparison_results,
        baseline=baseline_mmd,
        baseline_std=baseline_mmd_std,
    )
    ax_mmd.text(
        -0.08, 1.05, "(B.1)",
        transform=ax_mmd.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )
    ax_mmd.set_title("")  # Title handled by panel label
    if show_subplot_legends:
        create_subplot_legend_boxes(ax_mmd, lp_norms, include_best=True, loc="upper right")

    # =========================================================================
    # Row 1, Col 0: Per-zbin LPIPS (A.3) - shares x with A.1
    # =========================================================================
    ax_lpips_zbin = fig.add_subplot(gs_image[1, 0], sharex=ax_kid_zbin)
    if "lpips_zbin" in df_zbin.columns:
        plot_zbin_multiexperiment(
            df_zbin,
            "lpips_zbin",
            output_dir=None,
            baseline_real=baseline_lpips,
            baseline_std=baseline_lpips_std,
            show_legend=False,
            legend_outside=False,
            ax=ax_lpips_zbin,
        )
    else:
        ax_lpips_zbin.text(
            0.5, 0.5, "LPIPS data\nnot available",
            transform=ax_lpips_zbin.transAxes,
            ha="center", va="center",
            fontsize=PLOT_SETTINGS["font_size"],
            style="italic",
        )
        ax_lpips_zbin.set_xlabel("Z-bin", fontsize=PLOT_SETTINGS["axes_labelsize"])
        ax_lpips_zbin.set_ylabel("LPIPS", fontsize=PLOT_SETTINGS["axes_labelsize"])

    ax_lpips_zbin.text(
        -0.08, 1.05, "(A.3)",
        transform=ax_lpips_zbin.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )
    ax_lpips_zbin.set_ylim(0.0, 1.0)  # Fixed range for LPIPS
    if show_subplot_legends:
        create_subplot_legend_lines(ax_lpips_zbin, lp_norms, loc="upper right")

    if show_images_on_lpips and test_csv is not None:
        add_representative_images(
            ax_lpips_zbin,
            test_csv,
            zbins,
            image_step=PLOT_SETTINGS["image_step"],
            image_x_offset=image_x_offset,
            image_y_offset=image_y_offset,
        )

    # =========================================================================
    # Row 1, Col 1: Global LPIPS (A.4) - shares y with A.3
    # =========================================================================
    ax_lpips_global = fig.add_subplot(gs_image[1, 1], sharey=ax_lpips_zbin)
    if "lpips_global" in df_global.columns:
        plot_global_boxplots(
            df_global,
            "lpips_global",
            output_dir=None,
            comparison_results=lpips_comparison,
            baseline_real=baseline_lpips,
            baseline_std=baseline_lpips_std,
            show_significance=True,
            show_effect_sizes=show_effect_sizes,
            show_points=True,
            highlight_best=True,
            ax=ax_lpips_global,
        )
    else:
        ax_lpips_global.text(
            0.5, 0.5, "LPIPS data\nnot available",
            transform=ax_lpips_global.transAxes,
            ha="center", va="center",
            fontsize=PLOT_SETTINGS["font_size"],
            style="italic",
        )
        ax_lpips_global.set_ylabel("LPIPS", fontsize=PLOT_SETTINGS["axes_labelsize"])

    ax_lpips_global.text(
        -0.08, 1.05, "(A.4)",
        transform=ax_lpips_global.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )
    ax_lpips_global.set_ylabel("")
    plt.setp(ax_lpips_global.get_yticklabels(), visible=False)
    if show_subplot_legends:
        create_subplot_legend_boxes(ax_lpips_global, lp_norms, include_best=True, loc="upper right")

    # =========================================================================
    # Row 1, Col 2: Per-Feature Wasserstein Heatmap (B.2)
    # Uses transposed layout: features on Y-axis, configs on X-axis
    # X-axis uses colored markers to match B.1 layout
    # =========================================================================
    ax_heatmap = fig.add_subplot(gs_mask[1, 0])
    _plot_wasserstein_heatmap_transposed(
        ax=ax_heatmap,
        df=df_mask_wasserstein,
        show_annotations=False,  # No cell numbers to reduce clutter
    )
    ax_heatmap.text(
        -0.08, 1.05, "(B.2)",
        transform=ax_heatmap.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )

    # =========================================================================
    # Add vertical separator line between image metrics and mask metrics
    # The visual separation is achieved through spacing in the GridSpecs,
    # with a solid black line in the middle for clear delineation
    # =========================================================================
    # Get the position between the two GridSpecs in figure coordinates
    bbox_image = ax_kid_global.get_position()
    bbox_mask = ax_mmd.get_position()

    # Calculate x position for vertical line (midpoint of the gap)
    line_x = (bbox_image.x1 + bbox_mask.x0) / 2

    # Draw vertical line spanning the plot area - thick and black
    line = plt.Line2D(
        [line_x, line_x],
        [0.12, 0.88],  # Shortened to align with plot area
        transform=fig.transFigure,
        color="black",
        linestyle="-",
        linewidth=2.0,
        alpha=1.0,
    )
    fig.add_artist(line)

    # =========================================================================
    # Unified legend at bottom
    # =========================================================================
    create_unified_legend(
        fig,
        prediction_types=prediction_types,
        lp_norms=lp_norms,
        include_baseline=baseline_kid is not None or baseline_lpips is not None or baseline_mmd is not None,
        include_lp_norms=not show_subplot_legends,
        include_best=not show_subplot_legends,
        ncol=legend_ncol,
        bbox_y=-0.04,
    )

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = output_dir / f"icip2026_similarity_metrics_with_masks.{fmt}"
        fig.savefig(
            output_path,
            dpi=PLOT_SETTINGS["dpi_print"],
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print(f"Saved: {output_path}")

    plt.close(fig)


def create_compact_figure(
    df_global: pd.DataFrame,
    df_zbin: pd.DataFrame,
    output_dir: Path,
    test_csv: Path | None = None,
    comparison_results: dict[str, Any] | None = None,
    baseline_kid: float | None = None,
    baseline_lpips: float | None = None,
    formats: list[str] = ["pdf", "png"],
) -> None:
    """Create a compact single-column figure for IEEE single column.

    Layout:
        +---------------------------+
        |  Representative Images    |
        +---------------------------+
        |  Per-zbin KID             |
        +---------------------------+
        |  Global KID Boxplots      |
        +---------------------------+
        |       Legend              |
        +---------------------------+

    Args:
        df_global: DataFrame with global metric values.
        df_zbin: DataFrame with per-zbin metric values.
        output_dir: Output directory.
        test_csv: Path to test.csv for images.
        comparison_results: Statistical comparison results.
        baseline_kid: Real baseline KID value.
        baseline_lpips: Real baseline LPIPS value.
        formats: Output formats.
    """
    apply_ieee_style()

    # Single column width
    fig_width = PLOT_SETTINGS["figure_width_single"]
    fig_height = fig_width * 1.8

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1, 0.8],
        hspace=0.3,
        left=0.15,
        right=0.95,
        top=0.92,
        bottom=0.15,
    )

    zbins = np.array(sorted(df_zbin["zbin"].unique()))

    # Panel A: Per-zbin KID
    ax_zbin = fig.add_subplot(gs[0])
    plot_zbin_multiexperiment(
        df_zbin,
        "kid_zbin",
        output_dir=None,
        baseline_real=baseline_kid,
        show_legend=False,
        ax=ax_zbin,
    )
    ax_zbin.text(
        -0.15, 1.05, "(A)",
        transform=ax_zbin.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )

    # Add images
    if test_csv is not None:
        add_representative_images(ax_zbin, test_csv, zbins, image_step=8)

    # Panel B: Global KID boxplots
    ax_global = fig.add_subplot(gs[1])
    kid_comparison = None
    if comparison_results is not None:
        kid_comparison = comparison_results.get("kid_global", {}).get("between_group")

    plot_global_boxplots(
        df_global,
        "kid_global",
        output_dir=None,
        comparison_results=kid_comparison,
        baseline_real=baseline_kid,
        show_significance=True,
        show_effect_sizes=False,  # Compact version skips effect sizes
        show_points=True,
        highlight_best=True,
        ax=ax_global,
    )
    ax_global.text(
        -0.15, 1.05, "(B)",
        transform=ax_global.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )

    # Simplified legend
    prediction_types = sorted(df_global["prediction_type"].unique())
    lp_norms = sorted(df_global["lp_norm"].unique())

    handles = []
    labels = []

    for pred_type in prediction_types:
        color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")
        pred_label = PREDICTION_TYPE_LABELS_SHORT.get(pred_type, pred_type)
        patch = mpatches.Patch(facecolor=color, edgecolor="black", alpha=0.8)
        handles.append(patch)
        labels.append(pred_label)

    for lp_norm in lp_norms:
        linestyle = LP_NORM_STYLES.get(lp_norm, "-")
        lp_label = LP_NORM_LABELS.get(lp_norm, f"Lp={lp_norm}")
        line = Line2D([0], [0], color="black", linestyle=linestyle, linewidth=1.5)
        handles.append(line)
        labels.append(lp_label)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=6,
        fontsize=PLOT_SETTINGS["legend_fontsize"] - 1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = output_dir / f"icip2026_kid_compact.{fmt}"
        fig.savefig(
            output_path,
            dpi=PLOT_SETTINGS["dpi_print"],
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print(f"Saved: {output_path}")

    plt.close(fig)


def create_global_only_figure(
    df_global: pd.DataFrame,
    output_dir: Path,
    comparison_results: dict | None = None,
    baseline_kid: float | None = None,
    baseline_lpips: float | None = None,
    formats: list[str] = ["pdf", "png"],
) -> None:
    """Create publication figure with global boxplots only (no zbin data).

    Creates a 1x2 or 1x3 figure showing global KID, LPIPS, and optionally FID
    boxplots when per-zbin data is not available.

    Args:
        df_global: DataFrame with global metrics.
        output_dir: Directory for output files.
        comparison_results: Statistical comparison results.
        baseline_kid: Real baseline KID value.
        baseline_lpips: Real baseline LPIPS value.
        formats: Output formats.
    """
    apply_ieee_style()

    # Determine which metrics are available
    metrics = []
    if "kid_global" in df_global.columns:
        metrics.append(("kid_global", "KID", baseline_kid))
    if "lpips_global" in df_global.columns:
        metrics.append(("lpips_global", "LPIPS", baseline_lpips))
    if "fid_global" in df_global.columns:
        metrics.append(("fid_global", "FID", None))

    if not metrics:
        print("  No metrics available for global figure")
        return

    n_metrics = len(metrics)

    # Figure size
    fig_width = PLOT_SETTINGS["figure_width_double"]
    fig_height = fig_width * 0.4

    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, fig_height))
    if n_metrics == 1:
        axes = [axes]

    panel_labels = ["(A)", "(B)", "(C)"]

    for i, (metric_col, metric_name, baseline) in enumerate(metrics):
        ax = axes[i]

        metric_comparison = None
        if comparison_results is not None:
            metric_comparison = comparison_results.get(metric_col, {}).get("between_group")

        plot_global_boxplots(
            df_global,
            metric_col,
            output_dir=None,
            comparison_results=metric_comparison,
            baseline_real=baseline,
            show_significance=True,
            show_effect_sizes=False,
            show_points=True,
            highlight_best=True,
            ax=ax,
        )

        ax.text(
            -0.12, 1.05, panel_labels[i],
            transform=ax.transAxes,
            fontsize=PLOT_SETTINGS["panel_label_fontsize"],
            fontweight="bold",
        )

    # Unified legend
    prediction_types = sorted(df_global["prediction_type"].unique())
    lp_norms = sorted(df_global["lp_norm"].unique())

    handles = []
    labels = []

    for pred_type in prediction_types:
        color = PREDICTION_TYPE_COLORS.get(pred_type, "#888888")
        pred_label = PREDICTION_TYPE_LABELS_SHORT.get(pred_type, pred_type)
        patch = mpatches.Patch(facecolor=color, edgecolor="black", alpha=0.8)
        handles.append(patch)
        labels.append(pred_label)

    # Add baseline to legend
    if any(b is not None for _, _, b in metrics):
        line = Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5)
        handles.append(line)
        labels.append("Baseline")

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(handles),
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = output_dir / f"icip2026_global_only.{fmt}"
        fig.savefig(
            output_path,
            dpi=PLOT_SETTINGS["dpi_print"],
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print(f"Saved: {output_path}")

    plt.close(fig)


def generate_latex_metrics_table(
    df_global: pd.DataFrame,
    df_mask_global: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    metrics: list[str] | None = None,
    caption: str = "Global similarity metrics by prediction type and $L_p$ norm.",
    label: str = "tab:similarity_metrics",
    highlight_best: bool = True,
) -> str:
    """Generate a LaTeX table with global metrics (KID, LPIPS, MMD-MF).

    Creates a publication-ready LaTeX table with:
    - Rows: Prediction type × Lp norm configurations
    - Columns: Selected metrics (mean ± std)
    - Optional: Best value per column highlighted in bold

    Args:
        df_global: DataFrame with image metrics (kid_global, lpips_global).
        df_mask_global: DataFrame with mask metrics (mmd_mf_global). Optional.
        output_dir: Output directory to save .tex file. If None, only returns string.
        metrics: List of metrics to include. Default: ["kid_global", "lpips_global", "mmd_mf_global"].
        caption: LaTeX table caption.
        label: LaTeX table label for referencing.
        highlight_best: Whether to bold the best (lowest) value per metric.

    Returns:
        LaTeX table as a string.
    """
    if metrics is None:
        metrics = ["kid_global", "lpips_global", "mmd_mf_global"]

    # Merge image and mask metrics if both provided
    if df_mask_global is not None:
        # Merge on common columns
        merge_cols = ["prediction_type", "lp_norm", "replica_id"]
        if "self_cond_p" in df_global.columns and "self_cond_p" in df_mask_global.columns:
            merge_cols.append("self_cond_p")

        df_combined = pd.merge(
            df_global,
            df_mask_global[merge_cols + ["mmd_mf_global"]],
            on=merge_cols,
            how="outer",
        )
    else:
        df_combined = df_global.copy()

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in df_combined.columns]
    if not available_metrics:
        raise ValueError(f"No valid metrics found. Available columns: {df_combined.columns.tolist()}")

    # Aggregate by prediction_type and lp_norm
    agg_funcs = {m: ["mean", "std", "count"] for m in available_metrics}
    stats = df_combined.groupby(["prediction_type", "lp_norm"]).agg(agg_funcs)

    # Define ordering
    pred_order = ["epsilon", "velocity", "x0"]
    lp_order = [1.5, 2.0, 2.5]

    # Find best (lowest) values per metric
    best_values = {}
    if highlight_best:
        for metric in available_metrics:
            best_values[metric] = stats[(metric, "mean")].min()

    # Metric display names
    metric_names = {
        "kid_global": "KID",
        "lpips_global": "LPIPS",
        "mmd_mf_global": "MMD-MF",
        "fid_global": "FID",
    }

    # Prediction type display names
    pred_names = {
        "epsilon": r"$\epsilon$",
        "velocity": r"$\mathbf{v}$",
        "x0": r"$\mathbf{x}_0$",
    }

    # Build LaTeX table
    lines = []

    # Table header
    n_metrics = len(available_metrics)
    col_spec = "ll" + "c" * n_metrics
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{" + caption + "}")
    lines.append(r"  \label{" + label + "}")
    lines.append(r"  \begin{tabular}{" + col_spec + "}")
    lines.append(r"    \toprule")

    # Column headers
    header_cols = ["Pred. Type", "$L_p$"] + [metric_names.get(m, m) for m in available_metrics]
    lines.append("    " + " & ".join(header_cols) + r" \\")
    lines.append(r"    \midrule")

    # Data rows
    for pred_type in pred_order:
        for i, lp_norm in enumerate(lp_order):
            try:
                row_data = stats.loc[(pred_type, lp_norm)]
            except KeyError:
                continue

            # Prediction type name (only on first row of group)
            if i == 0:
                pred_display = pred_names.get(pred_type, pred_type)
            else:
                pred_display = ""

            # Lp norm
            lp_display = f"{lp_norm:.1f}"

            # Metric values
            metric_cells = []
            for metric in available_metrics:
                mean_val = row_data[(metric, "mean")]
                std_val = row_data[(metric, "std")]

                # Format based on metric scale
                if metric == "kid_global":
                    cell = f"{mean_val:.3f} $\\pm$ {std_val:.3f}"
                elif metric == "lpips_global":
                    cell = f"{mean_val:.3f} $\\pm$ {std_val:.3f}"
                elif metric == "mmd_mf_global":
                    cell = f"{mean_val:.2f} $\\pm$ {std_val:.2f}"
                else:
                    cell = f"{mean_val:.3f} $\\pm$ {std_val:.3f}"

                # Highlight best
                if highlight_best and abs(mean_val - best_values[metric]) < 1e-9:
                    cell = r"\textbf{" + cell + "}"

                metric_cells.append(cell)

            row = f"    {pred_display} & {lp_display} & " + " & ".join(metric_cells) + r" \\"
            lines.append(row)

        # Add midrule between prediction types (except after the last one)
        if pred_type != pred_order[-1]:
            lines.append(r"    \midrule")

    # Table footer
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    # Save to file if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "metrics_table.tex"
        with open(output_path, "w") as f:
            f.write(latex_str)
        print(f"Saved LaTeX table: {output_path}")

    return latex_str


def generate_plots_from_config(
    config_path: Path | str,
    output_subdir: str | None = None,
    add_mask_metrics: bool = False,
    mask_metrics_dir: Path | str | None = None,
) -> None:
    """Generate all ICIP 2026 plots from a config YAML file.

    This function loads all necessary data from the paths specified in the config:
    - Global metrics CSV from output_dir
    - Per-zbin metrics CSV from output_dir
    - Baseline values from output_dir/baseline_real_vs_real.json
    - Comparison results from output_dir/similarity_metrics_comparison.json
    - Representative images from cache_dir/test.csv

    When add_mask_metrics=True, also loads:
    - Mask metrics from mask_metrics_dir (or output_dir/../mask_metrics/)
    - Creates a combined 2x3 figure with image and mask metrics

    Args:
        config_path: Path to the YAML configuration file.
        output_subdir: Optional subdirectory within output_dir for plots
                       (default: "plots").
        add_mask_metrics: If True, include mask morphology metrics (MMD-MF and
                          per-feature Wasserstein heatmap) in a 2x3 layout.
        mask_metrics_dir: Directory containing mask metrics CSVs. If None,
                          defaults to {output_dir}/../mask_metrics/.

    Example:
        >>> generate_plots_from_config("config/icip2026.yaml")
        # Generates plots in {output_dir}/plots/

        >>> generate_plots_from_config("config/icip2026.yaml", add_mask_metrics=True)
        # Generates 2x3 figure with image + mask metrics
    """
    import json
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract paths
    paths = config.get("paths", {})
    cache_dir = Path(paths.get("cache_dir", ""))
    output_dir = Path(paths.get("output_dir", ""))

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Determine plots output directory
    if output_subdir is None:
        output_subdir = "plots"
    plots_dir = output_dir / output_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Extract plotting config
    plot_config = config.get("plotting", {})
    formats = plot_config.get("formats", ["pdf", "png"])

    # Image settings
    images_config = plot_config.get("images", {})
    show_images = images_config.get("enabled", True)
    image_x_offset = images_config.get("x_offset", None)
    image_y_offset = images_config.get("y_offset", None)

    # Override PLOT_SETTINGS with config values for boxplots
    boxplot_config = plot_config.get("boxplot", {})
    if "width_factor" in boxplot_config:
        PLOT_SETTINGS["boxplot_width_factor"] = boxplot_config["width_factor"]
    if "whis" in boxplot_config:
        whis = boxplot_config["whis"]
        PLOT_SETTINGS["boxplot_whis"] = tuple(whis) if isinstance(whis, list) else whis

    # Override image settings
    if images_config.get("step"):
        PLOT_SETTINGS["image_step"] = images_config["step"]
    if images_config.get("zoom"):
        PLOT_SETTINGS["image_zoom"] = images_config["zoom"]
    if images_config.get("x_offset") is not None:
        PLOT_SETTINGS["image_x_offset"] = images_config["x_offset"]
    if images_config.get("y_offset") is not None:
        PLOT_SETTINGS["image_y_offset"] = images_config["y_offset"]

    # ICIP2026 figure specific settings
    icip_config = plot_config.get("icip2026_figure", {})
    show_subplot_legends = icip_config.get("show_subplot_legends", True)
    show_effect_sizes = plot_config.get("show_effect_sizes", False)

    print("=" * 70)
    print("ICIP 2026 PLOT GENERATION FROM CONFIG")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Output dir: {output_dir}")
    print(f"Plots dir: {plots_dir}")
    print(f"Cache dir: {cache_dir}")
    print("=" * 70)

    # Determine the similarity metrics directory
    # Support both direct files in output_dir and files in similarity_metrics subdirectory
    sim_metrics_dir = output_dir
    if not (output_dir / "similarity_metrics_global.csv").exists():
        # Check if files are in similarity_metrics subdirectory
        if (output_dir / "similarity_metrics" / "similarity_metrics_global.csv").exists():
            sim_metrics_dir = output_dir / "similarity_metrics"
            print(f"Using similarity metrics from subdirectory: {sim_metrics_dir}")

    # Load global metrics CSV
    global_csv = sim_metrics_dir / "similarity_metrics_global.csv"
    if not global_csv.exists():
        raise FileNotFoundError(f"Global metrics CSV not found: {global_csv}")
    df_global = pd.read_csv(global_csv)
    print(f"Loaded global metrics: {len(df_global)} rows")

    # Filter by self_cond_p if specified in config
    ablation_config = config.get("ablation", {})
    include_filters = ablation_config.get("include", [])
    if include_filters and "self_cond_p" in df_global.columns:
        # Extract self_cond_p values to include
        sc_values = [f.get("self_cond_p") for f in include_filters if "self_cond_p" in f]
        if sc_values:
            df_global = df_global[df_global["self_cond_p"].isin(sc_values)]
            print(f"  Filtered to self_cond_p={sc_values}: {len(df_global)} rows")

    # Load per-zbin metrics CSV
    zbin_csv = sim_metrics_dir / "similarity_metrics_zbin.csv"
    df_zbin = None
    if zbin_csv.exists():
        df_zbin = pd.read_csv(zbin_csv)
        print(f"Loaded per-zbin metrics: {len(df_zbin)} rows")
    else:
        print("Warning: Per-zbin metrics CSV not found, skipping zbin plots")

    # Load baseline values
    baseline_json = sim_metrics_dir / "baseline_real_vs_real.json"
    baseline_kid = None
    baseline_kid_std = None
    baseline_lpips = None
    baseline_lpips_std = None

    if baseline_json.exists():
        with open(baseline_json) as f:
            baseline_data = json.load(f)
        baseline_kid = baseline_data.get("kid", {}).get("mean")
        baseline_kid_std = baseline_data.get("kid", {}).get("std")
        baseline_lpips = baseline_data.get("lpips", {}).get("mean")
        baseline_lpips_std = baseline_data.get("lpips", {}).get("std")
        print(f"Loaded baseline: KID={baseline_kid:.6f}, LPIPS={baseline_lpips:.6f}" if baseline_kid and baseline_lpips else "Loaded baseline data")
    else:
        print("Warning: Baseline JSON not found, plots will not show baseline")

    # Load comparison results (or compute if not found)
    comparison_json = sim_metrics_dir / "similarity_metrics_comparison.json"
    comparison_results = None
    if comparison_json.exists():
        with open(comparison_json) as f:
            comparison_results = json.load(f)
        print("Loaded statistical comparison results")
    else:
        # Auto-compute comparison results from global metrics
        print("Comparison JSON not found, computing statistical analysis...")
        try:
            from ..statistics.comparison import run_all_comparisons

            # Determine available metrics
            available_metrics = [m for m in ["kid_global", "fid_global", "lpips_global"]
                                if m in df_global.columns]

            if available_metrics:
                raw_results = run_all_comparisons(df_global, metrics=available_metrics)

                # Convert to JSON-serializable format
                comparison_results = {}
                for metric, data in raw_results.items():
                    comparison_results[metric] = {
                        "within_group": data["within_group"].to_dict(orient="records")
                            if hasattr(data["within_group"], "to_dict") else data["within_group"],
                        "between_group": {
                            k: (float(v) if hasattr(v, '__float__') and not isinstance(v, (bool, str)) else v)
                            for k, v in data["between_group"].items()
                        },
                    }

                # Save for future use
                with open(comparison_json, "w") as f:
                    json.dump(comparison_results, f, indent=2, default=str)
                print(f"  Computed and saved comparison results to {comparison_json}")
            else:
                print("  Warning: No valid metrics found for comparison")
        except Exception as e:
            print(f"  Warning: Failed to compute comparison: {e}")

    # Determine test.csv path for representative images
    test_csv = cache_dir / "test.csv" if cache_dir.exists() else None
    if test_csv and not test_csv.exists():
        test_csv = None
        print("Warning: test.csv not found, skipping representative images")

    # Load mask metrics if requested
    df_mask_global = None
    df_mask_wasserstein = None
    mask_comparison_results = None

    if add_mask_metrics:
        print("\n--- Loading mask metrics ---")

        # Determine mask metrics directory
        if mask_metrics_dir is not None:
            mask_dir = Path(mask_metrics_dir)
        else:
            # Default: sibling directory to sim_metrics_dir (e.g., .../self_cond_p_0.0/mask_metrics/)
            # or sibling to output_dir if sim_metrics_dir is the same as output_dir
            if sim_metrics_dir != output_dir:
                # Files are in similarity_metrics subdirectory, so look for mask_metrics sibling
                mask_dir = sim_metrics_dir.parent / "mask_metrics"
            else:
                mask_dir = output_dir / "mask_metrics"

        if not mask_dir.exists():
            print(f"Warning: Mask metrics directory not found: {mask_dir}")
            print("  Skipping mask metrics in combined figure")
            add_mask_metrics = False
        else:
            # Load mask global metrics
            mask_global_csv = mask_dir / "mask_metrics_global.csv"
            if mask_global_csv.exists():
                df_mask_global = pd.read_csv(mask_global_csv)
                print(f"  Loaded mask global metrics: {len(df_mask_global)} rows")

                # Filter by self_cond_p if needed
                if include_filters and "self_cond_p" in df_mask_global.columns:
                    sc_values = [f.get("self_cond_p") for f in include_filters if "self_cond_p" in f]
                    if sc_values:
                        df_mask_global = df_mask_global[df_mask_global["self_cond_p"].isin(sc_values)]
                        print(f"    Filtered to self_cond_p={sc_values}: {len(df_mask_global)} rows")
            else:
                print(f"  Warning: {mask_global_csv} not found")

            # Load mask Wasserstein features
            mask_wasserstein_csv = mask_dir / "mask_wasserstein_features.csv"
            if mask_wasserstein_csv.exists():
                df_mask_wasserstein = pd.read_csv(mask_wasserstein_csv)
                print(f"  Loaded mask Wasserstein features: {len(df_mask_wasserstein)} rows")

                # Filter by self_cond_p if needed
                if include_filters and "self_cond_p" in df_mask_wasserstein.columns:
                    sc_values = [f.get("self_cond_p") for f in include_filters if "self_cond_p" in f]
                    if sc_values:
                        df_mask_wasserstein = df_mask_wasserstein[df_mask_wasserstein["self_cond_p"].isin(sc_values)]
            else:
                print(f"  Warning: {mask_wasserstein_csv} not found")

            # Load mask comparison results
            mask_comparison_json = mask_dir / "mask_metrics_comparison.json"
            if mask_comparison_json.exists():
                with open(mask_comparison_json) as f:
                    mask_comparison_data = json.load(f)
                # Extract between_group comparison for mmd_mf_global
                if "mmd_mf_global" in mask_comparison_data:
                    mask_comparison_results = mask_comparison_data["mmd_mf_global"].get("between_group")
                print("  Loaded mask comparison results")
            else:
                print(f"  Warning: {mask_comparison_json} not found")

            # Check if we have the required data
            if df_mask_global is None or df_mask_wasserstein is None:
                print("  Warning: Missing required mask metrics data, disabling mask metrics")
                add_mask_metrics = False

    # Generate individual metric plots
    print("\n--- Generating individual plots ---")

    # Global boxplots
    for metric in ["kid_global", "lpips_global", "fid_global"]:
        if metric in df_global.columns:
            try:
                from .global_comparison import plot_global_boxplots

                metric_comparison = None
                if comparison_results and metric in comparison_results:
                    metric_comparison = comparison_results[metric].get("between_group")

                baseline = baseline_kid if "kid" in metric else baseline_lpips if "lpips" in metric else None

                plot_global_boxplots(
                    df_global,
                    metric_col=metric,
                    output_dir=plots_dir,
                    comparison_results=metric_comparison,
                    baseline_real=baseline,
                    formats=formats,
                )
                print(f"  Generated {metric} boxplot")
            except Exception as e:
                print(f"  Warning: Failed to generate {metric} boxplot: {e}")

    # Per-zbin line plots
    if df_zbin is not None:
        for metric in ["kid_zbin", "lpips_zbin"]:
            if metric in df_zbin.columns:
                try:
                    from .zbin_multiexp import plot_zbin_multiexperiment

                    baseline = baseline_kid if "kid" in metric else baseline_lpips if "lpips" in metric else None

                    plot_zbin_multiexperiment(
                        df_zbin,
                        metric_col=metric,
                        output_dir=plots_dir,
                        baseline_real=baseline,
                        test_csv=test_csv if show_images else None,
                        show_images=show_images,
                        formats=formats,
                    )
                    print(f"  Generated {metric} zbin plot")
                except Exception as e:
                    print(f"  Warning: Failed to generate {metric} zbin plot: {e}")

    # Generate ICIP 2026 publication figure
    if plot_config.get("icip2026_figure", {}).get("enabled", True):
        print("\n--- Generating ICIP 2026 publication figure ---")

        # Extract ICIP figure-specific settings
        show_images_on_lpips = icip_config.get("show_images_on_lpips", False)
        legend_ncol = icip_config.get("legend_ncol", 8)
        aspect_ratio = icip_config.get("aspect_ratio", 0.55)

        if df_zbin is not None:
            # Generate 2x3 figure with mask metrics if requested
            if add_mask_metrics and df_mask_global is not None and df_mask_wasserstein is not None:
                try:
                    create_icip2026_figure_with_masks(
                        df_global=df_global,
                        df_zbin=df_zbin,
                        df_mask_global=df_mask_global,
                        df_mask_wasserstein=df_mask_wasserstein,
                        output_dir=plots_dir,
                        test_csv=test_csv if show_images else None,
                        comparison_results=comparison_results,
                        mask_comparison_results=mask_comparison_results,
                        baseline_kid=baseline_kid,
                        baseline_kid_std=baseline_kid_std,
                        baseline_lpips=baseline_lpips,
                        baseline_lpips_std=baseline_lpips_std,
                        baseline_mmd=0.09,  # No baseline for mask metrics currently
                        baseline_mmd_std=0.02,
                        formats=formats,
                        show_images=show_images and test_csv is not None,
                        show_images_on_lpips=show_images_on_lpips,
                        image_x_offset=image_x_offset,
                        image_y_offset=image_y_offset,
                        show_subplot_legends=show_subplot_legends,
                        show_effect_sizes=show_effect_sizes,
                        legend_ncol=legend_ncol,
                        aspect_ratio=aspect_ratio,
                    )
                    print("  Generated ICIP 2026 2x3 figure with mask metrics")
                except Exception as e:
                    print(f"  Warning: Failed to generate ICIP 2026 figure with masks: {e}")
                    import traceback
                    traceback.print_exc()

            # Full 2x2 figure with zbin plots (always generate as fallback/alternative)
            try:
                create_icip2026_figure(
                    df_global=df_global,
                    df_zbin=df_zbin,
                    output_dir=plots_dir,
                    test_csv=test_csv if show_images else None,
                    comparison_results=comparison_results,
                    baseline_kid=baseline_kid,
                    baseline_kid_std=baseline_kid_std,
                    baseline_lpips=baseline_lpips,
                    baseline_lpips_std=baseline_lpips_std,
                    formats=formats,
                    show_images=show_images and test_csv is not None,
                    show_images_on_lpips=show_images_on_lpips,
                    image_x_offset=image_x_offset,
                    image_y_offset=image_y_offset,
                    show_subplot_legends=show_subplot_legends,
                    show_effect_sizes=show_effect_sizes,
                    legend_ncol=legend_ncol,
                    aspect_ratio=aspect_ratio,
                )
                print("  Generated ICIP 2026 2x2 figure")
            except Exception as e:
                print(f"  Warning: Failed to generate ICIP 2026 figure: {e}")
                import traceback
                traceback.print_exc()

            # Compact single-column version
            try:
                create_compact_figure(
                    df_global=df_global,
                    df_zbin=df_zbin,
                    output_dir=plots_dir,
                    test_csv=test_csv if show_images else None,
                    comparison_results=comparison_results,
                    baseline_kid=baseline_kid,
                    baseline_lpips=baseline_lpips,
                    formats=formats,
                )
                print("  Generated compact single-column figure")
            except Exception as e:
                print(f"  Warning: Failed to generate compact figure: {e}")
        else:
            # Global-only figure (no zbin data available)
            print("  No per-zbin data available, generating global-only figure...")
            try:
                create_global_only_figure(
                    df_global=df_global,
                    output_dir=plots_dir,
                    comparison_results=comparison_results,
                    baseline_kid=baseline_kid,
                    baseline_lpips=baseline_lpips,
                    formats=formats,
                )
                print("  Generated global-only publication figure")
            except Exception as e:
                print(f"  Warning: Failed to generate global-only figure: {e}")
                import traceback
                traceback.print_exc()

    # Generate summary table
    try:
        from .global_comparison import plot_metric_summary_table

        metrics = ["kid_global", "lpips_global", "fid_global"]
        available = [m for m in metrics if m in df_global.columns]
        if available:
            plot_metric_summary_table(df_global, metrics=available, output_dir=plots_dir, formats=formats)
            print("  Generated summary table")
    except Exception as e:
        print(f"  Warning: Failed to generate summary table: {e}")

    # Generate LaTeX metrics table
    print("\n--- Generating LaTeX metrics table ---")
    try:
        latex_metrics = ["kid_global", "lpips_global"]
        if add_mask_metrics and df_mask_global is not None:
            latex_metrics.append("mmd_mf_global")

        latex_str = generate_latex_metrics_table(
            df_global=df_global,
            df_mask_global=df_mask_global if add_mask_metrics else None,
            output_dir=plots_dir,
            metrics=latex_metrics,
            caption="Global similarity metrics by prediction type and $L_p$ norm. Best values per metric are highlighted in bold.",
            label="tab:similarity_metrics",
            highlight_best=True,
        )
        print("  Generated LaTeX table: metrics_table.tex")
    except Exception as e:
        print(f"  Warning: Failed to generate LaTeX table: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"PLOTS SAVED TO: {plots_dir}")
    print("=" * 70)
