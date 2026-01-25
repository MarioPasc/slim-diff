"""Main ICIP 2026 publication figure generation.

Creates a 2x2 subplot layout with:
- Top row: Per-zbin KID and LPIPS with representative MRI images
- Bottom row: Global KID and LPIPS boxplots with significance
- Unified legend outside all subplots

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


def create_unified_legend(
    fig: plt.Figure,
    prediction_types: list[str] | None = None,
    lp_norms: list[float] | None = None,
    include_baseline: bool = True,
    include_best: bool = True,
) -> None:
    """Create unified legend below all subplots.

    Args:
        fig: Matplotlib figure.
        prediction_types: List of prediction types (default: epsilon, velocity, x0).
        lp_norms: List of Lp norms (default: 1.5, 2.0, 2.5).
        include_baseline: Whether to include baseline in legend.
        include_best: Whether to include best marker in legend.
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

    # Lp norm legend entries (line styles with markers)
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

    # Hatch pattern legend (for boxplots)
    for lp_norm in lp_norms:
        hatch = LP_NORM_HATCHES.get(lp_norm, None)
        lp_label = LP_NORM_LABELS.get(lp_norm, f"Lp={lp_norm}") + " (box)"
        patch = mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            hatch=hatch if hatch else "",
            linewidth=0.5,
        )
        handles.append(patch)
        labels.append(lp_label)

    # Baseline
    if include_baseline:
        line = Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5)
        handles.append(line)
        labels.append("Real baseline")

    # Best marker
    if include_best:
        star = Line2D(
            [0], [0],
            marker="*",
            color="gold",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="None",
        )
        handles.append(star)
        labels.append("Best")

    # Place legend below figure
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(handles), 8),
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=PLOT_SETTINGS["legend_columnspacing"],
        handletextpad=PLOT_SETTINGS["legend_handletextpad"],
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
) -> None:
    """Create the main 2x2 publication figure for ICIP 2026.

    Layout:
        +--------------------------------------------------+
        |        Representative MRI Images (6 total)       |
        +--------------------------------------------------+
        | (A) Per-zbin KID        | (B) Per-zbin LPIPS     |
        +-------------------------+------------------------+
        | (C) Global KID Boxplots | (D) Global LPIPS       |
        +-------------------------+------------------------+
        |           Unified Legend (outside, bottom)       |
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
        show_images: Whether to show representative MRI images.
    """
    apply_ieee_style()

    # Figure dimensions (IEEE double column)
    fig_width = PLOT_SETTINGS["figure_width_double"]
    fig_height = fig_width * 1.0  # Tall for 2x2 + images + legend

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create GridSpec with space for images and legend
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1, 1],
        hspace=0.35,
        wspace=0.25,
        left=0.08,
        right=0.92,
        top=0.88 if show_images else 0.95,
        bottom=0.12,  # Space for legend
    )

    # Get z-bins for image positioning
    zbins = np.array(sorted(df_zbin["zbin"].unique()))

    # Panel A: Per-zbin KID (top-left)
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
        -0.12, 1.05, "(A)",
        transform=ax_kid_zbin.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )

    # Add representative images above panel A
    if show_images and test_csv is not None:
        add_representative_images(
            ax_kid_zbin,
            test_csv,
            zbins,
            image_step=PLOT_SETTINGS["image_step"],
        )

    # Panel B: Per-zbin LPIPS (top-right)
    ax_lpips_zbin = fig.add_subplot(gs[0, 1])
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
        -0.12, 1.05, "(B)",
        transform=ax_lpips_zbin.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )

    # Extract comparison results for each metric
    kid_comparison = None
    lpips_comparison = None
    if comparison_results is not None:
        kid_comparison = comparison_results.get("kid_global", {}).get("between_group")
        lpips_comparison = comparison_results.get("lpips_global", {}).get("between_group")

    # Panel C: Global KID boxplots (bottom-left)
    ax_kid_global = fig.add_subplot(gs[1, 0])
    plot_global_boxplots(
        df_global,
        "kid_global",
        output_dir=None,
        comparison_results=kid_comparison,
        baseline_real=baseline_kid,
        baseline_std=baseline_kid_std,
        show_significance=True,
        show_effect_sizes=True,
        show_points=True,
        highlight_best=True,
        ax=ax_kid_global,
    )
    ax_kid_global.text(
        -0.12, 1.05, "(C)",
        transform=ax_kid_global.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )

    # Panel D: Global LPIPS boxplots (bottom-right)
    ax_lpips_global = fig.add_subplot(gs[1, 1])
    if "lpips_global" in df_global.columns:
        plot_global_boxplots(
            df_global,
            "lpips_global",
            output_dir=None,
            comparison_results=lpips_comparison,
            baseline_real=baseline_lpips,
            baseline_std=baseline_lpips_std,
            show_significance=True,
            show_effect_sizes=True,
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
        -0.12, 1.05, "(D)",
        transform=ax_lpips_global.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )

    # Get unique values for legend
    prediction_types = sorted(df_global["prediction_type"].unique())
    lp_norms = sorted(df_global["lp_norm"].unique())

    # Unified legend at bottom
    create_unified_legend(
        fig,
        prediction_types=prediction_types,
        lp_norms=lp_norms,
        include_baseline=baseline_kid is not None or baseline_lpips is not None,
        include_best=True,
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
