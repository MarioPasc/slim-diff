"""Plotting utilities for similarity metrics.

ICIP 2026 version with:
- Paul Tol colorblind-friendly palettes
- LaTeX rendering for labels
- Representative MRI images above plot
- Boxplots instead of bar charts
- IEEE publication-ready formatting
"""

from .settings import (
    PLOT_SETTINGS,
    PREDICTION_TYPE_COLORS,
    PREDICTION_TYPE_LABELS,
    LP_NORM_LABELS,
    LP_NORM_STYLES,
    LP_NORM_LINESTYLES,
    LP_NORM_HATCHES,
    LP_NORM_MARKERS,
    apply_ieee_style,
    format_metric_label,
    get_significance_stars,
)
from .zbin_multiexp import (
    plot_zbin_multiexperiment,
    plot_zbin_by_prediction_type,
    add_representative_images,
    load_representative_images,
    create_legend_figure,
)
from .global_comparison import (
    plot_global_comparison,
    plot_global_boxplots,
    plot_metric_summary_table,
)
from .icip2026_figure import (
    create_icip2026_figure,
    create_compact_figure,
    create_unified_legend,
    create_subplot_legend_lines,
    create_subplot_legend_boxes,
    generate_plots_from_config,
    generate_latex_metrics_table,
)
from .nn_comparison import (
    plot_nn_boxplots,
    plot_nn_zbin_lines,
    plot_nn_histogram,
    create_nn_summary_figure,
)

__all__ = [
    # Settings
    "PLOT_SETTINGS",
    "PREDICTION_TYPE_COLORS",
    "PREDICTION_TYPE_LABELS",
    "LP_NORM_LABELS",
    "LP_NORM_STYLES",
    "LP_NORM_LINESTYLES",
    "LP_NORM_HATCHES",
    "LP_NORM_MARKERS",
    "apply_ieee_style",
    "format_metric_label",
    "get_significance_stars",
    # Per-zbin plots
    "plot_zbin_multiexperiment",
    "plot_zbin_by_prediction_type",
    "add_representative_images",
    "load_representative_images",
    "create_legend_figure",
    # Global comparison plots
    "plot_global_comparison",
    "plot_global_boxplots",
    "plot_metric_summary_table",
    # ICIP 2026 figures
    "create_icip2026_figure",
    "create_compact_figure",
    "create_unified_legend",
    "create_subplot_legend_lines",
    "create_subplot_legend_boxes",
    "generate_plots_from_config",
    "generate_latex_metrics_table",
    # NN distance plots
    "plot_nn_boxplots",
    "plot_nn_zbin_lines",
    "plot_nn_histogram",
    "create_nn_summary_figure",
]
