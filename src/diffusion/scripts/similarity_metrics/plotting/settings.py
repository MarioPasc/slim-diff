"""Publication-ready plot settings for ICIP 2026.

IEEE-compliant settings with Paul Tol colorblind-friendly palettes
and LaTeX rendering support.

References:
    - Paul Tol's color schemes: https://personal.sron.nl/~pault/
    - IEEE publication guidelines
    - scienceplots: https://github.com/garrettj403/SciencePlots
"""

from __future__ import annotations

# =============================================================================
# Paul Tol Color Palettes (SRON - colorblind safe)
# =============================================================================

PAUL_TOL_BRIGHT = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

PAUL_TOL_HIGH_CONTRAST = {
    "blue": "#004488",
    "yellow": "#DDAA33",
    "red": "#BB5566",
}

PAUL_TOL_MUTED = [
    "#CC6677",  # rose
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#117733",  # green
    "#88CCEE",  # cyan
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # purple
]

PAU_TOL_TENSORBOARD = {
    "orange": '#EE7733',
    "blue": '#0077BB',
    "cyan": '#33BBEE',
    "magenta": '#EE3377',
    "red": '#CC3311',
    "green": '#009988',
    "wine": '#882255',
    "grey": '#BBBBBB',
}



# =============================================================================
# Prediction Type Visual Encoding
# =============================================================================

PREDICTION_TYPE_COLORS = {
    "epsilon": PAU_TOL_TENSORBOARD["orange"],    # #EE7733
    "velocity": PAU_TOL_TENSORBOARD["green"],  # #009988
    "x0": PAU_TOL_TENSORBOARD["wine"],         # #882255
}

# LaTeX-formatted labels for prediction types
PREDICTION_TYPE_LABELS = {
    "epsilon": r"$\epsilon$-prediction",
    "velocity": r"$\mathbf{v}$-prediction",
    "x0": r"$\mathbf{x}_0$-prediction",
}

# Short labels for legends
PREDICTION_TYPE_LABELS_SHORT = {
    "epsilon": r"$\epsilon$",
    "velocity": r"$\mathbf{v}$",
    "x0": r"$\mathbf{x}_0$",
}

# =============================================================================
# Lp Norm Visual Encoding
# =============================================================================

LP_NORM_STYLES = {
    1.5: "-",     # Solid
    2.0: "--",    # Dashed
    2.5: ":",     # Dotted
}

LP_NORM_MARKERS = {
    1.5: "o",     # Circle
    2.0: "s",     # Square
    2.5: "^",     # Triangle
}

LP_NORM_ALPHAS = {
    1.5: 1.0,
    2.0: 0.85,
    2.5: 0.7,
}

# Hatch patterns for boxplots (grayscale compatible)
LP_NORM_HATCHES = {
    1.5: None,      # Solid fill
    2.0: "//",      # Forward diagonal
    2.5: "\\\\",    # Backward diagonal
}

# LaTeX-formatted labels for Lp norms
LP_NORM_LABELS = {
    1.5: r"$L_{1.5}$",
    2.0: r"$L_2$",
    2.5: r"$L_{2.5}$",
}

# =============================================================================
# IEEE ICIP Column Width Specifications
# From template: "Columns are to be 3.39 inches (86 mm) wide, with a 0.24 inch
# (6 mm) space between them." Print area: 7 inches wide by 9 inches high.
# =============================================================================

IEEE_COLUMN_WIDTH_INCHES = 3.39   # Single column (86 mm)
IEEE_COLUMN_GAP_INCHES = 0.24     # Gap between columns (6 mm)
IEEE_TEXT_WIDTH_INCHES = 7.0      # Full print area width (178 mm)
IEEE_TEXT_HEIGHT_INCHES = 9.0     # Full print area height (229 mm)

# =============================================================================
# Main Plot Settings Dictionary
# =============================================================================

PLOT_SETTINGS = {
    # Figure dimensions (IEEE ICIP compliant)
    "figure_width_single": IEEE_COLUMN_WIDTH_INCHES,  # 3.39 inches
    "figure_width_double": IEEE_TEXT_WIDTH_INCHES,    # 7.0 inches
    "figure_height_max": IEEE_TEXT_HEIGHT_INCHES,     # 9.0 inches (max)
    "figure_height_ratio": 0.75,  # Height = width * ratio (for plots)

    # Fonts (IEEE requires Times or similar serif)
    "font_family": "serif",
    "font_serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext_fontset": "stix",  # STIX for math (matches Times)
    "text_usetex": True,  # Set True if LaTeX is installed

    # Font sizes (IEEE guidelines)
    "font_size": 12,
    "axes_labelsize": 13,
    "axes_titlesize": 14,
    "tick_labelsize": 12,
    "legend_fontsize": 11,
    "annotation_fontsize": 10,
    "panel_label_fontsize": 12,

    # Line properties
    "line_width": 1.0,
    "line_width_thick": 1.5,
    "marker_size": 4,
    "marker_edge_width": 0.5,

    # Error bars
    "errorbar_capsize": 2,
    "errorbar_capthick": 0.8,
    "errorbar_linewidth": 0.8,

    # Error bands
    "error_band_alpha": 0.15,

    # Boxplot properties
    "boxplot_linewidth": 0.8,
    "boxplot_flier_size": 3,
    "boxplot_width": 0.6,

    # Grid
    "grid_alpha": 0.4,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,

    # Spines
    "spine_linewidth": 0.8,
    "spine_color": "0.2",

    # Ticks
    "tick_direction": "in",
    "tick_major_width": 0.8,
    "tick_minor_width": 0.5,
    "tick_major_length": 3.5,
    "tick_minor_length": 2.0,

    # Legend
    "legend_frameon": False,
    "legend_framealpha": 0.9,
    "legend_edgecolor": "0.8",
    "legend_borderpad": 0.4,
    "legend_columnspacing": 1.0,
    "legend_handletextpad": 0.5,

    # Image display (representative MRI images)
    "image_zoom": 0.12,
    "image_cmap": "gray",
    "image_x_offset": 0.0,   # Horizontal offset (fraction of axis width)
    "image_y_offset": 0.08,  # Vertical offset above axis (fraction)
    "image_step": 5,  # Sample every Nth z-bin

    # Boxplot whiskers and width
    "boxplot_whis": (5, 95),  # Whisker percentiles (5th to 95th)
    "boxplot_width_factor": 0.25,  # Box width as fraction of group spacing

    # DPI for output
    "dpi_print": 300,
    "dpi_screen": 150,

    # Significance annotations
    "significance_bracket_linewidth": 0.8,
    "significance_text_fontsize": 9,  # Stars (*, **, ***)
    "effect_size_fontsize": 8,        # Cliff's delta (d=0.XX)
}


def apply_ieee_style() -> None:
    """Apply IEEE publication style using scienceplots if available.

    Falls back to manual style settings if scienceplots is not installed.
    Overrides default color cycle with Paul Tol colorblind-safe palette.
    """
    import matplotlib.pyplot as plt

    # Try to use scienceplots if available
    try:
        plt.style.use(["science", "ieee"])
        _scienceplots_available = True
    except OSError:
        _scienceplots_available = False
        _apply_fallback_ieee_style()

    # Override with Paul Tol colors and custom settings
    plt.rcParams.update({
        "axes.prop_cycle": plt.cycler(
            color=list(PAUL_TOL_BRIGHT.values())[:6]
        ),
        # Ensure math rendering
        "mathtext.fontset": PLOT_SETTINGS["mathtext_fontset"],
        "font.family": PLOT_SETTINGS["font_family"],
        # Grid settings
        "axes.grid": True,
        "grid.alpha": PLOT_SETTINGS["grid_alpha"],
        "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
        "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
        # Tick settings
        "xtick.direction": PLOT_SETTINGS["tick_direction"],
        "ytick.direction": PLOT_SETTINGS["tick_direction"],
        "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
        "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
    })


def _apply_fallback_ieee_style() -> None:
    """Apply IEEE-like style without scienceplots."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # Fonts
        "font.family": PLOT_SETTINGS["font_family"],
        "font.serif": PLOT_SETTINGS["font_serif"],
        "font.size": PLOT_SETTINGS["font_size"],
        "mathtext.fontset": PLOT_SETTINGS["mathtext_fontset"],

        # Axes
        "axes.labelsize": PLOT_SETTINGS["axes_labelsize"],
        "axes.titlesize": PLOT_SETTINGS["axes_titlesize"],
        "axes.linewidth": PLOT_SETTINGS["spine_linewidth"],
        "axes.grid": True,

        # Ticks
        "xtick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "ytick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
        "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
        "xtick.minor.width": PLOT_SETTINGS["tick_minor_width"],
        "ytick.minor.width": PLOT_SETTINGS["tick_minor_width"],
        "xtick.direction": PLOT_SETTINGS["tick_direction"],
        "ytick.direction": PLOT_SETTINGS["tick_direction"],
        "xtick.major.size": PLOT_SETTINGS["tick_major_length"],
        "ytick.major.size": PLOT_SETTINGS["tick_major_length"],
        "xtick.minor.size": PLOT_SETTINGS["tick_minor_length"],
        "ytick.minor.size": PLOT_SETTINGS["tick_minor_length"],

        # Legend
        "legend.fontsize": PLOT_SETTINGS["legend_fontsize"],
        "legend.frameon": PLOT_SETTINGS["legend_frameon"],
        "legend.framealpha": PLOT_SETTINGS["legend_framealpha"],
        "legend.edgecolor": PLOT_SETTINGS["legend_edgecolor"],

        # Grid
        "grid.alpha": PLOT_SETTINGS["grid_alpha"],
        "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
        "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],

        # Figure
        "figure.figsize": (
            PLOT_SETTINGS["figure_width_double"],
            PLOT_SETTINGS["figure_width_double"] * PLOT_SETTINGS["figure_height_ratio"],
        ),
        "figure.dpi": PLOT_SETTINGS["dpi_screen"],
        "savefig.dpi": PLOT_SETTINGS["dpi_print"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def get_significance_stars(p_val: float) -> str:
    """Convert p-value to significance stars.

    Args:
        p_val: P-value from statistical test.

    Returns:
        String with stars: "***" (p<0.001), "**" (p<0.01),
        "*" (p<0.05), or "n.s." (not significant).
    """
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "n.s."


def format_metric_label(metric_col: str) -> str:
    """Format metric column name as a readable label.

    Args:
        metric_col: Column name like "kid_global" or "lpips_zbin".

    Returns:
        Formatted label like "KID" or "LPIPS".
    """
    metric = metric_col.replace("_global", "").replace("_zbin", "")
    return metric.upper()
