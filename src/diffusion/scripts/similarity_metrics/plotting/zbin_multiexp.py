"""Per-zbin multi-experiment plotting for similarity metrics.

Creates plots comparing metrics across z-bins for multiple experiments,
with visual encoding for prediction type and Lp norm.

ICIP 2026 version with:
- Paul Tol colorblind-friendly palettes
- LaTeX rendering for labels
- Representative MRI images above plot
- Legends outside plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from .settings import (
    PLOT_SETTINGS,
    PREDICTION_TYPE_COLORS,
    PREDICTION_TYPE_LABELS,
    PREDICTION_TYPE_LABELS_SHORT,
    LP_NORM_STYLES,
    LP_NORM_MARKERS,
    LP_NORM_ALPHAS,
    LP_NORM_LABELS,
    apply_ieee_style,
    format_metric_label,
)


def load_representative_images(
    test_csv: Path,
    zbins: list[int],
) -> dict[int, np.ndarray]:
    """Load representative images for specified z-bins from test.csv.

    Args:
        test_csv: Path to test.csv containing filepaths.
        zbins: List of z-bins to retrieve images for.

    Returns:
        Dictionary mapping zbin -> image array.
    """
    if not test_csv.exists():
        print(f"Warning: Test CSV not found at {test_csv}. Skipping images.")
        return {}

    try:
        df = pd.read_csv(test_csv)
    except Exception as e:
        print(f"Error reading test CSV: {e}")
        return {}

    # Filter for test split if available
    if "split" in df.columns:
        df = df[df["split"] == "test"]

    # Ensure z_bin column exists
    zbin_col = "z_bin" if "z_bin" in df.columns else "zbin"
    if zbin_col not in df.columns:
        print(f"Warning: Column '{zbin_col}' not found in {test_csv}. Skipping images.")
        return {}

    images = {}
    base_dir = test_csv.parent

    # Optimize: Group by zbin first
    df_grouped = df.groupby(zbin_col)

    for zbin in zbins:
        if zbin not in df_grouped.groups:
            continue

        # Get rows for this zbin
        zbin_rows = df_grouped.get_group(zbin)

        # Try to find a valid image file
        for _, row in zbin_rows.iterrows():
            fp = row.get("filepath")
            if not isinstance(fp, str) or not fp:
                continue

            full_path = base_dir / fp
            if full_path.exists():
                try:
                    data = np.load(full_path)
                    # Support both 'image' key and direct array
                    if isinstance(data, np.ndarray):
                        images[zbin] = data
                    elif "image" in data:
                        images[zbin] = data["image"]
                    else:
                        continue

                    break  # Found one, move to next zbin
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")
                    continue

    print(f"Loaded {len(images)} representative images.")
    return images


def add_representative_images(
    ax: plt.Axes,
    test_csv: Path,
    zbins: np.ndarray,
    image_step: int | None = None,
    image_zoom: float | None = None,
    image_y_offset: float | None = None,
) -> None:
    """Add representative MRI images above the plot.

    Uses AnnotationBbox with axes fraction coordinates to place images
    outside the plot area.

    Args:
        ax: Target axes (the per-zbin plot).
        test_csv: Path to test.csv with real slice paths.
        zbins: Array of all z-bin values in the data.
        image_step: Sample every Nth z-bin (default from PLOT_SETTINGS).
        image_zoom: Zoom factor for images (default from PLOT_SETTINGS).
        image_y_offset: Vertical offset above axis (default from PLOT_SETTINGS).
    """
    if image_step is None:
        image_step = PLOT_SETTINGS["image_step"]
    if image_zoom is None:
        image_zoom = PLOT_SETTINGS["image_zoom"]
    if image_y_offset is None:
        image_y_offset = PLOT_SETTINGS["image_y_offset"]

    sample_zbins = zbins[::image_step]
    images_map = load_representative_images(test_csv, sample_zbins.tolist())

    if not images_map:
        return

    x_min, x_max = zbins.min() - 1, zbins.max() + 1
    x_range = x_max - x_min

    for zb in sample_zbins:
        if zb not in images_map:
            continue

        img_arr = images_map[zb]
        # Normalize from [-1, 1] to [0, 1] for display
        img_disp = np.clip((img_arr + 1) / 2, 0, 1)

        im = OffsetImage(
            img_disp,
            zoom=image_zoom,
            cmap=PLOT_SETTINGS["image_cmap"],
        )

        # x position as fraction of axis width
        x_frac = (zb - x_min) / x_range

        # y = 1.0 is axis top, add offset to place above
        ab = AnnotationBbox(
            im,
            (x_frac, 1.0 + image_y_offset),
            xycoords="axes fraction",
            boxcoords="axes fraction",
            frameon=False,
            pad=0,
            box_alignment=(0.5, 0),  # Center horizontally, align bottom
        )
        ax.add_artist(ab)

        # Add z-bin label below image
        ax.annotate(
            f"z={zb}",
            xy=(x_frac, 1.0 + image_y_offset - 0.015),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
        )


def plot_zbin_multiexperiment(
    df_zbin: pd.DataFrame,
    metric_col: str,
    output_dir: Path,
    metric_name: str | None = None,
    baseline_real: float | None = None,
    baseline_std: float | None = None,
    figsize: tuple[float, float] | None = None,
    formats: list[str] = ["png", "pdf"],
    show_error_bands: bool = True,
    show_legend: bool = True,
    legend_outside: bool = True,
    title: str | None = None,
    test_csv: Path | None = None,
    show_images: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes | None:
    """Create per-zbin metric plot for all experiments.

    Visual encoding:
    - Color by prediction type (Paul Tol palette)
    - Line style by Lp: 1.5=solid, 2.0=dashed, 2.5=dotted
    - Markers by Lp: 1.5=circle, 2.0=square, 2.5=triangle
    - Error bands: +/-1 std (shaded area)
    - Baseline: horizontal dashed gray line

    Args:
        df_zbin: DataFrame with columns: experiment, prediction_type, lp_norm,
                 zbin, {metric_col}, {metric_col}_std.
        metric_col: Column name for metric values (e.g., "kid_zbin").
        output_dir: Output directory for plots.
        metric_name: Display name for metric (default: derived from col name).
        baseline_real: Optional baseline value (real vs real).
        baseline_std: Optional baseline std.
        figsize: Figure size (default: IEEE double column width).
        formats: Output formats.
        show_error_bands: Whether to show error bands.
        show_legend: Whether to show legend.
        legend_outside: Whether to place legend outside the plot.
        title: Optional title (default: auto-generated).
        test_csv: Path to test.csv for representative images.
        show_images: Whether to show representative images above plot.
        ax: Optional existing axes to plot on (for subplots).

    Returns:
        The axes object if ax was provided, None otherwise.
    """
    apply_ieee_style()

    if figsize is None:
        figsize = (
            PLOT_SETTINGS["figure_width_double"],
            PLOT_SETTINGS["figure_width_double"] * 0.5,
        )

    # Derive metric name from column if not provided
    if metric_name is None:
        metric_name = format_metric_label(metric_col)

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
    zbins = np.array(sorted(df_zbin["zbin"].unique()))

    # Create figure if no axes provided
    created_figure = ax is None
    if created_figure:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Plot baseline if provided
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
            ax.fill_between(
                [zbins[0] - 0.5, zbins[-1] + 0.5],
                baseline_real - baseline_std,
                baseline_real + baseline_std,
                color="gray",
                alpha=PLOT_SETTINGS["error_band_alpha"],
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
            marker = LP_NORM_MARKERS.get(lp_norm, "o")
            alpha = LP_NORM_ALPHAS.get(lp_norm, 1.0)

            # LaTeX label
            pred_label = PREDICTION_TYPE_LABELS_SHORT.get(pred_type, pred_type)
            lp_label = LP_NORM_LABELS.get(lp_norm, f"Lp={lp_norm}")
            label = f"{pred_label}, {lp_label}"

            # Plot line
            ax.plot(
                x, y,
                color=color,
                linestyle=linestyle,
                linewidth=PLOT_SETTINGS["line_width"],
                alpha=alpha,
                marker=marker,
                markersize=PLOT_SETTINGS["marker_size"],
                markeredgewidth=PLOT_SETTINGS["marker_edge_width"],
                markeredgecolor="white",
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
                    alpha=alpha * PLOT_SETTINGS["error_band_alpha"],
                    zorder=1,
                )

    # Add representative images if requested
    if show_images and test_csv is not None:
        add_representative_images(ax, test_csv, zbins)

    # Configure axes
    ax.set_xlabel("Z-bin", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_ylabel(metric_name, fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_xlim(zbins[0] - 0.5, zbins[-1] + 0.5)

    # Set title
    if title:
        ax.set_title(title, fontsize=PLOT_SETTINGS["axes_titlesize"])

    # Configure legend
    if show_legend:
        if legend_outside:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=PLOT_SETTINGS["legend_fontsize"],
                frameon=PLOT_SETTINGS["legend_frameon"],
                borderpad=PLOT_SETTINGS["legend_borderpad"],
            )
        else:
            ax.legend(
                loc="upper right",
                ncol=3,
                fontsize=PLOT_SETTINGS["legend_fontsize"],
                frameon=PLOT_SETTINGS["legend_frameon"],
            )

    # Save if we created the figure
    if created_figure:
        plt.tight_layout()

        base_name = f"{metric_col}_zbin_comparison"
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


def plot_zbin_by_prediction_type(
    df_zbin: pd.DataFrame,
    metric_col: str,
    output_dir: Path,
    metric_name: str | None = None,
    baseline_real: float | None = None,
    figsize: tuple[float, float] | None = None,
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
    apply_ieee_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metric_name is None:
        metric_name = format_metric_label(metric_col)

    if figsize is None:
        figsize = (PLOT_SETTINGS["figure_width_double"], 2.5)

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
                linewidth=PLOT_SETTINGS["line_width"],
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
            marker = LP_NORM_MARKERS.get(lp_norm, "o")
            alpha = LP_NORM_ALPHAS.get(lp_norm, 1.0)
            lp_label = LP_NORM_LABELS.get(lp_norm, f"Lp={lp_norm}")

            ax.plot(
                x, y,
                color=color,
                linestyle=linestyle,
                linewidth=PLOT_SETTINGS["line_width"],
                alpha=alpha,
                marker=marker,
                markersize=PLOT_SETTINGS["marker_size"] - 1,
                label=lp_label,
            )

        # LaTeX title
        pred_label = PREDICTION_TYPE_LABELS.get(pred_type, pred_type.capitalize())
        ax.set_title(pred_label, fontsize=PLOT_SETTINGS["axes_titlesize"])
        ax.set_xlabel("Z-bin", fontsize=PLOT_SETTINGS["axes_labelsize"])
        ax.set_xlim(zbins[0] - 0.5, zbins[-1] + 0.5)
        ax.grid(True, alpha=PLOT_SETTINGS["grid_alpha"], linestyle=PLOT_SETTINGS["grid_linestyle"])
        ax.legend(
            loc="upper right",
            fontsize=PLOT_SETTINGS["legend_fontsize"],
            frameon=PLOT_SETTINGS["legend_frameon"],
        )

    axes[0].set_ylabel(metric_name, fontsize=PLOT_SETTINGS["axes_labelsize"])

    plt.tight_layout()

    # Save
    base_name = f"{metric_col}_by_prediction_type"
    for fmt in formats:
        output_path = output_dir / f"{base_name}.{fmt}"
        plt.savefig(output_path, dpi=PLOT_SETTINGS["dpi_print"], bbox_inches="tight")
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
    apply_ieee_style()
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.axis("off")

    # Create dummy lines for legend
    lines = []
    labels = []

    # Prediction types
    for pred_type, color in PREDICTION_TYPE_COLORS.items():
        pred_label = PREDICTION_TYPE_LABELS.get(pred_type, pred_type)
        line, = ax.plot([], [], color=color, linewidth=2, linestyle="-")
        lines.append(line)
        labels.append(pred_label)

    # Lp norms
    for lp_norm in LP_NORM_STYLES.keys():
        linestyle = LP_NORM_STYLES[lp_norm]
        marker = LP_NORM_MARKERS[lp_norm]
        lp_label = LP_NORM_LABELS.get(lp_norm, f"Lp={lp_norm}")
        line, = ax.plot(
            [], [],
            color="black",
            linewidth=2,
            linestyle=linestyle,
            marker=marker,
            markersize=6,
        )
        lines.append(line)
        labels.append(lp_label)

    ax.legend(
        lines, labels,
        loc="center",
        ncol=6,
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=True,
    )

    plt.tight_layout()

    for fmt in formats:
        output_path = output_dir / f"legend.{fmt}"
        plt.savefig(output_path, dpi=PLOT_SETTINGS["dpi_print"], bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()
