"""Visualization script for KID evaluation results.

This script creates publication-quality plots showing:
1. Global KID (horizontal line with confidence interval)
2. Per-zbin KID with error bars and significance markers
3. Optional delta_kid plot showing bin vs rest deviations

Example usage:
    python -m src.diffusion.scripts.plot_kid_results \
        --global-csv /path/to/kid_replica_global.csv \
        --zbin-csv /path/to/kid_replica_zbin.csv \
        --output-dir /path/to/output \
        --show-delta \
        --format png pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


PLOT_SETTINGS = {
    "font_family": "serif",
    "font_serif": ["Times New Roman", "DejaVu Serif"],
    "font_size": 9,
    "axes_labelsize": 8,
    "axes_titlesize": 9,
    "axes_spine_width": 0.8,
    "axes_spine_color": "0.2",
    "tick_labelsize": 12,
    "tick_major_width": 0.6,
    "tick_minor_width": 0.4,
    "tick_direction": "in",
    "tick_length_major": 3.5,
    "tick_length_minor": 2.0,
    "legend_fontsize": 10,
    "legend_framealpha": 0.9,
    "legend_frameon": False,
    "legend_edgecolor": "0.8",
    "grid_linestyle": ":",
    "grid_alpha": 0.7,
    "grid_linewidth": 0.6,
    "line_width": 2.0,
    "axis_labelsize": 14,
    "xtick_fontsize": 12,
    "ytick_fontsize": 12,
    "xlabel_fontsize": 14,
    "ylabel_fontsize": 14,
    "title_fontsize": 16,
}

def apply_plot_settings():
    """Apply global matplotlib settings for consistent styling."""
    plt.rcParams.update({
        "font.family": PLOT_SETTINGS["font_family"],
        "font.serif": PLOT_SETTINGS["font_serif"],
        "font.size": PLOT_SETTINGS["font_size"],
        "axes.labelsize": PLOT_SETTINGS["axes_labelsize"],
        "axes.titlesize": PLOT_SETTINGS["axes_titlesize"],
        "xtick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "ytick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
        "xtick.minor.width": PLOT_SETTINGS["tick_minor_width"],
        "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
        "ytick.minor.width": PLOT_SETTINGS["tick_minor_width"],
        "xtick.direction": PLOT_SETTINGS["tick_direction"],
        "ytick.direction": PLOT_SETTINGS["tick_direction"],
        "legend.fontsize": PLOT_SETTINGS["legend_fontsize"],
        "legend.framealpha": PLOT_SETTINGS["legend_framealpha"],
        "legend.frameon": PLOT_SETTINGS["legend_frameon"],
        "legend.edgecolor": PLOT_SETTINGS["legend_edgecolor"],
        "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
        "grid.alpha": PLOT_SETTINGS["grid_alpha"],
        "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
        "axes.grid": True,
    })

apply_plot_settings()


def load_representative_images(
    test_csv: Path, zbins: list[int]
) -> dict[int, np.ndarray]:
    """Load representative images for specified z-bins from test.csv.
    
    Args:
        test_csv: Path to test.csv containing filepaths
        zbins: List of z-bins to retrieve images for
        
    Returns:
        Dictionary mapping zbin -> image array
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
                        
                    break # Found one, move to next zbin
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")
                    continue
    
    print(f"Loaded {len(images)} representative images.")
    return images


def plot_kid_results(
    df_global: pd.DataFrame,
    df_zbin: pd.DataFrame,
    output_dir: Path,
    title: str = "KID Evaluation: Synthetic vs Test",
    figsize: tuple[int, int] = (12, 6),
    show_delta: bool = False,
    formats: list[str] = ["png", "pdf"],
    include_images: bool = False,
    test_csv: Path | None = None,
    image_step: int = 3,
    image_zoom: float = 0.15,
    image_y_offset: float = 0.1,
    image_x_offset: float = 0.0,
    hide_significance: bool = False,
    brain_frac_csv: Path | None = None,
) -> None:
    """Create KID visualization plots.

    Args:
        df_global: DataFrame from kid_replica_global.csv
        df_zbin: DataFrame from kid_replica_zbin.csv
        output_dir: Output directory for plots
        title: Plot title (not displayed if empty string)
        figsize: Figure size (width, height)
        show_delta: Whether to show delta_kid subplot
        formats: List of output formats (png, pdf, svg)
        include_images: Whether to include representative images
        test_csv: Path to test.csv (required if include_images=True)
        image_step: Step size for sampling images
        image_zoom: Zoom level for images
        image_y_offset: Vertical offset fraction for images
        image_x_offset: Horizontal offset in data coordinates
        hide_significance: Whether to hide significance markers (*)
        brain_frac_csv: Path to CSV with mean_brain_frac per zbin (optional)
    """
    # Compute global KID statistics
    global_mean = df_global["kid_global"].mean()
    global_std = df_global["kid_global"].std()

    # Compute per-zbin statistics
    zbin_stats = df_zbin.groupby("zbin").agg(
        {
            "kid_zbin": ["mean", "std", "count"],
            "kid_rest": ["mean", "std"],
            "delta_kid": ["mean", "std"],
            "p_value": "first",  # Same for all replicas/groups of a bin
            "q_value_fdr": "first",
            "signif_code": "first",
        }
    )
    zbin_stats.columns = ["_".join(col).strip() for col in zbin_stats.columns.values]
    zbin_stats = zbin_stats.reset_index()

    zbins = zbin_stats["zbin"].values
    kid_zbin_mean = zbin_stats["kid_zbin_mean"].values
    kid_zbin_std = zbin_stats["kid_zbin_std"].values
    delta_mean = zbin_stats["delta_kid_mean"].values
    delta_std = zbin_stats["delta_kid_std"].values
    signif_codes = zbin_stats["signif_code_first"].values

    # Create figure
    if show_delta:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # ========== Main Plot: KID vs zbin ==========
    # Plot global KID with confidence interval
    ax1.axhline(global_mean, color="gray", linestyle="--", linewidth=PLOT_SETTINGS["line_width"], label=f"Global KID: {global_mean:.5f}", zorder=1)
    ax1.fill_between(
        [zbins[0] - 0.5, zbins[-1] + 0.5],
        global_mean - global_std,
        global_mean + global_std,
        color="gray",
        alpha=0.2,
        label=f"Global ±1 std: ±{global_std:.5f}",
        zorder=1,
    )

    # Plot per-zbin KID with error bars
    ax1.errorbar(
        zbins,
        kid_zbin_mean,
        yerr=kid_zbin_std,
        fmt="o",
        markersize=8,
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        label="Per-zbin KID (mean ± std)",
        zorder=2,
        color="steelblue",
        ecolor="steelblue",
        markeredgecolor="black",
        markeredgewidth=1,
        alpha=0.8,
    )

    # Calculate Y-axis range for positioning
    y_min, y_max = ax1.get_ylim()
    data_range = y_max - y_min
    
    # Track the maximum Y value reached by data + markers
    current_max_y = y_max

    # Add significance markers above points (if not hidden)
    if not hide_significance:
        y_offset = data_range * 0.05
        for zb, kid, std, code in zip(zbins, kid_zbin_mean, kid_zbin_std, signif_codes):
            if code and not pd.isna(code):
                text_y = kid + std + y_offset
                ax1.text(
                    zb,
                    text_y,
                    code,
                    ha="center",
                    va="bottom",
                    fontsize=PLOT_SETTINGS["legend_fontsize"],
                    fontweight="bold",
                    color="black",
                )
                current_max_y = max(current_max_y, text_y)

    # ========== Representative Images (outside plot, on top) ==========
    if include_images and test_csv:
        # Determine sampling z-bins
        sample_zbins = zbins[::image_step]
        images_map = load_representative_images(test_csv, sample_zbins)

        if images_map:
            # Calculate x positions in axes fraction coordinates
            x_min, x_max = zbins[0] - 1, zbins[-1] + 1
            x_range = x_max - x_min
            
            for zb in sample_zbins:
                if zb in images_map:
                    img_arr = images_map[zb]
                    
                    # Normalize to [0, 1] assuming [-1, 1] input range (standard for this project)
                    # If already [0, 1] or other, we just clip.
                    img_disp = np.clip((img_arr + 1) / 2, 0, 1)
                    
                    # Create OffsetImage
                    im = OffsetImage(img_disp, zoom=image_zoom, cmap="gray")
                    
                    # Calculate x position in axes fraction (0 to 1)
                    x_frac = (zb + image_x_offset - x_min) / x_range
                    
                    # Place image above the plot using axes fraction coordinates
                    # y > 1.0 means above the axes
                    ab = AnnotationBbox(
                        im, 
                        (x_frac, 1.0 + image_y_offset),
                        xycoords="axes fraction",
                        boxcoords="axes fraction",
                        frameon=False,
                        pad=0,
                        box_alignment=(0.5, 0),  # Center horizontally, align bottom
                    )
                    ax1.add_artist(ab)

    # ========== Secondary Y-axis: Mean Brain Fraction ==========
    ax1_twin = None
    if brain_frac_csv is not None and brain_frac_csv.exists():
        df_brain = pd.read_csv(brain_frac_csv)
        # Aggregate mean_brain_frac per zbin (across all rows: lesion_present, domain)
        brain_stats = df_brain.groupby("zbin").agg(
            total_slices=("n_slices", "sum"),
            weighted_brain_frac=("mean_brain_frac", lambda x: np.average(x, weights=df_brain.loc[x.index, "n_slices"]))
        ).reset_index()
        
        # Match zbins from KID data
        brain_zbins = brain_stats["zbin"].values
        brain_frac = brain_stats["weighted_brain_frac"].values
        
        # Create secondary y-axis
        ax1_twin = ax1.twinx()
        ax1_twin.plot(
            brain_zbins,
            brain_frac,
            color="forestgreen",
            linestyle="-",
            linewidth=2,
            marker="s",
            markersize=5,
            label="Mean Brain Fraction",
            alpha=0.7,
            zorder=1,
        )
        ax1_twin.set_ylabel("Mean Brain Fraction", fontsize=PLOT_SETTINGS["ylabel_fontsize"], fontweight="bold", color="forestgreen")
        ax1_twin.tick_params(axis="y", labelcolor="forestgreen")
        ax1_twin.spines["right"].set_color("forestgreen")

    ax1.set_ylabel("Kernel Inception Distance (KID)", fontsize=PLOT_SETTINGS["ylabel_fontsize"], fontweight="bold")
    if title:  # Only set title if not empty
        ax1.set_title(title, fontsize=PLOT_SETTINGS["title_fontsize"], fontweight="bold", pad=15)
    ax1.grid(True, alpha=PLOT_SETTINGS["grid_alpha"], linestyle=PLOT_SETTINGS["grid_linestyle"], linewidth=PLOT_SETTINGS["grid_linewidth"])
    # Ensure x-lim covers all bins
    ax1.set_xlim(zbins[0] - 1, zbins[-1] + 1)

    if not show_delta:
        ax1.set_xlabel("Z-bin", fontsize=PLOT_SETTINGS["xlabel_fontsize"], fontweight="bold")

    # ========== Delta Plot: delta_kid vs zbin ==========
    if show_delta:
        # Plot zero line
        ax2.axhline(0, color="black", linestyle="-", linewidth=1.5, alpha=0.5, zorder=1)

        # Plot delta_kid with error bars
        ax2.errorbar(
            zbins,
            delta_mean,
            yerr=delta_std,
            fmt="o",
            markersize=6,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            color="purple",
            ecolor="purple",
            alpha=0.7,
            zorder=2,
        )

        # Plot points (uniform color)
        ax2.scatter(zbins, delta_mean, s=80, c="purple", edgecolors="black", linewidths=1, zorder=3)

        ax2.set_xlabel("Z-bin", fontsize=PLOT_SETTINGS["xlabel_fontsize"], fontweight="bold")
        ax2.set_ylabel("Δ KID (bin - rest)", fontsize=PLOT_SETTINGS["ylabel_fontsize"], fontweight="bold")
        ax2.grid(True, alpha=PLOT_SETTINGS["grid_alpha"], linestyle=PLOT_SETTINGS["grid_linestyle"], linewidth=PLOT_SETTINGS["grid_linewidth"])
        ax2.set_xlim(zbins[0] - 0.5, zbins[-1] + 0.5)

        # Add shaded region for positive/negative delta
        ax2.fill_between(
            [zbins[0] - 0.5, zbins[-1] + 0.5],
            0,
            ax2.get_ylim()[1],
            color="green",
            alpha=0.05,
        )
        ax2.fill_between(
            [zbins[0] - 0.5, zbins[-1] + 0.5],
            ax2.get_ylim()[0],
            0,
            color="red",
            alpha=0.05,
        )

    # Collect legend handles from main axis
    handles, labels = ax1.get_legend_handles_labels()
    
    # Add twin axis legend if present
    if ax1_twin is not None:
        handles_twin, labels_twin = ax1_twin.get_legend_handles_labels()
        handles.extend(handles_twin)
        labels.extend(labels_twin)
    
    # Place combined legend at the bottom, outside the plot
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        ncol=len(handles),
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        framealpha=PLOT_SETTINGS["legend_framealpha"],
        bbox_to_anchor=(0.5, -0.02),
    )
    
    # Adjust layout to make room for legend at bottom
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save figure in requested formats
    for fmt in formats:
        output_path = output_dir / f"kid_results.{fmt}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def plot_kid_comparison(
    df_global: pd.DataFrame,
    df_zbin: pd.DataFrame,
    output_dir: Path,
    title: str = "KID: Bin vs Rest Comparison",
    figsize: tuple[int, int] = (14, 6),
    formats: list[str] = ["png", "pdf"],
) -> None:
    """Create side-by-side comparison of kid_zbin and kid_rest.

    Args:
        df_global: DataFrame from kid_replica_global.csv
        df_zbin: DataFrame from kid_replica_zbin.csv
        output_dir: Output directory for plots
        title: Plot title
        figsize: Figure size (width, height)
        formats: List of output formats (png, pdf, svg)
    """
    # Compute global KID statistics
    global_mean = df_global["kid_global"].mean()
    global_std = df_global["kid_global"].std()

    # Compute per-zbin statistics
    zbin_stats = df_zbin.groupby("zbin").agg(
        {
            "kid_zbin": ["mean", "std"],
            "kid_rest": ["mean", "std"],
            "signif_code": "first",
        }
    )
    zbin_stats.columns = ["_".join(col).strip() for col in zbin_stats.columns.values]
    zbin_stats = zbin_stats.reset_index()

    zbins = zbin_stats["zbin"].values
    kid_zbin_mean = zbin_stats["kid_zbin_mean"].values
    kid_zbin_std = zbin_stats["kid_zbin_std"].values
    kid_rest_mean = zbin_stats["kid_rest_mean"].values
    kid_rest_std = zbin_stats["kid_rest_std"].values
    signif_codes = zbin_stats["signif_code_first"].values

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot global KID reference
    ax.axhline(global_mean, color="gray", linestyle="--", linewidth=PLOT_SETTINGS["line_width"], label=f"Global KID: {global_mean:.5f}", zorder=1)
    ax.fill_between(
        [zbins[0] - 0.5, zbins[-1] + 0.5],
        global_mean - global_std,
        global_mean + global_std,
        color="gray",
        alpha=0.2,
        zorder=1,
    )

    # Plot kid_zbin
    ax.errorbar(
        zbins - 0.15,
        kid_zbin_mean,
        yerr=kid_zbin_std,
        fmt="o",
        markersize=6,
        capsize=4,
        label="KID (bin only)",
        color="steelblue",
        alpha=0.7,
        zorder=2,
    )

    # Plot kid_rest
    ax.errorbar(
        zbins + 0.15,
        kid_rest_mean,
        yerr=kid_rest_std,
        fmt="s",
        markersize=6,
        capsize=4,
        label="KID (rest, excluding bin)",
        color="coral",
        alpha=0.7,
        zorder=2,
    )

    # Add significance markers
    for zb, kid, code in zip(zbins, kid_zbin_mean, signif_codes):
        if code:
            color = "red" if code == "***" else "orange" if code == "**" else "gold"
            ax.scatter(zb - 0.15, kid, s=100, c=color, marker="*", edgecolors="black", linewidths=1, zorder=3)

    ax.set_xlabel("Z-bin", fontsize=PLOT_SETTINGS["xlabel_fontsize"], fontweight="bold")
    ax.set_ylabel("KID", fontsize=PLOT_SETTINGS["ylabel_fontsize"], fontweight="bold")
    ax.set_title(title, fontsize=PLOT_SETTINGS["title_fontsize"], fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=PLOT_SETTINGS["legend_fontsize"], framealpha=PLOT_SETTINGS["legend_framealpha"])
    ax.grid(True, alpha=PLOT_SETTINGS["grid_alpha"], linestyle=PLOT_SETTINGS["grid_linestyle"], linewidth=PLOT_SETTINGS["grid_linewidth"])
    ax.set_xlim(zbins[0] - 0.5, zbins[-1] + 0.5)

    plt.tight_layout()

    # Save figure
    for fmt in formats:
        output_path = output_dir / f"kid_bin_vs_rest_comparison.{fmt}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def print_summary_statistics(df_global: pd.DataFrame, df_zbin: pd.DataFrame) -> None:
    """Print summary statistics to console.

    Args:
        df_global: DataFrame from kid_replica_global.csv
        df_zbin: DataFrame from kid_replica_zbin.csv
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Global KID
    print("\nGlobal KID:")
    print(f"  Mean: {df_global['kid_global'].mean():.6f}")
    print(f"  Std:  {df_global['kid_global'].std():.6f}")
    print(f"  Min:  {df_global['kid_global'].min():.6f}")
    print(f"  Max:  {df_global['kid_global'].max():.6f}")
    print(f"  N replicas: {len(df_global)}")

    # Per-zbin KID
    zbin_stats = df_zbin.groupby("zbin").agg({"kid_zbin": ["mean", "std"], "signif_code": "first"})
    zbin_stats.columns = ["_".join(col).strip() for col in zbin_stats.columns.values]

    print("\nPer-zbin KID:")
    print(f"  Mean (across bins): {zbin_stats['kid_zbin_mean'].mean():.6f}")
    print(f"  Std (across bins):  {zbin_stats['kid_zbin_mean'].std():.6f}")
    print(f"  Min (across bins):  {zbin_stats['kid_zbin_mean'].min():.6f}")
    print(f"  Max (across bins):  {zbin_stats['kid_zbin_mean'].max():.6f}")

    # Significance counts
    signif_counts = df_zbin.groupby("zbin")["signif_code"].first().value_counts()
    print("\nSignificance counts:")
    print(f"  *** (q < 0.001): {signif_counts.get('***', 0)}")
    print(f"  **  (q < 0.01):  {signif_counts.get('**', 0)}")
    print(f"  *   (q < 0.05):  {signif_counts.get('*', 0)}")
    print(f"  n.s. (q ≥ 0.05): {signif_counts.get('', 0)}")

    # Top 5 bins with largest |delta_kid|
    delta_stats = df_zbin.groupby("zbin").agg({"delta_kid": "mean", "signif_code": "first"})
    top_deltas = delta_stats.reindex(delta_stats["delta_kid"].abs().sort_values(ascending=False).index).head(5)

    print("\nTop 5 bins with largest |Δ KID| (bin - rest):")
    for i, (zbin, row) in enumerate(top_deltas.iterrows()):
        delta = delta_stats.loc[zbin, "delta_kid"]
        code = delta_stats.loc[zbin, "signif_code"]
        print(f"  {i+1}. Bin {zbin:2d}: Δ={delta:+.6f} {code}")

    print("=" * 80 + "\n")


def main(args):
    """Main CLI entry point."""
    # Convert paths
    global_csv = Path(args.global_csv)
    zbin_csv = Path(args.zbin_csv)
    output_dir = Path(args.output_dir)
    test_csv = Path(args.test_csv) if args.test_csv else None

    # Validate image args
    if args.include_image_samples and not test_csv:
        print("Error: --test-csv is required when --include-image-samples is set.")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse figsize
    figsize = tuple(map(int, args.figsize.split(",")))

    # Parse formats
    formats = args.format.split(",")

    # Load CSVs
    print("Loading data...")
    df_global = pd.read_csv(global_csv)
    df_zbin = pd.read_csv(zbin_csv)
    print(f"  Global KID: {len(df_global)} replicas")
    print(f"  Per-zbin KID: {len(df_zbin)} rows")

    # Print summary statistics
    print_summary_statistics(df_global, df_zbin)

    # Generate plots
    print("\nGenerating plots...")

    # Load brain frac CSV if provided
    brain_frac_csv = Path(args.plot_mean_brain_frac) if args.plot_mean_brain_frac else None

    # Main KID plot
    plot_kid_results(
        df_global,
        df_zbin,
        output_dir,
        title=args.title,
        figsize=figsize,
        show_delta=args.show_delta,
        formats=formats,
        include_images=args.include_image_samples,
        test_csv=test_csv,
        image_step=args.image_step,
        image_zoom=args.image_zoom,
        image_y_offset=args.image_y_offset,
        image_x_offset=args.image_x_offset,
        hide_significance=args.hide_significance,
        brain_frac_csv=brain_frac_csv,
    )

    # Bin vs rest comparison plot
    if args.show_comparison:
        plot_kid_comparison(
            df_global,
            df_zbin,
            output_dir,
            title="KID: Bin vs Rest Comparison",
            figsize=(14, 6),
            formats=formats,
        )

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize KID evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python -m src.diffusion.scripts.plot_kid_results --global-csv /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/kid_quality/kid_replica_global.csv --zbin-csv /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/kid_quality/kid_replica_zbin_merged2.csv --output-dir /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/kid_quality --format png --test-csv /media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache/test.csv --include-image-samples --image-zoom 0.45 --image-x-offset 1 --plot-mean-brain-frac /home/mpascual/research/code/js-ddpm-epilepsy/docs/train_analysis/train_zbin_distribution.csv --hide-significance --image-y-offset 0.01
        """,
    )
    parser.add_argument(
        "--global-csv",
        type=str,
        required=True,
        help="Path to kid_replica_global.csv",
    )
    parser.add_argument(
        "--zbin-csv",
        type=str,
        required=True,
        help="Path to kid_replica_zbin.csv or kid_replica_zbin_mergedX.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Plot title (default: empty, no title displayed)",
    )
    parser.add_argument(
        "--hide-significance",
        action="store_true",
        help="Hide significance markers (*) on the plot",
    )
    parser.add_argument(
        "--plot-mean-brain-frac",
        type=str,
        help="Path to CSV with zbin distribution (e.g., train_zbin_distribution.csv) to plot mean brain fraction on secondary y-axis",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,6",
        help="Figure size as 'width,height' in inches (default: '12,6')",
    )
    parser.add_argument(
        "--show-delta",
        action="store_true",
        help="Show delta_kid (bin - rest) subplot",
    )
    parser.add_argument(
        "--show-comparison",
        action="store_true",
        help="Generate additional bin vs rest comparison plot",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png,pdf",
        help="Output format(s) as comma-separated list: png,pdf,svg (default: 'png,pdf')",
    )
    parser.add_argument(
        "--include-image-samples",
        action="store_true",
        help="Include representative images on the plot",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        help="Path to test.csv (required if --include-image-samples is set)",
    )
    parser.add_argument(
        "--image-step",
        type=int,
        default=3,
        help="Step size for z-bins to sample images (default: 3)",
    )
    parser.add_argument(
        "--image-zoom",
        type=float,
        default=0.15,
        help="Zoom level for image samples (default: 0.15)",
    )
    parser.add_argument(
        "--image-y-offset",
        type=float,
        default=0.1,
        help="Y-offset for images as fraction of data range (default: 0.1)",
    )
    parser.add_argument(
        "--image-x-offset",
        type=float,
        default=0.0,
        help="X-offset for images in data coordinates (default: 0.0)",
    )

    args = parser.parse_args()
    main(args)
