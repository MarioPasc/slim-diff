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
) -> None:
    """Create KID visualization plots.

    Args:
        df_global: DataFrame from kid_replica_global.csv
        df_zbin: DataFrame from kid_replica_zbin.csv
        output_dir: Output directory for plots
        title: Plot title
        figsize: Figure size (width, height)
        show_delta: Whether to show delta_kid subplot
        formats: List of output formats (png, pdf, svg)
        include_images: Whether to include representative images
        test_csv: Path to test.csv (required if include_images=True)
        image_step: Step size for sampling images
        image_zoom: Zoom level for images
        image_y_offset: Vertical offset fraction for images
        image_x_offset: Horizontal offset in data coordinates
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
    ax1.axhline(global_mean, color="gray", linestyle="--", linewidth=2, label=f"Global KID: {global_mean:.5f}", zorder=1)
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
    colors = []
    for code in signif_codes:
        if pd.isna(code):
            colors.append("steelblue")
        elif code == "***":
            colors.append("red")
        elif code == "**":
            colors.append("orange")
        elif code == "*":
            colors.append("gold")
        else:
            colors.append("steelblue")

    ax1.errorbar(
        zbins,
        kid_zbin_mean,
        yerr=kid_zbin_std,
        fmt="o",
        markersize=6,
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        label="Per-zbin KID (mean ± std)",
        zorder=2,
        color="steelblue",
        ecolor="steelblue",
        alpha=0.7,
    )

    # Color-code points by significance
    for i, (zb, kid, code, color) in enumerate(zip(zbins, kid_zbin_mean, signif_codes, colors)):
        ax1.scatter(zb, kid, s=80, c=color, edgecolors="black", linewidths=1, zorder=3)

    # Calculate Y-axis range for positioning
    y_min, y_max = ax1.get_ylim()
    data_range = y_max - y_min
    
    # Add significance markers above points
    # Logic: Only add if code is not NaN
    y_offset = data_range * 0.05
    
    # Track the maximum Y value reached by data + markers
    current_max_y = y_max

    for zb, kid, std, code in zip(zbins, kid_zbin_mean, kid_zbin_std, signif_codes):
        if code and not pd.isna(code):
            text_y = kid + std + y_offset
            ax1.text(
                zb,
                text_y,
                code,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="red" if code == "***" else "orange" if code == "**" else "gold",
            )
            current_max_y = max(current_max_y, text_y)

    # ========== Representative Images ==========
    if include_images and test_csv:
        # Determine sampling z-bins
        sample_zbins = zbins[::image_step]
        images_map = load_representative_images(test_csv, sample_zbins)

        if images_map:
            # Determine placement Y position (above highest data/marker)
            image_y_pos = current_max_y + (data_range * image_y_offset)
            
            for zb in sample_zbins:
                if zb in images_map:
                    img_arr = images_map[zb]
                    
                    # Normalize to [0, 1] assuming [-1, 1] input range (standard for this project)
                    # If already [0, 1] or other, we just clip.
                    img_disp = np.clip((img_arr + 1) / 2, 0, 1)
                    
                    # Create OffsetImage
                    im = OffsetImage(img_disp, zoom=image_zoom, cmap="gray")
                    
                    # Add AnnotationBbox
                    # xy is the position in data coordinates
                    ab = AnnotationBbox(
                        im, 
                        (zb + image_x_offset, image_y_pos),
                        xybox=(0, 0),
                        xycoords="data",
                        boxcoords="offset points",
                        frameon=False,
                        pad=0
                    )
                    ax1.add_artist(ab)
            
            # Extend Y-limit to accommodate images
            # We estimate image height in data coords roughly, or just add buffer
            # With data coords, the image size depends on zoom and dpi. 
            # A simpler way is to just extend Y max significantly.
            # Let's add 20% of range + the offset used
            ax1.set_ylim(y_min, image_y_pos + (data_range * 0.2))

    ax1.set_ylabel("KID", fontsize=12, fontweight="bold")
    ax1.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    # Ensure x-lim covers all bins
    ax1.set_xlim(zbins[0] - 1, zbins[-1] + 1)

    if not show_delta:
        ax1.set_xlabel("Z-bin", fontsize=12, fontweight="bold")

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

        # Color-code points by significance
        for i, (zb, delta, code, color) in enumerate(zip(zbins, delta_mean, signif_codes, colors)):
            ax2.scatter(zb, delta, s=80, c=color, edgecolors="black", linewidths=1, zorder=3)

        ax2.set_xlabel("Z-bin", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Δ KID (bin - rest)", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
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

    # Add custom legend for significance codes
    from matplotlib.patches import Patch

    sig_legend_elements = [
        Patch(facecolor="red", edgecolor="black", label="*** (q < 0.001)"),
        Patch(facecolor="orange", edgecolor="black", label="** (q < 0.01)"),
        Patch(facecolor="gold", edgecolor="black", label="* (q < 0.05)"),
        Patch(facecolor="steelblue", edgecolor="black", label="n.s. (q ≥ 0.05)"),
    ]

    # Add second legend for significance in lower bottom if any significance present
    any_significance = any(code and not pd.isna(code) for code in signif_codes)

    if any_significance:
        # Use figure legend to place it at the very bottom
        fig.legend(
            handles=sig_legend_elements,
            loc="lower center",
            ncol=4,
            fontsize=9,
            title="Significance",
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.02),
        )
        # Adjust layout to make room for legend at bottom
        # Use rect to reserve space at the bottom (left, bottom, right, top)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        plt.tight_layout()

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
    ax.axhline(global_mean, color="gray", linestyle="--", linewidth=2, label=f"Global KID: {global_mean:.5f}", zorder=1)
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

    ax.set_xlabel("Z-bin", fontsize=12, fontweight="bold")
    ax.set_ylabel("KID", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
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
    top_deltas = delta_stats.abs().sort_values("delta_kid", ascending=False).head(5)

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
  python -m src.diffusion.scripts.plot_kid_results \\
      --global-csv /path/to/kid_replica_global.csv \\
      --zbin-csv /path/to/kid_replica_zbin.csv \\
      --output-dir /path/to/output \\
      --show-delta \\
      --show-comparison \\
      --format png,pdf
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
        default="KID Evaluation: Synthetic vs Test",
        help="Plot title (default: 'KID Evaluation: Synthetic vs Test')",
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
