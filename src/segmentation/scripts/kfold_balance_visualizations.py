#!/usr/bin/env python3
"""K-fold balance visualizations for segmentation experiments.

This script generates comprehensive visualizations of k-fold data distributions,
including lesion/no-lesion balance, real/synthetic data mix, and z-bin distributions.

The visualizations are automatically generated when running kfold_segmentation.py
with --dry-run or during training setup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Color schemes
COLORS = {
    "real": "#2ecc71",  # Green
    "synthetic": "#9b59b6",  # Purple
    "lesion": "#e74c3c",  # Red
    "no_lesion": "#3498db",  # Blue
    "train": "#f39c12",  # Orange
    "val": "#1abc9c",  # Teal
}

FOLD_COLORS = sns.color_palette("husl", 5)


def load_data(
    stats_path: Path | str,
    csv_path: Path | str | None = None,
) -> tuple[list[dict], pd.DataFrame | None]:
    """Load k-fold statistics and optional CSV data.

    Args:
        stats_path: Path to kfold_statistics.json
        csv_path: Optional path to kfold_plan CSV

    Returns:
        Tuple of (statistics list, DataFrame or None)
    """
    with open(stats_path, "r") as f:
        stats = json.load(f)

    df = None
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)

    return stats, df


def create_fold_overview_plot(stats: list[dict], ax: Axes) -> None:
    """Create overview bar plot of samples per fold and split.

    Args:
        stats: List of fold statistics dictionaries
        ax: Matplotlib axes to plot on
    """
    folds = [s["fold"] for s in stats]
    train_totals = [s["train"]["total"] for s in stats]
    val_totals = [s["val"]["total"] for s in stats]
    train_real = [s["train"]["real"] for s in stats]
    train_synth = [s["train"]["synthetic"] for s in stats]

    x = np.arange(len(folds))
    width = 0.35

    # Stacked bars for training
    bars1 = ax.bar(
        x - width / 2,
        train_real,
        width,
        label="Train (Real)",
        color=COLORS["real"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x - width / 2,
        train_synth,
        width,
        bottom=train_real,
        label="Train (Synthetic)",
        color=COLORS["synthetic"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Validation bars
    bars3 = ax.bar(
        x + width / 2,
        val_totals,
        width,
        label="Validation",
        color=COLORS["val"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels
    for i, (real, synth, val) in enumerate(zip(train_real, train_synth, val_totals)):
        total_train = real + synth
        ax.annotate(
            f"{total_train:,}",
            xy=(i - width / 2, total_train),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        ax.annotate(
            f"{val:,}",
            xy=(i + width / 2, val),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Fold", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax.set_title("Samples per Fold (Train vs Validation)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))


def create_lesion_ratio_plot(stats: list[dict], ax: Axes) -> None:
    """Create lesion ratio comparison across folds.

    Args:
        stats: List of fold statistics dictionaries
        ax: Matplotlib axes to plot on
    """
    folds = [s["fold"] for s in stats]
    train_ratios = [s["train"]["lesion_ratio"] * 100 for s in stats]
    val_ratios = [s["val"]["lesion_ratio"] * 100 for s in stats]

    x = np.arange(len(folds))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        train_ratios,
        width,
        label="Train",
        color=COLORS["train"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        val_ratios,
        width,
        label="Validation",
        color=COLORS["val"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # Add horizontal line for average
    avg_train = np.mean(train_ratios)
    avg_val = np.mean(val_ratios)
    ax.axhline(float(avg_train), color=COLORS["train"], linestyle="--", alpha=0.7, linewidth=1.5)
    ax.axhline(float(avg_val), color=COLORS["val"], linestyle="--", alpha=0.7, linewidth=1.5)

    ax.set_xlabel("Fold", fontsize=11, fontweight="bold")
    ax.set_ylabel("Lesion Ratio (%)", fontsize=11, fontweight="bold")
    ax.set_title("Lesion Class Imbalance per Fold", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add note about imbalance
    ax.text(
        0.02,
        0.98,
        f"Train avg: {avg_train:.1f}%\nVal avg: {avg_val:.1f}%",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def create_zbin_heatmap(stats: list[dict], ax: Axes, split: str = "train") -> None:
    """Create heatmap of z-bin distribution across folds.

    Args:
        stats: List of fold statistics dictionaries
        ax: Matplotlib axes to plot on
        split: 'train' or 'val'
    """
    # Build matrix: rows = z-bins, columns = folds
    n_folds = len(stats)

    # Get all z-bins
    all_zbins = set()
    for s in stats:
        all_zbins.update(s[split]["zbins"].keys())

    zbins = sorted([int(z) for z in all_zbins])

    # Create matrix for lesion ratio per z-bin
    matrix = np.zeros((len(zbins), n_folds))

    for fold_idx, s in enumerate(stats):
        for zbin_idx, zbin in enumerate(zbins):
            zbin_data = s[split]["zbins"].get(str(zbin), {"lesion": 0, "no_lesion": 0})
            total = zbin_data.get("lesion", 0) + zbin_data.get("no_lesion", 0)
            if total > 0:
                matrix[zbin_idx, fold_idx] = zbin_data.get("lesion", 0) / total * 100

    # Create heatmap
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=100)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Lesion %", fontsize=10)

    # Labels
    ax.set_xticks(np.arange(n_folds))
    ax.set_xticklabels([f"F{i}" for i in range(n_folds)])
    ax.set_xlabel("Fold", fontsize=11, fontweight="bold")

    # Show every 5th z-bin label
    y_ticks = np.arange(0, len(zbins), 5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(zbins[i]) for i in y_ticks])
    ax.set_ylabel("Z-bin", fontsize=11, fontweight="bold")

    title = f"Lesion Distribution by Z-bin ({split.capitalize()})"
    ax.set_title(title, fontsize=12, fontweight="bold")


def create_source_pie_charts(stats: list[dict], ax: Axes) -> None:
    """Create pie chart showing real vs synthetic data composition.

    Args:
        stats: List of fold statistics dictionaries
        ax: Matplotlib axes to plot on
    """
    # Aggregate across all folds
    total_real = sum(s["train"]["real"] for s in stats)
    total_synth = sum(s["train"]["synthetic"] for s in stats)
    total_lesion = sum(s["train"]["lesion"] for s in stats)
    total_no_lesion = sum(s["train"]["no_lesion"] for s in stats)

    # Create two pie charts side by side
    ax.set_axis_off()

    # Source distribution (left)
    ax1 = ax.inset_axes((0.05, 0.1, 0.4, 0.8))
    sizes1 = [total_real, total_synth]
    labels1 = [f"Real\n({total_real:,})", f"Synthetic\n({total_synth:,})"]
    colors1 = [COLORS["real"], COLORS["synthetic"]]
    explode1 = (0.02, 0.02)

    wedges1, texts1, autotexts1 = ax1.pie(
        sizes1,
        labels=labels1,
        colors=colors1,
        autopct="%1.1f%%",
        explode=explode1,
        startangle=90,
        wedgeprops=dict(width=0.7, edgecolor="white"),
    )
    ax1.set_title("Data Source", fontsize=11, fontweight="bold")

    # Lesion distribution (right)
    ax2 = ax.inset_axes((0.55, 0.1, 0.4, 0.8))
    sizes2 = [total_lesion, total_no_lesion]
    labels2 = [f"Lesion\n({total_lesion:,})", f"No Lesion\n({total_no_lesion:,})"]
    colors2 = [COLORS["lesion"], COLORS["no_lesion"]]
    explode2 = (0.05, 0)

    wedges2, texts2, autotexts2 = ax2.pie(
        sizes2,
        labels=labels2,
        colors=colors2,
        autopct="%1.1f%%",
        explode=explode2,
        startangle=90,
        wedgeprops=dict(width=0.7, edgecolor="white"),
    )
    ax2.set_title("Class Distribution", fontsize=11, fontweight="bold")

    # Style the percentage text
    for autotext in autotexts1 + autotexts2:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")


def create_zbin_distribution_plot(
    df: pd.DataFrame | None,
    stats: list[dict],
    ax: Axes,
    fold: int = 0,
) -> None:
    """Create stacked bar plot of z-bin distribution for a specific fold.

    Args:
        df: DataFrame with sample data (optional)
        stats: List of fold statistics dictionaries
        ax: Matplotlib axes to plot on
        fold: Fold index to visualize
    """
    if df is not None and not df.empty:
        # Use DataFrame for detailed breakdown
        fold_data = df[(df["fold"] == fold) & (df["split"] == "train")]

        # Group by z_bin and source
        grouped = fold_data.groupby(["z_bin", "source", "has_lesion_slice"]).size().unstack(
            fill_value=0
        )

        zbins = sorted(fold_data["z_bin"].unique())
        x = np.arange(len(zbins))
        width = 0.6

        # Calculate heights for each category
        real_no_lesion = []
        real_lesion = []
        synth_no_lesion = []
        synth_lesion = []

        for zbin in zbins:
            try:
                rn = fold_data[
                    (fold_data["z_bin"] == zbin)
                    & (fold_data["source"] == "real")
                    & (fold_data["has_lesion_slice"] == False)  # noqa: E712
                ].shape[0]
            except Exception:
                rn = 0
            try:
                rl = fold_data[
                    (fold_data["z_bin"] == zbin)
                    & (fold_data["source"] == "real")
                    & (fold_data["has_lesion_slice"] == True)  # noqa: E712
                ].shape[0]
            except Exception:
                rl = 0
            try:
                sn = fold_data[
                    (fold_data["z_bin"] == zbin)
                    & (fold_data["source"] == "synthetic")
                    & (fold_data["has_lesion_slice"] == False)  # noqa: E712
                ].shape[0]
            except Exception:
                sn = 0
            try:
                sl = fold_data[
                    (fold_data["z_bin"] == zbin)
                    & (fold_data["source"] == "synthetic")
                    & (fold_data["has_lesion_slice"] == True)  # noqa: E712
                ].shape[0]
            except Exception:
                sl = 0

            real_no_lesion.append(rn)
            real_lesion.append(rl)
            synth_no_lesion.append(sn)
            synth_lesion.append(sl)

        # Stacked bar plot
        bottom1 = np.zeros(len(zbins))
        ax.bar(x, real_no_lesion, width, label="Real (No Lesion)", color="#27ae60", bottom=bottom1)
        bottom1 += real_no_lesion

        ax.bar(x, real_lesion, width, label="Real (Lesion)", color="#e74c3c", bottom=bottom1)
        bottom1 += real_lesion

        ax.bar(x, synth_no_lesion, width, label="Synth (No Lesion)", color="#8e44ad", bottom=bottom1)
        bottom1 += synth_no_lesion

        ax.bar(x, synth_lesion, width, label="Synth (Lesion)", color="#f39c12", bottom=bottom1)

    else:
        # Fall back to stats-based visualization
        fold_stats = stats[fold]["train"]["zbins"]
        zbins = sorted([int(z) for z in fold_stats.keys()])
        x = np.arange(len(zbins))
        width = 0.6

        lesion_counts = [fold_stats[str(z)].get("lesion", 0) for z in zbins]
        no_lesion_counts = [fold_stats[str(z)].get("no_lesion", 0) for z in zbins]

        ax.bar(x, no_lesion_counts, width, label="No Lesion", color=COLORS["no_lesion"])
        ax.bar(x, lesion_counts, width, bottom=no_lesion_counts, label="Lesion", color=COLORS["lesion"])

    ax.set_xlabel("Z-bin", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sample Count", fontsize=11, fontweight="bold")
    ax.set_title(f"Z-bin Distribution (Fold {fold}, Train)", fontsize=12, fontweight="bold")

    # Show every 5th label
    tick_positions = np.arange(0, len(zbins), 5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(zbins[i]) for i in tick_positions])

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_kfold_visualizations(
    stats_path: Path | str,
    csv_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    show: bool = False,
) -> Path:
    """Generate comprehensive k-fold visualization panel.

    Args:
        stats_path: Path to kfold_statistics.json
        csv_path: Optional path to kfold_plan CSV for detailed breakdowns
        output_dir: Output directory for saving figures (default: same as stats)
        show: Whether to display the figure

    Returns:
        Path to saved figure
    """
    # Load data
    stats, df = load_data(stats_path, csv_path)

    # Determine output path
    if output_dir is None:
        output_dir = Path(stats_path).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(14, 12), dpi=150)
    fig.suptitle(
        "K-Fold Cross-Validation Data Distribution",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Samples per fold overview
    ax1 = fig.add_subplot(gs[0, 0])
    create_fold_overview_plot(stats, ax1)
    ax1.text(-0.1, 1.05, "A", transform=ax1.transAxes, fontsize=14, fontweight="bold")

    # Panel B: Lesion ratio comparison
    ax2 = fig.add_subplot(gs[0, 1])
    create_lesion_ratio_plot(stats, ax2)
    ax2.text(-0.1, 1.05, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    # Panel C: Data composition (pie charts)
    ax3 = fig.add_subplot(gs[1, 0])
    create_source_pie_charts(stats, ax3)
    ax3.text(-0.05, 1.02, "C", transform=ax3.transAxes, fontsize=14, fontweight="bold")

    # Panel D: Z-bin distribution for fold 0
    ax4 = fig.add_subplot(gs[1, 1])
    create_zbin_distribution_plot(df, stats, ax4, fold=0)
    ax4.text(-0.1, 1.05, "D", transform=ax4.transAxes, fontsize=14, fontweight="bold")

    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    # Save figure
    fig_path = output_dir / "kfold_distribution_overview.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved k-fold visualization to: {fig_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig_path


def generate_detailed_zbin_plots(
    stats_path: Path | str,
    csv_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    """Generate detailed z-bin heatmaps for train and validation.

    Args:
        stats_path: Path to kfold_statistics.json
        csv_path: Optional path to kfold_plan CSV
        output_dir: Output directory for saving figures

    Returns:
        Path to saved figure
    """
    stats, df = load_data(stats_path, csv_path)

    if output_dir is None:
        output_dir = Path(stats_path).parent
    output_dir = Path(output_dir)

    # Create figure with train and val heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=150)
    fig.suptitle(
        "Z-bin Lesion Distribution Heatmaps",
        fontsize=14,
        fontweight="bold",
    )

    create_zbin_heatmap(stats, axes[0], split="train")
    create_zbin_heatmap(stats, axes[1], split="val")

    plt.tight_layout()

    fig_path = output_dir / "kfold_zbin_heatmaps.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved z-bin heatmaps to: {fig_path}")

    plt.close(fig)
    return fig_path


def generate_per_fold_breakdown(
    stats_path: Path | str,
    csv_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    """Generate per-fold detailed breakdown plots.

    Args:
        stats_path: Path to kfold_statistics.json
        csv_path: Optional path to kfold_plan CSV
        output_dir: Output directory for saving figures

    Returns:
        Path to saved figure
    """
    stats, df = load_data(stats_path, csv_path)
    n_folds = len(stats)

    if output_dir is None:
        output_dir = Path(stats_path).parent
    output_dir = Path(output_dir)

    # Create figure with one row per fold
    fig, axes = plt.subplots(n_folds, 2, figsize=(14, 4 * n_folds), dpi=150)
    fig.suptitle(
        "Per-Fold Z-bin Distributions",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for fold_idx in range(n_folds):
        # Train distribution
        create_zbin_distribution_plot(df, stats, axes[fold_idx, 0], fold=fold_idx)
        axes[fold_idx, 0].set_title(f"Fold {fold_idx} - Train", fontsize=11, fontweight="bold")

        # Validation - simplified view from stats
        fold_stats = stats[fold_idx]["val"]["zbins"]
        zbins = sorted([int(z) for z in fold_stats.keys()])
        x = np.arange(len(zbins))

        lesion_counts = [fold_stats[str(z)].get("lesion", 0) for z in zbins]
        no_lesion_counts = [fold_stats[str(z)].get("no_lesion", 0) for z in zbins]

        axes[fold_idx, 1].bar(x, no_lesion_counts, 0.6, label="No Lesion", color=COLORS["no_lesion"])
        axes[fold_idx, 1].bar(
            x, lesion_counts, 0.6, bottom=no_lesion_counts, label="Lesion", color=COLORS["lesion"]
        )
        axes[fold_idx, 1].set_xlabel("Z-bin", fontsize=10)
        axes[fold_idx, 1].set_ylabel("Count", fontsize=10)
        axes[fold_idx, 1].set_title(f"Fold {fold_idx} - Validation", fontsize=11, fontweight="bold")

        tick_positions = np.arange(0, len(zbins), 5)
        axes[fold_idx, 1].set_xticks(tick_positions)
        axes[fold_idx, 1].set_xticklabels([zbins[i] for i in tick_positions])
        axes[fold_idx, 1].legend(loc="upper right", fontsize=8)
        axes[fold_idx, 1].spines["top"].set_visible(False)
        axes[fold_idx, 1].spines["right"].set_visible(False)

    plt.tight_layout()

    fig_path = output_dir / "kfold_per_fold_breakdown.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved per-fold breakdown to: {fig_path}")

    plt.close(fig)
    return fig_path


def generate_all_visualizations(
    stats_path: Path | str,
    csv_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> list[Path]:
    """Generate all k-fold visualizations.

    Args:
        stats_path: Path to kfold_statistics.json
        csv_path: Optional path to kfold_plan CSV
        output_dir: Output directory for saving figures

    Returns:
        List of paths to saved figures
    """
    if output_dir is None:
        output_dir = Path(stats_path).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating k-fold visualizations...")

    paths = []

    # Main overview panel
    paths.append(generate_kfold_visualizations(stats_path, csv_path, output_dir))

    # Z-bin heatmaps
    paths.append(generate_detailed_zbin_plots(stats_path, csv_path, output_dir))

    # Per-fold breakdown
    paths.append(generate_per_fold_breakdown(stats_path, csv_path, output_dir))

    logger.info(f"Generated {len(paths)} visualization files in {output_dir}")

    return paths


# CLI interface for standalone usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate k-fold balance visualizations"
    )
    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="Path to kfold_statistics.json",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to kfold_plan CSV (optional, for detailed breakdowns)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as stats file)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively",
    )

    args = parser.parse_args()

    # Setup basic logging
    logging.basicConfig(level=logging.INFO)

    # Generate visualizations
    generate_all_visualizations(
        stats_path=args.stats,
        csv_path=args.csv,
        output_dir=args.output_dir,
    )
