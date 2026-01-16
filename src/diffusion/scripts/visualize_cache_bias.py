#!/usr/bin/env python3
"""Visualize slice cache statistics to analyze potential model bias.

This script analyzes a slice_cache directory and creates visualizations to help
answer the question: "Will the model be biased towards generating a better
specific type of image/lesion?"

Analyses performed:
1. Lesion area distribution through z-bins
2. Lesion count distribution through z-bins
3. Split comparisons (train/val/test)
4. Domain distribution (control vs epilepsy)
5. Token distribution analysis
6. Sample images per z-bin and pathology class

Usage:
    python src/diffusion/scripts/visualize_cache_bias.py --cache-dir /path/to/slice_cache
    python src/diffusion/scripts/visualize_cache_bias.py --cache-dir /path/to/slice_cache --no-show
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_all_splits(cache_dir: Path) -> pd.DataFrame:
    """Load all split CSVs from the cache directory.

    Args:
        cache_dir: Path to the slice cache directory.

    Returns:
        Combined DataFrame with all splits.
    """
    dfs = []
    for split in ["train", "val", "test"]:
        csv_path = cache_dir / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} slices from {split}.csv")
            dfs.append(df)
        else:
            logger.warning(f"{split}.csv not found, skipping...")

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {cache_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total slices loaded: {len(combined)}")
    return combined


def create_output_dir(cache_dir: Path) -> Path:
    """Create visualizations directory within cache directory.

    Args:
        cache_dir: Path to the slice cache directory.

    Returns:
        Path to the visualizations directory.
    """
    viz_dir = cache_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {viz_dir}")
    return viz_dir


def plot_lesion_area_by_zbin(
    df: pd.DataFrame,
    output_dir: Path,
    show: bool = True,
) -> None:
    """Plot lesion area distribution across z-bins.

    Args:
        df: DataFrame with slice data.
        output_dir: Output directory for plots.
        show: Whether to display plots.
    """
    # Filter for lesion slices only
    lesion_df = df[df["has_lesion"] == True].copy()

    if len(lesion_df) == 0:
        logger.warning("No lesion slices found, skipping lesion area analysis")
        return

    # Check if lesion_area_px column exists
    if "lesion_area_px" not in lesion_df.columns:
        logger.warning("lesion_area_px column not found, skipping lesion area analysis")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Lesion Area Distribution Analysis\n"
        "(Analyzing potential bias in lesion sizes)",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Box plot of lesion area by z-bin
    ax = axes[0, 0]
    zbins = sorted(lesion_df["z_bin"].unique())
    data_by_zbin = [lesion_df[lesion_df["z_bin"] == zb]["lesion_area_px"].values for zb in zbins]
    bp = ax.boxplot(data_by_zbin, positions=zbins, widths=0.6)
    ax.set_xlabel("Z-bin", fontweight="bold")
    ax.set_ylabel("Lesion Area (pixels)", fontweight="bold")
    ax.set_title("Lesion Area Distribution per Z-bin")
    ax.grid(axis="y", alpha=0.3)

    # 2. Violin plot by split
    ax = axes[0, 1]
    splits = ["train", "val", "test"]
    colors = ["#3498DB", "#E74C3C", "#2ECC71"]
    positions = []
    violins_data = []
    for i, split in enumerate(splits):
        split_lesions = lesion_df[lesion_df["split"] == split]["lesion_area_px"].values
        if len(split_lesions) > 0:
            violins_data.append(split_lesions)
            positions.append(i)

    if violins_data:
        parts = ax.violinplot(violins_data, positions=positions, showmeans=True, showmedians=True)
        ax.set_xticks(positions)
        ax.set_xticklabels([splits[p] for p in positions])
        ax.set_xlabel("Split", fontweight="bold")
        ax.set_ylabel("Lesion Area (pixels)", fontweight="bold")
        ax.set_title("Lesion Area Distribution by Split")
        ax.grid(axis="y", alpha=0.3)

    # 3. Histogram of lesion areas
    ax = axes[1, 0]
    for split, color in zip(splits, colors):
        split_areas = lesion_df[lesion_df["split"] == split]["lesion_area_px"].values
        if len(split_areas) > 0:
            ax.hist(
                split_areas,
                bins=30,
                alpha=0.5,
                label=f"{split} (n={len(split_areas)})",
                color=color,
                edgecolor="black",
            )
    ax.set_xlabel("Lesion Area (pixels)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Lesion Area Histogram by Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. Mean lesion area per z-bin by split
    ax = axes[1, 1]
    for split, color in zip(splits, colors):
        split_lesions = lesion_df[lesion_df["split"] == split]
        if len(split_lesions) > 0:
            mean_by_zbin = split_lesions.groupby("z_bin")["lesion_area_px"].mean()
            ax.plot(
                mean_by_zbin.index,
                mean_by_zbin.values,
                marker="o",
                label=split,
                color=color,
                linewidth=2,
            )
    ax.set_xlabel("Z-bin", fontweight="bold")
    ax.set_ylabel("Mean Lesion Area (pixels)", fontweight="bold")
    ax.set_title("Mean Lesion Area per Z-bin by Split")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "lesion_area_by_zbin.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_lesion_count_by_zbin(
    df: pd.DataFrame,
    output_dir: Path,
    show: bool = True,
) -> None:
    """Plot lesion count distribution across z-bins.

    Args:
        df: DataFrame with slice data.
        output_dir: Output directory for plots.
        show: Whether to display plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Lesion Count Distribution Analysis\n"
        "(Analyzing which z-bins have more lesion samples)",
        fontsize=14,
        fontweight="bold",
    )

    splits = ["train", "val", "test"]
    colors = ["#3498DB", "#E74C3C", "#2ECC71"]

    # 1. Stacked bar: lesion vs non-lesion per z-bin (all data)
    ax = axes[0, 0]
    zbins = sorted(df["z_bin"].unique())
    lesion_counts = df[df["has_lesion"] == True].groupby("z_bin").size()
    no_lesion_counts = df[df["has_lesion"] == False].groupby("z_bin").size()

    lesion_vals = [lesion_counts.get(zb, 0) for zb in zbins]
    no_lesion_vals = [no_lesion_counts.get(zb, 0) for zb in zbins]

    ax.bar(zbins, no_lesion_vals, label="No Lesion", color="#95A5A6", alpha=0.7)
    ax.bar(zbins, lesion_vals, bottom=no_lesion_vals, label="Lesion", color="#E74C3C", alpha=0.7)
    ax.set_xlabel("Z-bin", fontweight="bold")
    ax.set_ylabel("Number of Slices", fontweight="bold")
    ax.set_title("Slice Distribution: Lesion vs No Lesion")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. Lesion percentage per z-bin
    ax = axes[0, 1]
    total_by_zbin = df.groupby("z_bin").size()
    lesion_by_zbin = df[df["has_lesion"] == True].groupby("z_bin").size()
    lesion_pct = (lesion_by_zbin / total_by_zbin * 100).fillna(0)

    ax.bar(lesion_pct.index, lesion_pct.values, color="#E74C3C", alpha=0.7, edgecolor="black")
    ax.axhline(
        y=lesion_pct.mean(),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {lesion_pct.mean():.1f}%",
    )
    ax.set_xlabel("Z-bin", fontweight="bold")
    ax.set_ylabel("Lesion Percentage (%)", fontweight="bold")
    ax.set_title("Lesion Percentage per Z-bin")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. Lesion count per z-bin by split
    ax = axes[1, 0]
    for split, color in zip(splits, colors):
        split_lesions = df[(df["split"] == split) & (df["has_lesion"] == True)]
        lesion_count_by_zbin = split_lesions.groupby("z_bin").size()
        ax.plot(
            lesion_count_by_zbin.index,
            lesion_count_by_zbin.values,
            marker="o",
            label=split,
            color=color,
            linewidth=2,
        )
    ax.set_xlabel("Z-bin", fontweight="bold")
    ax.set_ylabel("Number of Lesion Slices", fontweight="bold")
    ax.set_title("Lesion Count per Z-bin by Split")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Lesion percentage per z-bin by split
    ax = axes[1, 1]
    for split, color in zip(splits, colors):
        split_df = df[df["split"] == split]
        if len(split_df) > 0:
            total = split_df.groupby("z_bin").size()
            lesion = split_df[split_df["has_lesion"] == True].groupby("z_bin").size()
            pct = (lesion / total * 100).fillna(0)
            ax.plot(pct.index, pct.values, marker="o", label=split, color=color, linewidth=2)
    ax.set_xlabel("Z-bin", fontweight="bold")
    ax.set_ylabel("Lesion Percentage (%)", fontweight="bold")
    ax.set_title("Lesion Percentage per Z-bin by Split")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "lesion_count_by_zbin.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_split_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    show: bool = True,
) -> None:
    """Compare distributions across train/val/test splits.

    Args:
        df: DataFrame with slice data.
        output_dir: Output directory for plots.
        show: Whether to display plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Split Comparison Analysis\n"
        "(Checking if train/val/test have similar distributions)",
        fontsize=14,
        fontweight="bold",
    )

    splits = ["train", "val", "test"]
    colors = ["#3498DB", "#E74C3C", "#2ECC71"]

    # 1. Pie chart: total slices per split
    ax = axes[0, 0]
    split_counts = df.groupby("split").size()
    ax.pie(
        [split_counts.get(s, 0) for s in splits],
        labels=splits,
        autopct="%1.1f%%",
        colors=colors,
        explode=[0.02] * 3,
    )
    ax.set_title("Total Slices per Split")

    # 2. Pie chart: lesion slices per split
    ax = axes[0, 1]
    lesion_df = df[df["has_lesion"] == True]
    lesion_split_counts = lesion_df.groupby("split").size()
    ax.pie(
        [lesion_split_counts.get(s, 0) for s in splits],
        labels=splits,
        autopct="%1.1f%%",
        colors=colors,
        explode=[0.02] * 3,
    )
    ax.set_title("Lesion Slices per Split")

    # 3. Bar chart: lesion vs no-lesion by split
    ax = axes[0, 2]
    x = np.arange(len(splits))
    width = 0.35
    lesion_counts = [len(df[(df["split"] == s) & (df["has_lesion"] == True)]) for s in splits]
    no_lesion_counts = [len(df[(df["split"] == s) & (df["has_lesion"] == False)]) for s in splits]
    ax.bar(x - width / 2, no_lesion_counts, width, label="No Lesion", color="#95A5A6")
    ax.bar(x + width / 2, lesion_counts, width, label="Lesion", color="#E74C3C")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_xlabel("Split", fontweight="bold")
    ax.set_ylabel("Number of Slices", fontweight="bold")
    ax.set_title("Lesion vs No-Lesion by Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4-6. Z-bin histograms per split
    for i, (split, color) in enumerate(zip(splits, colors)):
        ax = axes[1, i]
        split_df = df[df["split"] == split]
        zbins = sorted(split_df["z_bin"].unique())

        lesion_hist = split_df[split_df["has_lesion"] == True].groupby("z_bin").size()
        no_lesion_hist = split_df[split_df["has_lesion"] == False].groupby("z_bin").size()

        lesion_vals = [lesion_hist.get(zb, 0) for zb in zbins]
        no_lesion_vals = [no_lesion_hist.get(zb, 0) for zb in zbins]

        ax.bar(zbins, no_lesion_vals, label="No Lesion", color="#95A5A6", alpha=0.7)
        ax.bar(zbins, lesion_vals, bottom=no_lesion_vals, label="Lesion", color=color, alpha=0.7)
        ax.set_xlabel("Z-bin", fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title(f"{split.upper()} Z-bin Distribution")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "split_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_domain_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    show: bool = True,
) -> None:
    """Analyze domain distribution (control vs epilepsy).

    Args:
        df: DataFrame with slice data.
        output_dir: Output directory for plots.
        show: Whether to display plots.
    """
    if "source" not in df.columns:
        logger.warning("source column not found, skipping domain analysis")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Domain Distribution Analysis\n"
        "(Analyzing control vs epilepsy data balance)",
        fontsize=14,
        fontweight="bold",
    )

    domains = df["source"].unique().tolist()
    domain_colors = {"control": "#3498DB", "epilepsy": "#E74C3C"}

    # 1. Pie chart: domain distribution
    ax = axes[0, 0]
    domain_counts = df.groupby("source").size()
    ax.pie(
        domain_counts.values,
        labels=domain_counts.index,
        autopct="%1.1f%%",
        colors=[domain_colors.get(d, "#95A5A6") for d in domain_counts.index],
        explode=[0.02] * len(domain_counts),
    )
    ax.set_title("Overall Domain Distribution")

    # 2. Domain distribution by split
    ax = axes[0, 1]
    splits = ["train", "val", "test"]
    x = np.arange(len(splits))
    width = 0.35
    for i, domain in enumerate(domains):
        counts = [len(df[(df["split"] == s) & (df["source"] == domain)]) for s in splits]
        offset = width * (i - len(domains) / 2 + 0.5)
        ax.bar(
            x + offset,
            counts,
            width,
            label=domain,
            color=domain_colors.get(domain, "#95A5A6"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_xlabel("Split", fontweight="bold")
    ax.set_ylabel("Number of Slices", fontweight="bold")
    ax.set_title("Domain Distribution by Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. Domain x lesion breakdown
    ax = axes[1, 0]
    categories = []
    counts = []
    colors_list = []
    for domain in domains:
        for has_lesion in [False, True]:
            count = len(df[(df["source"] == domain) & (df["has_lesion"] == has_lesion)])
            lesion_str = "Lesion" if has_lesion else "No Lesion"
            categories.append(f"{domain}\n{lesion_str}")
            counts.append(count)
            base_color = domain_colors.get(domain, "#95A5A6")
            colors_list.append(base_color if has_lesion else "#95A5A6")
    ax.bar(categories, counts, color=colors_list, edgecolor="black")
    ax.set_ylabel("Number of Slices", fontweight="bold")
    ax.set_title("Domain x Lesion Breakdown")
    ax.grid(axis="y", alpha=0.3)

    # 4. Domain distribution per z-bin
    ax = axes[1, 1]
    zbins = sorted(df["z_bin"].unique())
    for domain in domains:
        domain_df = df[df["source"] == domain]
        counts_by_zbin = domain_df.groupby("z_bin").size()
        ax.plot(
            counts_by_zbin.index,
            counts_by_zbin.values,
            marker="o",
            label=domain,
            color=domain_colors.get(domain, "#95A5A6"),
            linewidth=2,
        )
    ax.set_xlabel("Z-bin", fontweight="bold")
    ax.set_ylabel("Number of Slices", fontweight="bold")
    ax.set_title("Domain Distribution per Z-bin")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "domain_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_token_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    show: bool = True,
) -> None:
    """Analyze token distribution (z_bin + pathology_class * z_bins).

    Args:
        df: DataFrame with slice data.
        output_dir: Output directory for plots.
        show: Whether to display plots.
    """
    if "token" not in df.columns:
        logger.warning("token column not found, skipping token analysis")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Token Distribution Analysis\n"
        "(Tokens encode z-bin and pathology class)",
        fontsize=14,
        fontweight="bold",
    )

    splits = ["train", "val", "test"]
    colors = ["#3498DB", "#E74C3C", "#2ECC71"]

    # 1. Token histogram (all data)
    ax = axes[0, 0]
    tokens = sorted(df["token"].unique())
    token_counts = df.groupby("token").size()
    ax.bar(token_counts.index, token_counts.values, color="#9B59B6", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Token", fontweight="bold")
    ax.set_ylabel("Number of Slices", fontweight="bold")
    ax.set_title("Token Distribution (All Data)")
    ax.grid(axis="y", alpha=0.3)

    # Add z_bins annotation
    n_zbins = df["z_bin"].max() + 1
    ax.axvline(x=n_zbins - 0.5, color="red", linestyle="--", linewidth=2)
    ax.text(
        n_zbins / 2,
        ax.get_ylim()[1] * 0.9,
        "No Lesion",
        ha="center",
        fontsize=10,
        color="gray",
    )
    ax.text(
        n_zbins * 1.5,
        ax.get_ylim()[1] * 0.9,
        "Lesion",
        ha="center",
        fontsize=10,
        color="gray",
    )

    # 2. Token distribution by split
    ax = axes[0, 1]
    for split, color in zip(splits, colors):
        split_df = df[df["split"] == split]
        token_counts = split_df.groupby("token").size()
        ax.plot(
            token_counts.index,
            token_counts.values,
            marker="o",
            label=split,
            color=color,
            linewidth=2,
            markersize=4,
        )
    ax.axvline(x=n_zbins - 0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Token", fontweight="bold")
    ax.set_ylabel("Number of Slices", fontweight="bold")
    ax.set_title("Token Distribution by Split")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Token imbalance ratio (train only)
    ax = axes[1, 0]
    train_df = df[df["split"] == "train"]
    train_token_counts = train_df.groupby("token").size()
    max_count = train_token_counts.max()
    imbalance_ratio = max_count / train_token_counts
    ax.bar(
        imbalance_ratio.index,
        imbalance_ratio.values,
        color="#E67E22",
        alpha=0.7,
        edgecolor="black",
    )
    ax.axhline(y=1, color="green", linestyle="--", linewidth=2, label="Balanced (ratio=1)")
    ax.set_xlabel("Token", fontweight="bold")
    ax.set_ylabel("Imbalance Ratio (max/count)", fontweight="bold")
    ax.set_title("Token Imbalance Ratio (Train Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. Token coverage heatmap
    ax = axes[1, 1]
    n_tokens = len(tokens)
    coverage_matrix = np.zeros((len(splits), n_tokens))
    for i, split in enumerate(splits):
        split_df = df[df["split"] == split]
        for j, token in enumerate(tokens):
            coverage_matrix[i, j] = len(split_df[split_df["token"] == token])

    im = ax.imshow(coverage_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels(splits)
    ax.set_xlabel("Token", fontweight="bold")
    ax.set_title("Token Coverage Heatmap")
    plt.colorbar(im, ax=ax, label="Count")

    plt.tight_layout()
    output_path = output_dir / "token_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_subject_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    show: bool = True,
) -> None:
    """Analyze subject-level statistics.

    Args:
        df: DataFrame with slice data.
        output_dir: Output directory for plots.
        show: Whether to display plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Subject-Level Analysis\n"
        "(Checking for subject-level biases)",
        fontsize=14,
        fontweight="bold",
    )

    splits = ["train", "val", "test"]
    colors = ["#3498DB", "#E74C3C", "#2ECC71"]

    # 1. Subjects per split
    ax = axes[0, 0]
    subject_counts = [df[df["split"] == s]["subject_id"].nunique() for s in splits]
    ax.bar(splits, subject_counts, color=colors, edgecolor="black")
    ax.set_xlabel("Split", fontweight="bold")
    ax.set_ylabel("Number of Subjects", fontweight="bold")
    ax.set_title("Unique Subjects per Split")
    for i, count in enumerate(subject_counts):
        ax.text(i, count + 0.5, str(count), ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 2. Slices per subject distribution
    ax = axes[0, 1]
    for split, color in zip(splits, colors):
        split_df = df[df["split"] == split]
        slices_per_subject = split_df.groupby("subject_id").size().values
        ax.hist(
            slices_per_subject,
            bins=20,
            alpha=0.5,
            label=f"{split} (n={len(slices_per_subject)})",
            color=color,
            edgecolor="black",
        )
    ax.set_xlabel("Slices per Subject", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of Slices per Subject")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. Lesion slices per subject (for subjects with lesions)
    ax = axes[1, 0]
    lesion_df = df[df["has_lesion"] == True]
    for split, color in zip(splits, colors):
        split_lesions = lesion_df[lesion_df["split"] == split]
        if len(split_lesions) > 0:
            lesion_slices_per_subject = split_lesions.groupby("subject_id").size().values
            ax.hist(
                lesion_slices_per_subject,
                bins=15,
                alpha=0.5,
                label=f"{split} (n={len(lesion_slices_per_subject)})",
                color=color,
                edgecolor="black",
            )
    ax.set_xlabel("Lesion Slices per Subject", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of Lesion Slices per Subject")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. Subject contribution to lesion data
    ax = axes[1, 1]
    lesion_per_subject = lesion_df.groupby("subject_id").size().sort_values(ascending=False)
    top_n = min(20, len(lesion_per_subject))
    top_subjects = lesion_per_subject.head(top_n)
    cumulative_pct = (top_subjects.cumsum() / lesion_per_subject.sum() * 100).values

    ax.bar(range(top_n), top_subjects.values, color="#E74C3C", alpha=0.7, edgecolor="black")
    ax2 = ax.twinx()
    ax2.plot(range(top_n), cumulative_pct, color="blue", marker="o", linewidth=2)
    ax2.set_ylabel("Cumulative %", color="blue", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="blue")

    ax.set_xlabel(f"Top {top_n} Subjects (sorted by lesion count)", fontweight="bold")
    ax.set_ylabel("Number of Lesion Slices", fontweight="bold")
    ax.set_title("Subject Contribution to Lesion Data")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "subject_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def generate_summary_report(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate a text summary report of the analysis.

    Args:
        df: DataFrame with slice data.
        output_dir: Output directory for the report.
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SLICE CACHE BIAS ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Overall statistics
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 40)
    report_lines.append(f"Total slices: {len(df)}")
    report_lines.append(f"Unique subjects: {df['subject_id'].nunique()}")
    report_lines.append(f"Z-bins: {df['z_bin'].nunique()} (0-{df['z_bin'].max()})")
    report_lines.append("")

    # Split breakdown
    report_lines.append("SPLIT BREAKDOWN")
    report_lines.append("-" * 40)
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        n_slices = len(split_df)
        n_subjects = split_df["subject_id"].nunique()
        n_lesion = len(split_df[split_df["has_lesion"] == True])
        lesion_pct = n_lesion / n_slices * 100 if n_slices > 0 else 0
        report_lines.append(f"{split.upper()}:")
        report_lines.append(f"  Slices: {n_slices} ({n_slices/len(df)*100:.1f}%)")
        report_lines.append(f"  Subjects: {n_subjects}")
        report_lines.append(f"  Lesion slices: {n_lesion} ({lesion_pct:.1f}%)")
    report_lines.append("")

    # Lesion analysis
    lesion_df = df[df["has_lesion"] == True]
    report_lines.append("LESION ANALYSIS")
    report_lines.append("-" * 40)
    report_lines.append(f"Total lesion slices: {len(lesion_df)}")
    report_lines.append(f"Lesion percentage: {len(lesion_df)/len(df)*100:.1f}%")

    if "lesion_area_px" in lesion_df.columns:
        report_lines.append(f"Mean lesion area: {lesion_df['lesion_area_px'].mean():.1f} px")
        report_lines.append(f"Median lesion area: {lesion_df['lesion_area_px'].median():.1f} px")
        report_lines.append(f"Min lesion area: {lesion_df['lesion_area_px'].min():.0f} px")
        report_lines.append(f"Max lesion area: {lesion_df['lesion_area_px'].max():.0f} px")
    report_lines.append("")

    # Z-bin distribution
    report_lines.append("Z-BIN DISTRIBUTION (Train Set)")
    report_lines.append("-" * 40)
    train_df = df[df["split"] == "train"]
    zbin_counts = train_df.groupby("z_bin").size()
    lesion_by_zbin = train_df[train_df["has_lesion"] == True].groupby("z_bin").size()
    for zbin in sorted(train_df["z_bin"].unique()):
        total = zbin_counts.get(zbin, 0)
        lesion = lesion_by_zbin.get(zbin, 0)
        pct = lesion / total * 100 if total > 0 else 0
        report_lines.append(f"  Z-bin {zbin:2d}: {total:4d} slices, {lesion:3d} lesion ({pct:5.1f}%)")
    report_lines.append("")

    # Potential bias warnings
    report_lines.append("POTENTIAL BIAS WARNINGS")
    report_lines.append("-" * 40)

    # Check for class imbalance
    lesion_pct = len(lesion_df) / len(df) * 100
    if lesion_pct < 10:
        report_lines.append(f"[WARNING] Low lesion percentage ({lesion_pct:.1f}%)")
        report_lines.append("  - Model may underfit lesion generation")
    elif lesion_pct > 50:
        report_lines.append(f"[WARNING] High lesion percentage ({lesion_pct:.1f}%)")
        report_lines.append("  - Model may overfit lesion patterns")

    # Check for z-bin imbalance
    train_zbin_counts = train_df.groupby("z_bin").size()
    zbin_ratio = train_zbin_counts.max() / train_zbin_counts.min()
    if zbin_ratio > 3:
        report_lines.append(f"[WARNING] Z-bin imbalance ratio: {zbin_ratio:.1f}x")
        report_lines.append("  - Model may generate better images for certain z-bins")

    # Check for split distribution mismatch
    for split in ["val", "test"]:
        split_df = df[df["split"] == split]
        split_lesion_pct = len(split_df[split_df["has_lesion"] == True]) / len(split_df) * 100
        train_lesion_pct = len(train_df[train_df["has_lesion"] == True]) / len(train_df) * 100
        if abs(split_lesion_pct - train_lesion_pct) > 10:
            report_lines.append(
                f"[WARNING] {split} lesion % ({split_lesion_pct:.1f}%) differs from "
                f"train ({train_lesion_pct:.1f}%)"
            )

    # Check for subject dominance
    lesion_per_subject = lesion_df.groupby("subject_id").size()
    top_subject_contribution = lesion_per_subject.max() / len(lesion_df) * 100
    if top_subject_contribution > 20:
        report_lines.append(
            f"[WARNING] Single subject contributes {top_subject_contribution:.1f}% "
            "of lesion data"
        )
        report_lines.append("  - Model may overfit to this subject's lesion patterns")

    if "lesion_area_px" in lesion_df.columns:
        # Check for lesion area distribution across z-bins
        area_by_zbin = lesion_df.groupby("z_bin")["lesion_area_px"].mean()
        area_ratio = area_by_zbin.max() / area_by_zbin.min()
        if area_ratio > 3:
            report_lines.append(f"[WARNING] Lesion area varies {area_ratio:.1f}x across z-bins")
            report_lines.append("  - Model may learn z-bin-specific lesion sizes")

    report_lines.append("")
    report_lines.append("=" * 70)

    # Save report
    report_text = "\n".join(report_lines)
    report_path = output_dir / "bias_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Saved: {report_path}")

    # Also print to console
    print(report_text)


def main() -> None:
    """Main entry point for cache bias visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize slice cache statistics to analyze potential model bias",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to the slice cache directory",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots (only save)",
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    # Load data
    df = load_all_splits(cache_dir)

    # Create output directory
    output_dir = create_output_dir(cache_dir)

    # Generate visualizations
    show = not args.no_show

    logger.info("Generating visualizations...")

    plot_lesion_area_by_zbin(df, output_dir, show)
    plot_lesion_count_by_zbin(df, output_dir, show)
    plot_split_comparison(df, output_dir, show)
    plot_domain_analysis(df, output_dir, show)
    plot_token_distribution(df, output_dir, show)
    plot_subject_analysis(df, output_dir, show)

    # Generate summary report
    generate_summary_report(df, output_dir)

    logger.info(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
