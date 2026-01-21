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
from scipy import stats

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


# =============================================================================
# Statistical Tests for Split Similarity
# =============================================================================


def compute_split_divergence(
    df: pd.DataFrame,
    thresholds: Optional[Dict] = None,
) -> Dict:
    """Compute statistical divergence metrics between train/val/test splits.

    Args:
        df: DataFrame with all splits.
        thresholds: Optional dict with threshold values for warnings.

    Returns:
        Dictionary with divergence metrics:
        - lesion_percentage: train/val/test percentages and differences
        - z_bin_distribution: chi-squared test results
        - lesion_area: Kolmogorov-Smirnov test results
    """
    if thresholds is None:
        thresholds = {
            "lesion_percentage_diff": 5.0,
            "chi_squared_p_value": 0.05,
            "ks_statistic": 0.1,
        }

    results = {}

    # Split DataFrames
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # 1. Lesion percentage comparison
    train_lesion_pct = train_df["has_lesion"].mean() * 100 if len(train_df) > 0 else 0
    val_lesion_pct = val_df["has_lesion"].mean() * 100 if len(val_df) > 0 else 0
    test_lesion_pct = test_df["has_lesion"].mean() * 100 if len(test_df) > 0 else 0

    train_val_diff = abs(train_lesion_pct - val_lesion_pct)
    train_test_diff = abs(train_lesion_pct - test_lesion_pct)

    results["lesion_percentage"] = {
        "train": train_lesion_pct,
        "val": val_lesion_pct,
        "test": test_lesion_pct,
        "train_val_diff": train_val_diff,
        "train_test_diff": train_test_diff,
        "train_val_significant": train_val_diff > thresholds["lesion_percentage_diff"],
        "train_test_significant": train_test_diff > thresholds["lesion_percentage_diff"],
    }

    # 2. Z-bin distribution chi-squared test (train vs val)
    if len(train_df) > 0 and len(val_df) > 0:
        all_zbins = sorted(set(train_df["z_bin"].unique()) | set(val_df["z_bin"].unique()))

        train_zbin_counts = train_df.groupby("z_bin").size()
        val_zbin_counts = val_df.groupby("z_bin").size()

        # Align counts to all z-bins
        train_aligned = np.array([train_zbin_counts.get(z, 0) for z in all_zbins])
        val_aligned = np.array([val_zbin_counts.get(z, 0) for z in all_zbins])

        # Normalize train to get expected frequencies for val
        train_freq = train_aligned / train_aligned.sum()
        val_expected = train_freq * val_aligned.sum()

        # Avoid division by zero - set minimum expected count
        val_expected = np.maximum(val_expected, 0.5)

        try:
            chi2_stat, chi2_p = stats.chisquare(val_aligned, val_expected)
        except Exception:
            chi2_stat, chi2_p = 0.0, 1.0

        results["z_bin_distribution"] = {
            "chi2_statistic": float(chi2_stat),
            "p_value": float(chi2_p),
            "significant": chi2_p < thresholds["chi_squared_p_value"],
            "n_zbins": len(all_zbins),
        }
    else:
        results["z_bin_distribution"] = {
            "chi2_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "n_zbins": 0,
        }

    # 3. Lesion area distribution (Kolmogorov-Smirnov test)
    if "lesion_area_px" in df.columns:
        train_lesion_areas = train_df[train_df["has_lesion"] == True]["lesion_area_px"].dropna()
        val_lesion_areas = val_df[val_df["has_lesion"] == True]["lesion_area_px"].dropna()

        if len(train_lesion_areas) > 5 and len(val_lesion_areas) > 5:
            ks_stat, ks_p = stats.ks_2samp(train_lesion_areas, val_lesion_areas)

            results["lesion_area"] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(ks_p),
                "significant": ks_stat > thresholds["ks_statistic"] or ks_p < 0.05,
                "train_mean": float(train_lesion_areas.mean()),
                "val_mean": float(val_lesion_areas.mean()),
                "train_median": float(train_lesion_areas.median()),
                "val_median": float(val_lesion_areas.median()),
            }
        else:
            results["lesion_area"] = {
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "train_mean": 0.0,
                "val_mean": 0.0,
                "train_median": 0.0,
                "val_median": 0.0,
            }
    else:
        results["lesion_area"] = None

    return results


def compute_per_zbin_lesion_comparison(
    df: pd.DataFrame,
    significance_threshold: float = 0.05,
    difference_threshold: float = 10.0,
) -> pd.DataFrame:
    """Compare lesion % per z-bin between train and val using Fisher's exact test.

    Args:
        df: DataFrame with all splits.
        significance_threshold: P-value threshold for significance.
        difference_threshold: Minimum % difference to consider significant.

    Returns:
        DataFrame with columns:
        - z_bin, train_lesion_pct, val_lesion_pct, difference,
          fisher_exact_p, significant
    """
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    results = []

    for z_bin in sorted(df["z_bin"].unique()):
        train_zbin = train_df[train_df["z_bin"] == z_bin]
        val_zbin = val_df[val_df["z_bin"] == z_bin]

        if len(train_zbin) == 0 or len(val_zbin) == 0:
            continue

        train_lesion = int(train_zbin["has_lesion"].sum())
        train_total = len(train_zbin)
        val_lesion = int(val_zbin["has_lesion"].sum())
        val_total = len(val_zbin)

        train_pct = train_lesion / train_total * 100
        val_pct = val_lesion / val_total * 100
        difference = abs(train_pct - val_pct)

        # Fisher's exact test for 2x2 contingency table
        contingency = [
            [train_lesion, train_total - train_lesion],
            [val_lesion, val_total - val_lesion],
        ]

        try:
            _, p_value = stats.fisher_exact(contingency)
        except Exception:
            p_value = 1.0

        results.append(
            {
                "z_bin": z_bin,
                "train_total": train_total,
                "train_lesion": train_lesion,
                "train_lesion_pct": train_pct,
                "val_total": val_total,
                "val_lesion": val_lesion,
                "val_lesion_pct": val_pct,
                "difference": difference,
                "fisher_exact_p": p_value,
                "significant": p_value < significance_threshold and difference > difference_threshold,
            }
        )

    return pd.DataFrame(results)


def generate_enhanced_warnings(
    divergence_metrics: Dict,
    zbin_comparison: pd.DataFrame,
    thresholds: Optional[Dict] = None,
) -> List[str]:
    """Generate actionable warnings based on bias analysis.

    Args:
        divergence_metrics: Output from compute_split_divergence().
        zbin_comparison: Output from compute_per_zbin_lesion_comparison().
        thresholds: Optional dict with threshold values.

    Returns:
        List of warning strings.
    """
    if thresholds is None:
        thresholds = {
            "lesion_percentage_diff": 5.0,
            "chi_squared_p_value": 0.05,
            "ks_statistic": 0.1,
            "significant_zbin_count": 3,
        }

    warnings = []

    # 1. Global lesion percentage
    lp = divergence_metrics["lesion_percentage"]
    if lp["train_val_significant"]:
        warnings.append(
            f"[BIAS] Val lesion % ({lp['val']:.1f}%) differs from train "
            f"({lp['train']:.1f}%) by {lp['train_val_diff']:.1f}% "
            f"(threshold: {thresholds['lesion_percentage_diff']:.1f}%)"
        )

    # 2. Z-bin distribution
    zd = divergence_metrics["z_bin_distribution"]
    if zd["significant"]:
        warnings.append(
            f"[BIAS] Z-bin distribution significantly differs between train/val "
            f"(chi2={zd['chi2_statistic']:.2f}, p={zd['p_value']:.4f})"
        )

    # 3. Lesion area distribution
    if divergence_metrics["lesion_area"] is not None:
        la = divergence_metrics["lesion_area"]
        if la["significant"]:
            warnings.append(
                f"[BIAS] Lesion area distribution differs significantly "
                f"(KS={la['ks_statistic']:.3f}, p={la['p_value']:.4f}, "
                f"train_mean={la['train_mean']:.1f}px, val_mean={la['val_mean']:.1f}px)"
            )

    # 4. Per-z-bin significant differences
    if len(zbin_comparison) > 0:
        n_significant = zbin_comparison["significant"].sum()
        if n_significant >= thresholds["significant_zbin_count"]:
            sig_zbins = zbin_comparison[zbin_comparison["significant"]]["z_bin"].tolist()
            warnings.append(
                f"[BIAS] {n_significant} z-bins have significantly different lesion rates: "
                f"{sig_zbins}"
            )

    return warnings


def plot_split_statistical_comparison(
    df: pd.DataFrame,
    divergence_metrics: Dict,
    zbin_comparison: pd.DataFrame,
    output_dir: Path,
    show: bool = True,
) -> None:
    """Create visualization showing statistical comparison between splits.

    Generates a 4-panel figure:
    1. Bar chart: Lesion % per split with 95% CI
    2. Heatmap: Z-bin lesion % difference (val - train)
    3. Box plot: Lesion area distribution comparison
    4. Text summary of statistical tests

    Args:
        df: DataFrame with all splits.
        divergence_metrics: Output from compute_split_divergence().
        zbin_comparison: Output from compute_per_zbin_lesion_comparison().
        output_dir: Output directory for plots.
        show: Whether to display plots.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(
        "Statistical Comparison of Train/Val/Test Splits\n"
        "(Detecting potential biases that affect model evaluation)",
        fontsize=14,
        fontweight="bold",
    )

    splits = ["train", "val", "test"]
    colors = ["#3498DB", "#E74C3C", "#2ECC71"]

    # 1. Lesion percentage with 95% confidence intervals
    ax1 = fig.add_subplot(gs[0, 0])

    lesion_pcts = []
    ci_lower = []
    ci_upper = []

    for split in splits:
        split_df = df[df["split"] == split]
        n = len(split_df)
        p = split_df["has_lesion"].mean() if n > 0 else 0

        # Wilson score interval for 95% CI
        if n > 0:
            z = 1.96
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator
            spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
            ci_low = max(0, center - spread)
            ci_high = min(1, center + spread)
        else:
            ci_low, ci_high = 0, 0

        lesion_pcts.append(p * 100)
        ci_lower.append(ci_low * 100)
        ci_upper.append(ci_high * 100)

    x = np.arange(len(splits))
    bars = ax1.bar(x, lesion_pcts, color=colors, alpha=0.7, edgecolor="black")

    # Add error bars
    yerr_lower = [lesion_pcts[i] - ci_lower[i] for i in range(len(splits))]
    yerr_upper = [ci_upper[i] - lesion_pcts[i] for i in range(len(splits))]
    ax1.errorbar(x, lesion_pcts, yerr=[yerr_lower, yerr_upper], fmt="none", color="black", capsize=5)

    # Add value labels
    for i, (pct, bar) in enumerate(zip(lesion_pcts, bars)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.1f}%", ha="center", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.set_ylabel("Lesion Percentage (%)", fontweight="bold")
    ax1.set_title("Lesion Percentage by Split (with 95% CI)")
    ax1.grid(axis="y", alpha=0.3)

    # Add significance annotation
    lp = divergence_metrics["lesion_percentage"]
    if lp["train_val_significant"]:
        ax1.annotate(
            f"Diff: {lp['train_val_diff']:.1f}%*",
            xy=(0.5, max(lesion_pcts) * 0.9),
            fontsize=10,
            color="red",
            fontweight="bold",
        )

    # 2. Z-bin lesion % difference heatmap
    ax2 = fig.add_subplot(gs[0, 1])

    if len(zbin_comparison) > 0:
        zbins = zbin_comparison["z_bin"].values
        diffs = zbin_comparison["val_lesion_pct"].values - zbin_comparison["train_lesion_pct"].values

        # Create heatmap-style bar chart
        colors_diff = ["#E74C3C" if d > 0 else "#3498DB" for d in diffs]
        bars = ax2.bar(zbins, diffs, color=colors_diff, alpha=0.7, edgecolor="black")

        # Highlight significant z-bins
        for i, (zb, sig) in enumerate(zip(zbin_comparison["z_bin"], zbin_comparison["significant"])):
            if sig:
                ax2.annotate("*", xy=(zb, diffs[i]), fontsize=14, ha="center", color="red", fontweight="bold")

        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax2.set_xlabel("Z-bin", fontweight="bold")
        ax2.set_ylabel("Val - Train Lesion % Difference", fontweight="bold")
        ax2.set_title("Per-Z-bin Lesion Rate Difference\n(* = significant, Fisher's exact p<0.05)")
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Per-Z-bin Lesion Rate Difference")

    # 3. Lesion area distribution comparison
    ax3 = fig.add_subplot(gs[1, 0])

    if "lesion_area_px" in df.columns:
        lesion_df = df[df["has_lesion"] == True]
        data_by_split = []
        labels = []

        for split in splits:
            split_areas = lesion_df[lesion_df["split"] == split]["lesion_area_px"].dropna()
            if len(split_areas) > 0:
                data_by_split.append(split_areas.values)
                labels.append(split)

        if data_by_split:
            bp = ax3.boxplot(data_by_split, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors[: len(labels)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax3.set_ylabel("Lesion Area (pixels)", fontweight="bold")
            ax3.set_title("Lesion Area Distribution by Split")
            ax3.grid(axis="y", alpha=0.3)

            # Add KS test annotation
            if divergence_metrics["lesion_area"] is not None:
                la = divergence_metrics["lesion_area"]
                sig_marker = "*" if la["significant"] else ""
                ax3.annotate(
                    f"KS={la['ks_statistic']:.3f}, p={la['p_value']:.3f}{sig_marker}",
                    xy=(0.02, 0.98),
                    xycoords="axes fraction",
                    fontsize=9,
                    va="top",
                    color="red" if la["significant"] else "black",
                )
        else:
            ax3.text(0.5, 0.5, "No lesion data", ha="center", va="center", transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, "No lesion_area_px column", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title("Lesion Area Distribution by Split")

    # 4. Statistical summary text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    summary_lines = [
        "STATISTICAL TEST SUMMARY",
        "=" * 40,
        "",
        "1. LESION PERCENTAGE TEST",
        f"   Train: {lp['train']:.1f}%",
        f"   Val:   {lp['val']:.1f}%",
        f"   Test:  {lp['test']:.1f}%",
        f"   Train-Val Diff: {lp['train_val_diff']:.1f}% {'[SIGNIFICANT]' if lp['train_val_significant'] else '[OK]'}",
        "",
        "2. Z-BIN DISTRIBUTION TEST (Chi-squared)",
    ]

    zd = divergence_metrics["z_bin_distribution"]
    summary_lines.extend(
        [
            f"   Chi2 statistic: {zd['chi2_statistic']:.2f}",
            f"   P-value: {zd['p_value']:.4f}",
            f"   Result: {'[SIGNIFICANT DIFFERENCE]' if zd['significant'] else '[OK - Similar distribution]'}",
            "",
            "3. LESION AREA TEST (Kolmogorov-Smirnov)",
        ]
    )

    if divergence_metrics["lesion_area"] is not None:
        la = divergence_metrics["lesion_area"]
        summary_lines.extend(
            [
                f"   KS statistic: {la['ks_statistic']:.3f}",
                f"   P-value: {la['p_value']:.4f}",
                f"   Train mean: {la['train_mean']:.1f}px, Val mean: {la['val_mean']:.1f}px",
                f"   Result: {'[SIGNIFICANT DIFFERENCE]' if la['significant'] else '[OK - Similar distribution]'}",
            ]
        )
    else:
        summary_lines.append("   [No lesion area data available]")

    # Add per-z-bin summary
    if len(zbin_comparison) > 0:
        n_sig = zbin_comparison["significant"].sum()
        summary_lines.extend(
            [
                "",
                "4. PER-Z-BIN ANALYSIS (Fisher's exact)",
                f"   Z-bins with significant differences: {n_sig}",
            ]
        )
        if n_sig > 0:
            sig_zbins = zbin_comparison[zbin_comparison["significant"]]["z_bin"].tolist()
            summary_lines.append(f"   Affected z-bins: {sig_zbins}")

    summary_text = "\n".join(summary_lines)
    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
    )

    plt.tight_layout()
    output_path = output_dir / "split_statistical_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def generate_summary_report(
    df: pd.DataFrame,
    output_dir: Path,
    divergence_metrics: Optional[Dict] = None,
    zbin_comparison: Optional[pd.DataFrame] = None,
) -> None:
    """Generate a text summary report of the analysis.

    Args:
        df: DataFrame with slice data.
        output_dir: Output directory for the report.
        divergence_metrics: Optional output from compute_split_divergence().
        zbin_comparison: Optional output from compute_per_zbin_lesion_comparison().
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

    if "lesion_area_px" in lesion_df.columns and len(lesion_df) > 0:
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

    # Statistical Tests Section (NEW)
    if divergence_metrics is not None:
        report_lines.append("STATISTICAL TESTS (Train vs Val)")
        report_lines.append("-" * 40)

        # Lesion percentage test
        lp = divergence_metrics["lesion_percentage"]
        report_lines.append("1. Lesion Percentage Comparison:")
        report_lines.append(f"   Train: {lp['train']:.1f}%, Val: {lp['val']:.1f}%, Test: {lp['test']:.1f}%")
        report_lines.append(f"   Train-Val difference: {lp['train_val_diff']:.1f}%")
        status = "SIGNIFICANT" if lp["train_val_significant"] else "OK"
        report_lines.append(f"   Status: [{status}] (threshold: 5.0%)")
        report_lines.append("")

        # Z-bin distribution test
        zd = divergence_metrics["z_bin_distribution"]
        report_lines.append("2. Z-bin Distribution (Chi-squared test):")
        report_lines.append(f"   Chi2 statistic: {zd['chi2_statistic']:.2f}")
        report_lines.append(f"   P-value: {zd['p_value']:.4f}")
        status = "SIGNIFICANT DIFFERENCE" if zd["significant"] else "OK - Similar distribution"
        report_lines.append(f"   Status: [{status}]")
        report_lines.append("")

        # Lesion area test
        if divergence_metrics["lesion_area"] is not None:
            la = divergence_metrics["lesion_area"]
            report_lines.append("3. Lesion Area Distribution (KS test):")
            report_lines.append(f"   KS statistic: {la['ks_statistic']:.3f}")
            report_lines.append(f"   P-value: {la['p_value']:.4f}")
            report_lines.append(f"   Train mean: {la['train_mean']:.1f}px, Val mean: {la['val_mean']:.1f}px")
            status = "SIGNIFICANT DIFFERENCE" if la["significant"] else "OK - Similar distribution"
            report_lines.append(f"   Status: [{status}]")
        report_lines.append("")

        # Per-z-bin analysis
        if zbin_comparison is not None and len(zbin_comparison) > 0:
            n_sig = zbin_comparison["significant"].sum()
            report_lines.append("4. Per-Z-bin Analysis (Fisher's exact test):")
            report_lines.append(f"   Z-bins with significant differences: {n_sig}")
            if n_sig > 0:
                sig_zbins = zbin_comparison[zbin_comparison["significant"]]["z_bin"].tolist()
                report_lines.append(f"   Affected z-bins: {sig_zbins}")
            report_lines.append("")

    # Potential bias warnings
    report_lines.append("POTENTIAL BIAS WARNINGS")
    report_lines.append("-" * 40)

    warnings_found = False

    # Generate enhanced warnings if statistical data available
    if divergence_metrics is not None and zbin_comparison is not None:
        enhanced_warnings = generate_enhanced_warnings(divergence_metrics, zbin_comparison)
        for warning in enhanced_warnings:
            report_lines.append(warning)
            warnings_found = True

    # Check for class imbalance (legacy)
    lesion_pct = len(lesion_df) / len(df) * 100
    if lesion_pct < 10:
        report_lines.append(f"[WARNING] Low lesion percentage ({lesion_pct:.1f}%)")
        report_lines.append("  - Model may underfit lesion generation")
        warnings_found = True
    elif lesion_pct > 50:
        report_lines.append(f"[WARNING] High lesion percentage ({lesion_pct:.1f}%)")
        report_lines.append("  - Model may overfit lesion patterns")
        warnings_found = True

    # Check for z-bin imbalance
    train_zbin_counts = train_df.groupby("z_bin").size()
    if len(train_zbin_counts) > 0:
        zbin_ratio = train_zbin_counts.max() / train_zbin_counts.min()
        if zbin_ratio > 3:
            report_lines.append(f"[WARNING] Z-bin imbalance ratio: {zbin_ratio:.1f}x")
            report_lines.append("  - Model may generate better images for certain z-bins")
            warnings_found = True

    # Check for subject dominance
    if len(lesion_df) > 0:
        lesion_per_subject = lesion_df.groupby("subject_id").size()
        top_subject_contribution = lesion_per_subject.max() / len(lesion_df) * 100
        if top_subject_contribution > 20:
            report_lines.append(
                f"[WARNING] Single subject contributes {top_subject_contribution:.1f}% "
                "of lesion data"
            )
            report_lines.append("  - Model may overfit to this subject's lesion patterns")
            warnings_found = True

    if "lesion_area_px" in lesion_df.columns and len(lesion_df) > 0:
        # Check for lesion area distribution across z-bins
        area_by_zbin = lesion_df.groupby("z_bin")["lesion_area_px"].mean()
        if len(area_by_zbin) > 0 and area_by_zbin.min() > 0:
            area_ratio = area_by_zbin.max() / area_by_zbin.min()
            if area_ratio > 3:
                report_lines.append(f"[WARNING] Lesion area varies {area_ratio:.1f}x across z-bins")
                report_lines.append("  - Model may learn z-bin-specific lesion sizes")
                warnings_found = True

    if not warnings_found:
        report_lines.append("[OK] No significant biases detected")

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


# =============================================================================
# Reusable Workflow Functions
# =============================================================================


def run_all_visualizations(
    df: pd.DataFrame,
    output_dir: Path,
    show: bool = True,
) -> Tuple[Dict, pd.DataFrame]:
    """Run all bias visualizations and return metrics.

    This is the main reusable function for generating all bias analysis
    visualizations. Can be called programmatically from the cache builder.

    Args:
        df: DataFrame with all splits (train, val, test).
        output_dir: Directory to save visualizations.
        show: Whether to display plots interactively.

    Returns:
        Tuple of (divergence_metrics, zbin_comparison_df)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating visualizations...")

    # Generate all standard visualizations
    plot_lesion_area_by_zbin(df, output_dir, show)
    plot_lesion_count_by_zbin(df, output_dir, show)
    plot_split_comparison(df, output_dir, show)
    plot_domain_analysis(df, output_dir, show)
    plot_token_distribution(df, output_dir, show)
    plot_subject_analysis(df, output_dir, show)

    # Run statistical tests
    logger.info("Running statistical tests...")
    divergence_metrics = compute_split_divergence(df)
    zbin_comparison = compute_per_zbin_lesion_comparison(df)

    # Generate statistical comparison visualization
    plot_split_statistical_comparison(df, divergence_metrics, zbin_comparison, output_dir, show)

    # Generate summary report
    generate_summary_report(df, output_dir, divergence_metrics, zbin_comparison)

    logger.info(f"Visualizations saved to: {output_dir}")

    return divergence_metrics, zbin_comparison


def generate_comparison_visualizations(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Generate visualizations comparing pre and post stratification.

    Creates:
    1. Side-by-side lesion % comparison (grouped bar chart)
    2. Z-bin distribution improvement heatmap
    3. Statistical test improvement table
    4. Summary text report with improvement metrics

    Args:
        df_pre: DataFrame with pre-stratification splits.
        df_post: DataFrame with post-stratification splits.
        output_dir: Directory to save comparison visualizations.
        show: Whether to display plots interactively.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating stratification comparison visualizations...")

    # Compute metrics for both
    pre_metrics = compute_split_divergence(df_pre)
    post_metrics = compute_split_divergence(df_post)

    pre_zbin = compute_per_zbin_lesion_comparison(df_pre)
    post_zbin = compute_per_zbin_lesion_comparison(df_post)

    # Generate comparison figure
    plot_stratification_comparison(
        df_pre, df_post,
        pre_metrics, post_metrics,
        pre_zbin, post_zbin,
        output_dir, show
    )

    # Generate comparison report
    generate_comparison_report(
        pre_metrics, post_metrics,
        pre_zbin, post_zbin,
        output_dir
    )

    logger.info(f"Comparison visualizations saved to: {output_dir}")


def plot_stratification_comparison(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    pre_metrics: Dict,
    post_metrics: Dict,
    pre_zbin: pd.DataFrame,
    post_zbin: pd.DataFrame,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Create 6-panel stratification comparison figure.

    Layout (2x3 grid):
    [1] Pre/Post Lesion % by Split (grouped bar)
    [2] Train-Val Difference Improvement (bar)
    [3] Z-bin Distribution Pre (line)
    [4] Z-bin Distribution Post (line)
    [5] Statistical Test Comparison (table-like)
    [6] Improvement Summary (text)
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(
        "Stratification Comparison: Pre vs Post\n"
        "(Evaluating improvement in train/val balance)",
        fontsize=14,
        fontweight="bold",
    )

    splits = ["train", "val", "test"]
    colors_pre = ["#3498DB", "#E74C3C", "#2ECC71"]
    colors_post = ["#1A5276", "#922B21", "#1D8348"]

    # Panel 1: Pre/Post Lesion % by Split (grouped bar)
    ax1 = fig.add_subplot(gs[0, 0])

    pre_pcts = [pre_metrics["lesion_percentage"][s] for s in splits]
    post_pcts = [post_metrics["lesion_percentage"][s] for s in splits]

    x = np.arange(len(splits))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, pre_pcts, width, label="Pre-stratification", color=colors_pre, alpha=0.7)
    bars2 = ax1.bar(x + width / 2, post_pcts, width, label="Post-stratification", color=colors_post, alpha=0.7)

    ax1.set_ylabel("Lesion %", fontweight="bold")
    ax1.set_title("Lesion % by Split")
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    # Panel 2: Train-Val Difference Improvement
    ax2 = fig.add_subplot(gs[0, 1])

    pre_diff = pre_metrics["lesion_percentage"]["train_val_diff"]
    post_diff = post_metrics["lesion_percentage"]["train_val_diff"]
    improvement = pre_diff - post_diff

    bars = ax2.bar(["Pre", "Post"], [pre_diff, post_diff],
                   color=["#E74C3C", "#27AE60"], alpha=0.7, edgecolor="black")

    ax2.set_ylabel("Train-Val Difference (%)", fontweight="bold")
    ax2.set_title(f"Train-Val Difference\n(Improvement: {improvement:.1f}%)")
    ax2.axhline(y=5.0, color="orange", linestyle="--", label="Threshold (5%)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")

    # Panel 3: Z-bin Distribution Pre
    ax3 = fig.add_subplot(gs[0, 2])

    if len(pre_zbin) > 0:
        ax3.plot(pre_zbin["z_bin"], pre_zbin["train_lesion_pct"], "b-o", label="Train", markersize=4)
        ax3.plot(pre_zbin["z_bin"], pre_zbin["val_lesion_pct"], "r-s", label="Val", markersize=4)
        ax3.fill_between(pre_zbin["z_bin"], pre_zbin["train_lesion_pct"], pre_zbin["val_lesion_pct"],
                        alpha=0.2, color="gray")

    ax3.set_xlabel("Z-bin", fontweight="bold")
    ax3.set_ylabel("Lesion %", fontweight="bold")
    ax3.set_title("Z-bin Distribution (PRE)")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Panel 4: Z-bin Distribution Post
    ax4 = fig.add_subplot(gs[1, 0])

    if len(post_zbin) > 0:
        ax4.plot(post_zbin["z_bin"], post_zbin["train_lesion_pct"], "b-o", label="Train", markersize=4)
        ax4.plot(post_zbin["z_bin"], post_zbin["val_lesion_pct"], "r-s", label="Val", markersize=4)
        ax4.fill_between(post_zbin["z_bin"], post_zbin["train_lesion_pct"], post_zbin["val_lesion_pct"],
                        alpha=0.2, color="gray")

    ax4.set_xlabel("Z-bin", fontweight="bold")
    ax4.set_ylabel("Lesion %", fontweight="bold")
    ax4.set_title("Z-bin Distribution (POST)")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Panel 5: Statistical Test Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis("off")

    # Build comparison table
    table_data = [
        ["Metric", "Pre", "Post", "Improved?"],
        ["─" * 12, "─" * 10, "─" * 10, "─" * 10],
        [
            "Lesion % Diff",
            f"{pre_metrics['lesion_percentage']['train_val_diff']:.1f}%",
            f"{post_metrics['lesion_percentage']['train_val_diff']:.1f}%",
            "YES" if post_metrics['lesion_percentage']['train_val_diff'] < pre_metrics['lesion_percentage']['train_val_diff'] else "NO"
        ],
        [
            "Chi² p-value",
            f"{pre_metrics['z_bin_distribution']['p_value']:.4f}",
            f"{post_metrics['z_bin_distribution']['p_value']:.4f}",
            "YES" if post_metrics['z_bin_distribution']['p_value'] > pre_metrics['z_bin_distribution']['p_value'] else "NO"
        ],
    ]

    if pre_metrics["lesion_area"] is not None and post_metrics["lesion_area"] is not None:
        table_data.append([
            "KS statistic",
            f"{pre_metrics['lesion_area']['ks_statistic']:.3f}",
            f"{post_metrics['lesion_area']['ks_statistic']:.3f}",
            "YES" if post_metrics['lesion_area']['ks_statistic'] < pre_metrics['lesion_area']['ks_statistic'] else "NO"
        ])

    # Add significant z-bins count
    pre_sig = pre_zbin["significant"].sum() if len(pre_zbin) > 0 else 0
    post_sig = post_zbin["significant"].sum() if len(post_zbin) > 0 else 0
    table_data.append([
        "Sig. Z-bins",
        str(pre_sig),
        str(post_sig),
        "YES" if post_sig < pre_sig else "NO" if post_sig > pre_sig else "SAME"
    ])

    table_text = "\n".join([f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]}" for row in table_data])

    ax5.text(0.1, 0.9, "STATISTICAL TEST COMPARISON",
            fontsize=11, fontweight="bold", transform=ax5.transAxes, va="top")
    ax5.text(0.1, 0.8, table_text,
            fontsize=10, fontfamily="monospace", transform=ax5.transAxes, va="top")

    # Panel 6: Improvement Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    # Compute overall improvement score
    improvements = []
    if post_metrics['lesion_percentage']['train_val_diff'] < pre_metrics['lesion_percentage']['train_val_diff']:
        improvements.append("Lesion % balance")
    if post_metrics['z_bin_distribution']['p_value'] > pre_metrics['z_bin_distribution']['p_value']:
        improvements.append("Z-bin distribution")
    if pre_metrics["lesion_area"] is not None and post_metrics["lesion_area"] is not None:
        if post_metrics['lesion_area']['ks_statistic'] < pre_metrics['lesion_area']['ks_statistic']:
            improvements.append("Lesion area distribution")
    if post_sig < pre_sig:
        improvements.append("Per-z-bin significance")

    summary_lines = [
        "IMPROVEMENT SUMMARY",
        "=" * 40,
        "",
        f"Areas improved: {len(improvements)}/4",
        "",
    ]

    if improvements:
        summary_lines.append("Improvements achieved:")
        for imp in improvements:
            summary_lines.append(f"  + {imp}")
    else:
        summary_lines.append("No significant improvements detected.")
        summary_lines.append("Consider adjusting stratification parameters.")

    summary_lines.extend([
        "",
        "Key metrics:",
        f"  Train-Val diff: {pre_diff:.1f}% -> {post_diff:.1f}%",
        f"  Reduction: {improvement:.1f}%",
    ])

    summary_text = "\n".join(summary_lines)
    ax6.text(0.1, 0.9, summary_text,
            fontsize=10, fontfamily="monospace", transform=ax6.transAxes, va="top")

    plt.tight_layout()
    output_path = output_dir / "stratification_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def generate_comparison_report(
    pre_metrics: Dict,
    post_metrics: Dict,
    pre_zbin: pd.DataFrame,
    post_zbin: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate text report comparing pre/post stratification."""

    pre_lp = pre_metrics["lesion_percentage"]
    post_lp = post_metrics["lesion_percentage"]
    pre_zd = pre_metrics["z_bin_distribution"]
    post_zd = post_metrics["z_bin_distribution"]

    report_lines = [
        "=" * 70,
        "STRATIFICATION IMPROVEMENT REPORT",
        "=" * 70,
        "",
        "LESION PERCENTAGE COMPARISON",
        "-" * 40,
        f"  PRE:  Train={pre_lp['train']:.1f}%, Val={pre_lp['val']:.1f}%, "
        f"Test={pre_lp['test']:.1f}%",
        f"        Train-Val Diff: {pre_lp['train_val_diff']:.1f}%",
        "",
        f"  POST: Train={post_lp['train']:.1f}%, Val={post_lp['val']:.1f}%, "
        f"Test={post_lp['test']:.1f}%",
        f"        Train-Val Diff: {post_lp['train_val_diff']:.1f}%",
        "",
        f"  IMPROVEMENT: {pre_lp['train_val_diff'] - post_lp['train_val_diff']:.1f}% reduction",
        f"  STATUS: {'PASS' if post_lp['train_val_diff'] <= 5.0 else 'NEEDS ATTENTION'} (threshold: 5%)",
        "",
        "Z-BIN DISTRIBUTION TEST (Chi-squared)",
        "-" * 40,
        f"  PRE:  Chi2={pre_zd['chi2_statistic']:.2f}, p-value={pre_zd['p_value']:.4f}",
        f"        {'SIGNIFICANT DIFFERENCE' if pre_zd['significant'] else 'OK'}",
        "",
        f"  POST: Chi2={post_zd['chi2_statistic']:.2f}, p-value={post_zd['p_value']:.4f}",
        f"        {'SIGNIFICANT DIFFERENCE' if post_zd['significant'] else 'OK'}",
        "",
        f"  IMPROVEMENT: p-value {'increased' if post_zd['p_value'] > pre_zd['p_value'] else 'decreased'}",
        "",
    ]

    # Lesion area comparison
    if pre_metrics["lesion_area"] is not None and post_metrics["lesion_area"] is not None:
        pre_la = pre_metrics["lesion_area"]
        post_la = post_metrics["lesion_area"]
        report_lines.extend([
            "LESION AREA DISTRIBUTION (KS test)",
            "-" * 40,
            f"  PRE:  KS={pre_la['ks_statistic']:.3f}, p-value={pre_la['p_value']:.4f}",
            f"        Train mean: {pre_la['train_mean']:.1f}px, Val mean: {pre_la['val_mean']:.1f}px",
            "",
            f"  POST: KS={post_la['ks_statistic']:.3f}, p-value={post_la['p_value']:.4f}",
            f"        Train mean: {post_la['train_mean']:.1f}px, Val mean: {post_la['val_mean']:.1f}px",
            "",
        ])

    # Per-z-bin comparison
    pre_sig = pre_zbin["significant"].sum() if len(pre_zbin) > 0 else 0
    post_sig = post_zbin["significant"].sum() if len(post_zbin) > 0 else 0
    report_lines.extend([
        "PER-Z-BIN ANALYSIS (Fisher's exact)",
        "-" * 40,
        f"  PRE:  {pre_sig} z-bins with significant differences",
        f"  POST: {post_sig} z-bins with significant differences",
        f"  CHANGE: {pre_sig - post_sig:+d} z-bins",
        "",
    ])

    # Overall assessment
    improvements = 0
    total_metrics = 4

    if post_lp['train_val_diff'] < pre_lp['train_val_diff']:
        improvements += 1
    if post_zd['p_value'] > pre_zd['p_value']:
        improvements += 1
    if pre_metrics["lesion_area"] is not None and post_metrics["lesion_area"] is not None:
        if post_metrics["lesion_area"]["ks_statistic"] < pre_metrics["lesion_area"]["ks_statistic"]:
            improvements += 1
    if post_sig <= pre_sig:
        improvements += 1

    report_lines.extend([
        "OVERALL ASSESSMENT",
        "-" * 40,
        f"  Metrics improved: {improvements}/{total_metrics}",
        "",
        f"  {'STRATIFICATION SUCCESSFUL' if improvements >= 2 else 'STRATIFICATION NEEDS REVIEW'}",
        "",
        "=" * 70,
    ])

    # Save report
    report_text = "\n".join(report_lines)
    report_path = output_dir / "improvement_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Saved: {report_path}")

    # Print to console
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

    # Use the refactored function to run all visualizations
    run_all_visualizations(df, output_dir, show=not args.no_show)

    logger.info(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
