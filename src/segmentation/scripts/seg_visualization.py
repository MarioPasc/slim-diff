"""Visualization script for comparing segmentation metrics across models and experiments.

Creates a 1x3 subplot comparing Dice/HD95 scores for UNet, DynUNet, and SwinUNetR
across different training strategies (real_only, real_synthetic_concat,
real_synthetic_balance, synthetic_only).

Usage:
    python seg_visualization.py /path/to/results /path/to/output \
        --synthetic-expansion-term x2 --metric dice --format png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# Plot Settings
# ==============================================================================

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


def apply_plot_settings() -> None:
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


# ==============================================================================
# Constants
# ==============================================================================

MODELS = [
    #"unet", 
    #"dynunet", 
    #"swinunetr", 
    "unetr", 
    "segresnet", 
    "attentionunet"
]
MODEL_DISPLAY_NAMES = {
    #"unet": "UNet",
    #"dynunet": "DynUNet",
    #"swinunetr": "SwinUNetR",
    "unetr": "UNetR",
    "segresnet": "SegResNet",
    "attentionunet": "AttentionUNet",
}

EXPERIMENTS = {
    r"$M_0$": "real_only",
    r"$M_1$": "real_traditional_augmentation",
    r"$M_2$": "real_synthetic_balance",
    r"$M_3$": "real_synthetic_traditional_augmentation",
    
}

METRIC_COLUMNS = {
    "dice": "test_dice",
    "hd95": "test_hd95",
}

METRIC_LABELS = {
    "dice": "Dice Score",
    "hd95": "HD95 (mm)",
}

BOXPLOT_COLORS = ['#EE7733', '#0077BB', '#CC3311', '#009988']


# ==============================================================================
# Data Loading Functions
# ==============================================================================

def get_experiment_folder(
    base_path: Path,
    experiment_name: str,
    model: str,
    synthetic_expansion: str,
) -> Path:
    """Get the folder path for an experiment-model combination.

    Args:
        base_path: Root folder containing all experiments.
        experiment_name: Name of the experiment (e.g., "real_only").
        model: Model name (e.g., "unet").
        synthetic_expansion: Expansion term for synthetic_only (e.g., "x2").

    Returns:
        Path to the experiment folder.
    """
    if experiment_name == "synthetic_only":
        return base_path / "synthetic" / synthetic_expansion / f"{experiment_name}_{model}"
    return base_path / f"{experiment_name}_{model}"


def load_test_results(experiment_folder: Path, metric: str) -> Optional[np.ndarray]:
    """Load test results from an experiment folder.

    Args:
        experiment_folder: Path to the experiment folder.
        metric: Metric to extract ("dice" or "hd95").

    Returns:
        Array of metric values for each fold, or None if not found.
    """
    csv_path = experiment_folder / "test_results.csv"

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    column = METRIC_COLUMNS[metric]

    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in {csv_path}")
        return None

    values = df[column].values

    # Handle NaN values for hd95
    valid_values = values[~np.isnan(values)]

    if len(valid_values) == 0:
        print(f"Warning: All values are NaN for {metric} in {experiment_folder}")
        return None

    if len(valid_values) < len(values):
        print(f"Note: {len(values) - len(valid_values)} NaN values skipped for {metric} in {experiment_folder.name}")

    return valid_values


def collect_data(
    base_path: Path,
    metric: str,
    synthetic_expansion: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Collect metric data for all experiments and models.

    Args:
        base_path: Root folder containing all experiments.
        metric: Metric to extract ("dice" or "hd95").
        synthetic_expansion: Expansion term for synthetic_only (e.g., "x2").

    Returns:
        Nested dict: {model: {experiment_alias: values}}.
    """
    data = {model: {} for model in MODELS}

    for alias, exp_name in EXPERIMENTS.items():
        for model in MODELS:
            folder = get_experiment_folder(base_path, exp_name, model, synthetic_expansion)
            values = load_test_results(folder, metric)

            if values is not None:
                data[model][alias] = values

    return data


# ==============================================================================
# Visualization Functions
# ==============================================================================

def create_boxplot_comparison(
    data: Dict[str, Dict[str, np.ndarray]],
    metric: str,
    output_path: Path,
    fmt: str,
    synthetic_expansion: str,
) -> None:
    """Create a 1x3 boxplot comparison figure.

    Args:
        data: Nested dict: {model: {experiment_alias: values}}.
        metric: Metric being plotted ("dice" or "hd95").
        output_path: Output folder for the figure.
        fmt: Output format ("png" or "pdf").
    """
    apply_plot_settings()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.subplots_adjust(wspace=0.1, left=0.08, right=0.98, top=0.88, bottom=0.12)

    experiment_aliases = list(EXPERIMENTS.keys())

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        model_data = data[model]

        # Prepare data for boxplot
        plot_data = []
        labels = []

        for alias in experiment_aliases:
            if alias in model_data:
                plot_data.append(model_data[alias])
                labels.append(alias)
            else:
                # Add empty data to maintain position
                plot_data.append([])
                labels.append(alias)

        # Create boxplot
        bp = ax.boxplot(
            plot_data,
            tick_labels=labels,
            patch_artist=True,
            widths=0.6,
            showfliers=True,
            flierprops=dict(marker='o', markersize=4, alpha=0.7),
        )

        # Color the boxes
        for patch, color in zip(bp['boxes'], BOXPLOT_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Style median lines
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        # Set title and labels
        ax.set_title(MODEL_DISPLAY_NAMES[model], fontsize=PLOT_SETTINGS["title_fontsize"])
        ax.tick_params(axis='x', labelsize=PLOT_SETTINGS["xtick_fontsize"])
        ax.tick_params(axis='y', labelsize=PLOT_SETTINGS["ytick_fontsize"])

        if idx == 0:
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=PLOT_SETTINGS["ylabel_fontsize"])

        # Add grid
        ax.yaxis.grid(True, linestyle=':', alpha=0.7)
        ax.set_axisbelow(True)

    # Save figure
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"test_{metric}_{synthetic_expansion}seg_vis.{fmt}"
    save_path = output_path / filename

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {save_path}")

    plt.close(fig)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize segmentation metrics across models and experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_folder",
        type=Path,
        help="Top folder containing all experiment results.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Output folder for the visualization.",
    )
    parser.add_argument(
        "--synthetic-expansion-term",
        type=str,
        default="x2",
        choices=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"],
        help="Expansion term for synthetic_only experiments (default: x2).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="dice",
        choices=["dice", "hd95"],
        help="Metric to plot (default: dice).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf"],
        dest="fmt",
        help="Output format (default: png).",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate input folder
    if not args.input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {args.input_folder}")

    print(f"Loading {args.metric} data from: {args.input_folder}")
    print(f"Synthetic expansion term: {args.synthetic_expansion_term}")

    # Collect data
    data = collect_data(
        args.input_folder,
        args.metric,
        args.synthetic_expansion_term,
    )

    # Check if we have any data
    has_data = any(
        len(model_data) > 0
        for model_data in data.values()
    )

    if not has_data:
        raise ValueError("No data found. Check input folder structure.")

    # Create visualization
    create_boxplot_comparison(
        data,
        args.metric,
        args.output_folder,
        args.fmt,
        args.synthetic_expansion_term,
    )


if __name__ == "__main__":
    main()
