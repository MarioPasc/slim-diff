"""Visualization script for comparing validation loss across models and experiments.

Creates a 1x3 subplot comparing validation loss curves for UNet, DynUNet, and SwinUNetR
across different training strategies. Shows mean loss with std as shaded region.

Usage:
    python seg_loss_visualization.py /path/to/results /path/to/output \
        --synthetic-expansion-term x2 --metric val/loss --format png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

MODELS = ["unet", "dynunet", "swinunetr"]
MODEL_DISPLAY_NAMES = {
    "unet": "UNet",
    "dynunet": "DynUNet",
    "swinunetr": "SwinUNetR",
}

EXPERIMENTS = {
    r"$M_0$": "real_only",
    r"$M_1$": "real_synthetic_concat",
    r"$M_2$": "real_synthetic_balance",
    r"$M_3$": "synthetic_only",
}

METRIC_LABELS = {
    "val/loss": "Validation Loss",
    "val/dice": "Validation Dice",
    "val/hd95": "Validation HD95 (mm)",
    "train/loss": "Training Loss",
}

# Colors for each experiment (same order as EXPERIMENTS)
EXPERIMENT_COLORS = ["#4682b4", "#ff7f0e", "#2ca02c", "#d62728"]
EXPERIMENT_LINESTYLES = ["-", "-", "-", "--"]


# ==============================================================================
# Data Loading Functions
# ==============================================================================

def get_experiment_folder(
    base_path: Path,
    experiment_name: str,
    model: str,
    synthetic_expansion: str,
) -> Path:
    """Get the folder path for an experiment-model combination."""
    if experiment_name == "synthetic_only":
        return base_path / "synthetic" / synthetic_expansion / f"{experiment_name}_{model}"
    return base_path / f"{experiment_name}_{model}"


def load_fold_metrics(fold_path: Path) -> Optional[pd.DataFrame]:
    """Load metrics from a single fold's csv_logs directory."""
    # Try csv_logs first, then lightning_logs
    csv_files = list(fold_path.glob("csv_logs/*.csv"))

    if csv_files:
        return pd.read_csv(csv_files[0])

    # Try lightning_logs
    lightning_csv = fold_path / "lightning_logs" / "version_0" / "metrics.csv"
    if lightning_csv.exists():
        return pd.read_csv(lightning_csv)

    return None


def load_experiment_metrics(
    experiment_folder: Path,
) -> Optional[Dict[int, pd.DataFrame]]:
    """Load metrics from all folds in an experiment.

    Returns:
        Dict mapping fold number to DataFrame, or None if not found.
    """
    if not experiment_folder.exists():
        print(f"Warning: {experiment_folder} not found")
        return None

    fold_data = {}
    for fold_dir in sorted(experiment_folder.glob("fold_*")):
        fold_num = int(fold_dir.name.split("_")[1])
        df = load_fold_metrics(fold_dir)
        if df is not None:
            fold_data[fold_num] = df

    if not fold_data:
        print(f"Warning: No fold data found in {experiment_folder}")
        return None

    return fold_data


def aggregate_folds(
    fold_data: Dict[int, pd.DataFrame],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate metric across folds, computing mean and std.

    Args:
        fold_data: Dict mapping fold number to DataFrame.
        metric: Column name to aggregate.

    Returns:
        Tuple of (epochs, mean_values, std_values).
    """
    # Find common epochs across all folds
    all_epochs = [set(df["epoch"].values) for df in fold_data.values()]
    common_epochs = sorted(set.intersection(*all_epochs))

    if not common_epochs:
        return np.array([]), np.array([]), np.array([])

    # Collect values for each epoch
    values_per_epoch = []
    for epoch in common_epochs:
        epoch_values = []
        for df in fold_data.values():
            val = df[df["epoch"] == epoch][metric].values
            if len(val) > 0 and not np.isnan(val[0]):
                epoch_values.append(val[0])
        values_per_epoch.append(epoch_values)

    # Compute mean and std
    epochs = np.array(common_epochs)
    means = np.array([np.mean(v) if v else np.nan for v in values_per_epoch])
    stds = np.array([np.std(v) if len(v) > 1 else 0.0 for v in values_per_epoch])

    # Remove NaN entries
    valid_mask = ~np.isnan(means)
    return epochs[valid_mask], means[valid_mask], stds[valid_mask]


def collect_all_data(
    base_path: Path,
    metric: str,
    synthetic_expansion: str,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Collect aggregated metric data for all experiments and models.

    Returns:
        Nested dict: {model: {experiment_alias: (epochs, means, stds)}}.
    """
    data = {model: {} for model in MODELS}

    for alias, exp_name in EXPERIMENTS.items():
        for model in MODELS:
            folder = get_experiment_folder(base_path, exp_name, model, synthetic_expansion)
            fold_data = load_experiment_metrics(folder)

            if fold_data is not None:
                epochs, means, stds = aggregate_folds(fold_data, metric)
                if len(epochs) > 0:
                    data[model][alias] = (epochs, means, stds)

    return data


def generate_master_csv(
    base_path: Path,
    output_path: Path,
    synthetic_expansion: str,
) -> pd.DataFrame:
    """Generate a master CSV with aggregated metrics across all experiments.

    The CSV has columns: epoch, fold, model, experiment, and metrics.
    """
    metrics_to_include = ["val/dice", "val/hd95", "val/loss"]

    all_data = []

    for model in MODELS:
        for alias, exp_name in EXPERIMENTS.items():
            folder = get_experiment_folder(base_path, exp_name, model, synthetic_expansion)
            fold_data = load_experiment_metrics(folder)

            if fold_data is None:
                continue

            for fold_num, df in fold_data.items():
                for _, row in df.iterrows():
                    entry = {
                        "model": model,
                        "experiment": exp_name,
                        "experiment_alias": alias,
                        "epoch": row["epoch"],
                        "fold": fold_num,
                    }
                    for metric in metrics_to_include:
                        if metric in df.columns:
                            entry[metric] = row[metric]
                    all_data.append(entry)

    master_df = pd.DataFrame(all_data)

    # Save master CSV
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"master_metrics_{synthetic_expansion}.csv"
    master_df.to_csv(csv_path, index=False)
    print(f"Saved master CSV to: {csv_path}")

    return master_df


# ==============================================================================
# Visualization Functions
# ==============================================================================

def create_loss_comparison(
    data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    metric: str,
    output_path: Path,
    fmt: str,
    synthetic_expansion: str,
) -> None:
    """Create a 1x3 line plot comparison figure with mean and std shading.

    Args:
        data: Nested dict: {model: {experiment_alias: (epochs, means, stds)}}.
        metric: Metric being plotted.
        output_path: Output folder for the figure.
        fmt: Output format ("png" or "pdf").
        synthetic_expansion: Expansion term for filename.
    """
    apply_plot_settings()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.subplots_adjust(wspace=0.1, left=0.07, right=0.98, top=0.88, bottom=0.14)

    experiment_aliases = list(EXPERIMENTS.keys())

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        model_data = data[model]

        for exp_idx, alias in enumerate(experiment_aliases):
            if alias not in model_data:
                continue

            epochs, means, stds = model_data[alias]
            color = EXPERIMENT_COLORS[exp_idx]
            linestyle = EXPERIMENT_LINESTYLES[exp_idx]

            # Plot mean line
            ax.plot(
                epochs,
                means,
                color=color,
                linestyle=linestyle,
                linewidth=PLOT_SETTINGS["line_width"],
                label=alias,
            )

            # Plot std shading
            ax.fill_between(
                epochs,
                means - stds,
                means + stds,
                color=color,
                alpha=0.2,
            )

        # Set title and labels
        ax.set_title(MODEL_DISPLAY_NAMES[model], fontsize=PLOT_SETTINGS["title_fontsize"])
        ax.set_xlabel("Epoch", fontsize=PLOT_SETTINGS["xlabel_fontsize"])
        ax.tick_params(axis='x', labelsize=PLOT_SETTINGS["xtick_fontsize"])
        ax.tick_params(axis='y', labelsize=PLOT_SETTINGS["ytick_fontsize"])

        if idx == 0:
            ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=PLOT_SETTINGS["ylabel_fontsize"])

        # Add grid
        ax.yaxis.grid(True, linestyle=':', alpha=0.7)
        ax.xaxis.grid(True, linestyle=':', alpha=0.7)
        ax.set_axisbelow(True)

        # Add legend only to last subplot
        if idx == 2:
            ax.legend(loc='upper right', fontsize=PLOT_SETTINGS["legend_fontsize"])

    # Save figure
    output_path.mkdir(parents=True, exist_ok=True)
    metric_name = metric.replace("/", "_")
    filename = f"{metric_name}_curves_{synthetic_expansion}.{fmt}"
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
        description="Visualize validation loss curves across models and experiments.",
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
        help="Output folder for the visualization and master CSV.",
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
        default="val/loss",
        choices=["val/loss", "val/dice", "val/hd95", "train/loss"],
        help="Metric to plot (default: val/loss).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf"],
        dest="fmt",
        help="Output format (default: png).",
    )
    parser.add_argument(
        "--generate-csv",
        action="store_true",
        help="Generate master CSV with all metrics.",
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

    # Generate master CSV if requested
    if args.generate_csv:
        generate_master_csv(
            args.input_folder,
            args.output_folder,
            args.synthetic_expansion_term,
        )

    # Collect data
    data = collect_all_data(
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
    create_loss_comparison(
        data,
        args.metric,
        args.output_folder,
        args.fmt,
        args.synthetic_expansion_term,
    )


if __name__ == "__main__":
    main()
