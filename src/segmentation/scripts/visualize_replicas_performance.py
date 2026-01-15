#!/usr/bin/env python3
"""Visualize TSTR (Train Synthetic Test Real) performance across dataset sizes.

This script plots segmentation performance (Dice or HD95) as a function of the
number of synthetic training images used. It aggregates results across k-fold
cross-validation and groups by model architecture.

The expected directory structure is:
    synthetic_only/
    ├── x1/
    │   ├── synth_x1_attentionunet/
    │   │   ├── kfold_results.json
    │   │   └── kfold_information/
    │   │       └── kfold_statistics.json
    │   ├── synth_x1_segresnet/
    │   └── synth_x1_unetr/
    ├── x2/
    │   └── ...
    └── x10/
        └── ...

Usage:
    python -m src.segmentation.scripts.visualize_replicas_performance \
        --results-dir /path/to/synthetic_only \
        --output-dir outputs/tstr_performance \
        --metric dice

Example:
    python -m src.segmentation.scripts.visualize_replicas_performance \
        --results-dir /media/mpascual/Sandisk2TB/research/epilepsy/results/segmentation/runs/synthetic_only \
        --output-dir outputs/tstr_performance \
        --metric dice
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model display names and colors
MODEL_CONFIG = {
    "attentionunet": {"display_name": "Attention U-Net", "color": "#1f77b4", "marker": "o"},
    "segresnet": {"display_name": "SegResNet", "color": "#ff7f0e", "marker": "s"},
    "unetr": {"display_name": "UNETR", "color": "#2ca02c", "marker": "^"},
}


@dataclass
class ExperimentResult:
    """Container for a single experiment's results."""

    multiplier: int
    model: str
    n_synthetic_samples: int
    metric_mean: float
    metric_std: float
    fold_values: list[float]


def extract_multiplier(folder_name: str) -> int | None:
    """Extract the multiplier from folder name like 'x1', 'x10'.

    Args:
        folder_name: Name of the folder (e.g., 'x1', 'x10')

    Returns:
        Integer multiplier or None if not matched
    """
    match = re.match(r"x(\d+)$", folder_name)
    if match:
        return int(match.group(1))
    return None


def load_kfold_results(experiment_dir: Path, metric: str) -> dict | None:
    """Load test results from kfold_results.json.

    Args:
        experiment_dir: Path to experiment directory (e.g., synth_x1_attentionunet)
        metric: Metric to extract ('dice' or 'hd95')

    Returns:
        Dictionary with mean, std, and per-fold values, or None if not found
    """
    results_path = experiment_dir / "kfold_results.json"

    if not results_path.exists():
        logger.debug(f"kfold_results.json not found at {results_path}")
        return None

    with open(results_path) as f:
        data = json.load(f)

    test_results = data.get("test", {})
    metric_data = test_results.get(metric, {})

    if not metric_data:
        logger.warning(f"Metric '{metric}' not found in {results_path}")
        return None

    fold_results = data.get("fold_results", [])
    fold_values = []
    for fold in fold_results:
        test_res = fold.get("test_results", {})
        value = test_res.get(f"test/{metric}")
        if value is not None:
            fold_values.append(value)

    return {
        "mean": metric_data.get("mean", 0.0),
        "std": metric_data.get("std", 0.0),
        "fold_values": fold_values,
    }


def load_sample_count(experiment_dir: Path) -> int | None:
    """Load synthetic sample count from kfold_statistics.json.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Mean synthetic sample count across folds, or None if not found
    """
    stats_path = experiment_dir / "kfold_information" / "kfold_statistics.json"

    if not stats_path.exists():
        logger.debug(f"kfold_statistics.json not found at {stats_path}")
        return None

    with open(stats_path) as f:
        data = json.load(f)

    synthetic_counts = []
    for fold_data in data:
        train = fold_data.get("train", {})
        synthetic = train.get("synthetic", 0)
        synthetic_counts.append(synthetic)

    if not synthetic_counts:
        return None

    return int(np.mean(synthetic_counts))


def scan_experiments(
    results_dir: Path,
    metric: str,
) -> list[ExperimentResult]:
    """Scan all experiments and collect results.

    Args:
        results_dir: Base directory containing xN folders
        metric: Metric to extract ('dice' or 'hd95')

    Returns:
        List of ExperimentResult objects
    """
    results = []

    # Find all xN directories
    for x_dir in sorted(results_dir.iterdir()):
        if not x_dir.is_dir():
            continue

        multiplier = extract_multiplier(x_dir.name)
        if multiplier is None:
            continue

        logger.info(f"Scanning {x_dir.name}...")

        # Find all experiment directories in this xN folder
        for exp_dir in x_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            # Extract model name from directory name (synth_xN_modelname)
            match = re.match(r"synth_x\d+_(\w+)$", exp_dir.name)
            if not match:
                continue

            model = match.group(1)

            # Load results
            metric_results = load_kfold_results(exp_dir, metric)
            if metric_results is None:
                logger.warning(f"No {metric} results found for {exp_dir.name}")
                continue

            sample_count = load_sample_count(exp_dir)
            if sample_count is None:
                logger.warning(f"No sample count found for {exp_dir.name}")
                continue

            result = ExperimentResult(
                multiplier=multiplier,
                model=model,
                n_synthetic_samples=sample_count,
                metric_mean=metric_results["mean"],
                metric_std=metric_results["std"],
                fold_values=metric_results["fold_values"],
            )
            results.append(result)

            logger.info(
                f"  {exp_dir.name}: {sample_count} samples, "
                f"{metric}={metric_results['mean']:.4f} +/- {metric_results['std']:.4f}"
            )

    return results


def create_performance_plot(
    results: list[ExperimentResult],
    metric: str,
    output_path: Path,
) -> None:
    """Create and save the performance plot.

    Args:
        results: List of experiment results
        metric: Metric name ('dice' or 'hd95')
        output_path: Path to save the plot
    """
    if not results:
        logger.error("No results to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group results by model
    models = sorted(set(r.model for r in results))

    for model in models:
        model_results = [r for r in results if r.model == model]
        model_results.sort(key=lambda r: r.n_synthetic_samples)

        if not model_results:
            continue

        config = MODEL_CONFIG.get(model, {"display_name": model, "color": "gray", "marker": "o"})

        x = [r.n_synthetic_samples for r in model_results]
        y = [r.metric_mean for r in model_results]
        yerr = [r.metric_std for r in model_results]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=config["display_name"],
            color=config["color"],
            marker=config["marker"],
            markersize=8,
            linewidth=2,
            capsize=4,
            capthick=1.5,
        )

    # Formatting
    metric_display = "Dice Score" if metric == "dice" else "HD95 (mm)"
    ax.set_xlabel("Number of Synthetic Training Images", fontsize=12)
    ax.set_ylabel(f"Test {metric_display}", fontsize=12)
    ax.set_title(f"TSTR Performance: {metric_display} vs Dataset Size", fontsize=14)

    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Use log scale for x-axis if range is large
    x_values = [r.n_synthetic_samples for r in results]
    if max(x_values) / min(x_values) > 5:
        ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot to {output_path}")


def save_results_csv(
    results: list[ExperimentResult],
    metric: str,
    output_path: Path,
) -> None:
    """Save results to CSV file.

    Args:
        results: List of experiment results
        metric: Metric name
        output_path: Path to save CSV
    """
    rows = []
    for r in results:
        rows.append(
            {
                "multiplier": r.multiplier,
                "model": r.model,
                "n_synthetic_samples": r.n_synthetic_samples,
                f"{metric}_mean": r.metric_mean,
                f"{metric}_std": r.metric_std,
                "n_folds": len(r.fold_values),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "n_synthetic_samples"])
    df.to_csv(output_path, index=False)

    logger.info(f"Saved CSV to {output_path}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Visualize TSTR performance across dataset sizes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.segmentation.scripts.visualize_replicas_performance \\
        --results-dir /path/to/synthetic_only \\
        --output-dir outputs/tstr_performance \\
        --metric dice
        """,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing xN folders with experiments",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for plots and CSV",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="dice",
        choices=["dice", "hd95"],
        help="Metric to plot (default: dice)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan and collect all results
    results = scan_experiments(results_dir, args.metric)

    if not results:
        logger.error("No valid experiment results found")
        return

    logger.info(f"Found {len(results)} experiment results")

    # Create plot
    plot_path = output_dir / f"tstr_performance_{args.metric}.png"
    create_performance_plot(results, args.metric, plot_path)

    # Save CSV
    csv_path = output_dir / f"tstr_performance_{args.metric}.csv"
    save_results_csv(results, args.metric, csv_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
