"""Publication-ready panel visualization.

Creates a single figure with three subplots:
1. Train/Val loss comparison
2. Precision (exp(-log sigma^2)) evolution
3. Per-timestep MSE heatmap with t@{timestep} labels
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from omegaconf import DictConfig

from ..data_loader import ComparisonDataLoader
from ..utils import PLOT_SETTINGS, get_model_color, smooth_series
from .base import BaseVisualization

logger = logging.getLogger(__name__)


class PublicationPanelVisualization(BaseVisualization):
    """Generate publication-ready panel with loss, precision, and MSE heatmap."""

    name = "publication_panel"

    def generate(self, data_loader: ComparisonDataLoader) -> list[Path]:
        """Generate publication panel figure.

        Creates a 1x3 subplot figure with:
        - Left: Train/Val loss curves
        - Middle: Precision evolution
        - Right: Per-timestep MSE heatmap

        Args:
            data_loader: ComparisonDataLoader instance.

        Returns:
            List of saved plot paths.
        """
        saved_paths = []

        # Get common settings
        max_epoch = data_loader.get_min_max_epoch()
        epochs = data_loader.get_epochs(max_epoch)
        smoothing_window = getattr(self.viz_cfg, "smoothing_window", 10)

        # Create figure with 3 subplots
        figsize = self._get_figsize((18, 5))
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)

        # Panel 1: Loss curves
        ax_loss = fig.add_subplot(gs[0, 0])
        self._plot_loss_panel(ax_loss, data_loader, epochs, max_epoch, smoothing_window)

        # Panel 2: Precision evolution
        ax_precision = fig.add_subplot(gs[0, 1])
        self._plot_precision_panel(ax_precision, data_loader, epochs, max_epoch, smoothing_window)

        # Panel 3: Timestep MSE heatmap
        ax_heatmap = fig.add_subplot(gs[0, 2])
        self._plot_timestep_mse_panel(ax_heatmap, data_loader, max_epoch)

        # Add panel labels
        for ax, label in zip([ax_loss, ax_precision, ax_heatmap], ["A", "B", "C"]):
            ax.text(
                -0.12, 1.05, label,
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="bottom",
            )

        fig.tight_layout()

        # Save figure
        saved_paths.extend(self._save_figure(fig, "publication_panel"))
        self._close_figure(fig)

        logger.info(f"Generated publication panel with {len(saved_paths)} outputs")
        return saved_paths

    def _plot_loss_panel(
        self,
        ax: plt.Axes,
        data_loader: ComparisonDataLoader,
        epochs: NDArray[np.int64],
        max_epoch: int,
        smoothing_window: int,
    ) -> None:
        """Plot train/val loss comparison.

        Args:
            ax: Matplotlib axes.
            data_loader: Data loader instance.
            epochs: Epoch array.
            max_epoch: Maximum epoch.
            smoothing_window: Smoothing window size.
        """
        loss_metrics = ["train/loss", "val/loss"]
        linestyles = ["-", "--"]

        for model_idx, model_name in enumerate(self.model_names):
            color = get_model_color(model_idx)

            for metric, linestyle in zip(loss_metrics, linestyles):
                try:
                    values = data_loader.get_metric_over_epochs(metric, max_epoch)
                    if model_name not in values:
                        continue

                    y = values[model_name]
                    x = epochs[: len(y)]

                    if smoothing_window > 1:
                        y = smooth_series(y, smoothing_window)

                    # Short label
                    split = "Train" if "train" in metric else "Val"
                    label = f"{model_name} ({split})"

                    ax.plot(
                        x, y,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.8,
                        label=label,
                        alpha=0.9,
                    )
                except Exception as e:
                    logger.warning(f"Failed to plot {metric} for {model_name}: {e}")

        ax.set_xlabel("Epoch", fontsize=PLOT_SETTINGS["xlabel_fontsize"])
        ax.set_ylabel("Loss", fontsize=PLOT_SETTINGS["ylabel_fontsize"])
        ax.set_title("Training Progress", fontsize=PLOT_SETTINGS["title_fontsize"])
        ax.legend(fontsize=8, loc="upper right", ncol=1)
        ax.grid(True, alpha=PLOT_SETTINGS["grid_alpha"])

    def _plot_precision_panel(
        self,
        ax: plt.Axes,
        data_loader: ComparisonDataLoader,
        epochs: NDArray[np.int64],
        max_epoch: int,
        smoothing_window: int,
    ) -> None:
        """Plot precision (exp(-log sigma^2)) evolution.

        Args:
            ax: Matplotlib axes.
            data_loader: Data loader instance.
            epochs: Epoch array.
            max_epoch: Maximum epoch.
            smoothing_window: Smoothing window size.
        """
        # Look for precision metrics - use the actual naming convention
        # Metrics are named precision_mse_group and precision_ffl_group
        precision_patterns = [
            "train/precision_mse_group",
            "train/precision_ffl_group",
        ]

        linestyles = ["-", "--", "-.", ":"]
        plotted_any = False

        for model_idx, model_name in enumerate(self.model_names):
            color = get_model_color(model_idx)
            metric_idx = 0

            for metric in precision_patterns:
                try:
                    values = data_loader.get_metric_over_epochs(metric, max_epoch)
                    if model_name not in values:
                        continue

                    y = values[model_name]
                    # Skip if all NaN
                    if np.all(np.isnan(y)):
                        continue

                    x = epochs[: len(y)]

                    if smoothing_window > 1:
                        y = smooth_series(y, smoothing_window)

                    # Extract group name for label
                    if "mse" in metric.lower():
                        group = "MSE"
                    elif "ffl" in metric.lower():
                        group = "FFL"
                    elif "image" in metric.lower():
                        group = "Image"
                    elif "mask" in metric.lower():
                        group = "Mask"
                    else:
                        group = metric.split("/")[-1]

                    label = f"{model_name} ({group})"
                    linestyle = linestyles[metric_idx % len(linestyles)]

                    ax.plot(
                        x, y,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.8,
                        label=label,
                        alpha=0.9,
                    )
                    plotted_any = True
                    metric_idx += 1
                except Exception as e:
                    logger.debug(f"Metric {metric} not available for {model_name}: {e}")

        ax.set_xlabel("Epoch", fontsize=PLOT_SETTINGS["xlabel_fontsize"])
        ax.set_ylabel(r"Precision $\exp(-\log\sigma^2)$", fontsize=PLOT_SETTINGS["ylabel_fontsize"])
        ax.set_title("Uncertainty Weighting", fontsize=PLOT_SETTINGS["title_fontsize"])

        if plotted_any:
            ax.legend(fontsize=8, loc="best", ncol=1)
        else:
            ax.text(
                0.5, 0.5, "No precision metrics\navailable",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=12, color="gray",
            )

        ax.grid(True, alpha=PLOT_SETTINGS["grid_alpha"])

    def _plot_timestep_mse_panel(
        self,
        ax: plt.Axes,
        data_loader: ComparisonDataLoader,
        max_epoch: int,
    ) -> None:
        """Plot per-timestep MSE heatmap.

        Args:
            ax: Matplotlib axes.
            data_loader: Data loader instance.
            max_epoch: Maximum epoch (for selecting final epoch values).
        """
        # Get common metrics and filter for timestep MSE
        common_metrics = data_loader.get_common_metrics()
        mse_pattern = re.compile(r"diagnostics/pred_mse_image_t(\d+)")

        # Extract timestep values and sort
        timestep_metrics = {}
        for m in common_metrics:
            match = mse_pattern.match(m)
            if match:
                t = int(match.group(1))
                timestep_metrics[t] = m

        if not timestep_metrics:
            ax.text(
                0.5, 0.5, "No timestep MSE\nmetrics available",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=12, color="gray",
            )
            return

        timesteps = sorted(timestep_metrics.keys())
        n_timesteps = len(timesteps)
        n_models = len(self.model_names)

        # Build data matrix: rows = models, cols = timesteps
        data_matrix = np.zeros((n_models, n_timesteps))

        for model_idx, model_name in enumerate(self.model_names):
            loader = data_loader.loaders[model_name]
            try:
                df = loader.load_performance_csv()
                target_epoch = loader.get_max_epoch()
                row = df[df["epoch"] == target_epoch]

                if row.empty:
                    continue

                for t_idx, t in enumerate(timesteps):
                    metric = timestep_metrics[t]
                    if metric in row.columns:
                        value = row[metric].values[0]
                        data_matrix[model_idx, t_idx] = value if not np.isnan(value) else 0
            except Exception as e:
                logger.warning(f"Failed to get timestep MSE for {model_name}: {e}")

        # Create heatmap
        cmap = getattr(self.viz_cfg, "heatmap_cmap", "viridis")
        im = ax.imshow(data_matrix, cmap=cmap, aspect="auto")

        # Set labels with t@{timestep} format
        ax.set_xticks(np.arange(n_timesteps))
        ax.set_xticklabels([f"t@{t}" for t in timesteps], rotation=45, ha="right", fontsize=9)
        ax.set_yticks(np.arange(n_models))
        ax.set_yticklabels(self.model_names, fontsize=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("MSE", fontsize=10)

        # Add value annotations (if not too many cells)
        if n_models * n_timesteps <= 50:
            for i in range(n_models):
                for j in range(n_timesteps):
                    value = data_matrix[i, j]
                    text_color = "white" if value > data_matrix.mean() else "black"
                    ax.text(
                        j, i, f"{value:.3f}",
                        ha="center", va="center",
                        color=text_color, fontsize=7,
                    )

        ax.set_xlabel("Timestep", fontsize=PLOT_SETTINGS["xlabel_fontsize"])
        ax.set_ylabel("Model", fontsize=PLOT_SETTINGS["ylabel_fontsize"])
        ax.set_title("Prediction MSE by Timestep", fontsize=PLOT_SETTINGS["title_fontsize"])
