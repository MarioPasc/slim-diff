"""Loss comparison curves visualization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig

from ..data_loader import ComparisonDataLoader
from ..utils import PLOT_SETTINGS, get_model_color, smooth_series
from .base import BaseVisualization

logger = logging.getLogger(__name__)


class LossCurvesVisualization(BaseVisualization):
    """Generate loss comparison curves across models."""

    name = "loss_curves"

    def generate(self, data_loader: ComparisonDataLoader) -> list[Path]:
        """Generate loss curve comparison plots.

        Creates:
        - Combined train/val loss plot
        - Per-channel loss plots (image/mask)

        Args:
            data_loader: ComparisonDataLoader instance.

        Returns:
            List of saved plot paths.
        """
        saved_paths = []

        # Get smoothing window from config
        smoothing_window = getattr(self.viz_cfg, "smoothing_window", 10)
        metrics = list(self.viz_cfg.metrics)

        # Get epochs (use minimum max epoch for fair comparison)
        max_epoch = data_loader.get_min_max_epoch()
        epochs = data_loader.get_epochs(max_epoch)

        # Group metrics by type
        train_metrics = [m for m in metrics if m.startswith("train/")]
        val_metrics = [m for m in metrics if m.startswith("val/")]

        # Plot 1: Combined train/val total loss
        total_metrics = ["train/loss", "val/loss"]
        available_total = [m for m in total_metrics if m in metrics]
        if available_total:
            fig = self._plot_metric_comparison(
                data_loader=data_loader,
                metrics=available_total,
                epochs=epochs,
                max_epoch=max_epoch,
                smoothing_window=smoothing_window,
                title="Training and Validation Loss Comparison",
                ylabel="Loss",
            )
            saved_paths.extend(self._save_figure(fig, "loss_curves_total"))
            self._close_figure(fig)

        # Plot 2: Per-channel comparison (image)
        image_metrics = [m for m in metrics if "image" in m]
        if image_metrics:
            fig = self._plot_metric_comparison(
                data_loader=data_loader,
                metrics=image_metrics,
                epochs=epochs,
                max_epoch=max_epoch,
                smoothing_window=smoothing_window,
                title="Image Channel Loss Comparison",
                ylabel="Loss (Image)",
            )
            saved_paths.extend(self._save_figure(fig, "loss_curves_image"))
            self._close_figure(fig)

        # Plot 3: Per-channel comparison (mask)
        mask_metrics = [m for m in metrics if "mask" in m]
        if mask_metrics:
            fig = self._plot_metric_comparison(
                data_loader=data_loader,
                metrics=mask_metrics,
                epochs=epochs,
                max_epoch=max_epoch,
                smoothing_window=smoothing_window,
                title="Mask Channel Loss Comparison",
                ylabel="Loss (Mask)",
            )
            saved_paths.extend(self._save_figure(fig, "loss_curves_mask"))
            self._close_figure(fig)

        # Plot 4: All metrics in subplot grid
        fig = self._plot_all_metrics_grid(
            data_loader=data_loader,
            metrics=metrics,
            epochs=epochs,
            max_epoch=max_epoch,
            smoothing_window=smoothing_window,
        )
        saved_paths.extend(self._save_figure(fig, "loss_curves_all"))
        self._close_figure(fig)

        logger.info(f"Generated {len(saved_paths)} loss curve plots")
        return saved_paths

    def _plot_metric_comparison(
        self,
        data_loader: ComparisonDataLoader,
        metrics: list[str],
        epochs: NDArray[np.int64],
        max_epoch: int,
        smoothing_window: int,
        title: str,
        ylabel: str,
    ) -> plt.Figure:
        """Create comparison plot for specific metrics.

        Args:
            data_loader: ComparisonDataLoader instance.
            metrics: Metric column names to plot.
            epochs: Epoch array.
            max_epoch: Maximum epoch to include.
            smoothing_window: Rolling average window.
            title: Plot title.
            ylabel: Y-axis label.

        Returns:
            Matplotlib figure.
        """
        figsize = self._get_figsize((14, 8))
        fig, ax = plt.subplots(figsize=figsize)

        # Line styles for different metrics
        linestyles = ["-", "--", "-.", ":"]

        for model_idx, model_name in enumerate(self.model_names):
            color = get_model_color(model_idx)

            for metric_idx, metric in enumerate(metrics):
                try:
                    values = data_loader.get_metric_over_epochs(metric, max_epoch)
                    if model_name not in values:
                        continue

                    y = values[model_name]
                    x = epochs[: len(y)]

                    # Apply smoothing
                    if smoothing_window > 1:
                        y = smooth_series(y, smoothing_window)

                    # Determine label
                    metric_short = metric.split("/")[-1]
                    label = f"{model_name} ({metric_short})"

                    # Plot with style
                    linestyle = linestyles[metric_idx % len(linestyles)]
                    ax.plot(
                        x,
                        y,
                        color=color,
                        linestyle=linestyle,
                        linewidth=PLOT_SETTINGS["line_width"],
                        label=label,
                        alpha=0.9,
                    )
                except Exception as e:
                    logger.warning(f"Failed to plot {metric} for {model_name}: {e}")

        ax.set_xlabel("Epoch", fontsize=PLOT_SETTINGS["xlabel_fontsize"])
        ax.set_ylabel(ylabel, fontsize=PLOT_SETTINGS["ylabel_fontsize"])
        ax.set_title(title, fontsize=PLOT_SETTINGS["title_fontsize"])
        ax.legend(loc="upper right", fontsize=PLOT_SETTINGS["legend_fontsize"])
        ax.grid(True, alpha=PLOT_SETTINGS["grid_alpha"])

        fig.tight_layout()
        return fig

    def _plot_all_metrics_grid(
        self,
        data_loader: ComparisonDataLoader,
        metrics: list[str],
        epochs: NDArray[np.int64],
        max_epoch: int,
        smoothing_window: int,
    ) -> plt.Figure:
        """Create subplot grid with all metrics.

        Args:
            data_loader: ComparisonDataLoader instance.
            metrics: All metric column names to plot.
            epochs: Epoch array.
            max_epoch: Maximum epoch to include.
            smoothing_window: Rolling average window.

        Returns:
            Matplotlib figure with subplots.
        """
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        figsize = (5 * n_cols, 4 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, metric in enumerate(metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            for model_idx, model_name in enumerate(self.model_names):
                color = get_model_color(model_idx)

                try:
                    values = data_loader.get_metric_over_epochs(metric, max_epoch)
                    if model_name not in values:
                        continue

                    y = values[model_name]
                    x = epochs[: len(y)]

                    if smoothing_window > 1:
                        y = smooth_series(y, smoothing_window)

                    ax.plot(
                        x,
                        y,
                        color=color,
                        linewidth=1.5,
                        label=model_name,
                        alpha=0.8,
                    )
                except Exception as e:
                    logger.warning(f"Failed to plot {metric} for {model_name}: {e}")

            # Format metric name for title
            metric_title = metric.replace("/", " / ").replace("_", " ").title()
            ax.set_title(metric_title, fontsize=10)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.grid(True, alpha=0.3)

            if idx == 0:
                ax.legend(fontsize=8, loc="upper right")

        # Hide empty subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        fig.suptitle("All Metrics Comparison", fontsize=14, y=1.02)
        fig.tight_layout()
        return fig
