"""Uncertainty weight evolution visualization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from ..data_loader import ComparisonDataLoader
from ..utils import PLOT_SETTINGS, get_model_color, smooth_series
from .base import BaseVisualization

logger = logging.getLogger(__name__)


class UncertaintyEvolutionVisualization(BaseVisualization):
    """Visualize how uncertainty weights evolve during training."""

    name = "uncertainty_evolution"

    def generate(self, data_loader: ComparisonDataLoader) -> list[Path]:
        """Generate uncertainty evolution plots.

        Creates:
        - log_var evolution across models
        - sigma evolution across models
        - precision evolution across models

        Args:
            data_loader: ComparisonDataLoader instance.

        Returns:
            List of saved plot paths.
        """
        saved_paths = []

        metrics = list(self.viz_cfg.metrics)
        max_epoch = data_loader.get_min_max_epoch()
        epochs = data_loader.get_epochs(max_epoch)

        # Group metrics by type
        log_var_metrics = [m for m in metrics if "log_var" in m]
        sigma_metrics = [m for m in metrics if "sigma" in m]
        precision_metrics = [m for m in metrics if "precision" in m]

        # Plot log variance evolution
        if log_var_metrics:
            fig = self._plot_uncertainty_type(
                data_loader=data_loader,
                metrics=log_var_metrics,
                epochs=epochs,
                max_epoch=max_epoch,
                title="Log Variance Evolution",
                ylabel=r"$\log(\sigma^2)$",
            )
            saved_paths.extend(self._save_figure(fig, "uncertainty_log_var"))
            self._close_figure(fig)

        # Plot sigma evolution
        if sigma_metrics:
            fig = self._plot_uncertainty_type(
                data_loader=data_loader,
                metrics=sigma_metrics,
                epochs=epochs,
                max_epoch=max_epoch,
                title="Sigma (Std Dev) Evolution",
                ylabel=r"$\sigma$",
            )
            saved_paths.extend(self._save_figure(fig, "uncertainty_sigma"))
            self._close_figure(fig)

        # Plot precision evolution
        if precision_metrics:
            fig = self._plot_uncertainty_type(
                data_loader=data_loader,
                metrics=precision_metrics,
                epochs=epochs,
                max_epoch=max_epoch,
                title="Precision (Task Weight) Evolution",
                ylabel=r"$\exp(-\log\sigma^2)$",
            )
            saved_paths.extend(self._save_figure(fig, "uncertainty_precision"))
            self._close_figure(fig)

        # Combined overview plot
        if metrics:
            fig = self._plot_combined_overview(
                data_loader=data_loader,
                metrics=metrics,
                epochs=epochs,
                max_epoch=max_epoch,
            )
            saved_paths.extend(self._save_figure(fig, "uncertainty_overview"))
            self._close_figure(fig)

        logger.info(f"Generated {len(saved_paths)} uncertainty evolution plots")
        return saved_paths

    def _plot_uncertainty_type(
        self,
        data_loader: ComparisonDataLoader,
        metrics: list[str],
        epochs: NDArray[np.int64],
        max_epoch: int,
        title: str,
        ylabel: str,
    ) -> plt.Figure:
        """Create plot for a specific uncertainty metric type.

        Args:
            data_loader: ComparisonDataLoader instance.
            metrics: Metric column names to plot.
            epochs: Epoch array.
            max_epoch: Maximum epoch to include.
            title: Plot title.
            ylabel: Y-axis label.

        Returns:
            Matplotlib figure.
        """
        figsize = self._get_figsize((14, 6))
        fig, ax = plt.subplots(figsize=figsize)

        # Line styles for different metrics (e.g., mse vs ffl)
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

                    # Determine label based on metric name
                    if "mse" in metric.lower():
                        group_label = "MSE"
                    elif "ffl" in metric.lower():
                        group_label = "FFL"
                    else:
                        group_label = metric.split("/")[-1]

                    label = f"{model_name} ({group_label})"
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
        ax.legend(loc="best", fontsize=PLOT_SETTINGS["legend_fontsize"])
        ax.grid(True, alpha=PLOT_SETTINGS["grid_alpha"])

        fig.tight_layout()
        return fig

    def _plot_combined_overview(
        self,
        data_loader: ComparisonDataLoader,
        metrics: list[str],
        epochs: NDArray[np.int64],
        max_epoch: int,
    ) -> plt.Figure:
        """Create combined subplot overview of all uncertainty metrics.

        Args:
            data_loader: ComparisonDataLoader instance.
            metrics: All metric column names.
            epochs: Epoch array.
            max_epoch: Maximum epoch to include.

        Returns:
            Matplotlib figure with subplots.
        """
        # Group by type
        log_var_metrics = [m for m in metrics if "log_var" in m]
        sigma_metrics = [m for m in metrics if "sigma" in m]
        precision_metrics = [m for m in metrics if "precision" in m]

        n_types = sum([len(log_var_metrics) > 0, len(sigma_metrics) > 0, len(precision_metrics) > 0])
        if n_types == 0:
            # Return empty figure if no metrics
            return plt.figure()

        fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5))
        if n_types == 1:
            axes = [axes]

        ax_idx = 0

        type_configs = [
            (log_var_metrics, r"$\log(\sigma^2)$", "Log Variance"),
            (sigma_metrics, r"$\sigma$", "Sigma"),
            (precision_metrics, r"Precision", "Precision"),
        ]

        for metric_list, ylabel, title in type_configs:
            if not metric_list:
                continue

            ax = axes[ax_idx]
            linestyles = ["-", "--"]

            for model_idx, model_name in enumerate(self.model_names):
                color = get_model_color(model_idx)

                for metric_idx, metric in enumerate(metric_list):
                    try:
                        values = data_loader.get_metric_over_epochs(metric, max_epoch)
                        if model_name not in values:
                            continue

                        y = values[model_name]
                        x = epochs[: len(y)]

                        # Short label
                        if "mse" in metric.lower():
                            group = "MSE"
                        elif "ffl" in metric.lower():
                            group = "FFL"
                        else:
                            group = ""

                        label = f"{model_name} {group}".strip()
                        linestyle = linestyles[metric_idx % len(linestyles)]

                        ax.plot(
                            x,
                            y,
                            color=color,
                            linestyle=linestyle,
                            linewidth=1.5,
                            label=label,
                            alpha=0.8,
                        )
                    except Exception:
                        pass

            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

            ax_idx += 1

        fig.suptitle("Uncertainty Weighting Evolution", fontsize=14, y=1.02)
        fig.tight_layout()
        return fig
