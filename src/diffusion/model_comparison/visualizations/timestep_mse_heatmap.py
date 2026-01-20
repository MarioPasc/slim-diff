"""Per-timestep MSE heatmap visualization."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from numpy.typing import NDArray
from omegaconf import DictConfig

from ..data_loader import ComparisonDataLoader
from ..utils import PLOT_SETTINGS, extract_channel_from_column, extract_timestep_from_column
from .base import BaseVisualization

logger = logging.getLogger(__name__)


class TimestepMSEHeatmapVisualization(BaseVisualization):
    """Generate heatmaps comparing per-timestep prediction MSE across models."""

    name = "timestep_mse_heatmap"

    def generate(self, data_loader: ComparisonDataLoader) -> list[Path]:
        """Generate per-timestep MSE heatmap comparison.

        Creates:
        - Heatmap for image channel MSE
        - Heatmap for mask channel MSE
        - Combined comparison heatmap

        Args:
            data_loader: ComparisonDataLoader instance.

        Returns:
            List of saved plot paths.
        """
        saved_paths = []

        # Get configuration
        channels = list(self.viz_cfg.channels)
        epoch = self.viz_cfg.epoch
        cmap = self.viz_cfg.cmap

        # Get common metrics matching the pattern
        common_metrics = data_loader.get_common_metrics()
        mse_pattern = re.compile(r"diagnostics/pred_mse_(image|mask)_t\d+")
        mse_metrics = [m for m in common_metrics if mse_pattern.match(m)]

        if not mse_metrics:
            logger.warning("No per-timestep MSE metrics found in common metrics")
            return saved_paths

        # Extract timestep bins and group by channel
        timestep_bins = sorted(set(
            extract_timestep_from_column(m) for m in mse_metrics
            if extract_timestep_from_column(m) is not None
        ))

        for channel in channels:
            channel_metrics = [m for m in mse_metrics if f"_{channel}_" in m]
            if not channel_metrics:
                logger.warning(f"No MSE metrics found for channel: {channel}")
                continue

            # Build data matrix: rows = models, cols = timesteps
            fig = self._create_heatmap(
                data_loader=data_loader,
                metrics=channel_metrics,
                timestep_bins=timestep_bins,
                epoch=epoch,
                channel=channel,
                cmap=cmap,
            )
            saved_paths.extend(self._save_figure(fig, f"timestep_mse_heatmap_{channel}"))
            self._close_figure(fig)

        # Combined comparison if both channels exist
        if len(channels) >= 2:
            fig = self._create_combined_heatmap(
                data_loader=data_loader,
                mse_metrics=mse_metrics,
                timestep_bins=timestep_bins,
                epoch=epoch,
                channels=channels,
                cmap=cmap,
            )
            saved_paths.extend(self._save_figure(fig, "timestep_mse_heatmap_combined"))
            self._close_figure(fig)

        logger.info(f"Generated {len(saved_paths)} timestep MSE heatmap plots")
        return saved_paths

    def _create_heatmap(
        self,
        data_loader: ComparisonDataLoader,
        metrics: list[str],
        timestep_bins: list[int],
        epoch: int,
        channel: str,
        cmap: str,
    ) -> plt.Figure:
        """Create heatmap for a single channel.

        Args:
            data_loader: ComparisonDataLoader instance.
            metrics: MSE metric column names for this channel.
            timestep_bins: Sorted list of timestep values.
            epoch: Epoch to use (-1 for final).
            channel: Channel name ('image' or 'mask').
            cmap: Colormap name.

        Returns:
            Matplotlib figure.
        """
        figsize = self._get_figsize((12, 6))
        fig, ax = plt.subplots(figsize=figsize)

        n_models = len(self.model_names)
        n_timesteps = len(timestep_bins)

        # Build data matrix
        data_matrix = np.zeros((n_models, n_timesteps))

        for model_idx, model_name in enumerate(self.model_names):
            loader = data_loader.loaders[model_name]
            df = loader.load_performance_csv()

            # Determine epoch
            if epoch == -1:
                target_epoch = loader.get_max_epoch()
            else:
                target_epoch = epoch

            row = df[df["epoch"] == target_epoch]
            if row.empty:
                logger.warning(f"No data for epoch {target_epoch} in {model_name}")
                continue

            for t_idx, t in enumerate(timestep_bins):
                metric_name = f"diagnostics/pred_mse_{channel}_t{t:04d}"
                if metric_name in row.columns:
                    value = row[metric_name].values[0]
                    data_matrix[model_idx, t_idx] = value if not np.isnan(value) else 0

        # Create heatmap
        im = ax.imshow(data_matrix, cmap=cmap, aspect="auto")

        # Set labels
        ax.set_xticks(np.arange(n_timesteps))
        ax.set_xticklabels([f"t={t}" for t in timestep_bins], rotation=45, ha="right")
        ax.set_yticks(np.arange(n_models))
        ax.set_yticklabels(self.model_names)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("MSE", fontsize=PLOT_SETTINGS["ylabel_fontsize"])

        # Add value annotations
        for i in range(n_models):
            for j in range(n_timesteps):
                value = data_matrix[i, j]
                text_color = "white" if value > data_matrix.mean() else "black"
                ax.text(
                    j, i, f"{value:.4f}",
                    ha="center", va="center",
                    color=text_color, fontsize=8,
                )

        ax.set_xlabel("Timestep", fontsize=PLOT_SETTINGS["xlabel_fontsize"])
        ax.set_ylabel("Model", fontsize=PLOT_SETTINGS["ylabel_fontsize"])
        ax.set_title(
            f"Per-Timestep Prediction MSE ({channel.title()} Channel)",
            fontsize=PLOT_SETTINGS["title_fontsize"],
        )

        fig.tight_layout()
        return fig

    def _create_combined_heatmap(
        self,
        data_loader: ComparisonDataLoader,
        mse_metrics: list[str],
        timestep_bins: list[int],
        epoch: int,
        channels: list[str],
        cmap: str,
    ) -> plt.Figure:
        """Create combined heatmap comparing both channels.

        Args:
            data_loader: ComparisonDataLoader instance.
            mse_metrics: All MSE metric column names.
            timestep_bins: Sorted list of timestep values.
            epoch: Epoch to use (-1 for final).
            channels: List of channel names.
            cmap: Colormap name.

        Returns:
            Matplotlib figure.
        """
        n_channels = len(channels)
        figsize = (6 * n_channels, 5)
        fig, axes = plt.subplots(1, n_channels, figsize=figsize)
        if n_channels == 1:
            axes = [axes]

        n_models = len(self.model_names)
        n_timesteps = len(timestep_bins)

        # Compute global min/max for consistent colorbar
        all_values = []

        for channel_idx, channel in enumerate(channels):
            data_matrix = np.zeros((n_models, n_timesteps))

            for model_idx, model_name in enumerate(self.model_names):
                loader = data_loader.loaders[model_name]
                df = loader.load_performance_csv()

                if epoch == -1:
                    target_epoch = loader.get_max_epoch()
                else:
                    target_epoch = epoch

                row = df[df["epoch"] == target_epoch]
                if row.empty:
                    continue

                for t_idx, t in enumerate(timestep_bins):
                    metric_name = f"diagnostics/pred_mse_{channel}_t{t:04d}"
                    if metric_name in row.columns:
                        value = row[metric_name].values[0]
                        data_matrix[model_idx, t_idx] = value if not np.isnan(value) else 0

            all_values.append(data_matrix)

        # Get global range
        vmin = min(m.min() for m in all_values)
        vmax = max(m.max() for m in all_values)

        # Plot each channel
        for channel_idx, (channel, data_matrix) in enumerate(zip(channels, all_values)):
            ax = axes[channel_idx]

            im = ax.imshow(
                data_matrix,
                cmap=cmap,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xticks(np.arange(n_timesteps))
            ax.set_xticklabels([f"t={t}" for t in timestep_bins], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(np.arange(n_models))
            ax.set_yticklabels(self.model_names, fontsize=9)

            ax.set_xlabel("Timestep", fontsize=10)
            if channel_idx == 0:
                ax.set_ylabel("Model", fontsize=10)
            ax.set_title(f"{channel.title()} Channel", fontsize=12)

        # Add single colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("MSE", fontsize=10)

        fig.suptitle("Per-Timestep Prediction MSE Comparison", fontsize=14, y=1.02)
        return fig
