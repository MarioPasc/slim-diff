"""Summary table visualization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import DictConfig

from ..data_loader import ComparisonDataLoader
from ..utils import PLOT_SETTINGS, compute_ranking, get_model_color
from .base import BaseVisualization

logger = logging.getLogger(__name__)


class SummaryTableVisualization(BaseVisualization):
    """Generate summary metrics table comparing all models."""

    name = "summary_table"

    def generate(self, data_loader: ComparisonDataLoader) -> list[Path]:
        """Generate summary table.

        Creates:
        - PNG/PDF table with final metrics for all models
        - CSV export of the same data

        Args:
            data_loader: ComparisonDataLoader instance.

        Returns:
            List of saved file paths.
        """
        saved_paths = []

        metrics = list(self.viz_cfg.metrics)
        epoch = self.viz_cfg.epoch
        primary_metric = self.viz_cfg.primary_metric
        lower_is_better = self.viz_cfg.lower_is_better

        # Get summary DataFrame
        summary_df = data_loader.get_summary_dataframe(metrics, epoch)

        if summary_df.empty:
            logger.warning("No data available for summary table")
            return saved_paths

        # Add ranking based on primary metric
        if primary_metric in summary_df.columns:
            rankings = compute_ranking(
                summary_df[primary_metric].to_dict(),
                lower_is_better=lower_is_better,
            )
            summary_df["Rank"] = summary_df.index.map(rankings)
            # Sort by rank
            summary_df = summary_df.sort_values("Rank")

        # Create table figure
        fig = self._create_table_figure(summary_df, primary_metric, lower_is_better)
        saved_paths.extend(self._save_figure(fig, "summary_table"))
        self._close_figure(fig)

        # Save CSV
        if self.cfg.output.save_data:
            csv_path = self._save_summary_csv(summary_df)
            saved_paths.append(csv_path)

        logger.info(f"Generated summary table with {len(saved_paths)} outputs")
        return saved_paths

    def _create_table_figure(
        self,
        summary_df: pd.DataFrame,
        primary_metric: str,
        lower_is_better: bool,
    ) -> plt.Figure:
        """Create table visualization using matplotlib.

        Args:
            summary_df: Summary DataFrame.
            primary_metric: Metric used for ranking.
            lower_is_better: Whether lower values are better.

        Returns:
            Matplotlib figure with table.
        """
        # Prepare display dataframe
        display_df = summary_df.copy()

        # Format numeric columns
        for col in display_df.columns:
            if col == "Rank":
                display_df[col] = display_df[col].astype(int)
            elif display_df[col].dtype in [np.float64, np.float32]:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")

        # Rename columns for display
        col_names = {col: col.replace("/", "\n") for col in display_df.columns}
        display_df = display_df.rename(columns=col_names)

        # Calculate figure size based on table dimensions
        n_rows = len(display_df)
        n_cols = len(display_df.columns)
        figsize = (max(12, n_cols * 1.5), max(4, n_rows * 0.6 + 2))

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        # Create table
        table_data = display_df.reset_index()
        table_data = table_data.rename(columns={"index": "Model"})

        col_labels = list(table_data.columns)
        cell_text = table_data.values.tolist()

        # Create color map for highlighting best values
        cell_colors = self._compute_cell_colors(
            summary_df,
            primary_metric,
            lower_is_better,
        )

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            cellColours=cell_colors,
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style header row
        for j, label in enumerate(col_labels):
            cell = table[(0, j)]
            cell.set_text_props(weight="bold", fontsize=11)
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")

        # Style model name column (bold)
        for i in range(len(cell_text)):
            cell = table[(i + 1, 0)]
            cell.set_text_props(weight="bold")

        ax.set_title(
            "Model Comparison Summary",
            fontsize=PLOT_SETTINGS["title_fontsize"],
            pad=20,
        )

        # Add subtitle with epoch info
        epoch = self.viz_cfg.epoch
        epoch_str = "Final Epoch" if epoch == -1 else f"Epoch {epoch}"
        ax.text(
            0.5, 0.02, f"({epoch_str}) | Primary Metric: {primary_metric}",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            style="italic",
        )

        fig.tight_layout()
        return fig

    def _compute_cell_colors(
        self,
        summary_df: pd.DataFrame,
        primary_metric: str,
        lower_is_better: bool,
    ) -> list[list[str]]:
        """Compute cell background colors for highlighting.

        Args:
            summary_df: Summary DataFrame.
            primary_metric: Metric used for ranking.
            lower_is_better: Whether lower values are better.

        Returns:
            2D list of color strings.
        """
        n_rows = len(summary_df)
        n_cols = len(summary_df.columns) + 1  # +1 for model name column

        # Initialize with white
        colors = [["white"] * n_cols for _ in range(n_rows)]

        # Color scheme
        best_color = "#90EE90"  # Light green for best
        good_color = "#E0FFE0"  # Very light green for second best
        neutral_color = "white"

        # Find best/worst for each metric
        for j, col in enumerate(summary_df.columns):
            col_idx = j + 1  # Account for model name column

            if col == "Rank":
                # Rank 1 = best
                for i, rank in enumerate(summary_df[col]):
                    if rank == 1:
                        colors[i][col_idx] = best_color
                    elif rank == 2:
                        colors[i][col_idx] = good_color
            elif summary_df[col].dtype in [np.float64, np.float32]:
                values = summary_df[col].dropna()
                if len(values) == 0:
                    continue

                if lower_is_better:
                    best_val = values.min()
                    second_best = values.nsmallest(2).iloc[-1] if len(values) > 1 else None
                else:
                    best_val = values.max()
                    second_best = values.nlargest(2).iloc[-1] if len(values) > 1 else None

                for i, val in enumerate(summary_df[col]):
                    if pd.isna(val):
                        continue
                    if val == best_val:
                        colors[i][col_idx] = best_color
                    elif second_best is not None and val == second_best:
                        colors[i][col_idx] = good_color

        return colors

    def _save_summary_csv(self, summary_df: pd.DataFrame) -> Path:
        """Save summary DataFrame to CSV.

        Args:
            summary_df: Summary DataFrame.

        Returns:
            Path to saved CSV file.
        """
        data_dir = self.output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        csv_path = data_dir / "summary_table.csv"

        summary_df.to_csv(csv_path)
        logger.info(f"Saved summary CSV to {csv_path}")
        return csv_path
