"""Base class for visualizations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from omegaconf import DictConfig

from ..utils import apply_plot_settings

logger = logging.getLogger(__name__)


class BaseVisualization(ABC):
    """Abstract base class for all visualizations."""

    # Name of the visualization (used for config lookup)
    name: str = "base"

    def __init__(
        self,
        cfg: DictConfig,
        output_dir: Path,
        model_names: list[str],
    ) -> None:
        """Initialize visualization.

        Args:
            cfg: Full configuration object.
            output_dir: Output directory for plots.
            model_names: List of model names being compared.
        """
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.model_names = model_names

        # Get visualization-specific config
        self.viz_cfg = getattr(cfg, self.name, DictConfig({}))

        # Common settings
        self.formats = cfg.visualizations.formats
        self.dpi = cfg.visualizations.dpi

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Apply plot settings
        apply_plot_settings()

    @abstractmethod
    def generate(self, data: dict[str, Any]) -> list[Path]:
        """Generate the visualization.

        Args:
            data: Loaded comparison data (structure depends on visualization).

        Returns:
            List of paths to generated plot files.
        """
        ...

    def _save_figure(
        self,
        fig: plt.Figure,
        name: str,
        formats: list[str] | None = None,
    ) -> list[Path]:
        """Save figure in multiple formats.

        Args:
            fig: Matplotlib figure.
            name: Base filename (without extension).
            formats: Output formats (default from config).

        Returns:
            List of saved file paths.
        """
        if formats is None:
            formats = self.formats

        saved_paths = []
        for fmt in formats:
            filepath = self.output_dir / f"{name}.{fmt}"
            fig.savefig(
                filepath,
                format=fmt,
                dpi=self.dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            saved_paths.append(filepath)
            logger.debug(f"Saved figure: {filepath}")

        return saved_paths

    def _get_figsize(self, default: tuple[float, float] = (12, 8)) -> tuple[float, float]:
        """Get figure size from config or default.

        Args:
            default: Default figure size.

        Returns:
            Figure size as (width, height) tuple.
        """
        if hasattr(self.viz_cfg, "figsize"):
            return tuple(self.viz_cfg.figsize)
        return default

    def _close_figure(self, fig: plt.Figure) -> None:
        """Close a matplotlib figure to free memory.

        Args:
            fig: Figure to close.
        """
        plt.close(fig)
