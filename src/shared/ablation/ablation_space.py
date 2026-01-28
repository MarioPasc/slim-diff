"""Ablation space definition for multi-axis experiment studies.

Provides classes for defining the full N-dimensional parameter space
and enumerating/filtering experiments within it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterator

from omegaconf import DictConfig

from src.shared.ablation.experiment_coords import ExperimentCoordinate


@dataclass
class AblationAxis:
    """Definition of a single axis in the ablation space.

    Attributes:
        name: Axis name (e.g., "prediction_type", "lp_norm", "self_cond_p").
        values: List of possible values along this axis.
        display_format: Format string for display names (e.g., "{:.1f}" for floats).
        folder_format: Format string for folder names (e.g., "self_cond_p{value}").
    """

    name: str
    values: list[Any]
    display_format: str = "{}"
    folder_format: str = "{value}"

    def format_value(self, value: Any) -> str:
        """Format a value using the display format.

        Args:
            value: The value to format.

        Returns:
            Formatted string.
        """
        return self.display_format.format(value)

    def format_folder(self, value: Any) -> str:
        """Format a value for folder naming.

        Args:
            value: The value to format.

        Returns:
            Folder name string.
        """
        formatted = self.format_value(value)
        return self.folder_format.format(value=formatted, name=self.name)


@dataclass
class AblationSpace:
    """Definition of the full N-dimensional ablation parameter space.

    Provides methods for enumerating all experiments in the space and
    filtering by axis values.

    Attributes:
        axes: Dictionary mapping axis names to AblationAxis definitions.
        hierarchy: Ordered list of axis names defining folder nesting order.
    """

    axes: dict[str, AblationAxis] = field(default_factory=dict)
    hierarchy: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set default hierarchy if not specified."""
        if not self.hierarchy:
            self.hierarchy = list(self.axes.keys())

    @classmethod
    def default(cls) -> AblationSpace:
        """Create the default ICIP 2026 ablation space.

        Returns:
            AblationSpace with standard 3-axis configuration.
        """
        return cls(
            axes={
                "self_cond_p": AblationAxis(
                    name="self_cond_p",
                    values=[0.0, 0.5, 0.8],
                    display_format="{:.1f}",
                    folder_format="self_cond_p_{value}",
                ),
                "prediction_type": AblationAxis(
                    name="prediction_type",
                    values=["epsilon", "velocity", "x0"],
                    display_format="{}",
                    folder_format="{value}",
                ),
                "lp_norm": AblationAxis(
                    name="lp_norm",
                    values=[1.5, 2.0, 2.5],
                    display_format="{:.1f}",
                    folder_format="lp_{value}",
                ),
            },
            hierarchy=["self_cond_p", "prediction_type_lp_norm"],
        )

    @classmethod
    def from_config(cls, cfg: DictConfig | dict) -> AblationSpace:
        """Create ablation space from configuration.

        Expected config structure:
        ```yaml
        ablation:
          axes:
            self_cond_p:
              values: [0.0, 0.5, 0.8]
              display_format: "{:.1f}"
              folder_format: "self_cond_p{value}"
            prediction_type:
              values: ["epsilon", "velocity", "x0"]
            lp_norm:
              values: [1.5, 2.0, 2.5]
              folder_format: "lp_{value}"
          hierarchy: ["self_cond_p", "prediction_type_lp_norm"]
        ```

        Args:
            cfg: Configuration dict or DictConfig with 'ablation' section.

        Returns:
            Configured AblationSpace instance.
        """
        if isinstance(cfg, DictConfig):
            cfg = dict(cfg)

        # Handle both top-level ablation key and direct ablation config
        if "ablation" in cfg:
            ablation_cfg = cfg["ablation"]
        else:
            ablation_cfg = cfg

        axes = {}
        axes_cfg = ablation_cfg.get("axes", {})

        for axis_name, axis_cfg in axes_cfg.items():
            if isinstance(axis_cfg, DictConfig):
                axis_cfg = dict(axis_cfg)

            axes[axis_name] = AblationAxis(
                name=axis_name,
                values=list(axis_cfg.get("values", [])),
                display_format=axis_cfg.get("display_format", "{}"),
                folder_format=axis_cfg.get("folder_format", "{value}"),
            )

        hierarchy = list(ablation_cfg.get("hierarchy", list(axes.keys())))

        return cls(axes=axes, hierarchy=hierarchy)

    def get_axis(self, name: str) -> AblationAxis:
        """Get an axis by name.

        Args:
            name: Axis name.

        Returns:
            AblationAxis instance.

        Raises:
            KeyError: If axis doesn't exist.
        """
        if name not in self.axes:
            raise KeyError(f"Unknown axis: {name}. Available: {list(self.axes.keys())}")
        return self.axes[name]

    def enumerate_coordinates(self) -> Iterator[ExperimentCoordinate]:
        """Enumerate all possible coordinates in the space.

        Yields:
            ExperimentCoordinate for each point in the parameter space.
        """
        # Get axis values in consistent order
        axis_names = ["prediction_type", "lp_norm", "self_cond_p"]
        axis_values = []

        for name in axis_names:
            if name in self.axes:
                axis_values.append(self.axes[name].values)
            else:
                # Use default values from ExperimentCoordinate
                defaults = {"prediction_type": ["epsilon", "velocity", "x0"],
                           "lp_norm": [1.5, 2.0, 2.5],
                           "self_cond_p": [0.5]}
                axis_values.append(defaults[name])

        for pred_type, lp_norm, self_cond_p in product(*axis_values):
            yield ExperimentCoordinate(
                prediction_type=pred_type,
                lp_norm=lp_norm,
                self_cond_p=self_cond_p,
            )

    def filter_coordinates(self, **filters: Any) -> Iterator[ExperimentCoordinate]:
        """Enumerate coordinates matching filter criteria.

        Args:
            **filters: Keyword arguments mapping axis names to required values.
                Values can be single values or lists of allowed values.

        Yields:
            ExperimentCoordinate instances matching all filter criteria.

        Examples:
            >>> space = AblationSpace.default()
            >>> list(space.filter_coordinates(prediction_type="x0"))
            [ExperimentCoordinate(prediction_type='x0', ...), ...]
        """
        for coord in self.enumerate_coordinates():
            if coord.matches_filter(**filters):
                yield coord

    def count_experiments(self) -> int:
        """Count total number of experiments in the space.

        Returns:
            Total number of coordinate combinations.
        """
        count = 1
        for axis in self.axes.values():
            count *= len(axis.values)
        return count

    def get_axis_values(self, axis_name: str) -> list[Any]:
        """Get all values for a specific axis.

        Args:
            axis_name: Name of the axis.

        Returns:
            List of values for that axis.
        """
        return list(self.get_axis(axis_name).values)

    def __repr__(self) -> str:
        """Return string representation."""
        axis_summary = ", ".join(
            f"{name}({len(ax.values)})" for name, ax in self.axes.items()
        )
        return f"AblationSpace({axis_summary}, total={self.count_experiments()})"
