"""Cross-experiment comparison specifications for ablation studies.

Provides classes for defining and executing comparisons between
experiments that differ along specific axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Iterator

from omegaconf import DictConfig

from src.shared.ablation.ablation_space import AblationSpace
from src.shared.ablation.experiment_coords import ExperimentCoordinate


@dataclass
class ComparisonSpec:
    """Specification for a comparison analysis.

    Defines which axis to vary and which axes to hold fixed.

    Attributes:
        name: Descriptive name for this comparison.
        varying_axis: Name of the axis being compared (e.g., "prediction_type").
        fixed_axes: Dict of axis names to fixed values. Empty dict means
            generate comparisons for all combinations of fixed axis values.
    """

    name: str
    varying_axis: str
    fixed_axes: dict[str, Any] = field(default_factory=dict)

    def get_coordinates(
        self, space: AblationSpace
    ) -> list[ExperimentCoordinate]:
        """Get all coordinates involved in this comparison.

        Args:
            space: AblationSpace defining the parameter space.

        Returns:
            List of coordinates matching the fixed axes constraints.
        """
        coords = list(space.filter_coordinates(**self.fixed_axes))
        return sorted(coords, key=lambda c: (c.self_cond_p, c.prediction_type, c.lp_norm))

    def get_groups(
        self, space: AblationSpace
    ) -> dict[Any, list[ExperimentCoordinate]]:
        """Get coordinates grouped by the varying axis value.

        Args:
            space: AblationSpace defining the parameter space.

        Returns:
            Dict mapping varying axis values to lists of coordinates.
        """
        groups: dict[Any, list[ExperimentCoordinate]] = {}
        for coord in self.get_coordinates(space):
            key = getattr(coord, self.varying_axis)
            if key not in groups:
                groups[key] = []
            groups[key].append(coord)
        return groups

    def get_pairs(
        self, space: AblationSpace
    ) -> list[tuple[ExperimentCoordinate, ExperimentCoordinate]]:
        """Get all pairs of coordinates that differ only in the varying axis.

        Args:
            space: AblationSpace defining the parameter space.

        Returns:
            List of coordinate pairs for pairwise comparison.
        """
        pairs = []
        coords = self.get_coordinates(space)

        for coord_a, coord_b in combinations(coords, 2):
            # Check that they differ only in the varying axis
            diff_axes = _get_differing_axes(coord_a, coord_b)
            if diff_axes == [self.varying_axis]:
                pairs.append((coord_a, coord_b))

        return pairs


def _get_differing_axes(
    coord_a: ExperimentCoordinate, coord_b: ExperimentCoordinate
) -> list[str]:
    """Get list of axis names that differ between two coordinates.

    Args:
        coord_a: First coordinate.
        coord_b: Second coordinate.

    Returns:
        List of axis names with different values.
    """
    diff = []
    dict_a = coord_a.to_dict()
    dict_b = coord_b.to_dict()
    for key in dict_a:
        if dict_a[key] != dict_b[key]:
            diff.append(key)
    return diff


def differ_by_one_axis(
    coord_a: ExperimentCoordinate, coord_b: ExperimentCoordinate
) -> str | None:
    """Check if two coordinates differ by exactly one axis.

    Args:
        coord_a: First coordinate.
        coord_b: Second coordinate.

    Returns:
        Name of the single differing axis, or None if 0 or >1 differ.
    """
    diff = _get_differing_axes(coord_a, coord_b)
    return diff[0] if len(diff) == 1 else None


def generate_all_1d_comparisons(space: AblationSpace) -> list[ComparisonSpec]:
    """Generate comparison specs for all single-axis comparisons.

    For each axis, generates a comparison that varies that axis while
    fixing all others at each combination of values.

    Args:
        space: AblationSpace defining the parameter space.

    Returns:
        List of ComparisonSpec instances.
    """
    comparisons = []
    axis_names = list(space.axes.keys())

    for varying_axis in axis_names:
        # Generate spec with no fixed axes - will compare across all combinations
        comparisons.append(ComparisonSpec(
            name=f"by_{varying_axis}",
            varying_axis=varying_axis,
            fixed_axes={},
        ))

    return comparisons


def generate_pairwise_deltas(
    coordinates: list[ExperimentCoordinate],
) -> Iterator[tuple[ExperimentCoordinate, ExperimentCoordinate, str]]:
    """Generate all pairs of coordinates that differ by exactly one axis.

    Args:
        coordinates: List of experiment coordinates.

    Yields:
        Tuples of (coord_a, coord_b, varying_axis) for each valid pair.
    """
    for coord_a, coord_b in combinations(coordinates, 2):
        diff_axis = differ_by_one_axis(coord_a, coord_b)
        if diff_axis is not None:
            yield coord_a, coord_b, diff_axis


@dataclass
class ComparisonResult:
    """Result of a comparison analysis.

    Stores coordinates involved and computed metrics/deltas.

    Attributes:
        spec: The ComparisonSpec that generated this result.
        coordinates: List of coordinates in the comparison.
        metrics: Dict of metric names to per-coordinate values.
        deltas: Dict of delta results for pairwise comparisons.
    """

    spec: ComparisonSpec
    coordinates: list[ExperimentCoordinate]
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    deltas: list[dict[str, Any]] = field(default_factory=list)

    def add_metric(self, metric_name: str, values: dict[str, float]) -> None:
        """Add metric values for coordinates.

        Args:
            metric_name: Name of the metric.
            values: Dict mapping display names to metric values.
        """
        self.metrics[metric_name] = values

    def add_delta(
        self,
        coord_a: ExperimentCoordinate,
        coord_b: ExperimentCoordinate,
        varying_axis: str,
        metric_deltas: dict[str, float],
    ) -> None:
        """Add a pairwise delta result.

        Args:
            coord_a: First coordinate.
            coord_b: Second coordinate.
            varying_axis: Name of the axis that differs.
            metric_deltas: Dict mapping metric names to delta values (b - a).
        """
        self.deltas.append({
            "exp_a": coord_a.to_display_name(),
            "exp_b": coord_b.to_display_name(),
            "varying_axis": varying_axis,
            "value_a": getattr(coord_a, varying_axis),
            "value_b": getattr(coord_b, varying_axis),
            "fixed_axes": {
                k: v for k, v in coord_a.to_dict().items() if k != varying_axis
            },
            "deltas": metric_deltas,
        })


def comparison_specs_from_config(cfg: DictConfig | dict) -> list[ComparisonSpec]:
    """Load comparison specifications from configuration.

    Expected config structure:
    ```yaml
    comparison:
      analyses:
        - name: "by_prediction_type"
          varying_axis: "prediction_type"
          fixed_axes: {}
        - name: "x0_self_cond_sweep"
          varying_axis: "self_cond_p"
          fixed_axes:
            prediction_type: "x0"
            lp_norm: 1.5
    ```

    Args:
        cfg: Configuration with 'comparison.analyses' section.

    Returns:
        List of ComparisonSpec instances.
    """
    if isinstance(cfg, DictConfig):
        cfg = dict(cfg)

    comparison_cfg = cfg.get("comparison", {})
    analyses = comparison_cfg.get("analyses", [])

    specs = []
    for analysis in analyses:
        if isinstance(analysis, DictConfig):
            analysis = dict(analysis)

        specs.append(ComparisonSpec(
            name=analysis.get("name", "unnamed"),
            varying_axis=analysis["varying_axis"],
            fixed_axes=dict(analysis.get("fixed_axes", {})),
        ))

    return specs
