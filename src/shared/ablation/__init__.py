"""Multi-axis ablation study experiment management.

This module provides a flexible system for managing experiments across
multiple ablation axes (e.g., prediction_type, lp_norm, self_cond_p).

Key classes:
- ExperimentCoordinate: Immutable coordinate in the parameter space
- AblationSpace: Definition of the full N-dimensional parameter space
- ExperimentDiscoverer: Filesystem discovery of experiments
- ComparisonSpec: Specification for cross-experiment comparisons
"""

from src.shared.ablation.experiment_coords import ExperimentCoordinate
from src.shared.ablation.ablation_space import AblationAxis, AblationSpace
from src.shared.ablation.discovery import ExperimentDiscoverer
from src.shared.ablation.comparison import ComparisonSpec

__all__ = [
    "ExperimentCoordinate",
    "AblationAxis",
    "AblationSpace",
    "ExperimentDiscoverer",
    "ComparisonSpec",
]
