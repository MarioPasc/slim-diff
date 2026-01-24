"""Explainable AI tools (GradCAM, feature probes)."""

from src.classification.diagnostics.xai.aggregation import (
    AggregatedHeatmaps,
    aggregate_heatmaps,
    compute_attention_difference,
    plot_gradcam_results,
    radial_attention_profile,
)
from src.classification.diagnostics.xai.gradcam import (
    GradCAM,
    GradCAMResult,
    run_gradcam_analysis,
)

__all__ = [
    "AggregatedHeatmaps",
    "GradCAM",
    "GradCAMResult",
    "aggregate_heatmaps",
    "compute_attention_difference",
    "plot_gradcam_results",
    "radial_attention_profile",
    "run_gradcam_analysis",
]
