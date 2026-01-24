"""Explainable AI tools (GradCAM, channel decomposition, spectral, feature space, IG)."""

from src.classification.diagnostics.xai.aggregation import (
    AggregatedHeatmaps,
    aggregate_heatmaps,
    compute_attention_difference,
    plot_gradcam_results,
    radial_attention_profile,
)
from src.classification.diagnostics.xai.channel_decomposition import (
    run_channel_decomposition,
)
from src.classification.diagnostics.xai.feature_space import (
    run_feature_space_analysis,
)
from src.classification.diagnostics.xai.gradcam import (
    GradCAM,
    GradCAMResult,
    run_gradcam_analysis,
)
from src.classification.diagnostics.xai.integrated_gradients import (
    IntegratedGradients,
    run_integrated_gradients,
)
from src.classification.diagnostics.xai.spectral_attribution import (
    run_spectral_attribution,
)

__all__ = [
    "AggregatedHeatmaps",
    "GradCAM",
    "GradCAMResult",
    "IntegratedGradients",
    "aggregate_heatmaps",
    "compute_attention_difference",
    "plot_gradcam_results",
    "radial_attention_profile",
    "run_channel_decomposition",
    "run_feature_space_analysis",
    "run_gradcam_analysis",
    "run_integrated_gradients",
    "run_spectral_attribution",
]
