"""Feature-level probes for spectral, texture, and frequency analysis."""

from src.classification.diagnostics.feature_probes.frequency_bands import (
    band_analysis,
    bandpass_filter_2d,
    compute_band_power,
    run_band_analysis,
)
from src.classification.diagnostics.feature_probes.spectral import (
    azimuthal_average,
    compute_2d_psd,
    run_spectral_analysis,
    spectral_analysis,
    spectral_divergence,
    spectral_slope,
)
from src.classification.diagnostics.feature_probes.texture import (
    compute_glcm_features,
    compute_gradient_magnitude_stats,
    compute_lbp_histogram,
    run_texture_analysis,
    texture_comparison,
)

__all__ = [
    # Spectral
    "compute_2d_psd",
    "azimuthal_average",
    "spectral_slope",
    "spectral_divergence",
    "spectral_analysis",
    "run_spectral_analysis",
    # Texture
    "compute_glcm_features",
    "compute_lbp_histogram",
    "compute_gradient_magnitude_stats",
    "texture_comparison",
    "run_texture_analysis",
    # Frequency bands
    "bandpass_filter_2d",
    "compute_band_power",
    "band_analysis",
    "run_band_analysis",
]
