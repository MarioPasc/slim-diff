"""Similarity metrics pipeline for evaluating synthetic image quality.

This module provides tools for computing KID, FID, and LPIPS metrics
between real and synthetic images, with statistical comparison and
publication-quality plotting.

Note: Metric classes (KIDComputer, FIDComputer, LPIPSComputer) require
torchmetrics to be installed. Plotting functionality works without it.
"""

# Lazy imports to avoid requiring torchmetrics for plotting-only usage
def __getattr__(name):
    if name == "InceptionFeatureExtractor":
        from .metrics.kid import InceptionFeatureExtractor
        return InceptionFeatureExtractor
    elif name == "KIDComputer":
        from .metrics.kid import KIDComputer
        return KIDComputer
    elif name == "FIDComputer":
        from .metrics.fid import FIDComputer
        return FIDComputer
    elif name == "LPIPSComputer":
        from .metrics.lpips import LPIPSComputer
        return LPIPSComputer
    elif name == "ICIPExperimentLoader":
        from .data.loaders import ICIPExperimentLoader
        return ICIPExperimentLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "InceptionFeatureExtractor",
    "KIDComputer",
    "FIDComputer",
    "LPIPSComputer",
    "ICIPExperimentLoader",
]
