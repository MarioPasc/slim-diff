"""Similarity metrics pipeline for evaluating synthetic image quality.

This module provides tools for computing KID, FID, and LPIPS metrics
between real and synthetic images, with statistical comparison and
publication-quality plotting.
"""

from .metrics.kid import InceptionFeatureExtractor, KIDComputer
from .metrics.fid import FIDComputer
from .metrics.lpips import LPIPSComputer
from .data.loaders import ICIPExperimentLoader

__all__ = [
    "InceptionFeatureExtractor",
    "KIDComputer",
    "FIDComputer",
    "LPIPSComputer",
    "ICIPExperimentLoader",
]
