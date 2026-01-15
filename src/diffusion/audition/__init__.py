"""Real vs Synthetic Image Audition System.

This module implements a classifier-based audition system to evaluate the
distinguishability between real and synthetic epilepsy lesion patches.
A classifier's performance (AUC, PR) indicates how easily real and synthetic
images can be differentiated.

Key components:
- Patch extraction: Extract lesion-centered patches from both datasets
- Classifier: Simple CNN for binary classification (real vs synthetic)
- Evaluation: AUC-ROC, PR-AUC, and per-zbin analysis

Usage:
    # Step 1: Extract patches
    python -m src.diffusion.audition.scripts.extract_patches --config path/to/config.yaml

    # Step 2: Train classifier
    python -m src.diffusion.audition.scripts.train_classifier --config path/to/config.yaml

    # Step 3: Evaluate
    python -m src.diffusion.audition.scripts.evaluate_audition --config path/to/config.yaml
"""

from .data.patch_extractor import PatchExtractor
from .models.classifier import AuditionClassifier

__all__ = ["PatchExtractor", "AuditionClassifier"]
