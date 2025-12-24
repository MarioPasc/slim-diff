"""Seeding utilities for reproducibility."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, enable deterministic CUDA operations.
            May impact performance but ensures reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For CUDA >= 10.2
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")


def get_generator(seed: int | None = None) -> torch.Generator:
    """Create a PyTorch generator with optional seed.

    Args:
        seed: Optional seed for the generator.

    Returns:
        Configured torch.Generator instance.
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator
