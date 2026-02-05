"""Loss functions for JS-DDPM training."""

from src.diffusion.losses.diffusion_losses import DiffusionLoss
from src.diffusion.losses.focal_frequency_loss import FocalFrequencyLoss
from src.diffusion.losses.perceptual_loss import LPIPSLoss, create_lpips_loss
from src.diffusion.losses.uncertainty import (
    GroupUncertaintyWeightedLoss,
    UncertaintyWeightedLoss,
)

__all__ = [
    "DiffusionLoss",
    "FocalFrequencyLoss",
    "GroupUncertaintyWeightedLoss",
    "LPIPSLoss",
    "UncertaintyWeightedLoss",
    "create_lpips_loss",
]
