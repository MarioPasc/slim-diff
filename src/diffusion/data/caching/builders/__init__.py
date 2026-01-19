"""Dataset-specific slice cache builders.

Each builder implements the abstract methods of SliceCacheBuilder for
a specific dataset (epilepsy, BraTS-MEN, etc.).
"""

# Builders are auto-registered via @register_dataset decorator
# Import them here to trigger registration
from .epilepsy import EpilepsySliceCacheBuilder
from .brats_men import BraTSMenSliceCacheBuilder

__all__ = [
    "EpilepsySliceCacheBuilder",
    "BraTSMenSliceCacheBuilder",
]
