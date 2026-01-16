"""Slice-level dataset for JS-DDPM training.

Provides a PyTorch Dataset that reads cached slices from .npz files
and returns samples for diffusion model training.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.utils.data import Dataset, WeightedRandomSampler

from src.diffusion.utils.io import load_sample_npz

logger = logging.getLogger(__name__)


class SliceDataset(Dataset):
    """Dataset for loading cached 2D slices.

    Each sample contains:
    - image: (1, H, W) tensor in [-1, 1]
    - mask: (1, H, W) tensor in {-1, +1}
    - token: int64 conditioning token
    - metadata: dict with subject_id, z_index, etc.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        split: str = "train",
        transform: Any | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            cache_dir: Path to slice cache directory.
            split: One of "train", "val", "test".
            transform: Optional additional transforms to apply.
        """
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.transform = transform

        # Load index CSV
        csv_path = self.cache_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Index CSV not found: {csv_path}. "
                "Run cache builder first."
            )

        self.samples = self._load_index(csv_path)
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_index(self, csv_path: Path) -> list[dict[str, Any]]:
        """Load sample metadata from CSV index.

        Args:
            csv_path: Path to CSV file.

        Returns:
            List of sample metadata dictionaries.
        """
        samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert types
                sample = {
                    "filepath": self.cache_dir / row["filepath"],
                    "subject_id": row["subject_id"],
                    "z_index": int(row["z_index"]),
                    "z_bin": int(row["z_bin"]),
                    "pathology_class": int(row["pathology_class"]),
                    "token": int(row["token"]),
                    "source": row["source"],
                    "has_lesion": row["has_lesion"].lower() == "true",
                }
                samples.append(sample)
        return samples

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with image, mask, token, and metadata.
        """
        sample_meta = self.samples[idx]

        # Load from npz
        data = load_sample_npz(sample_meta["filepath"])

        # Ensure channel dimension and convert to tensor
        image = data["image"]
        mask = data["mask"]

        if image.ndim == 2:
            image = image[np.newaxis]
        if mask.ndim == 2:
            mask = mask[np.newaxis]

        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))

        # Token as int64
        token = torch.tensor(sample_meta["token"], dtype=torch.int64)

        result = {
            "image": image,  # (1, H, W)
            "mask": mask,    # (1, H, W)
            "token": token,  # scalar
            "subject_id": sample_meta["subject_id"],
            "z_index": sample_meta["z_index"],
            "z_bin": sample_meta["z_bin"],
            "pathology_class": sample_meta["pathology_class"],
            "source": sample_meta["source"],
            "has_lesion": sample_meta["has_lesion"],
        }

        if self.transform is not None:
            result = self.transform(result)

        return result


def get_weighted_sampler(
    dataset: SliceDataset,
    mode: str = "balance",
    lesion_weight: float = 5.0,
) -> WeightedRandomSampler:
    """Create a weighted random sampler for class balancing.

    Args:
        dataset: SliceDataset instance.
        mode: Oversampling mode. Options:
            - "balance": Automatically compute weights for 50/50 lesion/non-lesion
              sampling. This ensures the model sees both classes at the same rate.
            - "weight": Use the provided lesion_weight as a fixed multiplier.
        lesion_weight: Weight multiplier for lesion slices (only used when mode="weight").

    Returns:
        WeightedRandomSampler for the dataloader.
    """
    # Count lesion and non-lesion samples
    n_lesion = sum(1 for s in dataset.samples if s["has_lesion"])
    n_non_lesion = len(dataset.samples) - n_lesion

    if n_lesion == 0:
        logger.warning("No lesion samples found. Weighted sampling will have no effect.")
        effective_lesion_weight = 1.0
    elif n_non_lesion == 0:
        logger.warning("No non-lesion samples found. Weighted sampling will have no effect.")
        effective_lesion_weight = 1.0
    elif mode == "balance":
        # Compute weight to achieve 50/50 sampling
        # For equal probability: w_lesion * n_lesion = w_non_lesion * n_non_lesion
        # With w_non_lesion = 1.0: w_lesion = n_non_lesion / n_lesion
        effective_lesion_weight = n_non_lesion / n_lesion
        logger.info(
            f"Lesion oversampling mode='balance': {n_lesion} lesion, "
            f"{n_non_lesion} non-lesion samples. "
            f"Computed lesion weight: {effective_lesion_weight:.2f}x for 50/50 sampling."
        )
    elif mode == "weight":
        effective_lesion_weight = lesion_weight
        logger.info(
            f"Lesion oversampling mode='weight': Using fixed weight={lesion_weight}x. "
            f"({n_lesion} lesion, {n_non_lesion} non-lesion samples)"
        )
    else:
        raise ValueError(f"Unknown lesion_oversampling mode: {mode}. Use 'balance' or 'weight'.")

    # Compute weights based on lesion presence
    weights = []
    for sample in dataset.samples:
        if sample["has_lesion"]:
            weights.append(effective_lesion_weight)
        else:
            weights.append(1.0)

    weights = torch.tensor(weights, dtype=torch.float64)

    # Normalize weights (optional, but helps with interpretation)
    weights = weights / weights.sum() * len(weights)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for batching.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary.
    """
    # Stack tensors
    images = torch.stack([s["image"] for s in batch])
    masks = torch.stack([s["mask"] for s in batch])
    tokens = torch.stack([s["token"] for s in batch])

    # Collect metadata
    metadata = {
        "subject_id": [s["subject_id"] for s in batch],
        "z_index": [s["z_index"] for s in batch],
        "z_bin": [s["z_bin"] for s in batch],
        "pathology_class": [s["pathology_class"] for s in batch],
        "source": [s["source"] for s in batch],
        "has_lesion": [s["has_lesion"] for s in batch],
    }

    return {
        "image": images,  # (B, 1, H, W)
        "mask": masks,    # (B, 1, H, W)
        "token": tokens,  # (B,)
        "metadata": metadata,
    }


def create_dataloader(
    cfg: DictConfig,
    split: str = "train",
    shuffle: bool | None = None,
    use_weighted_sampler: bool | None = None,
) -> torch.utils.data.DataLoader:
    """Create a dataloader for the specified split.

    Args:
        cfg: Configuration object.
        split: One of "train", "val", "test".
        shuffle: Whether to shuffle (default: True for train).
        use_weighted_sampler: Whether to use lesion oversampling
            (default: from config for train, False otherwise).

    Returns:
        Configured DataLoader instance.
    """
    # Create dataset
    dataset = SliceDataset(
        cache_dir=cfg.data.cache_dir,
        split=split,
    )

    # Determine shuffle and sampler settings
    if shuffle is None:
        shuffle = split == "train"

    if use_weighted_sampler is None:
        use_weighted_sampler = (
            split == "train" and cfg.data.lesion_oversampling.enabled
        )

    # Create sampler if needed
    sampler = None
    if use_weighted_sampler:
        # Get mode with fallback for backward compatibility
        mode = cfg.data.lesion_oversampling.get("mode", "weight")
        sampler = get_weighted_sampler(
            dataset,
            mode=mode,
            lesion_weight=cfg.data.lesion_oversampling.weight,
        )
        shuffle = False  # Sampler handles randomization

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )

    logger.info(
        f"Created {split} dataloader: "
        f"{len(dataset)} samples, "
        f"batch_size={cfg.training.batch_size}, "
        f"weighted_sampler={use_weighted_sampler}"
    )

    return dataloader
