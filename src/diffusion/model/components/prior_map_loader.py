"""Flexible prior map loader for anatomical conditioning.

Supports loading anatomical prior maps with arbitrary number of channels,
enabling tissue-aware conditioning (e.g., WM, GM, CSF, ventricles).

Supports two NPZ formats:
1. Legacy binary: `bin_{z_bin}` = (H, W) uint8 boolean array
2. Multi-channel: `prior_{z_bin}` = (C, H, W) float32 probability maps
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PriorMapLoader:
    """Loader for anatomical prior maps with flexible channel support.

    This loader handles both legacy binary priors (single channel brain/background)
    and multi-channel tissue probability maps. The number of channels is inferred
    from the channel_mapping configuration.

    Args:
        cache_dir: Directory containing the priors NPZ file.
        filename: Name of the NPZ file (e.g., "zbin_priors_brain_roi.npz").
        n_bins: Number of z-bins expected in the priors file.
        channel_mapping: Dictionary mapping channel indices to tissue names.
            Example: {0: "background", 1: "brain"} for binary,
            or {0: "background", 1: "csf", 2: "gm", 3: "wm", 4: "ventricles"}
            for tissue maps.
        fallback_to_binary: If True and multi-channel priors not found,
            attempt to load legacy binary priors and expand to expected channels.
        normalize_range: Tuple (min, max) for normalizing prior values.
            Default is (-1.0, 1.0) to match model input range.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        filename: str,
        n_bins: int,
        channel_mapping: Dict[int, str],
        fallback_to_binary: bool = True,
        normalize_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self.cache_dir = Path(cache_dir)
        self.filename = filename
        self.n_bins = n_bins
        self.channel_mapping = channel_mapping
        self.n_channels = len(channel_mapping)
        self.fallback_to_binary = fallback_to_binary
        self.normalize_range = normalize_range

        # Load priors from file
        self._priors: Dict[int, np.ndarray] = {}
        self._format: str = "unknown"
        self._load_priors()

    def _load_priors(self) -> None:
        """Load priors from NPZ file."""
        filepath = self.cache_dir / self.filename
        if not filepath.exists():
            raise FileNotFoundError(f"Priors file not found: {filepath}")

        data = np.load(filepath)

        # Detect format: look for multi-channel keys first
        if "prior_0" in data:
            self._format = "multichannel"
            self._load_multichannel_priors(data)
        elif "bin_0" in data:
            self._format = "binary"
            self._load_binary_priors(data)
        else:
            # Try generic naming convention
            available_keys = list(data.keys())
            raise ValueError(
                f"Cannot detect priors format. Available keys: {available_keys}. "
                f"Expected 'prior_{{i}}' or 'bin_{{i}}' naming."
            )

        logger.info(
            f"Loaded {len(self._priors)} priors in {self._format} format "
            f"with {self.n_channels} channels from {filepath}"
        )

    def _load_multichannel_priors(self, data: np.lib.npyio.NpzFile) -> None:
        """Load multi-channel priors (prior_{z_bin} format)."""
        for z_bin in range(self.n_bins):
            key = f"prior_{z_bin}"
            if key not in data:
                logger.warning(f"Missing prior for z_bin {z_bin}")
                continue

            prior = data[key].astype(np.float32)

            # Validate shape
            if prior.ndim == 2:
                # Single channel, add channel dimension
                prior = prior[np.newaxis, ...]
            elif prior.ndim != 3:
                raise ValueError(
                    f"Invalid prior shape for {key}: {prior.shape}. "
                    f"Expected (C, H, W) or (H, W)."
                )

            # Check channel count matches
            if prior.shape[0] != self.n_channels:
                logger.warning(
                    f"Prior {key} has {prior.shape[0]} channels, "
                    f"expected {self.n_channels}. Attempting to adapt."
                )
                prior = self._adapt_channels(prior)

            self._priors[z_bin] = prior

    def _load_binary_priors(self, data: np.lib.npyio.NpzFile) -> None:
        """Load legacy binary priors (bin_{z_bin} format)."""
        for z_bin in range(self.n_bins):
            key = f"bin_{z_bin}"
            if key not in data:
                logger.warning(f"Missing prior for z_bin {z_bin}")
                continue

            # Load binary mask: (H, W) boolean/uint8
            binary_mask = data[key].astype(np.float32)
            if binary_mask.ndim != 2:
                raise ValueError(f"Invalid binary prior shape: {binary_mask.shape}")

            # Convert to multi-channel format
            prior = self._binary_to_multichannel(binary_mask)
            self._priors[z_bin] = prior

    def _binary_to_multichannel(self, binary_mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to multi-channel format.

        For a simple 2-channel case (background, brain):
        - Channel 0 (background): inverted mask (1 - mask)
        - Channel 1 (brain): the mask itself

        For more channels, we repeat the brain channel and set background.

        Args:
            binary_mask: Binary mask of shape (H, W) with values in [0, 1].

        Returns:
            Multi-channel array of shape (C, H, W).
        """
        H, W = binary_mask.shape

        if self.n_channels == 1:
            # Single channel: just the brain mask
            return binary_mask[np.newaxis, ...]

        elif self.n_channels == 2:
            # Two channels: [background, brain]
            prior = np.zeros((2, H, W), dtype=np.float32)
            prior[0] = 1.0 - binary_mask  # background
            prior[1] = binary_mask  # brain
            return prior

        else:
            # More channels: background + replicated brain for all tissue classes
            # This is a fallback; real tissue maps should be used for accuracy
            prior = np.zeros((self.n_channels, H, W), dtype=np.float32)
            prior[0] = 1.0 - binary_mask  # background

            # Distribute brain equally across remaining channels
            n_tissue_channels = self.n_channels - 1
            for i in range(1, self.n_channels):
                prior[i] = binary_mask / n_tissue_channels

            logger.warning(
                f"Using fallback: distributing binary brain mask across "
                f"{n_tissue_channels} tissue channels. Consider providing "
                f"proper tissue priors for better results."
            )
            return prior

    def _adapt_channels(self, prior: np.ndarray) -> np.ndarray:
        """Adapt prior to expected number of channels.

        Args:
            prior: Prior array of shape (C_in, H, W).

        Returns:
            Adapted prior of shape (C_out, H, W) where C_out = self.n_channels.
        """
        C_in, H, W = prior.shape

        if C_in == self.n_channels:
            return prior

        if C_in > self.n_channels:
            # Truncate extra channels
            return prior[: self.n_channels]

        # C_in < self.n_channels: pad with zeros
        adapted = np.zeros((self.n_channels, H, W), dtype=np.float32)
        adapted[:C_in] = prior
        return adapted

    def get_prior(self, z_bin: int) -> np.ndarray:
        """Get prior for a single z-bin.

        Args:
            z_bin: Z-bin index.

        Returns:
            Prior array of shape (C, H, W).
        """
        if z_bin not in self._priors:
            raise KeyError(f"No prior for z_bin {z_bin}")
        return self._priors[z_bin]

    def get_tensor(
        self,
        z_bins: Sequence[int],
        device: Union[str, torch.device] = "cuda",
        normalize: bool = True,
    ) -> torch.Tensor:
        """Get prior tensors for a batch of z-bins.

        Args:
            z_bins: Sequence of z-bin indices.
            device: Device to place tensors on.
            normalize: If True, normalize priors to normalize_range.

        Returns:
            Tensor of shape (B, C, H, W) where B = len(z_bins).
        """
        priors = []
        for z_bin in z_bins:
            prior = self.get_prior(z_bin)
            priors.append(prior)

        # Stack into batch
        batch = np.stack(priors, axis=0)  # (B, C, H, W)

        # Convert to tensor
        tensor = torch.from_numpy(batch).to(device=device)

        # Normalize if requested
        if normalize:
            # Prior values are typically in [0, 1]
            # Normalize to [min, max] (default [-1, 1])
            min_val, max_val = self.normalize_range
            tensor = tensor * (max_val - min_val) + min_val

        return tensor

    @property
    def channel_names(self) -> List[str]:
        """Get ordered list of channel names."""
        return [self.channel_mapping[i] for i in sorted(self.channel_mapping.keys())]

    @property
    def format(self) -> str:
        """Get detected format of loaded priors."""
        return self._format

    @property
    def spatial_shape(self) -> Optional[tuple[int, int]]:
        """Get spatial shape (H, W) of priors, if available."""
        if self._priors:
            first_prior = next(iter(self._priors.values()))
            return first_prior.shape[1], first_prior.shape[2]
        return None

    def __len__(self) -> int:
        """Return number of loaded priors."""
        return len(self._priors)

    def __contains__(self, z_bin: int) -> bool:
        """Check if z_bin has a prior."""
        return z_bin in self._priors


def load_prior_map_loader(
    cache_dir: Union[str, Path],
    filename: str,
    n_bins: int,
    channel_mapping: Optional[Dict[int, str]] = None,
    fallback_to_binary: bool = True,
    normalize_range: tuple[float, float] = (-1.0, 1.0),
) -> PriorMapLoader:
    """Factory function to create a PriorMapLoader.

    Args:
        cache_dir: Directory containing priors NPZ file.
        filename: Name of the NPZ file.
        n_bins: Number of z-bins.
        channel_mapping: Channel to tissue name mapping.
            Default: {0: "brain"} (single channel binary).
        fallback_to_binary: Whether to fall back to binary priors.
        normalize_range: Range for normalization.

    Returns:
        Configured PriorMapLoader instance.
    """
    if channel_mapping is None:
        channel_mapping = {0: "brain"}

    return PriorMapLoader(
        cache_dir=cache_dir,
        filename=filename,
        n_bins=n_bins,
        channel_mapping=channel_mapping,
        fallback_to_binary=fallback_to_binary,
        normalize_range=normalize_range,
    )
