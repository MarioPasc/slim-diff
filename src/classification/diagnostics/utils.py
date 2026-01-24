"""Shared utilities for diagnostic analyses."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_patches(
    patches_dir: Path,
    experiment_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-extracted patches for an experiment.

    Args:
        patches_dir: Base directory containing per-experiment patch subdirs.
        experiment_name: Name of the experiment (e.g., 'velocity_lp_1.5').

    Returns:
        Tuple of (real_patches, synth_patches, real_zbins, synth_zbins).
        Patches are float32 with shape (N, 2, H, W).
    """
    exp_dir = Path(patches_dir) / experiment_name
    real_path = exp_dir / "real_patches.npz"
    synth_path = exp_dir / "synthetic_patches.npz"

    if not real_path.exists():
        raise FileNotFoundError(f"Real patches not found: {real_path}")
    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic patches not found: {synth_path}")

    real_data = np.load(real_path, allow_pickle=True)
    synth_data = np.load(synth_path, allow_pickle=True)

    real_patches = real_data["patches"].astype(np.float32)
    synth_patches = synth_data["patches"].astype(np.float32)
    real_zbins = real_data["z_bins"].astype(np.int32)
    synth_zbins = synth_data["z_bins"].astype(np.int32)

    logger.info(
        f"Loaded patches for '{experiment_name}': "
        f"real={real_patches.shape}, synth={synth_patches.shape}"
    )
    return real_patches, synth_patches, real_zbins, synth_zbins


def load_full_replicas(
    replicas_base_dir: Path,
    experiment_name: str,
    replica_ids: list[int],
    lesion_only: bool = True,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load full 160x160 images from replica NPZ files.

    Args:
        replicas_base_dir: Base directory containing per-experiment run dirs.
        experiment_name: Name of the experiment.
        replica_ids: List of replica indices to load.
        lesion_only: If True, only load samples with lesion_present=1.
        max_samples: Maximum number of samples to load.

    Returns:
        Tuple of (images, masks, zbins, lesion_present).
        Images/masks are float32, shape (N, 160, 160).
    """
    exp_dir = Path(replicas_base_dir) / experiment_name / "replicas"
    all_images, all_masks, all_zbins, all_lesion = [], [], [], []

    for rid in replica_ids:
        replica_path = exp_dir / f"replica_{rid:03d}.npz"
        if not replica_path.exists():
            logger.warning(f"Replica not found: {replica_path}")
            continue

        data = np.load(replica_path)
        images = data["images"].astype(np.float32)
        masks = data["masks"].astype(np.float32)
        zbins = data["zbin"].astype(np.int32)
        lesion_present = data["lesion_present"].astype(np.int32)

        if lesion_only:
            lesion_mask = lesion_present == 1
            images = images[lesion_mask]
            masks = masks[lesion_mask]
            zbins = zbins[lesion_mask]
            lesion_present = lesion_present[lesion_mask]

        all_images.append(images)
        all_masks.append(masks)
        all_zbins.append(zbins)
        all_lesion.append(lesion_present)

    images = np.concatenate(all_images, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    zbins = np.concatenate(all_zbins, axis=0)
    lesion_present = np.concatenate(all_lesion, axis=0)

    if max_samples is not None and len(images) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(images), max_samples, replace=False)
        images, masks, zbins, lesion_present = (
            images[idx], masks[idx], zbins[idx], lesion_present[idx]
        )

    logger.info(
        f"Loaded full replicas for '{experiment_name}': "
        f"{len(images)} samples from {len(replica_ids)} replicas"
    )
    return images, masks, zbins, lesion_present


def load_real_slices(
    cache_dir: Path,
    lesion_only: bool = True,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real slices from the slice cache.

    Args:
        cache_dir: Path to the slice cache directory.
        lesion_only: If True, only load slices with lesions.
        max_samples: Maximum number of slices to load.

    Returns:
        Tuple of (images, masks, zbins). All float32.
    """
    cache_path = Path(cache_dir)
    csv_paths = [cache_path / "train.csv", cache_path / "val.csv"]

    all_images, all_masks, all_zbins = [], [], []

    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)

        if lesion_only and "lesion_present" in df.columns:
            df = df[df["lesion_present"] == 1]

        for _, row in df.iterrows():
            slice_path = cache_path / row["filepath"]
            if not slice_path.exists():
                continue
            data = np.load(slice_path)
            all_images.append(data["image"].astype(np.float32))
            all_masks.append(data["mask"].astype(np.float32))
            if "z_bin" in row:
                all_zbins.append(int(row["z_bin"]))
            elif "zbin" in row:
                all_zbins.append(int(row["zbin"]))
            else:
                all_zbins.append(0)

    images = np.stack(all_images, axis=0)
    masks = np.stack(all_masks, axis=0)
    zbins = np.array(all_zbins, dtype=np.int32)

    if max_samples is not None and len(images) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(images), max_samples, replace=False)
        images, masks, zbins = images[idx], masks[idx], zbins[idx]

    logger.info(f"Loaded {len(images)} real slices from cache")
    return images, masks, zbins


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_result_json(result: dict[str, Any], output_path: Path) -> None:
    """Save analysis result to JSON with numpy serialization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, cls=NumpyEncoder, indent=2)
    logger.info(f"Saved results to {output_path}")


def ensure_output_dir(base_dir: Path, experiment_name: str, analysis_name: str) -> Path:
    """Create and return the output directory for an analysis.

    Structure: {base_dir}/{experiment_name}/{analysis_name}/
    """
    out = Path(base_dir) / experiment_name / analysis_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame to CSV for inter-experiment analysis.

    Args:
        df: DataFrame to save.
        output_path: Full path for the CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved CSV to {output_path}")


def save_figure(fig: Any, output_dir: Path, name: str, formats: list[str] | None = None, dpi: int = 300) -> None:
    """Save a matplotlib figure in multiple formats.

    Args:
        fig: Matplotlib figure.
        output_dir: Directory to save to.
        name: Base filename (without extension).
        formats: List of formats (default: ['png', 'pdf']).
        dpi: Resolution for raster formats.
    """
    if formats is None:
        formats = ["png", "pdf"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved figure '{name}' to {output_dir}")
