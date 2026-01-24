"""Shared utilities for diagnostic analyses."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


# ──────────────────────────────────────────────────────────────────────────────
# Model loading utilities (shared by all XAI analyses)
# ──────────────────────────────────────────────────────────────────────────────


def load_model_from_checkpoint(
    ckpt_path: Path,
    in_channels: int,
    device: str,
) -> nn.Module:
    """Load a trained classifier from a checkpoint.

    Creates a minimal compatible config for ClassificationLightningModule
    and loads the model weights from the checkpoint.

    Args:
        ckpt_path: Path to the .ckpt file.
        in_channels: Number of input channels for the model.
        device: Target device string.

    Returns:
        The loaded model in eval mode on the specified device.
    """
    from omegaconf import OmegaConf

    from src.classification.training.lit_module import ClassificationLightningModule

    train_cfg = OmegaConf.create({
        "training": {
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "max_epochs": 50,
            "scheduler": {"type": "reduce_on_plateau", "factor": 0.5, "patience": 5, "min_lr": 1e-6},
            "early_stopping": {"monitor": "val/auc", "patience": 10, "min_delta": 0.001},
        },
        "model": {"config_path": "models/simple_cnn.yaml"},
    })

    lit_module = ClassificationLightningModule.load_from_checkpoint(
        str(ckpt_path),
        cfg=train_cfg,
        in_channels=in_channels,
        map_location=device,
    )
    lit_module.eval()
    model = lit_module.model.to(device)
    model.eval()
    return model


def discover_checkpoint(
    checkpoints_base_dir: Path,
    experiment_name: str,
    fold_idx: int,
) -> Path | None:
    """Discover the best checkpoint file for a given experiment and fold.

    Searches all subdirectories under the experiment's checkpoint directory
    for fold checkpoint files. Handles versioned checkpoints (e.g., -v1, -v2)
    by preferring the base name without version suffix.

    Args:
        checkpoints_base_dir: Base directory containing experiment subdirs.
        experiment_name: Experiment name (e.g., 'epsilon_lp_1.5').
        fold_idx: Fold index to find checkpoint for.

    Returns:
        Path to the best checkpoint file, or None if not found.
    """
    exp_dir = Path(checkpoints_base_dir) / experiment_name
    if not exp_dir.exists():
        return None

    base_name = f"fold{fold_idx}_best.ckpt"
    candidates: list[Path] = []

    for subdir in sorted(exp_dir.rglob("*.ckpt")):
        # Skip non-matching filenames
        pass

    # Walk all subdirectories
    for subdir in sorted(exp_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Check recursively within subdirs
        for ckpt in sorted(subdir.rglob(base_name)):
            candidates.append(ckpt)
        if not candidates or candidates[-1].parent != subdir:
            # Look for versioned checkpoints
            for ckpt in sorted(subdir.rglob(f"fold{fold_idx}_best-v*.ckpt")):
                candidates.append(ckpt)

    if not candidates:
        return None

    # Prefer non-versioned, then latest by modification time
    non_versioned = [c for c in candidates if "-v" not in c.name]
    if non_versioned:
        return max(non_versioned, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def determine_in_channels(mode: str) -> int:
    """Map input mode string to channel count.

    Args:
        mode: One of 'joint', 'image_only', 'mask_only'.

    Returns:
        Number of input channels (2 for joint, 1 otherwise).
    """
    if mode == "joint":
        return 2
    elif mode in ("image_only", "mask_only"):
        return 1
    else:
        raise ValueError(f"Unknown input mode: {mode}")


def select_patches_by_mode(
    patches: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Select channels from patches based on input mode.

    Args:
        patches: Array of shape (N, 2, H, W).
        mode: One of 'joint', 'image_only', 'mask_only'.

    Returns:
        Array of shape (N, C, H, W) where C depends on mode.
    """
    if mode == "joint":
        return patches
    elif mode == "image_only":
        return patches[:, 0:1, :, :]
    elif mode == "mask_only":
        return patches[:, 1:2, :, :]
    else:
        raise ValueError(f"Unknown input mode: {mode}")
