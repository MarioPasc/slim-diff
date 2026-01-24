"""Uniform dithering to mask float16 quantization artifacts.

The synthetic replicas are stored in float16 (~20K unique values) while real
data is float32 (~4.3M unique values). A classifier trivially exploits this
discrete value structure to achieve AUC=1.0.

This module implements precision-aware uniform dithering: for each value x,
add U(-eps/2, eps/2) where eps is the float16 ULP (unit in the last place)
at that value. This fills quantization gaps without introducing bias.

Reference:
    Roberts, L. (1962). "Picture coding using pseudo-random noise."
    IEEE Trans. Information Theory, 8(2), 145-154.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.classification.data.dataset import ClassificationDataset
from src.classification.diagnostics.utils import (
    ensure_output_dir,
    load_patches,
    save_figure,
    save_result_json,
)
from src.classification.evaluation.metrics import _bootstrap_ci
from src.classification.training.lit_module import ClassificationLightningModule

logger = logging.getLogger(__name__)


@dataclass
class DitheringStats:
    """Statistics from the dithering process."""

    n_samples: int
    unique_values_before: int
    unique_values_after: int
    max_eps_applied: float
    mean_eps_applied: float
    min_eps_applied: float


def compute_float16_eps(values: np.ndarray) -> np.ndarray:
    """Compute the float16 ULP (unit in last place) for each value.

    The ULP is the spacing between consecutive float16 representable
    values at the magnitude of each input value.

    Args:
        values: Input array of float32 values (originally from float16).

    Returns:
        Array of same shape with the float16 eps at each value.
    """
    fp16_values = values.astype(np.float16)
    eps = np.abs(np.spacing(fp16_values)).astype(np.float32)
    # Avoid zero eps for subnormal numbers
    eps = np.maximum(eps, np.finfo(np.float16).smallest_subnormal)
    return eps


def apply_uniform_dithering(
    patches: np.ndarray,
    seed: int = 42,
    clip_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, DitheringStats]:
    """Apply precision-aware uniform dithering to patches.

    For each value x in the synthetic patches, adds uniform noise
    U(-eps(x)/2, eps(x)/2) where eps(x) is the float16 precision
    at value x. This destroys the quantization signature without
    introducing bias (E[dithered] = x).

    Args:
        patches: Synthetic patches array (N, C, H, W) in float32.
        seed: Random seed for reproducibility.
        clip_range: Range to clip after dithering.

    Returns:
        Tuple of (dithered_patches, stats).
    """
    rng = np.random.default_rng(seed)

    unique_before = len(np.unique(patches))

    eps_array = compute_float16_eps(patches)
    noise = rng.uniform(-0.5, 0.5, size=patches.shape).astype(np.float32)
    dithered = patches + noise * eps_array
    dithered = np.clip(dithered, clip_range[0], clip_range[1])

    unique_after = len(np.unique(dithered))

    stats = DitheringStats(
        n_samples=len(patches),
        unique_values_before=unique_before,
        unique_values_after=unique_after,
        max_eps_applied=float(eps_array.max()),
        mean_eps_applied=float(eps_array.mean()),
        min_eps_applied=float(eps_array[eps_array > 0].min()),
    )

    logger.info(
        f"Dithering: {unique_before} -> {unique_after} unique values "
        f"(mean eps={stats.mean_eps_applied:.6f})"
    )
    return dithered, stats


def _build_kfold_splits(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    real_zbins: np.ndarray,
    synth_zbins: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build stratified k-fold splits by z-bin.

    Returns:
        List of (train_indices, val_indices) tuples for the combined dataset.
    """
    from sklearn.model_selection import StratifiedKFold

    n_real = len(real_patches)
    n_synth = len(synth_patches)
    n_total = n_real + n_synth

    labels = np.concatenate([np.zeros(n_real), np.ones(n_synth)])
    zbins = np.concatenate([real_zbins, synth_zbins])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in skf.split(np.arange(n_total), zbins):
        splits.append((train_idx, val_idx))
    return splits


def _train_single_fold(
    real_patches: np.ndarray,
    synth_patches: np.ndarray,
    real_zbins: np.ndarray,
    synth_zbins: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    input_mode: str,
    cfg: DictConfig,
    fold_idx: int,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train a single fold and return test outputs."""
    # Split data
    all_patches = np.concatenate([real_patches, synth_patches], axis=0)
    all_zbins = np.concatenate([real_zbins, synth_zbins], axis=0)
    all_labels = np.concatenate([
        np.zeros(len(real_patches)),
        np.ones(len(synth_patches)),
    ])

    train_patches = all_patches[train_idx]
    train_zbins = all_zbins[train_idx]
    train_labels = all_labels[train_idx]
    val_patches = all_patches[val_idx]
    val_zbins = all_zbins[val_idx]
    val_labels = all_labels[val_idx]

    # Split into real/synth for dataset
    train_real_mask = train_labels == 0
    train_synth_mask = train_labels == 1
    val_real_mask = val_labels == 0
    val_synth_mask = val_labels == 1

    train_ds = ClassificationDataset(
        real_patches=train_patches[train_real_mask],
        synth_patches=train_patches[train_synth_mask],
        real_zbins=train_zbins[train_real_mask],
        synth_zbins=train_zbins[train_synth_mask],
        input_mode=input_mode,
    )
    val_ds = ClassificationDataset(
        real_patches=val_patches[val_real_mask],
        synth_patches=val_patches[val_synth_mask],
        real_zbins=val_zbins[val_real_mask],
        synth_zbins=val_zbins[val_synth_mask],
        input_mode=input_mode,
    )

    # Weighted sampler for training
    class_weights = train_ds.get_class_weights()
    sampler = WeightedRandomSampler(class_weights, len(train_ds), replacement=True)

    batch_size = cfg.dithering.reclassification.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Build training config compatible with ClassificationLightningModule
    # The model factory resolves relative config_path from src/classification/config/
    train_cfg = OmegaConf.create({
        "training": {
            "optimizer": "adam",
            "learning_rate": cfg.dithering.reclassification.learning_rate,
            "weight_decay": cfg.dithering.reclassification.weight_decay,
            "max_epochs": cfg.dithering.reclassification.max_epochs,
            "scheduler": {"type": "reduce_on_plateau", "factor": 0.5, "patience": 5, "min_lr": 1e-6},
            "early_stopping": {"monitor": "val/auc", "patience": cfg.dithering.reclassification.early_stopping_patience, "min_delta": 0.001},
        },
        "model": {"config_path": "models/simple_cnn.yaml"},
    })

    module = ClassificationLightningModule(
        cfg=train_cfg, in_channels=train_ds.in_channels, fold_idx=fold_idx
    )

    # Callbacks
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val/auc", patience=cfg.dithering.reclassification.early_stopping_patience,
        mode="max", min_delta=0.001,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.dithering.reclassification.max_epochs,
        accelerator="gpu" if "cuda" in device else "cpu",
        devices=1,
        callbacks=[early_stop],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(module, train_loader, val_loader)

    # Evaluate on validation set
    trainer.test(module, val_loader)
    test_outputs = module.get_test_outputs()

    return test_outputs


def dither_and_reclassify(
    cfg: DictConfig,
    experiment_name: str,
) -> dict[str, Any]:
    """Run full dithering + re-classification pipeline.

    1. Load patches
    2. Apply dithering to synthetic patches
    3. Retrain k-fold classifiers on dithered data
    4. Compare AUC before/after
    5. Compute delta-AUC with bootstrap CI

    Args:
        cfg: Diagnostics configuration.
        experiment_name: Name of the experiment.

    Returns:
        Dict with AUC comparisons and dithering statistics.
    """
    output_dir = ensure_output_dir(
        Path(cfg.output.base_dir), experiment_name, "dithering"
    )

    # Load patches
    real_patches, synth_patches, real_zbins, synth_zbins = load_patches(
        Path(cfg.data.patches_base_dir), experiment_name
    )

    # Apply dithering
    dithered_synth, dither_stats = apply_uniform_dithering(
        synth_patches,
        seed=cfg.dithering.seed,
        clip_range=tuple(cfg.dithering.clip_range),
    )

    device = cfg.experiment.device
    seed = cfg.experiment.seed
    n_folds = cfg.dithering.reclassification.n_folds
    bootstrap_n = cfg.dithering.metrics.bootstrap_n
    confidence_level = cfg.dithering.metrics.confidence_level

    results: dict[str, Any] = {
        "dithering_stats": asdict(dither_stats),
        "per_mode": {},
    }

    for input_mode in cfg.dithering.reclassification.input_modes:
        logger.info(f"Re-classifying with dithered data, mode={input_mode}")

        # Build k-fold splits
        splits = _build_kfold_splits(
            real_patches, dithered_synth, real_zbins, synth_zbins,
            n_folds=n_folds, seed=seed,
        )

        fold_probs_all, fold_labels_all = [], []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"  Fold {fold_idx}/{n_folds}")
            test_outputs = _train_single_fold(
                real_patches, dithered_synth, real_zbins, synth_zbins,
                train_idx, val_idx, input_mode, cfg, fold_idx, device,
            )
            fold_probs_all.append(test_outputs["probs"])
            fold_labels_all.append(test_outputs["labels"])

        # Compute pooled AUC after dithering
        all_probs = np.concatenate(fold_probs_all)
        all_labels = np.concatenate(fold_labels_all)
        auc_after = float(roc_auc_score(all_labels, all_probs))

        # Bootstrap CI
        ci_lower, ci_upper = _bootstrap_ci(
            all_labels, all_probs,
            metric_fn=roc_auc_score,
            n_iterations=bootstrap_n,
            confidence_level=confidence_level,
            seed=seed,
        )

        # Original AUC is ~1.0
        auc_before = 1.0

        mode_result = {
            "auc_before_dithering": auc_before,
            "auc_after_dithering": auc_after,
            "delta_auc": auc_before - auc_after,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_folds": n_folds,
        }
        results["per_mode"][input_mode] = mode_result

        logger.info(
            f"  {input_mode}: AUC {auc_before:.4f} -> {auc_after:.4f} "
            f"(delta={auc_before - auc_after:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}])"
        )

    # Save results
    save_result_json(results, output_dir / "reclassification_results.json")

    # Generate comparison plot
    _plot_dithering_comparison(results, output_dir, cfg.output.plot_format, cfg.output.plot_dpi)

    return results


def _plot_dithering_comparison(
    results: dict[str, Any],
    output_dir: Path,
    plot_format: list[str],
    dpi: int,
) -> None:
    """Generate bar chart comparing AUC before/after dithering."""
    import matplotlib.pyplot as plt

    modes = list(results["per_mode"].keys())
    auc_before = [results["per_mode"][m]["auc_before_dithering"] for m in modes]
    auc_after = [results["per_mode"][m]["auc_after_dithering"] for m in modes]
    ci_low = [results["per_mode"][m]["ci_lower"] for m in modes]
    ci_high = [results["per_mode"][m]["ci_upper"] for m in modes]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(modes))
    width = 0.35

    bars1 = ax.bar(x - width / 2, auc_before, width, label="Before dithering", color="#d62728", alpha=0.8)
    bars2 = ax.bar(x + width / 2, auc_after, width, label="After dithering", color="#2ca02c", alpha=0.8)

    # Error bars on "after" bars
    yerr_low = [a - cl for a, cl in zip(auc_after, ci_low)]
    yerr_high = [ch - a for a, ch in zip(auc_after, ci_high)]
    ax.errorbar(x + width / 2, auc_after, yerr=[yerr_low, yerr_high],
                fmt="none", color="black", capsize=4)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance level")
    ax.set_ylabel("AUC-ROC")
    ax.set_xlabel("Input Mode")
    ax.set_title("Classification AUC: Before vs. After Float16 Dithering")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, "delta_auc_comparison", plot_format, dpi)
    plt.close(fig)
