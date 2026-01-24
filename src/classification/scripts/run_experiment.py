"""CLI script for running k-fold classification on one experiment.

Usage:
    python -m src.classification run \
        --config <path> --experiment <name> --input-mode joint
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.classification.data.data_module import KFoldClassificationDataModule, ControlDataModule
from src.classification.evaluation.metrics import (
    compute_fold_metrics,
    aggregate_fold_metrics,
    ExperimentResult,
)
from src.classification.evaluation.statistical_tests import permutation_test_auc
from src.classification.evaluation.reporting import save_experiment_result
from src.classification.training.lit_module import ClassificationLightningModule

logger = logging.getLogger(__name__)


def run_experiment(args: argparse.Namespace) -> ExperimentResult:
    """Run k-fold classification for one experiment and input mode."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = OmegaConf.load(args.config)
    experiment_name = args.experiment
    input_mode: Literal["joint", "image_only", "mask_only"] = args.input_mode
    n_folds = cfg.data.kfold.n_folds

    # Optional flags
    use_dithering = getattr(args, "dithering", False)
    use_full_image = getattr(args, "full_image", False)

    # Determine which folds to run
    if args.folds:
        fold_indices = [int(f) for f in args.folds.split(",")]
    else:
        fold_indices = list(range(n_folds))

    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Setup data module
    is_control = experiment_name == "control"

    # Determine patches directory based on full-image vs patch mode
    patches_subdir = cfg.output.get("full_images_subdir", "full_images") if use_full_image else cfg.output.patches_subdir
    if is_control:
        patches_dir = Path(cfg.output.base_dir) / patches_subdir / "control"
    else:
        patches_dir = Path(cfg.output.base_dir) / patches_subdir / experiment_name

    # Output directory for this run (include dithering/full-image suffix)
    mode_suffix = input_mode
    if use_dithering:
        mode_suffix += "_dithered"
    if use_full_image:
        mode_suffix += "_fullimg"

    results_dir = (
        Path(cfg.output.base_dir) / cfg.output.results_subdir / experiment_name / mode_suffix
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    if use_dithering:
        logger.info("Dithering ENABLED: will apply uniform dithering to synthetic data")
    if use_full_image:
        logger.info("Full-image mode ENABLED: using 160x160 images instead of patches")

    fold_results = []
    for fold_idx in fold_indices:
        logger.info(f"--- Fold {fold_idx}/{n_folds - 1} ---")

        # Create data module for this fold
        if is_control:
            dm = ControlDataModule(
                cfg=cfg, patches_dir=patches_dir, input_mode=input_mode, repeat_idx=0
            )
        else:
            dm = KFoldClassificationDataModule(
                cfg=cfg, experiment_name=experiment_name,
                input_mode=input_mode, patches_dir=patches_dir,
                dithering=use_dithering,
                dithering_seed=cfg.experiment.seed,
            )
        dm.set_fold(fold_idx)
        dm.prepare_data()
        dm.setup()

        # Create model
        model = ClassificationLightningModule(
            cfg=cfg, in_channels=dm.in_channels, fold_idx=fold_idx
        )

        # Callbacks
        ckpt_dir = Path(cfg.output.base_dir) / cfg.output.checkpoints_subdir / experiment_name / mode_suffix
        callbacks = [
            EarlyStopping(
                monitor=cfg.training.early_stopping.monitor,
                mode=cfg.training.early_stopping.mode,
                patience=cfg.training.early_stopping.patience,
                min_delta=cfg.training.early_stopping.min_delta,
            ),
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename=f"fold{fold_idx}_best",
                monitor=cfg.training.early_stopping.monitor,
                mode=cfg.training.early_stopping.mode,
                save_top_k=1,
            ),
        ]

        # Trainer
        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            callbacks=callbacks,
            precision=cfg.training.precision,
            enable_progress_bar=True,
            enable_model_summary=(fold_idx == 0),
            logger=False,  # We handle our own logging
            deterministic=True,
        )

        # Train
        trainer.fit(model, datamodule=dm)

        # Evaluate on validation fold (using best checkpoint)
        best_path = callbacks[1].best_model_path
        if best_path:
            model = ClassificationLightningModule.load_from_checkpoint(
                best_path, cfg=cfg, in_channels=dm.in_channels, fold_idx=fold_idx
            )

        model.clear_test_outputs()
        trainer.test(model, dataloaders=dm.val_dataloader())
        outputs = model.get_test_outputs()

        # Compute metrics for this fold
        fold_result = compute_fold_metrics(
            probs=outputs["probs"],
            labels=outputs["labels"],
            z_bins=outputs["z_bins"],
            fold_idx=fold_idx,
            bootstrap_n=cfg.evaluation.bootstrap.n_iterations,
            confidence_level=cfg.evaluation.bootstrap.confidence_level,
            bootstrap_seed=cfg.evaluation.bootstrap.seed + fold_idx,
            min_samples_per_zbin=cfg.evaluation.per_zbin.min_samples,
        )
        fold_results.append(fold_result)

        logger.info(
            f"Fold {fold_idx}: AUC={fold_result.global_metrics.auc_roc:.4f} "
            f"[{fold_result.global_metrics.auc_roc_ci_lower:.4f}, "
            f"{fold_result.global_metrics.auc_roc_ci_upper:.4f}]"
        )

        # Free GPU memory between folds
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate across folds
    experiment_result = aggregate_fold_metrics(
        fold_results=fold_results,
        experiment_name=experiment_name,
        input_mode=input_mode,
        bootstrap_n=cfg.evaluation.bootstrap.n_iterations,
        confidence_level=cfg.evaluation.bootstrap.confidence_level,
        bootstrap_seed=cfg.evaluation.bootstrap.seed,
    )

    logger.info(
        f"Experiment {experiment_name} ({input_mode}): "
        f"AUC={experiment_result.mean_auc:.4f} +/- {experiment_result.std_auc:.4f} "
        f"[{experiment_result.pooled_ci_lower:.4f}, {experiment_result.pooled_ci_upper:.4f}]"
    )

    # Permutation test
    if cfg.evaluation.permutation_test.n_permutations > 0:
        all_probs = experiment_result.fold_results[0].probs
        all_labels = experiment_result.fold_results[0].labels
        if len(experiment_result.fold_results) > 1:
            import numpy as np
            all_probs = np.concatenate([fr.probs for fr in experiment_result.fold_results])
            all_labels = np.concatenate([fr.labels for fr in experiment_result.fold_results])

        perm_result = permutation_test_auc(
            probs=all_probs,
            labels=all_labels,
            n_permutations=cfg.evaluation.permutation_test.n_permutations,
            seed=cfg.evaluation.permutation_test.seed,
            alpha=cfg.evaluation.permutation_test.alpha,
        )
        logger.info(f"Permutation test: p={perm_result.p_value:.4f}")

    # Save results
    save_experiment_result(
        experiment_result, results_dir / "experiment_result.json"
    )

    return experiment_result
