"""Training runners for k-fold segmentation."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.segmentation.callbacks.logging_callbacks import CSVLoggingCallback
from src.segmentation.data.dataset import PlannedFoldDataset
from src.segmentation.data.kfold_planner import KFoldPlanner
from src.segmentation.data.transforms import SegmentationTransforms
from src.segmentation.training.lit_module import SegmentationLitModule
from src.segmentation.utils.seeding import seed_everything

logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*num_workers.*bottleneck.*")
warnings.filterwarnings("ignore", message=".*Precision 16-mixed.*")
warnings.filterwarnings("ignore", message=".*always_return_as_numpy.*")
warnings.filterwarnings("ignore", message=".*single channel prediction.*")
warnings.filterwarnings("ignore", message=".*ground truth of class 0 is all 0.*")
warnings.filterwarnings("ignore", message=".*prediction of class 0 is all 0.*")

# Set matmul precision for Tensor Cores
torch.set_float32_matmul_precision('medium')


class KFoldSegmentationRunner:
    """Runner for k-fold cross-validation segmentation."""

    def __init__(self, cfg: DictConfig):
        """Initialize runner.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.output_dir = Path(cfg.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create k-fold planner (handles real + synthetic data mixing)
        self.planner = KFoldPlanner(cfg)

        # Determine which folds to run
        self.folds_to_run = cfg.k_fold.folds_to_run or list(
            range(cfg.k_fold.n_folds)
        )

    def run(self):
        """Run k-fold training."""
        logger.info(
            f"Starting {len(self.folds_to_run)}-fold cross-validation"
        )

        # Store fold results
        fold_results = []

        for fold_idx in self.folds_to_run:
            logger.info(f"\n{'='*80}")
            logger.info(f"FOLD {fold_idx}/{self.cfg.k_fold.n_folds - 1}")
            logger.info(f"{'='*80}\n")

            # Print fold statistics
            self.planner.print_fold_statistics(fold_idx)

            # Train fold
            fold_result = self.train_fold(fold_idx)
            fold_results.append(fold_result)

        # Aggregate results across folds
        self._aggregate_results(fold_results)

        logger.info("\nK-fold cross-validation complete!")

    def train_fold(self, fold_idx: int):
        """Train a single fold.

        Args:
            fold_idx: Fold index

        Returns:
            Dict with fold results
        """
        # Set seed for reproducibility
        seed_everything(self.cfg.experiment.seed + fold_idx)

        # Create fold-specific output directory
        fold_dir = self.output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Update config for this fold
        fold_cfg = OmegaConf.create(self.cfg)
        fold_cfg.experiment.name = (
            f"{self.cfg.experiment.name}_fold{fold_idx}"
        )
        fold_cfg.experiment.output_dir = str(fold_dir)

        # Save fold config
        OmegaConf.save(fold_cfg, fold_dir / "config.yaml")

        # Create dataloaders
        train_loader, val_loader = self._create_fold_dataloaders(fold_idx)

        # Create model
        model = SegmentationLitModule(fold_cfg)

        # Create callbacks
        callbacks = self._build_callbacks(fold_cfg, fold_idx)

        # Create trainer (no wandb logger)
        trainer = pl.Trainer(
            default_root_dir=fold_dir,
            max_epochs=fold_cfg.training.max_epochs,
            max_steps=fold_cfg.training.max_steps or -1,
            precision=fold_cfg.training.precision,
            gradient_clip_val=fold_cfg.training.gradient_clip_val,
            gradient_clip_algorithm=fold_cfg.training.gradient_clip_algorithm,
            accumulate_grad_batches=fold_cfg.training.accumulate_grad_batches,
            val_check_interval=fold_cfg.training.val_check_interval,
            check_val_every_n_epoch=fold_cfg.training.check_val_every_n_epoch,
            log_every_n_steps=fold_cfg.logging.log_every_n_steps,
            callbacks=callbacks,
            logger=False,  # Disable default logger, using CSV only
            enable_progress_bar=True,
            accelerator="auto",
            devices="auto",
            strategy="auto",
        )

        # Train
        logger.info(f"Training fold {fold_idx}...")
        trainer.fit(model, train_loader, val_loader)

        # Extract best validation metrics
        best_score = trainer.checkpoint_callback.best_model_score
        if best_score is not None:
            best_val_dice = best_score.item()
        else:
            logger.warning(f"Fold {fold_idx}: No best model score found (likely all NaN)")
            best_val_dice = float("nan")
        best_model_path = trainer.checkpoint_callback.best_model_path

        logger.info(
            f"Fold {fold_idx} training complete. Best Val Dice: {best_val_dice:.4f}"
        )

        # Evaluate on test set
        test_results = self._evaluate_test_set(
            trainer, model, fold_idx, fold_dir, best_model_path
        )

        fold_result = {
            "fold": fold_idx,
            "best_val_dice": best_val_dice,
            "best_model_path": best_model_path,
            "test_results": test_results,
        }

        return fold_result

    def _create_fold_dataloaders(self, fold_idx: int):
        """Create train/val dataloaders for a fold.

        Args:
            fold_idx: Fold index

        Returns:
            (train_loader, val_loader) tuple
        """
        # Get samples from planner (handles real + synthetic mixing)
        train_samples, val_samples = self.planner.get_fold(fold_idx)

        # Build transforms
        train_transforms = SegmentationTransforms.build_train_transforms(
            self.cfg
        )
        val_transforms = SegmentationTransforms.build_val_transforms(
            self.cfg
        )

        # Create datasets using PlannedFoldDataset (works with NPZ replicas)
        train_dataset = PlannedFoldDataset(
            samples=train_samples,
            real_cache_dir=Path(self.cfg.data.real.cache_dir),
            synthetic_dir=(
                Path(self.cfg.data.synthetic.samples_dir)
                if self.cfg.data.synthetic.enabled
                else None
            ),
            transform=train_transforms,
            mask_threshold=self.cfg.data.mask.binarize_threshold,
        )

        val_dataset = PlannedFoldDataset(
            samples=val_samples,
            real_cache_dir=Path(self.cfg.data.real.cache_dir),
            synthetic_dir=None,  # Validation uses only real data
            transform=val_transforms,
            mask_threshold=self.cfg.data.mask.binarize_threshold,
        )

        # Determine sampler for training
        sampler = None
        shuffle = True

        if self.cfg.training.class_balancing.enabled:
            if (
                self.cfg.training.class_balancing.method
                == "weighted_sampler"
            ):
                sampler = self._create_weighted_sampler(train_dataset)
                shuffle = False

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
            drop_last=False,
        )

        logger.info(
            f"Created dataloaders: train={len(train_dataset)}, "
            f"val={len(val_dataset)}"
        )

        return train_loader, val_loader

    def _evaluate_test_set(
        self,
        trainer: pl.Trainer,
        model: pl.LightningModule,
        fold_idx: int,
        fold_dir: Path,
        best_model_path: str,
    ) -> dict:
        """Evaluate best model on test set.

        Args:
            trainer: Lightning trainer
            model: Lightning module
            fold_idx: Fold index
            fold_dir: Fold output directory
            best_model_path: Path to best checkpoint

        Returns:
            Dictionary with test metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATING ON TEST SET - Fold {fold_idx}")
        logger.info(f"{'='*80}")

        # Create test dataloader
        test_loader = self._create_test_dataloader()

        if test_loader is None:
            logger.warning("No test set available, skipping test evaluation")
            return {
                "test/dice": float("nan"),
                "test/hd95": float("nan"),
                "test/loss": float("nan"),
            }

        # Load best checkpoint if available
        if best_model_path and Path(best_model_path).exists():
            logger.info(f"Loading best checkpoint: {best_model_path}")
            # Load checkpoint - weights_only=False is safe for checkpoints we created ourselves
            checkpoint = torch.load(best_model_path, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            logger.warning("No checkpoint found, using current model weights")

        # Run test
        logger.info(f"Running test evaluation on {len(test_loader.dataset)} samples...")
        test_results = trainer.test(model, test_loader, verbose=False)

        # Extract metrics (test_results is a list with one dict)
        if test_results and len(test_results) > 0:
            test_metrics = test_results[0]
        else:
            logger.warning("No test results returned")
            test_metrics = {}

        # Save test results to JSON
        test_results_path = fold_dir / "test_results.json"
        with open(test_results_path, "w") as f:
            json.dump({
                "fold": fold_idx,
                "best_model_path": str(best_model_path),
                "metrics": test_metrics,
            }, f, indent=2)

        logger.info(f"Test results saved to: {test_results_path}")
        logger.info(f"Test Dice: {test_metrics.get('test/dice', float('nan')):.4f}")
        logger.info(f"Test HD95: {test_metrics.get('test/hd95', float('nan')):.2f}")
        logger.info(f"Test Loss: {test_metrics.get('test/loss', float('nan')):.4f}")

        return test_metrics

    def _create_test_dataloader(self):
        """Create test dataloader (same for all folds).

        Returns:
            DataLoader for test set
        """
        # Get test samples from planner (same across all folds)
        test_samples = self.planner.test_samples

        if not test_samples:
            logger.warning("No test samples found")
            return None

        # Build transforms (use validation transforms for test)
        test_transforms = SegmentationTransforms.build_val_transforms(
            self.cfg
        )

        # Create dataset - test uses only real data
        test_dataset = PlannedFoldDataset(
            samples=test_samples,
            real_cache_dir=Path(self.cfg.data.real.cache_dir),
            synthetic_dir=None,  # Test uses only real data
            transform=test_transforms,
            mask_threshold=self.cfg.data.mask.binarize_threshold,
        )

        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
            drop_last=False,
        )

        logger.info(f"Created test dataloader: {len(test_dataset)} samples")

        return test_loader

    def _create_weighted_sampler(self, dataset):
        """Create weighted sampler for class balancing.

        Args:
            dataset: Training dataset (PlannedFoldDataset)

        Returns:
            WeightedRandomSampler
        """
        import torch

        weights = []
        lesion_weight = self.cfg.training.class_balancing.lesion_weight

        for sample in dataset.samples:
            # SampleRecord uses .has_lesion attribute
            if sample.has_lesion:
                weights.append(lesion_weight)
            else:
                weights.append(1.0)

        weights = torch.tensor(weights, dtype=torch.float64)

        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

    def _build_callbacks(self, cfg: DictConfig, fold_idx: int):
        """Build callbacks for training.

        Args:
            cfg: Configuration
            fold_idx: Fold index

        Returns:
            List of callbacks
        """
        callbacks = []

        # Checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(cfg.experiment.output_dir) / "checkpoints",
            filename=cfg.logging.checkpointing.filename,
            save_top_k=cfg.logging.checkpointing.save_top_k,
            monitor=cfg.logging.checkpointing.monitor,
            mode=cfg.logging.checkpointing.mode,
            save_last=cfg.logging.checkpointing.save_last,
            every_n_epochs=cfg.logging.checkpointing.every_n_epochs,
        )
        callbacks.append(checkpoint_callback)

        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # CSV logging
        if cfg.logging.csv.enabled:
            callbacks.append(CSVLoggingCallback(cfg, fold_idx))

        # Early stopping
        if cfg.training.early_stopping.enabled:
            callbacks.append(
                EarlyStopping(
                    monitor=cfg.training.early_stopping.monitor,
                    patience=cfg.training.early_stopping.patience,
                    mode=cfg.training.early_stopping.mode,
                    min_delta=0.0001,  # Minimum change to qualify as improvement
                    check_finite=False,  # Don't stop on NaN, let training continue
                    verbose=True,
                )
            )

        return callbacks

    def _aggregate_results(self, fold_results: list):
        """Aggregate results across folds.

        Args:
            fold_results: List of fold result dicts
        """
        # Validation metrics
        val_dice_scores = [r["best_val_dice"] for r in fold_results]
        val_dice_scores_clean = [d for d in val_dice_scores if not np.isnan(d)]

        # Test metrics
        test_dice_scores = [
            r["test_results"].get("test/dice", float("nan"))
            for r in fold_results
        ]
        test_dice_scores_clean = [d for d in test_dice_scores if not np.isnan(d)]

        test_hd95_scores = [
            r["test_results"].get("test/hd95", float("nan"))
            for r in fold_results
        ]
        test_hd95_scores_clean = [h for h in test_hd95_scores if not np.isnan(h)]

        test_loss_scores = [
            r["test_results"].get("test/loss", float("nan"))
            for r in fold_results
        ]
        test_loss_scores_clean = [l for l in test_loss_scores if not np.isnan(l)]

        # Validation summary
        val_summary = {
            "mean": float(np.mean(val_dice_scores_clean)) if val_dice_scores_clean else float("nan"),
            "std": float(np.std(val_dice_scores_clean)) if val_dice_scores_clean else float("nan"),
            "min": float(np.min(val_dice_scores_clean)) if val_dice_scores_clean else float("nan"),
            "max": float(np.max(val_dice_scores_clean)) if val_dice_scores_clean else float("nan"),
        }

        # Test summary
        test_summary = {
            "dice": {
                "mean": float(np.mean(test_dice_scores_clean)) if test_dice_scores_clean else float("nan"),
                "std": float(np.std(test_dice_scores_clean)) if test_dice_scores_clean else float("nan"),
                "min": float(np.min(test_dice_scores_clean)) if test_dice_scores_clean else float("nan"),
                "max": float(np.max(test_dice_scores_clean)) if test_dice_scores_clean else float("nan"),
            },
            "hd95": {
                "mean": float(np.mean(test_hd95_scores_clean)) if test_hd95_scores_clean else float("nan"),
                "std": float(np.std(test_hd95_scores_clean)) if test_hd95_scores_clean else float("nan"),
                "min": float(np.min(test_hd95_scores_clean)) if test_hd95_scores_clean else float("nan"),
                "max": float(np.max(test_hd95_scores_clean)) if test_hd95_scores_clean else float("nan"),
            },
            "loss": {
                "mean": float(np.mean(test_loss_scores_clean)) if test_loss_scores_clean else float("nan"),
                "std": float(np.std(test_loss_scores_clean)) if test_loss_scores_clean else float("nan"),
                "min": float(np.min(test_loss_scores_clean)) if test_loss_scores_clean else float("nan"),
                "max": float(np.max(test_loss_scores_clean)) if test_loss_scores_clean else float("nan"),
            },
        }

        results_summary = {
            "validation": val_summary,
            "test": test_summary,
            "fold_results": fold_results,
        }

        # Save aggregated results to JSON
        with open(self.output_dir / "kfold_results.json", "w") as f:
            json.dump(results_summary, f, indent=2)

        # Save test results CSV
        self._save_test_results_csv(fold_results)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("K-FOLD RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info("\nVALIDATION METRICS:")
        logger.info(
            f"  Mean Dice: {val_summary['mean']:.4f} ± {val_summary['std']:.4f}"
        )
        logger.info(
            f"  Range: [{val_summary['min']:.4f}, {val_summary['max']:.4f}]"
        )
        logger.info("\nTEST METRICS:")
        logger.info(
            f"  Dice: {test_summary['dice']['mean']:.4f} ± {test_summary['dice']['std']:.4f}"
        )
        logger.info(
            f"  HD95: {test_summary['hd95']['mean']:.2f} ± {test_summary['hd95']['std']:.2f} mm"
        )
        logger.info(
            f"  Loss: {test_summary['loss']['mean']:.4f} ± {test_summary['loss']['std']:.4f}"
        )
        logger.info("=" * 80)

    def _save_test_results_csv(self, fold_results: list):
        """Save test results to CSV.

        Args:
            fold_results: List of fold result dicts
        """
        import csv as csv_module

        csv_path = self.output_dir / "test_results.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv_module.writer(f)
            # Header
            writer.writerow([
                "fold",
                "best_val_dice",
                "test_dice",
                "test_hd95",
                "test_loss",
                "best_model_path",
            ])

            # Data rows
            for result in fold_results:
                test_metrics = result["test_results"]
                writer.writerow([
                    result["fold"],
                    result["best_val_dice"],
                    test_metrics.get("test/dice", float("nan")),
                    test_metrics.get("test/hd95", float("nan")),
                    test_metrics.get("test/loss", float("nan")),
                    result["best_model_path"],
                ])

        logger.info(f"Test results CSV saved to: {csv_path}")
