"""Training runners for k-fold segmentation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.segmentation.callbacks.logging_callbacks import CSVLoggingCallback
from src.segmentation.data.dataset import PlannedFoldDataset
from src.segmentation.data.kfold_planner import KFoldPlanner
from src.segmentation.data.transforms import SegmentationTransforms
from src.segmentation.training.lit_module import SegmentationLitModule
from src.segmentation.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


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

        # Create logger
        wandb_logger = self._build_logger(fold_cfg, fold_idx)

        # Create trainer
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
            logger=wandb_logger,
            enable_progress_bar=True,
            accelerator="auto",
            devices="auto",
            strategy="auto",
        )

        # Train
        logger.info(f"Training fold {fold_idx}...")
        trainer.fit(model, train_loader, val_loader)

        # Extract best metrics
        best_dice = trainer.checkpoint_callback.best_model_score.item()
        best_model_path = trainer.checkpoint_callback.best_model_path

        fold_result = {
            "fold": fold_idx,
            "best_dice": best_dice,
            "best_model_path": best_model_path,
        }

        logger.info(
            f"Fold {fold_idx} complete. Best Dice: {best_dice:.4f}"
        )

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
                )
            )

        return callbacks

    def _build_logger(self, cfg: DictConfig, fold_idx: int):
        """Build W&B logger.

        Args:
            cfg: Configuration
            fold_idx: Fold index

        Returns:
            WandbLogger or None
        """
        if not cfg.logging.wandb.enabled:
            return None

        return WandbLogger(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.experiment.name,
            save_dir=Path(cfg.experiment.output_dir) / "logs",
            offline=cfg.logging.wandb.offline,
            tags=list(cfg.logging.wandb.tags) + [f"fold_{fold_idx}"],
            group=cfg.model.name,
            job_type=f"fold_{fold_idx}",
        )

    def _aggregate_results(self, fold_results: list):
        """Aggregate results across folds.

        Args:
            fold_results: List of fold result dicts
        """
        dice_scores = [r["best_dice"] for r in fold_results]

        results_summary = {
            "mean_dice": float(np.mean(dice_scores)),
            "std_dice": float(np.std(dice_scores)),
            "min_dice": float(np.min(dice_scores)),
            "max_dice": float(np.max(dice_scores)),
            "fold_results": fold_results,
        }

        # Save to JSON
        with open(self.output_dir / "kfold_results.json", "w") as f:
            json.dump(results_summary, f, indent=2)

        logger.info("\n" + "=" * 80)
        logger.info("K-FOLD RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"Mean Dice: {results_summary['mean_dice']:.4f} ± "
            f"{results_summary['std_dice']:.4f}"
        )
        logger.info(
            f"Range: [{results_summary['min_dice']:.4f}, "
            f"{results_summary['max_dice']:.4f}]"
        )
        logger.info("=" * 80)
