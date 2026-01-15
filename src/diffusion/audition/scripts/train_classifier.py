#!/usr/bin/env python3
"""Train the real vs synthetic classifier.

This script trains a CNN classifier to distinguish between real and synthetic
lesion patches. The resulting model is used to evaluate synthetic data quality.

Usage:
    python -m src.diffusion.audition.scripts.train_classifier --config path/to/audition.yaml

Example:
    python -m src.diffusion.audition.scripts.train_classifier \
        --config src/diffusion/audition/config/audition.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from ..data.data_module import AuditionDataModule
from ..training.callbacks import CSVLoggingCallback, MetricsLoggerCallback
from ..training.lit_module import AuditionLightningModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_callbacks(cfg: OmegaConf) -> list[pl.Callback]:
    """Build training callbacks from configuration.

    Args:
        cfg: Configuration dictionary.

    Returns:
        List of Lightning callbacks.
    """
    callbacks = []

    # Checkpointing
    checkpoint_dir = Path(cfg.output.checkpoints_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cfg = cfg.logging.checkpointing
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=ckpt_cfg.filename,
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            save_top_k=ckpt_cfg.save_top_k,
            save_last=True,
        )
    )

    # Early stopping
    if cfg.training.early_stopping.enabled:
        es_cfg = cfg.training.early_stopping
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.monitor,
                mode=es_cfg.mode,
                patience=es_cfg.patience,
                min_delta=es_cfg.min_delta,
            )
        )

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # CSV logging
    if cfg.logging.csv_logging.enabled:
        callbacks.append(CSVLoggingCallback(cfg))

    # Metrics logger
    callbacks.append(MetricsLoggerCallback(log_every_n_epochs=1))

    return callbacks


def main() -> None:
    """Main entry point for classifier training."""
    parser = argparse.ArgumentParser(
        description="Train real vs synthetic classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.diffusion.audition.scripts.train_classifier \\
        --config src/diffusion/audition/config/audition.yaml

    python -m src.diffusion.audition.scripts.train_classifier \\
        --config src/diffusion/audition/config/audition.yaml \\
        --max-epochs 100
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to audition configuration YAML file",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override maximum epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Apply overrides
    if args.max_epochs:
        cfg.training.max_epochs = args.max_epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size

    # Set seed
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Create data module
    logger.info("Initializing data module...")
    data_module = AuditionDataModule(cfg)

    # Prepare data (creates splits)
    data_module.prepare_data()

    # Create model
    logger.info("Creating model...")
    model = AuditionLightningModule(cfg)

    # Build callbacks
    callbacks = build_callbacks(cfg)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        precision=cfg.training.precision,
        deterministic=True,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume,
    )

    # Test
    logger.info("Running test evaluation...")
    trainer.test(model, datamodule=data_module)

    # Print best checkpoint
    best_ckpt = trainer.checkpoint_callback.best_model_path
    logger.info(f"Best checkpoint: {best_ckpt}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
