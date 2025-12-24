"""Training runner for JS-DDPM.

CLI entrypoint for training the diffusion model from YAML configuration.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.diffusion.data.dataset import create_dataloader
from src.diffusion.training.callbacks.epoch_callbacks import VisualizationCallback
from src.diffusion.training.callbacks.step_callbacks import GradientNormCallback
from src.diffusion.training.lit_modules import JSDDPMLightningModule
from src.diffusion.utils.logging import setup_logger
from src.diffusion.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def build_callbacks(cfg: DictConfig) -> list[pl.Callback]:
    """Build training callbacks from configuration.

    Args:
        cfg: Configuration object.

    Returns:
        List of Lightning callbacks.
    """
    callbacks = []

    # Checkpointing
    ckpt_cfg = cfg.logging.checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.experiment.output_dir) / "checkpoints",
        filename=ckpt_cfg.filename,
        save_top_k=ckpt_cfg.save_top_k,
        monitor=ckpt_cfg.monitor,
        mode=ckpt_cfg.mode,
        save_last=ckpt_cfg.save_last,
        every_n_epochs=ckpt_cfg.every_n_epochs,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Progress bar
    callbacks.append(RichProgressBar())

    # Visualization callback
    if cfg.visualization.enabled:
        callbacks.append(VisualizationCallback(cfg))

    # Early stopping (optional)
    if cfg.training.early_stopping.enabled:
        es_cfg = cfg.training.early_stopping
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.monitor,
                patience=es_cfg.patience,
                mode=es_cfg.mode,
            )
        )

    # Gradient norm monitoring (optional)
    callbacks.append(GradientNormCallback(log_every_n_steps=100))

    return callbacks


def build_logger(cfg: DictConfig) -> pl.loggers.Logger:
    """Build training logger from configuration.

    Args:
        cfg: Configuration object.

    Returns:
        Lightning logger instance.
    """
    log_cfg = cfg.logging.logger
    output_dir = Path(cfg.experiment.output_dir)

    if log_cfg.type == "tensorboard":
        return TensorBoardLogger(
            save_dir=output_dir / "logs",
            name=cfg.experiment.name,
        )
    elif log_cfg.type == "wandb":
        return WandbLogger(
            project=log_cfg.wandb.project,
            entity=log_cfg.wandb.entity,
            name=cfg.experiment.name,
            save_dir=output_dir / "logs",
        )
    else:
        raise ValueError(f"Unknown logger type: {log_cfg.type}")


def train(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Configuration object.
    """
    # Create output directory
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # Set seed
    seed_everything(cfg.experiment.seed)

    logger.info(f"Starting training: {cfg.experiment.name}")
    logger.info(f"Output directory: {output_dir}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = create_dataloader(cfg, split="train")
    val_loader = create_dataloader(cfg, split="val", shuffle=False)

    # Create model
    logger.info("Creating model...")
    model = JSDDPMLightningModule(cfg)

    # Create callbacks and logger
    callbacks = build_callbacks(cfg)
    training_logger = build_logger(cfg)

    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.max_steps or -1,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        gradient_clip_algorithm=cfg.training.gradient_clip_algorithm,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        callbacks=callbacks,
        logger=training_logger,
        enable_progress_bar=True,
        accelerator="auto",
        devices="auto",
        strategy="auto",
    )

    # Train
    logger.info("Starting training loop...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    logger.info("Training complete!")
    logger.info(f"Best model path: {trainer.checkpoint_callback.best_model_path}")


def main() -> None:
    """CLI entrypoint for training."""
    parser = argparse.ArgumentParser(
        description="Train JS-DDPM diffusion model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/diffusion/config/jsddpm.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Apply overrides
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    if args.output_dir is not None:
        cfg.experiment.output_dir = args.output_dir

    # Setup logging
    setup_logger("jsddpm", level=logging.INFO)

    # Train
    train(cfg)


if __name__ == "__main__":
    main()
