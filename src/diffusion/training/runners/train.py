"""Training runner for JS-DDPM.

CLI entrypoint for training the diffusion model from YAML configuration.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from src.diffusion.data.dataset import create_dataloader
from src.diffusion.training.callbacks.diagnostics_callbacks import (
    DataStatisticsCallback,
    DiagnosticsCallback,
    PredictionQualityCallback,
    SNRCallback,
    WandbSummaryCallback,
)
from src.diffusion.training.callbacks.csv_callback import CSVLoggingCallback
from src.diffusion.training.callbacks.epoch_callbacks import (
    EMACallback,
    VisualizationCallback,
)
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

    # Progress bar - DISABLED for supercomputer (tqdm is misleading in cluster environments)
    # callbacks.append(RichProgressBar())

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

    # Gradient norm monitoring
    callbacks.append(GradientNormCallback(log_every_n_steps=100))

    # Wandb summary tracking (best metrics)
    callbacks.append(WandbSummaryCallback())

    # Data statistics logging
    callbacks.append(DataStatisticsCallback())

    # Advanced diagnostics callbacks
    callbacks.append(DiagnosticsCallback(
        cfg=cfg,
        log_every_n_epochs=1,
        log_histograms=True,
    ))

    callbacks.append(SNRCallback(
        log_every_n_epochs=5,
        n_samples=100,
    ))

    callbacks.append(PredictionQualityCallback(
        log_every_n_epochs=5,
        n_samples=100,
        timestep_bins=10,
    ))

    callbacks.append(CSVLoggingCallback(cfg=cfg))

    # EMA callback (if enabled)
    if cfg.training.ema.enabled:
        callbacks.append(EMACallback(
            decay=cfg.training.ema.decay,
            update_every=cfg.training.ema.update_every,
            use_for_validation=cfg.training.ema.use_for_validation,
        ))
        logger.info(
            f"EMA enabled: decay={cfg.training.ema.decay}, "
            f"update_every={cfg.training.ema.update_every}"
        )

    return callbacks


def setup_wandb_config(logger: WandbLogger, cfg: DictConfig) -> None:
    """Setup wandb configuration with all hyperparameters.

    Logs comprehensive hyperparameters to wandb.config for run comparison
    and reproducibility.

    Args:
        logger: WandbLogger instance.
        cfg: Configuration object.
    """
    import wandb

    # Experiment metadata
    logger.experiment.config.update({
        "experiment/name": cfg.experiment.name,
        "experiment/seed": cfg.experiment.seed,
    })

    # Data configuration
    logger.experiment.config.update({
        "data/z_range_min": cfg.data.slice_sampling.z_range[0],
        "data/z_range_max": cfg.data.slice_sampling.z_range[1],
        "data/target_spacing": cfg.data.transforms.target_spacing[0],
        "data/roi_size": cfg.data.transforms.roi_size[0],
        "data/lesion_oversampling_enabled": cfg.data.lesion_oversampling.enabled,
        "data/lesion_oversampling_weight": cfg.data.lesion_oversampling.weight,
    })

    # Model configuration
    logger.experiment.config.update({
        "model/type": cfg.model.type,
        "model/in_channels": cfg.model.in_channels,
        "model/out_channels": cfg.model.out_channels,
        "model/channels": cfg.model.channels,
        "model/num_res_blocks": cfg.model.num_res_blocks,
        "model/num_head_channels": cfg.model.num_head_channels,
        "model/dropout": cfg.model.dropout,
    })

    # Conditioning configuration
    logger.experiment.config.update({
        "conditioning/z_bins": cfg.conditioning.z_bins,
        "conditioning/cfg_enabled": cfg.conditioning.cfg.enabled,
        "conditioning/cfg_dropout_prob": cfg.conditioning.cfg.dropout_prob,
    })

    # Scheduler configuration
    logger.experiment.config.update({
        "scheduler/type": cfg.scheduler.type,
        "scheduler/num_train_timesteps": cfg.scheduler.num_train_timesteps,
        "scheduler/schedule": cfg.scheduler.schedule,
        "scheduler/beta_start": cfg.scheduler.beta_start,
        "scheduler/beta_end": cfg.scheduler.beta_end,
        "scheduler/prediction_type": cfg.scheduler.prediction_type,
    })

    # Sampler configuration
    logger.experiment.config.update({
        "sampler/type": cfg.sampler.type,
        "sampler/num_inference_steps": cfg.sampler.num_inference_steps,
        "sampler/eta": cfg.sampler.eta,
        "sampler/guidance_scale": cfg.sampler.guidance_scale,
    })

    # Training configuration
    logger.experiment.config.update({
        "training/batch_size": cfg.training.batch_size,
        "training/max_epochs": cfg.training.max_epochs,
        "training/precision": cfg.training.precision,
        "training/gradient_clip_val": cfg.training.gradient_clip_val,
        "training/accumulate_grad_batches": cfg.training.accumulate_grad_batches,
        "training/ema_enabled": cfg.training.ema.enabled,
        "training/ema_decay": cfg.training.ema.decay,
        "training/ema_update_every": cfg.training.ema.update_every,
    })

    # Optimizer configuration
    logger.experiment.config.update({
        "optimizer/type": cfg.training.optimizer.type,
        "optimizer/lr": cfg.training.optimizer.lr,
        "optimizer/weight_decay": cfg.training.optimizer.weight_decay,
        "optimizer/betas": cfg.training.optimizer.betas,
        "optimizer/eps": cfg.training.optimizer.eps,
    })

    # Learning rate scheduler
    logger.experiment.config.update({
        "lr_scheduler/type": cfg.training.lr_scheduler.type,
        "lr_scheduler/eta_min": cfg.training.lr_scheduler.eta_min,
    })

    # Loss configuration
    logger.experiment.config.update({
        "loss/uncertainty_weighting_enabled": cfg.loss.uncertainty_weighting.enabled,
        "loss/uncertainty_initial_log_vars": cfg.loss.uncertainty_weighting.initial_log_vars,
        "loss/lesion_weighted_mask_enabled": cfg.loss.lesion_weighted_mask.enabled,
        "loss/lesion_weight": cfg.loss.lesion_weighted_mask.lesion_weight,
        "loss/background_weight": cfg.loss.lesion_weighted_mask.background_weight,
    })

    # Define custom metrics for proper x-axis handling
    logger.experiment.define_metric("train/*", step_metric="trainer/global_step")
    logger.experiment.define_metric("val/*", step_metric="epoch")
    logger.experiment.define_metric("diagnostics/*", step_metric="epoch")

    import logging as log
    log.getLogger(__name__).info("Wandb config and metrics defined successfully")


def build_logger(cfg: DictConfig) -> WandbLogger:
    """Build wandb logger from configuration.

    Args:
        cfg: Configuration object.

    Returns:
        WandbLogger instance.
    """
    log_cfg = cfg.logging.logger
    output_dir = Path(cfg.experiment.output_dir)

    if log_cfg.type != "wandb":
        raise ValueError(
            f"Only wandb logger is supported. Got: {log_cfg.type}. "
            f"Please set logging.logger.type='wandb' in your config."
        )

    wandb_logger = WandbLogger(
        project=log_cfg.wandb.project,
        entity=log_cfg.wandb.entity,
        name=cfg.experiment.name,
        save_dir=output_dir / "logs",
        offline=log_cfg.wandb.get("offline", False),
        tags=log_cfg.wandb.get("tags", None),
        notes=log_cfg.wandb.get("notes", None),
    )

    # Setup wandb config and define metrics
    setup_wandb_config(wandb_logger, cfg)

    return wandb_logger


def train(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Configuration object.
    """
    # Fix CUDA multiprocessing issue: use 'spawn' instead of 'fork'
    # This prevents CUDA context inheritance in DataLoader workers
    # Only set if not already set (to avoid conflicts)
    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
            logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")
    except RuntimeError:
        # Start method already set, just log a warning
        logger.warning(
            f"Multiprocessing start method already set to '{multiprocessing.get_start_method()}'. "
            "Cannot change to 'spawn'. If you encounter CUDA errors with num_workers > 0, "
            "set num_workers=0 in your config."
        )

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
        enable_progress_bar=False,  # Disabled for supercomputer (tqdm misleading in cluster)
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
