#!/usr/bin/env python3
"""CLI for k-fold segmentation training.

This script executes the segmentation k-fold cross-validation strategy.
It loads the master configuration file and optional model-specific configs,
creates k-fold splits, and optionally runs the full training pipeline.

Usage:
    # Dry run: only create k-fold CSVs without training
    python -m src.segmentation.cli.kfold_segmentation \
        --model unet \
        --config src/segmentation/config/master.yaml \
        --dry-run

    # Full training
    python -m src.segmentation.cli.kfold_segmentation \
        --model unet \
        --config src/segmentation/config/master.yaml \
        --folds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.segmentation.data.kfold_planner import KFoldPlanner
from src.segmentation.scripts.kfold_balance_visualizations import generate_all_visualizations
from src.segmentation.training.runners import KFoldSegmentationRunner
from src.segmentation.utils.config import load_and_merge_configs
from src.segmentation.utils.logging import setup_logger
from src.segmentation.utils.seeding import seed_everything


# Available models (must match files in config/models/)
AVAILABLE_MODELS = ["unet", "dynunet", "unetplusplus", "swinunetr"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="K-fold segmentation training for epilepsy lesions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - only generate k-fold CSVs
  python -m src.segmentation.cli.kfold_segmentation --model unet --dry-run

  # Full training with all folds
  python -m src.segmentation.cli.kfold_segmentation --model unet

  # Train specific folds only
  python -m src.segmentation.cli.kfold_segmentation --model unet --folds 0 2 4

  # Override configuration options
  python -m src.segmentation.cli.kfold_segmentation --model unet \\
      --max-epochs 50 --batch-size 16 --output-dir ./my_outputs
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=AVAILABLE_MODELS,
        help="Model architecture to use",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="src/segmentation/config/master.yaml",
        help="Path to master configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="Path to experiment configuration file (optional, used by orchestrator)",
    )

    # Dry run mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only create k-fold CSVs and statistics without training",
    )

    # K-fold options
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific folds to run (default: all folds from config)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Number of folds (overrides config)",
    )

    # Data options
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Use only real data (disable synthetic)",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Use only synthetic data (disable real)",
    )
    parser.add_argument(
        "--merging-strategy",
        type=str,
        choices=["concat", "balance"],
        default=None,
        help="Strategy for merging real and synthetic data",
    )
    parser.add_argument(
        "--replicas",
        type=str,
        nargs="+",
        default=None,
        help="List of replica NPZ files to use (e.g., replica_019.npz replica_018.npz)",
    )

    # Training overrides
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of dataloader workers (overrides config)",
    )

    # Logging options
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        default=None,
        help="Run W&B in offline mode",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser.parse_args()


def build_cli_overrides(args: argparse.Namespace) -> dict:
    """Build configuration overrides from CLI arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary of configuration overrides
    """
    overrides = {}

    # Experiment overrides
    experiment_overrides = {}
    if args.output_dir is not None:
        experiment_overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        experiment_overrides["seed"] = args.seed
    if experiment_overrides:
        overrides["experiment"] = experiment_overrides

    # K-fold overrides
    kfold_overrides = {}
    if args.folds is not None:
        kfold_overrides["folds_to_run"] = args.folds
    if args.n_folds is not None:
        kfold_overrides["n_folds"] = args.n_folds
    if kfold_overrides:
        overrides["k_fold"] = kfold_overrides

    # Data overrides
    data_overrides = {}
    if args.real_only:
        data_overrides["real"] = {"enabled": True}
        data_overrides["synthetic"] = {"enabled": False}
    elif args.synthetic_only:
        data_overrides["real"] = {"enabled": False}
        data_overrides["synthetic"] = {"enabled": True}

    if args.merging_strategy is not None:
        if "synthetic" not in data_overrides:
            data_overrides["synthetic"] = {}
        data_overrides["synthetic"]["merging_strategy"] = args.merging_strategy

    if args.replicas is not None:
        if "synthetic" not in data_overrides:
            data_overrides["synthetic"] = {}
        data_overrides["synthetic"]["replicas"] = args.replicas

    if data_overrides:
        overrides["data"] = data_overrides

    # Training overrides
    training_overrides = {}
    if args.max_epochs is not None:
        training_overrides["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        training_overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        training_overrides["num_workers"] = args.num_workers
    if args.learning_rate is not None:
        training_overrides["optimizer"] = {"lr": args.learning_rate}
    if training_overrides:
        overrides["training"] = training_overrides

    # Logging overrides
    logging_overrides = {}
    if args.no_wandb:
        logging_overrides["wandb"] = {"enabled": False}
    elif args.wandb_offline:
        logging_overrides["wandb"] = {"offline": True}
    if logging_overrides:
        overrides["logging"] = logging_overrides

    return overrides


def run_dry_run(cfg, logger: logging.Logger) -> None:
    """Execute dry run: create k-fold CSVs without training.

    Args:
        cfg: Configuration object
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("DRY RUN MODE - Creating k-fold plan only")
    logger.info("=" * 80)

    # Ensure output directory exists
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create kfold_information subdirectory for all k-fold related outputs
    kfold_info_dir = output_dir / "kfold_information"
    kfold_info_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration used
    config_path = kfold_info_dir / "config_dry_run.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Saved configuration to: {config_path}")

    # Create the k-fold planner (temporarily override output_dir for plan generation)
    logger.info("\nInitializing KFoldPlanner...")
    
    # Create a modified config for the planner to output to kfold_information
    planner_cfg = OmegaConf.merge(cfg, {"experiment": {"output_dir": str(kfold_info_dir)}})
    planner = KFoldPlanner(planner_cfg)  # type: ignore[arg-type]

    # Generate the fold plan CSV
    logger.info("\nGenerating k-fold plan CSV...")
    csv_path = planner.plan()
    logger.info(f"K-fold plan saved to: {csv_path}")

    # Print statistics for each fold
    logger.info("\n" + "=" * 80)
    logger.info("FOLD STATISTICS")
    logger.info("=" * 80)

    all_stats = []
    for fold_idx in range(cfg.k_fold.n_folds):
        stats = planner.get_fold_statistics(fold_idx)
        all_stats.append(stats)
        planner.print_fold_statistics(fold_idx)

    # Save statistics to JSON
    stats_path = kfold_info_dir / "kfold_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"\nSaved fold statistics to: {stats_path}")

    # Generate visualizations
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    try:
        viz_paths = generate_all_visualizations(
            stats_path=stats_path,
            csv_path=csv_path,
            output_dir=kfold_info_dir,
        )
        logger.info(f"Generated {len(viz_paths)} visualization files:")
        for vp in viz_paths:
            logger.info(f"  - {vp.name}")
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")
        logger.warning("Continuing without visualizations...")
        viz_paths = []

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("DRY RUN SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"K-fold information: {kfold_info_dir}")
    logger.info(f"  - Configuration: {config_path.name}")
    logger.info(f"  - K-fold plan CSV: {csv_path.name}")
    logger.info(f"  - Fold statistics: {stats_path.name}")
    for vp in viz_paths:
        logger.info(f"  - Visualization: {vp.name}")
    logger.info(f"Number of folds: {cfg.k_fold.n_folds}")

    # Data mode summary
    if cfg.data.real.enabled and cfg.data.synthetic.enabled:
        mode = f"Real + Synthetic ({cfg.data.synthetic.merging_strategy})"
        logger.info(f"Data mode: {mode}")
        logger.info(f"Replicas: {cfg.data.synthetic.replicas}")
    elif cfg.data.real.enabled:
        logger.info("Data mode: Real only")
    elif cfg.data.synthetic.enabled:
        logger.info("Data mode: Synthetic only")
    else:
        logger.warning("Data mode: No data enabled!")

    # Per-fold summary table
    logger.info("\nPer-fold training set summary:")
    logger.info("-" * 60)
    logger.info(f"{'Fold':<6} {'Total':<10} {'Real':<10} {'Synth':<10} {'Lesion%':<10}")
    logger.info("-" * 60)

    for stats in all_stats:
        train = stats["train"]
        logger.info(
            f"{stats['fold']:<6} "
            f"{train['total']:<10} "
            f"{train['real']:<10} "
            f"{train['synthetic']:<10} "
            f"{train['lesion_ratio']*100:>6.1f}%"
        )

    logger.info("-" * 60)
    logger.info("\nDry run complete. Review the outputs before running full training.")


def run_training(cfg, logger: logging.Logger) -> None:
    """Execute full k-fold training.

    Args:
        cfg: Configuration object
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("K-FOLD SEGMENTATION TRAINING")
    logger.info("=" * 80)

    # Ensure output directory exists
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create kfold_information subdirectory
    kfold_info_dir = output_dir / "kfold_information"
    kfold_info_dir.mkdir(parents=True, exist_ok=True)

    # Generate k-fold plan and visualizations before training
    logger.info("\nGenerating k-fold plan and visualizations...")
    
    # Create a modified config for the planner
    planner_cfg = OmegaConf.merge(cfg, {"experiment": {"output_dir": str(kfold_info_dir)}})
    planner = KFoldPlanner(planner_cfg)  # type: ignore[arg-type]
    
    # Generate CSV
    csv_path = planner.plan()
    logger.info(f"K-fold plan saved to: {csv_path}")
    
    # Save statistics
    all_stats = [planner.get_fold_statistics(i) for i in range(cfg.k_fold.n_folds)]
    stats_path = kfold_info_dir / "kfold_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Fold statistics saved to: {stats_path}")
    
    # Generate visualizations
    try:
        viz_paths = generate_all_visualizations(
            stats_path=stats_path,
            csv_path=csv_path,
            output_dir=kfold_info_dir,
        )
        logger.info(f"Generated {len(viz_paths)} visualization files")
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")

    # Save training configuration
    config_path = kfold_info_dir / "config_training.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Training configuration saved to: {config_path}")

    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    # Run k-fold training
    runner = KFoldSegmentationRunner(cfg)
    runner.run()

    logger.info("\nTraining complete!")


def main():
    """Main entry point for k-fold segmentation CLI."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("segmentation", level=log_level)

    # Banner
    logger.info("=" * 80)
    logger.info("EPILEPSY LESION SEGMENTATION - K-FOLD CROSS-VALIDATION")
    logger.info("=" * 80)

    # Validate arguments
    if args.real_only and args.synthetic_only:
        logger.error("Cannot specify both --real-only and --synthetic-only")
        sys.exit(1)

    # Load and merge configurations
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    if args.experiment_config:
        logger.info(f"Experiment config: {args.experiment_config}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'TRAINING'}")

    # Load experiment config first if provided (used by orchestrator)
    exp_overrides = {}
    if args.experiment_config:
        exp_cfg = OmegaConf.load(args.experiment_config)
        logger.info(f"Loading experiment config: {args.experiment_config}")
        exp_overrides = OmegaConf.to_container(exp_cfg, resolve=True)

    # Build CLI overrides (these take precedence over experiment config)
    cli_overrides = build_cli_overrides(args)

    # Merge all overrides: experiment config first, then CLI overrides
    all_overrides = OmegaConf.merge(exp_overrides, cli_overrides)

    if all_overrides:
        logger.info("Configuration overrides:")
        if exp_overrides:
            logger.info("  From experiment config:")
            for key, value in exp_overrides.items():
                logger.info(f"    {key}: {value}")
        if cli_overrides:
            logger.info("  From CLI (takes precedence):")
            for key, value in cli_overrides.items():
                logger.info(f"    {key}: {value}")

    # Load configuration with all overrides
    try:
        cfg = load_and_merge_configs(
            master_path=args.config,
            model_name=args.model,
            cli_overrides=all_overrides,
        )

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Set experiment name
    cfg.experiment.name = f"seg_{args.model}"

    # Log configuration summary
    logger.info("\nConfiguration summary:")
    logger.info(f"  Experiment: {cfg.experiment.name}")
    logger.info(f"  Output dir: {cfg.experiment.output_dir}")
    logger.info(f"  Seed: {cfg.experiment.seed}")
    logger.info(f"  N folds: {cfg.k_fold.n_folds}")
    logger.info(f"  Folds to run: {cfg.k_fold.folds_to_run or 'all'}")
    logger.info(f"  Real data: enabled={cfg.data.real.enabled}")
    logger.info(f"  Synthetic data: enabled={cfg.data.synthetic.enabled}")

    if cfg.data.synthetic.enabled:
        logger.info(f"    Merging strategy: {cfg.data.synthetic.merging_strategy}")
        logger.info(f"    Replicas: {cfg.data.synthetic.replicas}")

    if not args.dry_run:
        logger.info(f"  Model: {cfg.model.name}")
        logger.info(f"  Batch size: {cfg.training.batch_size}")
        logger.info(f"  Max epochs: {cfg.training.max_epochs}")
        logger.info(f"  Learning rate: {cfg.training.optimizer.lr}")

    logger.info("=" * 80)

    # Set random seed for reproducibility
    seed_everything(cfg.experiment.seed)

    # Execute
    if args.dry_run:
        run_dry_run(cfg, logger)
    else:
        run_training(cfg, logger)


if __name__ == "__main__":
    main()
