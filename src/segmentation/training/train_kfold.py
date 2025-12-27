"""CLI for k-fold segmentation training."""

from __future__ import annotations

import argparse
import logging

from src.segmentation.training.runners import KFoldSegmentationRunner
from src.segmentation.utils.config import load_and_merge_configs
from src.segmentation.utils.logging import setup_logger


def main():
    """Main entry point for k-fold training."""
    parser = argparse.ArgumentParser(
        description="K-fold segmentation training for epilepsy lesions"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["unet", "dynunet", "unetplusplus", "swinunetr"],
        help="Model architecture",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/segmentation/config/master.yaml",
        help="Path to master config",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific folds to run (default: all)",
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=None,
        help="Ratio of synthetic to real data (overrides config)",
    )
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
        help="Random seed override",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum epochs override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger("segmentation", level=logging.INFO)
    logger = logging.getLogger("segmentation")

    logger.info("="*80)
    logger.info("K-FOLD SEGMENTATION TRAINING")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")

    # Load and merge configs
    cli_overrides = {}

    if args.folds is not None:
        cli_overrides["k_fold"] = {"folds_to_run": args.folds}

    if args.synthetic_ratio is not None:
        cli_overrides["data"] = {
            "synthetic": {
                "enabled": args.synthetic_ratio > 0,
                "ratio": args.synthetic_ratio,
            }
        }

    if args.output_dir is not None:
        cli_overrides["experiment"] = {"output_dir": args.output_dir}

    if args.seed is not None:
        if "experiment" not in cli_overrides:
            cli_overrides["experiment"] = {}
        cli_overrides["experiment"]["seed"] = args.seed

    if args.max_epochs is not None:
        cli_overrides["training"] = {"max_epochs": args.max_epochs}

    if args.batch_size is not None:
        if "training" not in cli_overrides:
            cli_overrides["training"] = {}
        cli_overrides["training"]["batch_size"] = args.batch_size

    # Load configuration
    cfg = load_and_merge_configs(
        master_path=args.config,
        model_name=args.model,
        cli_overrides=cli_overrides,
    )

    # Set experiment name
    cfg.experiment.name = f"seg_{args.model}"

    logger.info(f"Experiment: {cfg.experiment.name}")
    logger.info(f"Output dir: {cfg.experiment.output_dir}")
    logger.info(f"Seed: {cfg.experiment.seed}")
    logger.info(f"Folds: {cfg.k_fold.folds_to_run or 'all'}")
    logger.info(
        f"Synthetic: enabled={cfg.data.synthetic.enabled}, "
        f"ratio={cfg.data.synthetic.ratio}"
    )
    logger.info("="*80)

    # Run k-fold training
    runner = KFoldSegmentationRunner(cfg)
    runner.run()

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
