"""CLI script for patch extraction.

Usage:
    python -m src.classification extract --config <path> --experiment <name>
    python -m src.classification extract --config <path> --all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from src.classification.data.patch_extractor import PatchExtractor

logger = logging.getLogger(__name__)


def run_extraction(args: argparse.Namespace) -> None:
    """Run patch extraction for one or all experiments."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = OmegaConf.load(args.config)
    base_output = Path(cfg.output.base_dir) / cfg.output.patches_subdir

    if args.all:
        experiments = [exp.name for exp in cfg.data.synthetic.experiments]
    elif args.experiment:
        experiments = [args.experiment]
    else:
        # Extract real patches only (for control experiment)
        logger.info("No experiment specified; extracting real patches only.")
        extractor = PatchExtractor(cfg, experiment_name=None)
        output_dir = base_output / "control"
        extractor.extract_real_only(output_dir)
        logger.info(f"Real patches saved to {output_dir}")
        return

    for exp_name in experiments:
        logger.info(f"{'='*60}")
        logger.info(f"Extracting patches for: {exp_name}")
        logger.info(f"{'='*60}")

        extractor = PatchExtractor(cfg, experiment_name=exp_name)
        output_dir = base_output / exp_name
        stats = extractor.extract_all(output_dir)

        logger.info(
            f"Done: {stats.n_real} real, {stats.n_synthetic} synthetic patches "
            f"(patch_size={stats.patch_size})"
        )
