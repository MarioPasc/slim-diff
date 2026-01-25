"""CLI script for full-image extraction (no patch cropping).

Usage:
    python -m src.classification extract-full --config <path> --experiment <name>
    python -m src.classification extract-full --config <path> --all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from src.classification.data.full_image_extractor import FullImageExtractor

logger = logging.getLogger(__name__)


def run_full_extraction(args: argparse.Namespace) -> None:
    """Run full-image extraction for one or all experiments."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = OmegaConf.load(args.config)
    base_output = Path(cfg.output.base_dir) / cfg.output.get("full_images_subdir", "full_images")

    if args.all:
        experiments = [exp.name for exp in cfg.data.synthetic.experiments]
    elif args.experiment:
        experiments = [args.experiment]
    else:
        logger.error("Specify --experiment <name> or --all for full-image extraction.")
        return

    for exp_name in experiments:
        logger.info(f"{'='*60}")
        logger.info(f"Extracting full images for: {exp_name}")
        logger.info(f"{'='*60}")

        extractor = FullImageExtractor(cfg, experiment_name=exp_name)
        output_dir = base_output / exp_name
        stats = extractor.extract_all(output_dir)

        logger.info(
            f"Done: {stats['n_real']} real, {stats['n_synthetic']} synthetic "
            f"full images ({stats['image_size']}x{stats['image_size']})"
        )

    # Run dataset analysis if not skipped
    skip_analysis = getattr(args, "skip_analysis", False)
    if not skip_analysis and len(experiments) > 0:
        logger.info(f"{'='*60}")
        logger.info("Running dataset analysis...")
        logger.info(f"{'='*60}")

        from src.classification.data.dataset_analysis import run_dataset_analysis

        analysis_output = base_output / "analysis"
        run_dataset_analysis(
            patches_dir=base_output,
            output_dir=analysis_output,
            experiments=experiments,
            dpi=150,
        )
        logger.info(f"Analysis complete. Output saved to {analysis_output}")
