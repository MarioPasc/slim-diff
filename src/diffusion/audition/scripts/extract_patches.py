#!/usr/bin/env python3
"""Extract lesion patches from real and synthetic datasets.

This script scans both real and synthetic datasets, finds all lesion samples,
computes the optimal patch size based on maximum lesion dimensions, and
extracts centered patches for classifier training.

Usage:
    python -m src.diffusion.audition.scripts.extract_patches --config path/to/audition.yaml

Example:
    python -m src.diffusion.audition.scripts.extract_patches \
        --config src/diffusion/audition/config/audition.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from ..data.patch_extractor import PatchExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for patch extraction."""
    parser = argparse.ArgumentParser(
        description="Extract lesion patches from real and synthetic datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.diffusion.audition.scripts.extract_patches \\
        --config src/diffusion/audition/config/audition.yaml

    python -m src.diffusion.audition.scripts.extract_patches \\
        --config src/diffusion/audition/config/audition.yaml \\
        --no-balance  # Keep all synthetic samples
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to audition configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for patches",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Don't balance classes (keep all synthetic samples)",
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

    # Override output directory if specified
    if args.output_dir:
        cfg.output.patches_dir = args.output_dir

    # Create output directory
    output_dir = Path(cfg.output.patches_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run extraction
    logger.info("Starting patch extraction...")
    extractor = PatchExtractor(cfg)
    stats = extractor.extract_all(
        output_dir=output_dir,
        balance_classes=not args.no_balance,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Real patches: {stats.n_real}")
    logger.info(f"Synthetic patches: {stats.n_synthetic}")
    logger.info(f"Patch size: {stats.patch_size}x{stats.patch_size}")
    logger.info(f"Max lesion dimensions: {stats.max_lesion_height}x{stats.max_lesion_width}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
