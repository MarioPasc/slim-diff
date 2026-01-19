"""Command-line interface for building slice caches.

Usage:
    # New format
    python -m src.diffusion.data.caching.cli --config configs/cache/epilepsy.yaml

    # Legacy format (auto-migration)
    python -m src.diffusion.data.caching.cli --config src/diffusion/config/jsddpm.yaml --legacy

    # Override dataset type
    python -m src.diffusion.data.caching.cli --config configs/cache/my_config.yaml --dataset-type epilepsy
"""

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from .registry import get_registry
from .utils.config_utils import load_cache_config, migrate_legacy_config
from src.diffusion.utils.logging import setup_logger
from src.diffusion.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def main():
    """Main entry point for cache building CLI."""
    parser = argparse.ArgumentParser(
        description="Build slice cache for JS-DDPM datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build cache for epilepsy dataset
  python -m src.diffusion.data.caching.cli --config configs/cache/epilepsy.yaml

  # Build cache for BraTS-MEN dataset
  python -m src.diffusion.data.caching.cli --config configs/cache/brats_men.yaml

  # Use legacy config format (auto-migrates)
  python -m src.diffusion.data.caching.cli --config src/diffusion/config/jsddpm.yaml --legacy

  # Override dataset type from config
  python -m src.diffusion.data.caching.cli --config configs/cache/my_config.yaml --dataset-type brats_men
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to cache configuration YAML file",
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        help="Override dataset type from config (e.g., 'epilepsy', 'brats_men')",
    )

    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy config format (jsddpm.yaml) with automatic migration",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (overrides config if provided)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(level=args.log_level)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    logger.info("=" * 80)
    logger.info("JS-DDPM Slice Cache Builder")
    logger.info("=" * 80)
    logger.info(f"Config file: {config_path}")

    try:
        if args.legacy:
            logger.info("Loading legacy config format...")
            cfg = OmegaConf.load(config_path)
            cache_cfg = migrate_legacy_config(cfg)
            logger.info("Legacy config migrated successfully")
        else:
            logger.info("Loading cache config...")
            cache_cfg = load_cache_config(config_path)

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # Override dataset type if specified
    dataset_type = args.dataset_type or cache_cfg.dataset_type

    # Set random seed
    seed = args.seed or cache_cfg.datasets.get(dataset_type, {}).get("splits", {}).get("seed", 42)
    seed_everything(seed)
    logger.info(f"Random seed: {seed}")

    # List available datasets
    registry = get_registry()
    available = registry.list_available()
    logger.info(f"Available datasets: {', '.join(available)}")

    # Validate dataset type
    if not registry.is_registered(dataset_type):
        logger.error(
            f"Unknown dataset type: '{dataset_type}'. "
            f"Available: {', '.join(available)}"
        )
        return 1

    # Create builder
    logger.info(f"Creating builder for dataset: {dataset_type}")
    try:
        builder = registry.create(dataset_type, cache_cfg)
    except Exception as e:
        logger.error(f"Failed to create builder: {e}")
        return 1

    # Display configuration summary
    logger.info("-" * 80)
    logger.info("Configuration Summary:")
    logger.info(f"  Dataset type: {dataset_type}")
    logger.info(f"  Cache directory: {cache_cfg.cache_dir}")
    logger.info(f"  Z-bins: {cache_cfg.z_bins}")
    logger.info(f"  Z-range: {builder.z_range}")
    logger.info(f"  Lesion area threshold: {builder.lesion_area_min_pixels} pixels")
    logger.info(f"  Drop healthy patients: {builder.drop_healthy_patients}")
    logger.info("-" * 80)

    # Build cache
    try:
        builder.build_cache()
        logger.info("\n" + "=" * 80)
        logger.info("✓ Cache build complete!")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"\n✗ Cache build failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
