"""CLI script for patch extraction.

Usage:
    python -m src.classification extract --config <path> --experiment <name>
    python -m src.classification extract --config <path> --all
    python -m src.classification extract --config <path> --filter prediction_type=x0

Experiment names can be:
    - Display name: sc_0.5__x0_lp_1.5
    - Legacy name: x0_lp_1.5 (uses default self_cond_p from config)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from src.classification.data.patch_extractor import PatchExtractor
from src.shared.ablation import ExperimentCoordinate, ExperimentDiscoverer, AblationSpace

logger = logging.getLogger(__name__)


def _parse_filters(filter_args: list[str] | None) -> dict:
    """Parse filter arguments like 'prediction_type=x0' into dict."""
    if not filter_args:
        return {}

    filters = {}
    for f in filter_args:
        if "=" not in f:
            raise ValueError(f"Invalid filter format: {f}. Expected 'key=value'.")
        key, value = f.split("=", 1)
        # Try to parse as number
        try:
            value = float(value)
        except ValueError:
            pass
        filters[key] = value
    return filters


def run_extraction(args: argparse.Namespace) -> None:
    """Run patch extraction for one or all experiments."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = OmegaConf.load(args.config)
    base_output = Path(cfg.output.base_dir) / cfg.output.patches_subdir

    # Handle control experiment (real patches only)
    if not args.all and not args.experiment and not getattr(args, "filter", None):
        logger.info("No experiment specified; extracting real patches only.")
        extractor = PatchExtractor(cfg, experiment=None)
        output_dir = base_output / "control"
        extractor.extract_real_only(output_dir)
        logger.info(f"Real patches saved to {output_dir}")
        return

    # Get experiments to process
    coordinates: list[ExperimentCoordinate] = []

    if args.experiment:
        # Single experiment - resolve name to coordinate
        extractor = PatchExtractor(cfg, experiment=args.experiment)
        if extractor.experiment_coord:
            coordinates = [extractor.experiment_coord]
    elif args.all or getattr(args, "filter", None):
        # Discover experiments from filesystem
        try:
            space = AblationSpace.from_config(cfg)
        except (KeyError, TypeError):
            space = AblationSpace.default()

        default_sc = cfg.get("ablation", {}).get("default_self_cond_p", 0.5)
        discoverer = ExperimentDiscoverer(
            base_dir=Path(cfg.data.synthetic.results_base_dir),
            space=space,
            default_self_cond_p=default_sc,
        )

        # Apply filters if provided
        filters = _parse_filters(getattr(args, "filter", None))
        if filters:
            coordinates = discoverer.discover_matching(**filters)
        else:
            coordinates = discoverer.discover_all()

    if not coordinates:
        logger.warning("No experiments found to process.")
        return

    logger.info(f"Found {len(coordinates)} experiments to process")

    # Process each experiment
    experiment_names = []
    for coord in coordinates:
        exp_name = coord.to_display_name()
        experiment_names.append(exp_name)

        logger.info(f"{'='*60}")
        logger.info(f"Extracting patches for: {exp_name}")
        logger.info(f"{'='*60}")

        extractor = PatchExtractor(cfg, experiment=coord)
        output_dir = base_output / exp_name
        stats = extractor.extract_all(output_dir)

        logger.info(
            f"Done: {stats.n_real} real, {stats.n_synthetic} synthetic patches "
            f"(patch_size={stats.patch_size})"
        )

    # Run dataset analysis if not skipped
    skip_analysis = getattr(args, "skip_analysis", False)
    if not skip_analysis and experiment_names:
        logger.info(f"{'='*60}")
        logger.info("Running dataset analysis...")
        logger.info(f"{'='*60}")

        from src.classification.data.dataset_analysis import run_dataset_analysis

        analysis_output = base_output / "analysis"
        run_dataset_analysis(
            patches_dir=base_output,
            output_dir=analysis_output,
            experiments=experiment_names,
            dpi=150,
        )
        logger.info(f"Analysis complete. Output saved to {analysis_output}")
