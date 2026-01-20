"""Main entry point for model comparison."""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .config import load_config, validate_model_paths
from .data_loader import ComparisonDataLoader
from .visualizations import (
    LossCurvesVisualization,
    PublicationPanelVisualization,
    ReconstructionComparisonVisualization,
    SummaryTableVisualization,
    TimestepMSEHeatmapVisualization,
    UncertaintyEvolutionVisualization,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Visualization registry
VISUALIZATION_REGISTRY = {
    "loss_curves": LossCurvesVisualization,
    "uncertainty_evolution": UncertaintyEvolutionVisualization,
    "timestep_mse_heatmap": TimestepMSEHeatmapVisualization,
    "reconstruction_comparison": ReconstructionComparisonVisualization,
    "summary_table": SummaryTableVisualization,
    "publication_panel": PublicationPanelVisualization,
}


class ModelComparisonRunner:
    """Main runner for model comparison analysis."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize runner with configuration.

        Args:
            cfg: Validated configuration object.
        """
        self.cfg = cfg
        self.model_paths = validate_model_paths(cfg)
        self.output_dir = self._setup_output_dir()
        self.data_loader = ComparisonDataLoader(cfg)

        logger.info(f"Initialized ModelComparisonRunner")
        logger.info(f"Models: {list(self.model_paths.keys())}")
        logger.info(f"Output directory: {self.output_dir}")

    def _setup_output_dir(self) -> Path:
        """Setup output directory with optional timestamp.

        Returns:
            Output directory path.
        """
        base_dir = Path(self.cfg.output.output_dir)

        if self.cfg.output.timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            output_dir = base_dir / timestamp
        else:
            output_dir = base_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / "plots").mkdir(exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)

        # Save config copy
        config_copy_path = output_dir / "config.yaml"
        OmegaConf.save(self.cfg, config_copy_path)
        logger.info(f"Saved config to {config_copy_path}")

        return output_dir

    def run(self) -> dict[str, list[Path]]:
        """Run all enabled visualizations.

        Returns:
            Dictionary mapping visualization_name -> list of output paths.
        """
        results: dict[str, list[Path]] = {}
        enabled_visualizations = list(self.cfg.visualizations.enabled)

        logger.info(f"Running {len(enabled_visualizations)} visualizations")

        for viz_name in enabled_visualizations:
            try:
                paths = self._run_visualization(viz_name)
                results[viz_name] = paths
                logger.info(f"Completed {viz_name}: {len(paths)} outputs")
            except Exception as e:
                logger.error(f"Failed to run {viz_name}: {e}")
                import traceback
                traceback.print_exc()
                results[viz_name] = []

        # Print summary
        self._print_summary(results)

        return results

    def _run_visualization(self, viz_name: str) -> list[Path]:
        """Run a single visualization.

        Args:
            viz_name: Visualization name.

        Returns:
            List of output paths.
        """
        if viz_name not in VISUALIZATION_REGISTRY:
            logger.warning(f"Unknown visualization: {viz_name}")
            return []

        viz_class = VISUALIZATION_REGISTRY[viz_name]
        plots_dir = self.output_dir / "plots"

        # Create visualization instance
        viz = viz_class(
            cfg=self.cfg,
            output_dir=plots_dir,
            model_names=list(self.model_paths.keys()),
        )

        # Special handling for reconstruction comparison
        if viz_name == "reconstruction_comparison":
            viz.set_model_paths(self.model_paths)

        # Generate visualization
        paths = viz.generate(self.data_loader)

        return paths

    def _print_summary(self, results: dict[str, list[Path]]) -> None:
        """Print summary of generated outputs.

        Args:
            results: Dictionary of visualization results.
        """
        total_files = sum(len(paths) for paths in results.values())

        print("\n" + "=" * 60)
        print("MODEL COMPARISON COMPLETE")
        print("=" * 60)
        print(f"\nModels compared: {len(self.model_paths)}")
        for name in self.model_paths:
            print(f"  - {name}")

        print(f"\nVisualizations generated:")
        for viz_name, paths in results.items():
            status = "OK" if paths else "FAILED"
            print(f"  - {viz_name}: {len(paths)} files [{status}]")

        print(f"\nTotal files: {total_files}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60 + "\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare multiple JS-DDPM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python -m src.diffusion.model_comparison.runner --config path/to/config.yaml

  # Run specific visualizations only
  python -m src.diffusion.model_comparison.runner \\
      --config config.yaml \\
      --visualizations reconstruction_comparison summary_table

  # Override output directory
  python -m src.diffusion.model_comparison.runner \\
      --config config.yaml \\
      --output-dir ./my_comparison_output
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model comparison YAML config",
    )
    parser.add_argument(
        "--visualizations",
        nargs="+",
        default=None,
        help="Specific visualizations to run (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Do not append timestamp to output directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load and validate config
    cfg = load_config(Path(args.config))

    # Override with CLI args
    if args.visualizations:
        cfg.visualizations.enabled = args.visualizations
    if args.output_dir:
        cfg.output.output_dir = args.output_dir
    if args.no_timestamp:
        cfg.output.timestamp = False

    # Run comparison
    runner = ModelComparisonRunner(cfg)
    results = runner.run()

    # Exit with error if any visualization failed
    failed = [name for name, paths in results.items() if not paths]
    if failed:
        logger.warning(f"Some visualizations failed: {failed}")


if __name__ == "__main__":
    main()
