"""CLI entry point for the classification module.

Usage:
    python -m src.classification extract --config <path> [--experiment <name>] [--all]
    python -m src.classification extract-full --config <path> --experiment <name>
    python -m src.classification run --config <path> --experiment <name> [--input-mode joint] [--dithering] [--full-image]
    python -m src.classification run-all --config <path> [--dithering] [--full-image]
    python -m src.classification report --config <path>
    python -m src.classification plot --patches-dir <path> [--output-dir <path>] [--publication]
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="src.classification",
        description="Lesion quality classification: real vs. synthetic.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- extract ---
    p_extract = subparsers.add_parser("extract", help="Extract lesion-centered patches.")
    p_extract.add_argument("--config", required=True, help="Path to classification_task.yaml")
    p_extract.add_argument("--experiment", help="Single experiment name to extract")
    p_extract.add_argument("--all", action="store_true", help="Extract for all experiments")
    p_extract.add_argument("--skip-analysis", action="store_true",
                           help="Skip generating dataset analysis plots")

    # --- extract-full ---
    p_extract_full = subparsers.add_parser(
        "extract-full", help="Extract full 160x160 images (no patch cropping)."
    )
    p_extract_full.add_argument("--config", required=True, help="Path to classification_task.yaml")
    p_extract_full.add_argument("--experiment", help="Single experiment name to extract")
    p_extract_full.add_argument("--all", action="store_true", help="Extract for all experiments")
    p_extract_full.add_argument("--skip-analysis", action="store_true",
                                help="Skip generating dataset analysis plots")

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run k-fold classification for one experiment.")
    p_run.add_argument("--config", required=True, help="Path to classification_task.yaml")
    p_run.add_argument("--experiment", required=True, help="Experiment name")
    p_run.add_argument("--input-mode", default="joint", choices=["joint", "image_only", "mask_only"])
    p_run.add_argument("--folds", default=None, help="Comma-separated fold indices (default: all)")
    p_run.add_argument("--dithering", action="store_true",
                       help="Apply uniform dithering to synthetic data before classification")
    p_run.add_argument("--full-image", action="store_true",
                       help="Use full 160x160 images instead of patches (requires extract-full first)")

    # --- run-all ---
    p_all = subparsers.add_parser("run-all", help="Run all experiments and input modes.")
    p_all.add_argument("--config", required=True, help="Path to classification_task.yaml")
    p_all.add_argument("--include-control", action="store_true", help="Run real-vs-real control")
    p_all.add_argument("--dithering", action="store_true",
                       help="Apply uniform dithering to synthetic data before classification")
    p_all.add_argument("--full-image", action="store_true",
                       help="Use full 160x160 images instead of patches (requires extract-full first)")

    # --- report ---
    p_report = subparsers.add_parser("report", help="Generate comparison tables and figures.")
    p_report.add_argument("--config", required=True, help="Path to classification_task.yaml")
    p_report.add_argument("--format", default="latex", choices=["latex", "markdown", "csv"])

    # --- diagnose ---
    p_diag = subparsers.add_parser("diagnose", help="Run diagnostic analyses on real vs. synthetic.")
    p_diag.add_argument("--config", required=True, help="Path to diagnostics.yaml")
    p_diag.add_argument("--experiment", required=True, help="Experiment name")
    p_diag.add_argument(
        "--component", default="all",
        choices=["all", "dither", "gradcam", "spectral", "texture", "stats", "full-image", "report"],
    )
    p_diag.add_argument("--gpu", type=int, default=0, help="GPU device index")

    # --- plot ---
    p_plot = subparsers.add_parser(
        "plot",
        help="Generate analysis plots from extracted full images.",
    )
    p_plot.add_argument(
        "--patches-dir",
        type=str,
        required=True,
        help="Directory containing experiment subdirectories with patches.",
    )
    p_plot.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: patches_dir/analysis",
    )
    p_plot.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to include. Default: all found.",
    )
    p_plot.add_argument(
        "--publication",
        action="store_true",
        help="Generate publication-ready plots (IEEE style, Paul Tol colors).",
    )
    p_plot.add_argument(
        "--n-zbins",
        type=int,
        default=6,
        help="Number of representative z-bins (default: 6).",
    )
    p_plot.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["pdf", "png"],
        help="Output formats (default: pdf png).",
    )

    args = parser.parse_args()

    if args.command == "extract":
        from src.classification.scripts.extract_patches import run_extraction
        run_extraction(args)
    elif args.command == "extract-full":
        from src.classification.scripts.extract_full_images import run_full_extraction
        run_full_extraction(args)
    elif args.command == "run":
        from src.classification.scripts.run_experiment import run_experiment
        run_experiment(args)
    elif args.command == "run-all":
        from src.classification.scripts.run_all_experiments import run_all
        run_all(args)
    elif args.command == "report":
        from src.classification.scripts.run_all_experiments import generate_report
        generate_report(args)
    elif args.command == "diagnose":
        from src.classification.diagnostics.cli import run_from_args
        run_from_args(args)
    elif args.command == "plot":
        from pathlib import Path
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s | %(name)s | %(message)s",
        )

        patches_dir = Path(args.patches_dir)
        output_dir = Path(args.output_dir) if args.output_dir else patches_dir / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.publication:
            # Publication-ready plots (IEEE style)
            from src.classification.plotting.image_grid import plot_publication_image_grid

            plot_publication_image_grid(
                patches_dir=patches_dir,
                output_path=output_dir / "image_grid_publication",
                experiments=args.experiments,
                n_zbins=args.n_zbins,
                show_lesion=True,
                formats=args.formats,
            )
        else:
            # Standard analysis plots
            from src.classification.data.dataset_analysis import run_dataset_analysis

            run_dataset_analysis(
                patches_dir=patches_dir,
                output_dir=output_dir,
                experiments=args.experiments,
                dpi=150,
            )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
