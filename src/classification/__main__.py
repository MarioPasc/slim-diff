"""CLI entry point for the classification module.

Usage:
    python -m src.classification extract --config <path> [--experiment <name>] [--all]
    python -m src.classification run --config <path> --experiment <name> [--input-mode joint]
    python -m src.classification run-all --config <path>
    python -m src.classification report --config <path>
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

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run k-fold classification for one experiment.")
    p_run.add_argument("--config", required=True, help="Path to classification_task.yaml")
    p_run.add_argument("--experiment", required=True, help="Experiment name")
    p_run.add_argument("--input-mode", default="joint", choices=["joint", "image_only", "mask_only"])
    p_run.add_argument("--folds", default=None, help="Comma-separated fold indices (default: all)")

    # --- run-all ---
    p_all = subparsers.add_parser("run-all", help="Run all experiments and input modes.")
    p_all.add_argument("--config", required=True, help="Path to classification_task.yaml")
    p_all.add_argument("--include-control", action="store_true", help="Run real-vs-real control")

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

    args = parser.parse_args()

    if args.command == "extract":
        from src.classification.scripts.extract_patches import run_extraction
        run_extraction(args)
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
