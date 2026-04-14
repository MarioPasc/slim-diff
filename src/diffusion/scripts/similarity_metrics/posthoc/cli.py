"""Unified CLI for the TASK-05 post-hoc analyses.

Modes:

* ``--only tau_sensitivity`` — run the tau sweep over every cell.
* ``--only comparison``      — statistical comparison from ``fold_metrics.csv``.
* ``--only tables``          — emit LaTeX tables from available inputs.
* ``--only all`` (default)   — run all three sequentially.

Example (full run, Picasso-style paths)::

    python -m src.diffusion.scripts.similarity_metrics.posthoc.cli \\
        --fold-metrics /path/to/eval_output/fold_metrics.csv \\
        --results-root /path/to/results \\
        --cache-dir   /path/to/slice_cache \\
        --output-dir  /path/to/posthoc_output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .cross_fold_comparison import cross_fold_comparison
from .latex_tables import generate_all_tables
from .tau_sensitivity import DEFAULT_TAU_VALUES, run_tau_sensitivity_grid

logger = logging.getLogger(__name__)

VALID_MODES = ("tau_sensitivity", "comparison", "tables", "all")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slimdiff-posthoc",
        description=(
            "ICIP 2026 camera-ready post-hoc analyses "
            "(tau sensitivity + cross-fold comparison + LaTeX tables)."
        ),
    )
    p.add_argument(
        "--only",
        choices=VALID_MODES,
        default="all",
        help="Which analysis to run (default: all).",
    )
    p.add_argument("--fold-metrics", type=str, default=None,
                   help="Path to fold_metrics.csv from TASK-04.")
    p.add_argument("--results-root", type=str, default=None,
                   help="Parent of slimdiff_cr_* directories.")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Slice cache root (has folds/).")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Where to write posthoc artifacts.")
    p.add_argument("--cell-dir-template", type=str, default=None,
                   help="Override {results_root}/slimdiff_cr_{architecture}_fold_{fold}.")

    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--architectures", nargs="+",
                   default=["shared", "decoupled"])

    # tau_sensitivity knobs
    p.add_argument("--tau-values", type=float, nargs="+",
                   default=list(DEFAULT_TAU_VALUES),
                   help="Tau values to sweep (default: the 9-point sweep).")
    p.add_argument("--max-replicas", type=int, default=None)
    p.add_argument("--min-lesion-size-px", type=int, default=5)
    p.add_argument("--subset-size", type=int, default=500)
    p.add_argument("--num-subsets", type=int, default=100)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--no-normalize-features", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # comparison knobs
    p.add_argument("--early-stopping-csv", type=str, default=None,
                   help="Optional per-cell early-stopping epochs (CSV or JSON).")

    # tables knobs
    p.add_argument("--tau-csv", type=str, default=None,
                   help="Override the tau_sensitivity.csv path for table emission.")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _run_tau_sensitivity(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.results_root is None or args.cache_dir is None:
        logger.error(
            "--results-root and --cache-dir are required for tau_sensitivity.",
        )
        return None
    run_tau_sensitivity_grid(
        folds=list(range(int(args.n_folds))),
        architectures=list(args.architectures),
        cache_dir=Path(args.cache_dir),
        results_root=Path(args.results_root),
        output_dir=output_dir,
        tau_values=args.tau_values,
        max_replicas=args.max_replicas,
        cell_dir_template=args.cell_dir_template,
        min_lesion_size_px=args.min_lesion_size_px,
        subset_size=args.subset_size,
        num_subsets=args.num_subsets,
        degree=args.degree,
        normalize_features=not args.no_normalize_features,
        seed=args.seed,
    )
    return output_dir / "tau_sensitivity.csv"


def _run_comparison(args: argparse.Namespace, output_dir: Path) -> None:
    if args.fold_metrics is None:
        logger.error("--fold-metrics is required for comparison.")
        return
    cross_fold_comparison(
        fold_metrics_csv=Path(args.fold_metrics),
        output_json=output_dir / "cross_fold_comparison.json",
        early_stopping_csv=args.early_stopping_csv,
    )


def _run_tables(
    args: argparse.Namespace,
    output_dir: Path,
    tau_csv_override: Path | None = None,
) -> None:
    if args.fold_metrics is None:
        logger.error("--fold-metrics is required for tables.")
        return
    tau_csv = (
        Path(args.tau_csv) if args.tau_csv
        else tau_csv_override
        if tau_csv_override and tau_csv_override.exists()
        else None
    )
    generate_all_tables(
        fold_metrics_csv=Path(args.fold_metrics),
        output_dir=output_dir,
        tau_csv=tau_csv,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tau_csv_for_tables: Path | None = None

    if args.only in ("tau_sensitivity", "all"):
        tau_csv_for_tables = _run_tau_sensitivity(args, output_dir)

    if args.only in ("comparison", "all"):
        _run_comparison(args, output_dir)

    if args.only in ("tables", "all"):
        _run_tables(args, output_dir, tau_csv_override=tau_csv_for_tables)

    return 0


if __name__ == "__main__":
    sys.exit(main())
