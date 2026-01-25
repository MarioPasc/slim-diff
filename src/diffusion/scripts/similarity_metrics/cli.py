"""CLI for similarity metrics computation.

Entry point: jsddpm-similarity-metrics

Commands:
    compute-all: Full pipeline for ICIP 2026 experiments
    baseline: Compute real train-test baseline
    compare: Statistical comparison from existing CSV
    plot: Generate plots from existing CSV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_config(config_path: Path | str | None) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file (optional).

    Returns:
        Config dict (empty if no config).
    """
    if config_path is None:
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """Merge config file with CLI arguments (CLI takes precedence).

    Args:
        config: Config dict from YAML.
        args: Parsed CLI arguments.

    Returns:
        Merged config dict.
    """
    merged = {}

    # Paths
    paths = config.get("paths", {})
    merged["runs_dir"] = args.runs_dir or paths.get("runs_dir")
    merged["cache_dir"] = args.cache_dir or paths.get("cache_dir")
    merged["output_dir"] = args.output_dir or paths.get("output_dir")

    # Metrics config
    metrics_config = config.get("metrics", {})
    merged["kid"] = metrics_config.get("kid", {"enabled": True})
    merged["fid"] = metrics_config.get("fid", {"enabled": True})
    merged["lpips"] = metrics_config.get("lpips", {"enabled": True})

    # Compute config
    compute_config = config.get("compute", {})
    merged["device"] = getattr(args, "device", None) or compute_config.get("device", "cuda:0")
    merged["batch_size"] = getattr(args, "batch_size", None) or compute_config.get("batch_size", 32)

    # Plotting config
    merged["plotting"] = config.get("plotting", {})

    return merged


def cmd_compute_all(args: argparse.Namespace) -> int:
    """Run full metric computation pipeline.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from .run_icip2026 import run_full_pipeline

    # Load and merge config
    config = load_config(args.config)
    merged = merge_config_with_args(config, args)

    # Validate required paths
    if not merged["runs_dir"]:
        print("Error: --runs-dir is required (or set in config)")
        return 1
    if not merged["cache_dir"]:
        print("Error: --cache-dir is required (or set in config)")
        return 1
    if not merged["output_dir"]:
        print("Error: --output-dir is required (or set in config)")
        return 1

    # Parse metrics to compute
    metrics = args.metrics if args.metrics else ["kid", "fid", "lpips"]

    # Run pipeline
    try:
        run_full_pipeline(
            runs_dir=Path(merged["runs_dir"]),
            cache_dir=Path(merged["cache_dir"]),
            output_dir=Path(merged["output_dir"]),
            metrics=metrics,
            device=merged["device"],
            batch_size=merged["batch_size"],
            compute_per_zbin=not args.skip_zbin,
            n_lpips_pairs=args.n_lpips_pairs,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_baseline(args: argparse.Namespace) -> int:
    """Compute real vs real baseline metrics.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from .run_icip2026 import compute_baseline_metrics

    # Load config
    config = load_config(args.config)
    paths = config.get("paths", {})

    cache_dir = args.cache_dir or paths.get("cache_dir")
    output_dir = args.output_dir or paths.get("output_dir")

    if not cache_dir:
        print("Error: --cache-dir is required")
        return 1
    if not output_dir:
        print("Error: --output-dir is required")
        return 1

    try:
        compute_baseline_metrics(
            cache_dir=Path(cache_dir),
            output_dir=Path(output_dir),
            device=args.device,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Run statistical comparison on existing CSV.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    import pandas as pd
    from .statistics.comparison import run_all_comparisons, comparison_results_to_dataframe

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        print(f"Error: Input CSV not found: {input_csv}")
        return 1

    try:
        # Load metrics
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} rows from {input_csv}")

        # Run comparisons
        metrics = ["kid_global", "fid_global", "lpips_global"]
        available_metrics = [m for m in metrics if m in df.columns]

        if not available_metrics:
            print("Error: No valid metric columns found")
            return 1

        results = run_all_comparisons(df, metrics=available_metrics)

        # Convert to DataFrame and save
        comparison_df = comparison_results_to_dataframe(results)
        output_csv = output_dir / "similarity_metrics_comparison.csv"
        comparison_df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")

        # Also save detailed JSON
        output_json = output_dir / "similarity_metrics_comparison.json"
        # Convert to JSON-serializable format
        json_results = {}
        for metric, data in results.items():
            json_results[metric] = {
                "within_group": data["within_group"].to_dict(orient="records"),
                "between_group": data["between_group"],
            }
        with open(output_json, "w") as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"Saved: {output_json}")

        # Print summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        for metric in available_metrics:
            print(f"\n{metric}:")
            bg = results[metric]["between_group"]
            print(f"  Best prediction type: {bg['best_group']}")
            print(f"  Kruskal-Wallis p-value: {bg['p_value']:.4g}")
            if bg["significant"]:
                print("  Significant differences between prediction types!")

        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_plot(args: argparse.Namespace) -> int:
    """Generate plots from existing CSV files.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    import pandas as pd
    from .plotting.zbin_multiexp import plot_zbin_multiexperiment
    from .plotting.global_comparison import plot_global_comparison, plot_metric_summary_table

    # Load config for plotting settings
    config = load_config(args.config)
    plot_config = config.get("plotting", {})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = args.format.split(",") if args.format else plot_config.get("formats", ["png", "pdf"])

    try:
        # Load global metrics
        if args.global_csv:
            df_global = pd.read_csv(args.global_csv)
            print(f"Loaded {len(df_global)} rows from global CSV")

            # Generate global comparison plots
            for metric in ["kid_global", "fid_global", "lpips_global"]:
                if metric in df_global.columns:
                    plot_global_comparison(
                        df_global,
                        metric_col=metric,
                        output_dir=output_dir,
                        baseline_real=args.baseline_kid if "kid" in metric else None,
                        formats=formats,
                    )

            # Summary table
            plot_metric_summary_table(
                df_global,
                metrics=["kid_global", "fid_global", "lpips_global"],
                output_dir=output_dir,
                formats=formats,
            )

        # Load per-zbin metrics
        if args.zbin_csv:
            df_zbin = pd.read_csv(args.zbin_csv)
            print(f"Loaded {len(df_zbin)} rows from zbin CSV")

            # Generate per-zbin plots
            for metric in ["kid_zbin", "lpips_zbin"]:
                if metric in df_zbin.columns:
                    plot_zbin_multiexperiment(
                        df_zbin,
                        metric_col=metric,
                        output_dir=output_dir,
                        baseline_real=args.baseline_kid if "kid" in metric else None,
                        formats=formats,
                    )

        print(f"\nPlots saved to: {output_dir}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="jsddpm-similarity-metrics",
        description="Compute and analyze similarity metrics (KID, FID, LPIPS) for JSDDPM experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with config file
    jsddpm-similarity-metrics compute-all --config config/icip2026.yaml

    # Full pipeline with CLI args
    jsddpm-similarity-metrics compute-all \\
        --runs-dir /path/to/icip2026/runs \\
        --cache-dir /path/to/slice_cache \\
        --output-dir /path/to/output

    # Compute baseline only
    jsddpm-similarity-metrics baseline \\
        --cache-dir /path/to/slice_cache \\
        --output-dir /path/to/output

    # Generate plots from existing CSVs
    jsddpm-similarity-metrics plot \\
        --global-csv output/similarity_metrics_global.csv \\
        --zbin-csv output/similarity_metrics_zbin.csv \\
        --output-dir output/plots
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ===== compute-all =====
    p_all = subparsers.add_parser(
        "compute-all",
        help="Run full similarity metrics pipeline",
    )
    p_all.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML config file",
    )
    p_all.add_argument(
        "--runs-dir",
        type=str,
        help="Path to ICIP runs directory (overrides config)",
    )
    p_all.add_argument(
        "--cache-dir",
        type=str,
        help="Path to slice cache directory (overrides config)",
    )
    p_all.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (overrides config)",
    )
    p_all.add_argument(
        "--metrics",
        nargs="+",
        choices=["kid", "fid", "lpips"],
        help="Metrics to compute (default: all)",
    )
    p_all.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    p_all.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction (default: 32)",
    )
    p_all.add_argument(
        "--n-lpips-pairs",
        type=int,
        default=1000,
        help="Number of pairs for LPIPS computation (default: 1000)",
    )
    p_all.add_argument(
        "--skip-zbin",
        action="store_true",
        help="Skip per-zbin metric computation (faster)",
    )
    p_all.set_defaults(func=cmd_compute_all)

    # ===== baseline =====
    p_base = subparsers.add_parser(
        "baseline",
        help="Compute real train-test baseline metrics",
    )
    p_base.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML config file",
    )
    p_base.add_argument(
        "--cache-dir",
        type=str,
        help="Path to slice cache directory",
    )
    p_base.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results",
    )
    p_base.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    p_base.set_defaults(func=cmd_baseline)

    # ===== compare =====
    p_comp = subparsers.add_parser(
        "compare",
        help="Run statistical comparison on existing metrics CSV",
    )
    p_comp.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to similarity_metrics_global.csv",
    )
    p_comp.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for comparison results",
    )
    p_comp.set_defaults(func=cmd_compare)

    # ===== plot =====
    p_plot = subparsers.add_parser(
        "plot",
        help="Generate plots from existing CSV files",
    )
    p_plot.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML config file (for plotting settings)",
    )
    p_plot.add_argument(
        "--global-csv",
        type=str,
        help="Path to similarity_metrics_global.csv",
    )
    p_plot.add_argument(
        "--zbin-csv",
        type=str,
        help="Path to similarity_metrics_zbin.csv",
    )
    p_plot.add_argument(
        "--comparison-csv",
        type=str,
        help="Path to similarity_metrics_comparison.csv",
    )
    p_plot.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for plots",
    )
    p_plot.add_argument(
        "--format",
        type=str,
        default="png,pdf",
        help="Output formats (comma-separated, default: png,pdf)",
    )
    p_plot.add_argument(
        "--baseline-kid",
        type=float,
        help="Baseline KID value (real vs real) to show on plots",
    )
    p_plot.set_defaults(func=cmd_plot)

    # Parse and execute
    args = parser.parse_args()

    if hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
