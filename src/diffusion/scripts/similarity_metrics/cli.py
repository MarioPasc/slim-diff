"""CLI for similarity metrics computation.

Entry point: jsddpm-similarity-metrics

Commands:
    image-metrics: Image-to-image metrics (KID, FID, LPIPS)
    mask-metrics: Mask morphology metrics (MMD-MF)
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


def cmd_image_metrics(args: argparse.Namespace) -> int:
    """Run image-to-image metrics pipeline (KID, FID, LPIPS).

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


def cmd_mask_plot(args: argparse.Namespace) -> int:
    """Regenerate mask quality plots from saved CSV/JSON files.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    import json
    import pandas as pd
    from .plotting.mask_comparison import create_mask_quality_figure

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"

    # Load required files
    global_csv = input_dir / "mask_metrics_global.csv"
    wasserstein_csv = input_dir / "mask_wasserstein_features.csv"
    comparison_json = input_dir / "mask_metrics_comparison.json"

    if not global_csv.exists():
        print(f"Error: {global_csv} not found")
        return 1
    if not wasserstein_csv.exists():
        print(f"Error: {wasserstein_csv} not found")
        return 1

    print(f"Loading data from {input_dir}...")
    df_global = pd.read_csv(global_csv)
    df_wasserstein = pd.read_csv(wasserstein_csv)

    # Load comparison results (optional - for significance brackets)
    comparison_results = None
    if comparison_json.exists():
        with open(comparison_json) as f:
            comparison_data = json.load(f)
        if "mmd_mf_global" in comparison_data:
            comparison_results = comparison_data["mmd_mf_global"].get("between_group")
        print(f"Loaded comparison results from {comparison_json}")
    else:
        print(f"Warning: {comparison_json} not found, plots will lack significance brackets")

    # Determine output formats
    formats = args.format.split(",") if args.format else ["pdf", "png"]

    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating plots to {output_dir}...")

    try:
        create_mask_quality_figure(
            df_global=df_global,
            wasserstein_df=df_wasserstein,
            output_dir=output_dir,
            comparison_results=comparison_results,
            formats=formats,
        )
        print("Done!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_mask_metrics(args: argparse.Namespace) -> int:
    """Run mask morphology metrics (MMD-MF) only.

    Computes MMD on morphological features extracted from lesion masks,
    without recomputing KID/FID/LPIPS for images.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from .run_mask_metrics import main as run_mask_metrics_main, setup_logging

    # Default paths (same as in run_mask_metrics.py)
    DEFAULT_RUNS_DIR = "/media/mpascual/Sandisk2TB/research/jsddpm/results/epilepsy/icip2026/runs/self_cond_ablation"
    DEFAULT_CACHE_DIR = "/media/mpascual/Sandisk2TB/research/jsddpm/data/epilepsy/slice_cache"
    DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/jsddpm/results/epilepsy/icip2026/mask_metrics"

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Load config if provided
    config = load_config(args.config)
    paths = config.get("paths", {})

    # Determine paths (CLI args override config, config overrides defaults)
    runs_dir = args.runs_dir or paths.get("runs_dir") or DEFAULT_RUNS_DIR
    cache_dir = args.cache_dir or paths.get("cache_dir") or DEFAULT_CACHE_DIR
    output_dir = args.output_dir or paths.get("output_dir")

    # Use mask_metrics subdirectory if using config's output_dir
    if output_dir and not args.output_dir:
        output_dir = str(Path(output_dir).parent / "mask_metrics")
    elif not output_dir:
        output_dir = DEFAULT_OUTPUT_DIR

    # Get mask_morphology config
    mask_config = config.get("metrics", {}).get("mask_morphology", {})

    # Create args namespace for run_mask_metrics
    mask_args = argparse.Namespace(
        runs_dir=runs_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        self_cond_p=args.self_cond_p,
        min_lesion_size=args.min_lesion_size or mask_config.get("min_lesion_size_px", 5),
        subset_size=args.subset_size or mask_config.get("subset_size", 500),
        num_subsets=args.num_subsets or mask_config.get("num_subsets", 100),
        verbose=args.verbose,
    )

    return run_mask_metrics_main(mask_args)


def cmd_plot(args: argparse.Namespace) -> int:
    """Generate plots from existing CSV files or config.

    If only --config is provided, generates all plots using paths from config.
    If individual CSVs are provided, generates plots from those files.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    # Check if we should use config-only mode
    config = load_config(args.config) if args.config else {}

    # Config-only mode: generate everything from config
    if args.config and not args.global_csv and not args.zbin_csv:
        print("Using config-only mode: generating all plots from config paths")
        try:
            from .plotting.icip2026_figure import generate_plots_from_config

            generate_plots_from_config(
                config_path=args.config,
                output_subdir=args.output_subdir,
            )
            return 0
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Legacy mode: use explicit CSV paths
    import pandas as pd
    from .plotting.zbin_multiexp import plot_zbin_multiexperiment
    from .plotting.global_comparison import plot_global_comparison, plot_metric_summary_table

    plot_config = config.get("plotting", {})

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif config.get("paths", {}).get("output_dir"):
        output_dir = Path(config["paths"]["output_dir"]) / "plots"
    else:
        print("Error: --output-dir is required (or set output_dir in config)")
        return 1

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

            # Determine test_csv for images
            test_csv = None
            if args.test_csv:
                test_csv = Path(args.test_csv)
            elif config.get("paths", {}).get("cache_dir"):
                potential_test_csv = Path(config["paths"]["cache_dir"]) / "test.csv"
                if potential_test_csv.exists():
                    test_csv = potential_test_csv

            # Generate per-zbin plots
            for metric in ["kid_zbin", "lpips_zbin"]:
                if metric in df_zbin.columns:
                    plot_zbin_multiexperiment(
                        df_zbin,
                        metric_col=metric,
                        output_dir=output_dir,
                        baseline_real=args.baseline_kid if "kid" in metric else None,
                        test_csv=test_csv,
                        show_images=test_csv is not None,
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
        description="Compute and analyze similarity metrics (KID, FID, LPIPS, MMD-MF) for JSDDPM experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Image-to-image metrics (KID, FID, LPIPS)
    jsddpm-similarity-metrics image-metrics --config config/icip2026.yaml

    # Mask metrics only (MMD-MF) - fast, no GPU needed
    jsddpm-similarity-metrics mask-metrics --config config/pred_type_lp_norm.yaml

    # Mask metrics with CLI args
    jsddpm-similarity-metrics mask-metrics \\
        --runs-dir /path/to/runs \\
        --cache-dir /path/to/cache \\
        --self-cond-p 0.5

    # Compute baseline only
    jsddpm-similarity-metrics baseline \\
        --cache-dir /path/to/slice_cache \\
        --output-dir /path/to/output

    # Generate all plots from config (recommended)
    jsddpm-similarity-metrics plot --config config/icip2026.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ===== image-metrics =====
    p_all = subparsers.add_parser(
        "image-metrics",
        help="Compute image-to-image similarity metrics (KID, FID, LPIPS)",
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
    p_all.set_defaults(func=cmd_image_metrics)

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

    # ===== mask-plot =====
    p_mask_plot = subparsers.add_parser(
        "mask-plot",
        help="Regenerate mask quality plots from saved data",
        description="""
Regenerate mask quality plots from previously saved CSV/JSON files.
Use this to adjust plot appearance without recomputing metrics.

Required files in input-dir:
  - mask_metrics_global.csv
  - mask_wasserstein_features.csv
  - mask_metrics_comparison.json (optional, for significance brackets)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_mask_plot.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Directory containing saved mask metrics files",
    )
    p_mask_plot.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for plots (default: input-dir/plots)",
    )
    p_mask_plot.add_argument(
        "--format",
        type=str,
        default="pdf,png",
        help="Output formats, comma-separated (default: pdf,png)",
    )
    p_mask_plot.set_defaults(func=cmd_mask_plot)

    # ===== mask-metrics =====
    p_mask = subparsers.add_parser(
        "mask-metrics",
        help="Compute mask morphology metrics (MMD-MF) only",
        description="""
Compute MMD-MF (Maximum Mean Discrepancy on Morphological Features) for
evaluating generated lesion masks. This runs independently of KID/FID/LPIPS.

Answers the questions:
- Which prediction type yields masks most similar to real masks?
- Within each type, which Lp norm maximizes this similarity?
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_mask.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML config file",
    )
    p_mask.add_argument(
        "--runs-dir",
        type=str,
        help="Path to runs directory (overrides config)",
    )
    p_mask.add_argument(
        "--cache-dir",
        type=str,
        help="Path to cache directory (overrides config)",
    )
    p_mask.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (overrides config)",
    )
    p_mask.add_argument(
        "--self-cond-p",
        type=float,
        default=0.5,
        help="Self-conditioning probability to analyze (default: 0.5)",
    )
    p_mask.add_argument(
        "--min-lesion-size",
        type=int,
        help="Minimum lesion size in pixels (default: 5)",
    )
    p_mask.add_argument(
        "--subset-size",
        type=int,
        help="MMD subset size (default: 500)",
    )
    p_mask.add_argument(
        "--num-subsets",
        type=int,
        help="Number of MMD subsets (default: 100)",
    )
    p_mask.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    p_mask.set_defaults(func=cmd_mask_metrics)

    # ===== plot =====
    p_plot = subparsers.add_parser(
        "plot",
        help="Generate plots from config or existing CSV files",
        description="""
Generate ICIP 2026 publication-ready plots.

Two modes of operation:
1. Config-only mode: Just provide --config, plots are generated from paths in config
2. CSV mode: Provide explicit --global-csv and/or --zbin-csv paths

Examples:
    # Config-only mode (recommended)
    jsddpm-similarity-metrics plot --config config/icip2026.yaml

    # With custom output subdirectory
    jsddpm-similarity-metrics plot --config config/icip2026.yaml --output-subdir icip_figures

    # CSV mode (legacy)
    jsddpm-similarity-metrics plot --global-csv metrics.csv --output-dir plots/
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_plot.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML config file. If provided alone, generates all plots from config paths.",
    )
    p_plot.add_argument(
        "--global-csv",
        type=str,
        help="Path to similarity_metrics_global.csv (overrides config)",
    )
    p_plot.add_argument(
        "--zbin-csv",
        type=str,
        help="Path to similarity_metrics_zbin.csv (overrides config)",
    )
    p_plot.add_argument(
        "--test-csv",
        type=str,
        help="Path to test.csv for representative images (overrides config cache_dir)",
    )
    p_plot.add_argument(
        "--comparison-csv",
        type=str,
        help="Path to similarity_metrics_comparison.csv (overrides config)",
    )
    p_plot.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for plots (required if not using config-only mode)",
    )
    p_plot.add_argument(
        "--output-subdir",
        type=str,
        default="plots",
        help="Subdirectory within output_dir for plots (default: plots)",
    )
    p_plot.add_argument(
        "--format",
        type=str,
        help="Output formats (comma-separated, e.g., 'png,pdf'). Overrides config.",
    )
    p_plot.add_argument(
        "--baseline-kid",
        type=float,
        help="Baseline KID value (real vs real) to show on plots. Auto-loaded from config output_dir if not specified.",
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
