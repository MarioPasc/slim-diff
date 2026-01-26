"""End-to-end orchestration for ICIP 2026 similarity metrics pipeline.

Executes the full pipeline:
1. Load real data
2. Compute metrics per experiment (KID, FID, LPIPS)
3. Compute baseline (real vs real)
4. Compute per-zbin metrics
5. Statistical analysis
6. Generate plots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data.loaders import ICIPExperimentLoader
from src.shared.ablation import AblationSpace
from .metrics.kid import KIDComputer, compute_per_zbin_kid
from .metrics.fid import FIDComputer
from .metrics.lpips import LPIPSComputer, compute_per_zbin_lpips
from .statistics.comparison import run_all_comparisons, comparison_results_to_dataframe
from .plotting.zbin_multiexp import plot_zbin_multiexperiment
from .plotting.global_comparison import plot_global_comparison, plot_metric_summary_table
from .plotting.icip2026_figure import create_icip2026_figure, create_compact_figure


def run_full_pipeline(
    runs_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    metrics: list[str] = ["kid", "fid", "lpips"],
    device: str = "cuda:0",
    batch_size: int = 32,
    compute_per_zbin: bool = True,
    n_lpips_pairs: int = 1000,
    test_csv: Path | None = None,
    create_publication_figure: bool = True,
) -> dict[str, Any]:
    """Execute full ICIP 2026 similarity metrics pipeline.

    Args:
        runs_dir: Path to ICIP runs directory containing experiment folders.
        cache_dir: Path to slice cache directory with test.csv.
        output_dir: Output directory for results.
        metrics: List of metrics to compute ("kid", "fid", "lpips").
        device: Device to use for computation.
        batch_size: Batch size for feature extraction.
        compute_per_zbin: Whether to compute per-zbin metrics.
        n_lpips_pairs: Number of pairs for LPIPS computation.
        test_csv: Path to test.csv for representative images (default: cache_dir/test.csv).
        create_publication_figure: Whether to create the ICIP 2026 publication figure.

    Returns:
        Dict with paths to output files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default test_csv path
    if test_csv is None:
        test_csv = Path(cache_dir) / "test.csv"
    else:
        test_csv = Path(test_csv)

    print("=" * 70)
    print("ICIP 2026 SIMILARITY METRICS PIPELINE")
    print("=" * 70)
    print(f"Runs directory: {runs_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Test CSV: {test_csv}")
    print(f"Metrics: {metrics}")
    print(f"Device: {device}")
    print("=" * 70)

    # Initialize experiment loader
    loader = ICIPExperimentLoader(runs_dir, cache_dir)
    print(f"\nDiscovered {len(loader.experiments)} experiments:")
    for coord in loader.experiments:
        print(f"  - {coord.to_display_name()}")

    # Print experiment summary
    summary_df = loader.get_experiment_summary()
    print("\nExperiment summary:")
    print(summary_df.to_string(index=False))

    # ===== PHASE 1: Load real data =====
    print("\n" + "=" * 70)
    print("PHASE 1: Loading real data")
    print("=" * 70)

    real_images, real_zbins = loader.load_real_data(splits=["test"])
    print(f"Loaded {len(real_images)} real test images")

    # ===== PHASE 2: Compute metrics per experiment =====
    print("\n" + "=" * 70)
    print("PHASE 2: Computing metrics per experiment")
    print("=" * 70)

    # Initialize metric computers
    kid_computer = KIDComputer(device=device, batch_size=batch_size) if "kid" in metrics else None
    fid_computer = FIDComputer(device=device, batch_size=batch_size) if "fid" in metrics else None
    lpips_computer = LPIPSComputer(device=device, batch_size=batch_size) if "lpips" in metrics else None

    global_results = []

    for coord in tqdm(
        list(loader.iter_experiments()),
        desc="Experiments",
    ):
        exp_name = coord.to_display_name()
        print(f"\n--- {exp_name} ---")

        replica_paths = loader.get_replica_paths(coord)
        print(f"  Found {len(replica_paths)} replicas")

        for replica_path in replica_paths:
            replica_id = int(replica_path.stem.split("_")[-1])
            print(f"  Replica {replica_id}...", end=" ", flush=True)

            # Load replica
            synth_images, synth_zbins = loader.load_replica(coord, replica_id)

            result_row = {
                "experiment": exp_name,
                "prediction_type": coord.prediction_type,
                "lp_norm": coord.lp_norm,
                "self_cond_p": coord.self_cond_p,
                "replica_id": replica_id,
                "n_real": len(real_images),
                "n_synth": len(synth_images),
            }

            # Compute KID
            if kid_computer is not None:
                kid_result = kid_computer.compute(real_images, synth_images, show_progress=False)
                result_row["kid_global"] = kid_result.value
                result_row["kid_global_std"] = kid_result.std
                print(f"KID={kid_result.value:.5f}", end=" ", flush=True)

            # Compute FID
            if fid_computer is not None:
                fid_result = fid_computer.compute(real_images, synth_images, show_progress=False)
                result_row["fid_global"] = fid_result.value
                print(f"FID={fid_result.value:.2f}", end=" ", flush=True)

            # Compute LPIPS
            if lpips_computer is not None:
                lpips_result = lpips_computer.compute_pairwise(
                    real_images, synth_images,
                    n_pairs=n_lpips_pairs,
                    show_progress=False,
                )
                result_row["lpips_global"] = lpips_result.value
                result_row["lpips_global_std"] = lpips_result.std
                print(f"LPIPS={lpips_result.value:.4f}", end=" ", flush=True)

            print()  # Newline
            global_results.append(result_row)

            # Clear GPU memory
            torch.cuda.empty_cache()

    # Save global results
    df_global = pd.DataFrame(global_results)
    global_csv_path = output_dir / "similarity_metrics_global.csv"
    df_global.to_csv(global_csv_path, index=False)
    print(f"\nSaved global metrics: {global_csv_path}")

    # ===== PHASE 3: Compute baseline =====
    print("\n" + "=" * 70)
    print("PHASE 3: Computing baseline (real vs real)")
    print("=" * 70)

    baseline_results = compute_baseline_metrics(
        cache_dir=cache_dir,
        output_dir=output_dir,
        device=device,
        metrics=metrics,
    )

    # ===== PHASE 4: Compute per-zbin metrics =====
    zbin_results_list = []
    if compute_per_zbin:
        print("\n" + "=" * 70)
        print("PHASE 4: Computing per-zbin metrics")
        print("=" * 70)

        for coord in tqdm(
            list(loader.iter_experiments()),
            desc="Per-zbin metrics",
        ):
            exp_name = coord.to_display_name()
            print(f"\n--- {exp_name} ---")

            # Load and merge all replicas for this experiment
            synth_images, synth_zbins, _ = loader.load_all_replicas(coord)
            print(f"  Merged {len(synth_images)} synthetic samples")

            # Per-zbin KID
            if "kid" in metrics:
                print("  Computing per-zbin KID...")
                kid_zbin_results = compute_per_zbin_kid(
                    real_images, real_zbins,
                    synth_images, synth_zbins,
                    subset_size=250,  # Smaller for per-zbin
                    num_subsets=50,
                    device=device,
                    batch_size=batch_size,
                )

                for row in kid_zbin_results:
                    row.update({
                        "experiment": exp_name,
                        "prediction_type": coord.prediction_type,
                        "lp_norm": coord.lp_norm,
                        "self_cond_p": coord.self_cond_p,
                        "kid_zbin": row.pop("kid"),
                        "kid_zbin_std": row.pop("kid_std"),
                    })
                    zbin_results_list.append(row)

            # Per-zbin LPIPS
            if "lpips" in metrics:
                print("  Computing per-zbin LPIPS...")
                lpips_zbin_results = compute_per_zbin_lpips(
                    real_images, real_zbins,
                    synth_images, synth_zbins,
                    n_pairs_per_zbin=100,
                    device=device,
                    batch_size=batch_size,
                )

                # Merge with KID results or create new
                for lpips_row in lpips_zbin_results:
                    zbin = lpips_row["zbin"]
                    # Find matching row
                    for row in zbin_results_list:
                        if (row["experiment"] == exp_name and row["zbin"] == zbin):
                            row["lpips_zbin"] = lpips_row["lpips"]
                            row["lpips_zbin_std"] = lpips_row["lpips_std"]
                            break

            torch.cuda.empty_cache()

        # Save per-zbin results
        if zbin_results_list:
            df_zbin = pd.DataFrame(zbin_results_list)
            zbin_csv_path = output_dir / "similarity_metrics_zbin.csv"
            df_zbin.to_csv(zbin_csv_path, index=False)
            print(f"\nSaved per-zbin metrics: {zbin_csv_path}")

    # ===== PHASE 5: Statistical analysis =====
    print("\n" + "=" * 70)
    print("PHASE 5: Statistical analysis")
    print("=" * 70)

    available_metrics = [f"{m}_global" for m in metrics if f"{m}_global" in df_global.columns]
    if available_metrics:
        comparison_results = run_all_comparisons(df_global, metrics=available_metrics)

        # Convert to DataFrame and save
        comparison_df = comparison_results_to_dataframe(comparison_results)
        comparison_csv_path = output_dir / "similarity_metrics_comparison.csv"
        comparison_df.to_csv(comparison_csv_path, index=False)
        print(f"Saved comparison results: {comparison_csv_path}")

        # Print summary
        for metric in available_metrics:
            print(f"\n{metric}:")
            bg = comparison_results[metric]["between_group"]
            print(f"  Best prediction type: {bg['best_group']}")
            print(f"  Kruskal-Wallis p-value: {bg['p_value']:.4g}")
            if bg["significant"]:
                print("  *** Significant differences between prediction types ***")

    # ===== PHASE 6: Generate plots =====
    print("\n" + "=" * 70)
    print("PHASE 6: Generating plots")
    print("=" * 70)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Global comparison plots
    baseline_kid = baseline_results.get("kid", {}).get("mean")
    for metric in available_metrics:
        try:
            comparison_data = comparison_results.get(metric, {}).get("between_group")
            plot_global_comparison(
                df_global,
                metric_col=metric,
                output_dir=plots_dir,
                comparison_results=comparison_data,
                baseline_real=baseline_kid if "kid" in metric else None,
            )
        except Exception as e:
            print(f"Warning: Failed to create global plot for {metric}: {e}")

    # Per-zbin plots
    if compute_per_zbin and zbin_results_list:
        df_zbin = pd.DataFrame(zbin_results_list)
        for metric in ["kid_zbin", "lpips_zbin"]:
            if metric in df_zbin.columns:
                try:
                    plot_zbin_multiexperiment(
                        df_zbin,
                        metric_col=metric,
                        output_dir=plots_dir,
                        baseline_real=baseline_kid if "kid" in metric else None,
                    )
                except Exception as e:
                    print(f"Warning: Failed to create zbin plot for {metric}: {e}")

    # Summary table
    try:
        plot_metric_summary_table(
            df_global,
            metrics=available_metrics,
            output_dir=plots_dir,
        )
    except Exception as e:
        print(f"Warning: Failed to create summary table: {e}")

    # ICIP 2026 Publication Figure (2x2 layout)
    if create_publication_figure and compute_per_zbin and zbin_results_list:
        print("\nGenerating ICIP 2026 publication figure...")
        try:
            # Extract baseline values
            baseline_kid_mean = baseline_results.get("kid", {}).get("mean")
            baseline_kid_std = baseline_results.get("kid", {}).get("std")
            baseline_lpips_mean = baseline_results.get("lpips", {}).get("mean")
            baseline_lpips_std = baseline_results.get("lpips", {}).get("std")

            # Create main 2x2 figure
            create_icip2026_figure(
                df_global=df_global,
                df_zbin=df_zbin,
                output_dir=plots_dir,
                test_csv=test_csv if test_csv.exists() else None,
                comparison_results=comparison_results if available_metrics else None,
                baseline_kid=baseline_kid_mean,
                baseline_kid_std=baseline_kid_std,
                baseline_lpips=baseline_lpips_mean,
                baseline_lpips_std=baseline_lpips_std,
                formats=["pdf", "png"],
                show_images=test_csv.exists(),
            )

            # Create compact single-column figure
            create_compact_figure(
                df_global=df_global,
                df_zbin=df_zbin,
                output_dir=plots_dir,
                test_csv=test_csv if test_csv.exists() else None,
                comparison_results=comparison_results if available_metrics else None,
                baseline_kid=baseline_kid_mean,
                baseline_lpips=baseline_lpips_mean,
                formats=["pdf", "png"],
            )
            print("Publication figures generated successfully!")
        except Exception as e:
            print(f"Warning: Failed to create publication figure: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Global metrics: {global_csv_path}")
    if compute_per_zbin and zbin_results_list:
        print(f"  Per-zbin metrics: {zbin_csv_path}")
    print(f"  Comparison results: {comparison_csv_path}")
    print(f"  Plots: {plots_dir}")

    return {
        "global_csv": str(global_csv_path),
        "zbin_csv": str(output_dir / "similarity_metrics_zbin.csv") if compute_per_zbin else None,
        "comparison_csv": str(comparison_csv_path),
        "plots_dir": str(plots_dir),
        "baseline": baseline_results,
    }


def compute_baseline_metrics(
    cache_dir: Path,
    output_dir: Path,
    device: str = "cuda:0",
    metrics: list[str] = ["kid", "lpips"],
) -> dict[str, Any]:
    """Compute baseline metrics (real train vs real test).

    Args:
        cache_dir: Path to slice cache directory.
        output_dir: Output directory.
        device: Device to use.
        metrics: Metrics to compute.

    Returns:
        Dict with baseline values per metric.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary loader just for data loading
    loader = ICIPExperimentLoader(
        runs_dir=output_dir,  # Dummy, not used
        cache_dir=cache_dir,
    )

    # Load train and test data
    print("Loading train data...")
    train_images, train_zbins = loader.load_real_data(splits=["train"])
    print(f"Loaded {len(train_images)} train images")

    print("Loading test data...")
    test_images, test_zbins = loader.load_real_data(splits=["test"])
    print(f"Loaded {len(test_images)} test images")

    baseline_results = {}

    # Compute KID baseline
    if "kid" in metrics:
        print("\nComputing KID baseline (train vs test)...")
        kid_computer = KIDComputer(device=device)
        kid_result = kid_computer.compute(train_images, test_images, show_progress=True)
        baseline_results["kid"] = {
            "mean": kid_result.value,
            "std": kid_result.std,
            "n_train": len(train_images),
            "n_test": len(test_images),
        }
        print(f"  KID baseline: {kid_result.value:.6f} ± {kid_result.std:.6f}")
        torch.cuda.empty_cache()

    # Compute LPIPS baseline
    if "lpips" in metrics:
        print("\nComputing LPIPS baseline (train vs test)...")
        lpips_computer = LPIPSComputer(device=device)
        lpips_result = lpips_computer.compute_pairwise(
            train_images, test_images,
            n_pairs=1000,
            show_progress=True,
        )
        baseline_results["lpips"] = {
            "mean": lpips_result.value,
            "std": lpips_result.std,
            "n_train": len(train_images),
            "n_test": len(test_images),
        }
        print(f"  LPIPS baseline: {lpips_result.value:.6f} ± {lpips_result.std:.6f}")
        lpips_computer.clear_model()
        torch.cuda.empty_cache()

    # Save baseline results
    baseline_csv_path = output_dir / "baseline_real_vs_real.csv"
    baseline_rows = []
    for metric, data in baseline_results.items():
        baseline_rows.append({
            "metric": metric,
            "mean": data["mean"],
            "std": data["std"],
            "n_train": data["n_train"],
            "n_test": data["n_test"],
        })
    pd.DataFrame(baseline_rows).to_csv(baseline_csv_path, index=False)
    print(f"\nSaved baseline results: {baseline_csv_path}")

    # Also save as JSON
    baseline_json_path = output_dir / "baseline_real_vs_real.json"
    with open(baseline_json_path, "w") as f:
        json.dump(baseline_results, f, indent=2)

    return baseline_results
