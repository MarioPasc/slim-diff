#!/usr/bin/env python
"""Standalone script for computing mask morphology metrics.

This script computes MMD-MF (Maximum Mean Discrepancy on Morphological Features)
for evaluating generated lesion masks, without recomputing KID/FID/LPIPS.

Usage:
    jsddpm-similarity-metrics mask-metrics --self-cond-p 0.5

    # Or as module:
    python -m src.diffusion.scripts.similarity_metrics.run_mask_metrics \
        --runs-dir /path/to/runs \
        --cache-dir /path/to/cache \
        --output-dir /path/to/output \
        --self-cond-p 0.5

The script expects folder structure:
    runs_dir/self_cond_p_{X}/{pred_type}_lp_{Y}/replicas/replica_*.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics.mask_morphology import MaskMorphologyDistanceComputer
from .plotting.mask_comparison import create_mask_quality_figure
from .statistics.comparison import run_all_comparisons, comparison_results_to_dataframe

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_real_masks(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load real masks from cache directory.

    Args:
        cache_dir: Path to cache dir with test.csv

    Returns:
        masks: (N, H, W) array in {-1, +1}
        zbins: (N,) array of z-bin indices
    """
    csv_path = cache_dir / "test.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"test.csv not found at {csv_path}")

    logger.info(f"Loading real masks from {csv_path}")
    df = pd.read_csv(csv_path)

    masks_list = []
    zbins_list = []
    skipped = 0

    zbin_col = "z_bin" if "z_bin" in df.columns else "zbin"

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading real masks"):
        filepath = cache_dir / row["filepath"]
        if not filepath.exists():
            skipped += 1
            continue

        data = np.load(filepath)
        masks_list.append(data["mask"].astype(np.float32))
        zbins_list.append(row[zbin_col])

    if skipped > 0:
        logger.warning(f"Skipped {skipped} missing files")

    logger.info(f"Loaded {len(masks_list)} real masks")
    return np.stack(masks_list), np.array(zbins_list, dtype=np.int32)


def discover_experiments(runs_dir: Path, self_cond_p: float) -> list[dict]:
    """Discover experiments for a given self_cond_p value.

    Args:
        runs_dir: Path to runs directory
        self_cond_p: Self-conditioning probability to filter by

    Returns:
        List of experiment dicts with keys: name, pred_type, lp_norm, path
    """
    # Look for self_cond_p_X directory
    sc_dir = runs_dir / f"self_cond_p_{self_cond_p}"
    if not sc_dir.exists():
        logger.warning(f"Directory not found: {sc_dir}")
        return []

    logger.info(f"Scanning {sc_dir} for experiments...")
    experiments = []
    for exp_dir in sorted(sc_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        # Parse folder name: {pred_type}_lp_{lp_norm}
        name = exp_dir.name
        if "_lp_" not in name:
            logger.debug(f"Skipping {name} (no _lp_ pattern)")
            continue

        parts = name.split("_lp_")
        pred_type = parts[0]
        try:
            lp_norm = float(parts[1])
        except ValueError:
            logger.debug(f"Skipping {name} (cannot parse lp_norm)")
            continue

        replicas_dir = exp_dir / "replicas"
        if not replicas_dir.exists():
            logger.debug(f"Skipping {name} (no replicas dir)")
            continue

        # Match replica_XXX.npz files (exclude *_meta.json)
        replica_files = sorted(
            f for f in replicas_dir.glob("replica_*.npz")
            if not f.name.endswith("_meta.json")
        )
        if not replica_files:
            logger.debug(f"Skipping {name} (no replica files)")
            continue

        experiments.append({
            "name": f"sc_{self_cond_p}__{pred_type}_lp_{lp_norm}",
            "prediction_type": pred_type,
            "lp_norm": lp_norm,
            "self_cond_p": self_cond_p,
            "path": exp_dir,
            "replica_files": replica_files,
        })
        logger.debug(f"Found experiment: {pred_type}_lp_{lp_norm} ({len(replica_files)} replicas)")

    return experiments


def load_replica_masks(replica_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load masks from a replica NPZ file.

    Args:
        replica_path: Path to replica_*.npz

    Returns:
        masks: (N, H, W) array
        zbins: (N,) array
    """
    data = np.load(replica_path)
    masks = data["masks"].astype(np.float32)
    zbins = data["zbin"].astype(np.int32)
    return masks, zbins


def run_mask_metrics(
    runs_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    self_cond_p: float = 0.5,
    min_lesion_size_px: int = 5,
    subset_size: int = 500,
    num_subsets: int = 100,
) -> dict:
    """Run mask morphology metrics pipeline.

    Args:
        runs_dir: Path to runs directory
        cache_dir: Path to cache directory
        output_dir: Output directory
        self_cond_p: Self-conditioning probability to analyze
        min_lesion_size_px: Minimum lesion size for feature extraction
        subset_size: MMD subset size
        num_subsets: Number of MMD subsets

    Returns:
        Dict with output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MASK MORPHOLOGY METRICS PIPELINE (MMD-MF)")
    logger.info("=" * 70)
    logger.info(f"Runs directory:    {runs_dir}")
    logger.info(f"Cache directory:   {cache_dir}")
    logger.info(f"Output directory:  {output_dir}")
    logger.info(f"Self-cond prob:    {self_cond_p}")
    logger.info(f"Min lesion size:   {min_lesion_size_px} px")
    logger.info(f"MMD subset size:   {subset_size}")
    logger.info(f"MMD num subsets:   {num_subsets}")
    logger.info("=" * 70)

    # Discover experiments
    logger.info(f"Discovering experiments for self_cond_p={self_cond_p}...")
    experiments = discover_experiments(runs_dir, self_cond_p)
    logger.info(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        logger.info(f"  - {exp['prediction_type']}_lp_{exp['lp_norm']} ({len(exp['replica_files'])} replicas)")

    if not experiments:
        logger.error("No experiments found! Check runs_dir and folder structure.")
        return {}

    # Load real masks
    logger.info("-" * 70)
    logger.info("PHASE 1: Loading real masks")
    logger.info("-" * 70)
    real_masks, real_zbins = load_real_masks(cache_dir)

    # Initialize MMD computer
    mmd_computer = MaskMorphologyDistanceComputer(
        min_lesion_size_px=min_lesion_size_px,
        subset_size=subset_size,
        num_subsets=num_subsets,
    )

    # Pre-extract real features once (expensive operation)
    logger.info("-" * 70)
    logger.info("PHASE 1b: Extracting real mask features (one-time)")
    logger.info("-" * 70)
    import time
    t0 = time.time()
    real_features = mmd_computer.get_real_features(real_masks, show_progress=True)
    logger.info(f"Extracted {real_features.shape[0]} lesion features from {len(real_masks)} masks")
    logger.info(f"Feature extraction took {time.time() - t0:.1f}s")

    # Compute metrics
    logger.info("-" * 70)
    logger.info("PHASE 2: Computing mask morphology metrics")
    logger.info("-" * 70)

    global_results = []
    wasserstein_results = []
    total_replicas = sum(len(exp["replica_files"]) for exp in experiments)
    processed = 0

    for exp in experiments:
        logger.info(f"Processing: {exp['prediction_type']}_lp_{exp['lp_norm']}")

        for replica_path in exp["replica_files"]:
            # Parse replica ID from filename (e.g., replica_000.npz -> 0)
            replica_id = int(replica_path.stem.split("_")[1])
            processed += 1

            logger.info(f"  [{processed}/{total_replicas}] Replica {replica_id}...")

            # Load synthetic masks
            t0 = time.time()
            synth_masks, synth_zbins = load_replica_masks(replica_path)
            logger.debug(f"    Loaded {len(synth_masks)} synthetic masks in {time.time()-t0:.1f}s")

            # Extract synthetic features (show progress since this takes ~1s per replica)
            t0 = time.time()
            synth_features = mmd_computer.extract_features(
                synth_masks, show_progress=True, desc=f"    Extracting features (replica {replica_id})"
            )
            logger.info(f"    Extracted {synth_features.shape[0]} synth lesions in {time.time()-t0:.1f}s")

            # Compute MMD-MF (reuse pre-computed features)
            t0 = time.time()
            mmd_result, _, _ = mmd_computer.compute(
                real_masks, synth_masks,
                show_progress=False,
                real_features=real_features,
                synth_features=synth_features,
            )
            logger.info(f"    MMD-MF = {mmd_result.value:.5f} +/- {mmd_result.std:.5f} ({time.time()-t0:.1f}s)")

            global_results.append({
                "experiment": exp["name"],
                "prediction_type": exp["prediction_type"],
                "lp_norm": exp["lp_norm"],
                "self_cond_p": exp["self_cond_p"],
                "replica_id": replica_id,
                "mmd_mf_global": mmd_result.value,
                "mmd_mf_global_std": mmd_result.std,
                "n_real_lesions": mmd_result.metadata.get("n_real_lesions", 0),
                "n_synth_lesions": mmd_result.metadata.get("n_synth_lesions", 0),
            })

            # Compute per-feature Wasserstein (reuse features - no re-extraction!)
            t0 = time.time()
            wasserstein_dists = mmd_computer.compute_per_feature_wasserstein(
                real_features=real_features,
                synth_features=synth_features,
            )
            geom_mean = wasserstein_dists.get("geometric_mean", float("nan"))
            logger.debug(f"    Wasserstein geom_mean = {geom_mean:.4f} ({time.time()-t0:.2f}s)")

            wasserstein_results.append({
                "experiment": exp["name"],
                "prediction_type": exp["prediction_type"],
                "lp_norm": exp["lp_norm"],
                "self_cond_p": exp["self_cond_p"],
                "replica_id": replica_id,
                **wasserstein_dists,
            })

    # Save results
    logger.info("-" * 70)
    logger.info("PHASE 3: Saving results")
    logger.info("-" * 70)

    df_global = pd.DataFrame(global_results)
    global_csv = output_dir / "mask_metrics_global.csv"
    df_global.to_csv(global_csv, index=False)
    logger.info(f"Saved: {global_csv}")

    df_wasserstein = pd.DataFrame(wasserstein_results)
    wasserstein_csv = output_dir / "mask_wasserstein_features.csv"
    df_wasserstein.to_csv(wasserstein_csv, index=False)
    logger.info(f"Saved: {wasserstein_csv}")

    # Statistical analysis
    logger.info("-" * 70)
    logger.info("PHASE 4: Statistical analysis")
    logger.info("-" * 70)

    comparison_results = run_all_comparisons(df_global, metrics=["mmd_mf_global"])

    # Save flattened comparison CSV
    comparison_df = comparison_results_to_dataframe(comparison_results)
    comparison_csv = output_dir / "mask_metrics_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info(f"Saved: {comparison_csv}")

    # Save full comparison results as JSON (needed for significance brackets in plots)
    comparison_json = output_dir / "mask_metrics_comparison.json"
    json_results = {}
    for metric, data in comparison_results.items():
        json_results[metric] = {
            "within_group": data["within_group"].to_dict(orient="records") if hasattr(data["within_group"], "to_dict") else data["within_group"],
            "between_group": {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in data["between_group"].items()
            },
        }
    with open(comparison_json, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    logger.info(f"Saved: {comparison_json}")

    # Save run metadata (for reproducibility)
    metadata = {
        "self_cond_p": self_cond_p,
        "min_lesion_size_px": min_lesion_size_px,
        "subset_size": subset_size,
        "num_subsets": num_subsets,
        "n_experiments": len(experiments),
        "n_replicas_total": total_replicas,
        "n_real_masks": len(real_masks),
        "n_real_lesions": int(real_features.shape[0]),
        "experiments": [
            {
                "name": exp["name"],
                "prediction_type": exp["prediction_type"],
                "lp_norm": exp["lp_norm"],
                "n_replicas": len(exp["replica_files"]),
            }
            for exp in experiments
        ],
    }
    metadata_json = output_dir / "mask_metrics_metadata.json"
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved: {metadata_json}")

    # Print summary
    if "mmd_mf_global" in comparison_results:
        bg = comparison_results["mmd_mf_global"]["between_group"]
        logger.info(f"Best prediction type: {bg['best_group']}")
        logger.info(f"Kruskal-Wallis p-value: {bg['p_value']:.4g}")
        if bg["significant"]:
            logger.info("*** SIGNIFICANT differences between prediction types ***")

    # Generate plots
    logger.info("-" * 70)
    logger.info("PHASE 5: Generating plots")
    logger.info("-" * 70)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    try:
        mask_comparison = None
        if "mmd_mf_global" in comparison_results:
            mask_comparison = comparison_results["mmd_mf_global"].get("between_group")

        create_mask_quality_figure(
            df_global=df_global,
            wasserstein_df=df_wasserstein,
            output_dir=plots_dir,
            comparison_results=mask_comparison,
            formats=["pdf", "png"],
        )
        logger.info("Mask quality figure generated!")
    except Exception as e:
        logger.error(f"Failed to create figure: {e}")
        import traceback
        traceback.print_exc()

    logger.info("=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output files:")
    logger.info(f"  Global metrics:       {global_csv}")
    logger.info(f"  Wasserstein features: {wasserstein_csv}")
    logger.info(f"  Comparison CSV:       {comparison_csv}")
    logger.info(f"  Comparison JSON:      {comparison_json}")
    logger.info(f"  Metadata:             {metadata_json}")
    logger.info(f"  Plots:                {plots_dir}")
    logger.info("")
    logger.info("To regenerate plots without recomputing:")
    logger.info(f"  jsddpm-similarity-metrics mask-plot --input-dir {output_dir}")

    return {
        "global_csv": str(global_csv),
        "wasserstein_csv": str(wasserstein_csv),
        "comparison_csv": str(comparison_csv),
        "comparison_json": str(comparison_json),
        "metadata_json": str(metadata_json),
        "plots_dir": str(plots_dir),
    }


def main(args: argparse.Namespace | None = None) -> int:
    """Main entry point.

    Args:
        args: Parsed arguments (if None, parses from command line).

    Returns:
        Exit code (0=success, 1=error).
    """
    if args is None:
        parser = argparse.ArgumentParser(
            description="Compute mask morphology metrics (MMD-MF) for JSDDPM experiments",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
    # Using CLI entry point
    jsddpm-similarity-metrics mask-metrics --self-cond-p 0.5

    # Using default paths (from config)
    python -m src.diffusion.scripts.similarity_metrics.run_mask_metrics

    # Custom paths
    python -m src.diffusion.scripts.similarity_metrics.run_mask_metrics \\
        --runs-dir /path/to/runs \\
        --cache-dir /path/to/cache \\
        --output-dir /path/to/output
            """,
        )
        parser.add_argument(
            "--runs-dir",
            type=str,
            default="/media/mpascual/Sandisk2TB/research/jsddpm/results/epilepsy/icip2026/runs/self_cond_ablation",
            help="Path to runs directory",
        )
        parser.add_argument(
            "--cache-dir",
            type=str,
            default="/media/mpascual/Sandisk2TB/research/jsddpm/data/epilepsy/slice_cache",
            help="Path to cache directory",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="/media/mpascual/Sandisk2TB/research/jsddpm/results/epilepsy/icip2026/mask_metrics",
            help="Output directory",
        )
        parser.add_argument(
            "--self-cond-p",
            type=float,
            default=0.5,
            help="Self-conditioning probability to analyze (default: 0.5)",
        )
        parser.add_argument(
            "--min-lesion-size",
            type=int,
            default=5,
            help="Minimum lesion size in pixels (default: 5)",
        )
        parser.add_argument(
            "--subset-size",
            type=int,
            default=500,
            help="MMD subset size (default: 500)",
        )
        parser.add_argument(
            "--num-subsets",
            type=int,
            default=100,
            help="Number of MMD subsets (default: 100)",
        )
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Enable verbose (DEBUG level) logging",
        )
        args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=getattr(args, "verbose", False))

    try:
        run_mask_metrics(
            runs_dir=Path(args.runs_dir),
            cache_dir=Path(args.cache_dir),
            output_dir=Path(args.output_dir),
            self_cond_p=args.self_cond_p,
            min_lesion_size_px=args.min_lesion_size,
            subset_size=args.subset_size,
            num_subsets=args.num_subsets,
        )
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
