"""CLI for computing global and per-zbin KID metrics with statistical analysis.

This module orchestrates KID computation across all replicas, performs statistical
tests (Wilcoxon signed-rank + FDR correction), and generates CSV outputs.

Supports two modes:
1. Test-only mode (original): Compare replicas against test set distribution
2. Multi-split mode: Compare replicas against union of train/val/test splits
   (useful for uniformly-sampled synthetic replicas)

Usage (test-only, backwards compatible):
    python -m src.diffusion.scripts.kid.zbin_global_kid \
        --replicas-dir /path/to/replicas \
        --test-slices-csv /path/to/slice_cache/test.csv \
        --test-dist-csv docs/test_analysis/test_zbin_distribution.csv \
        --output-dir /path/to/quality_report \
        --subset-size 1000 \
        --num-subsets 100 \
        --degree 3 \
        --batch-size 32 \
        --device cuda \
        --merge-replicas 5

Usage (multi-split, for uniform replicas):
    python -m src.diffusion.scripts.kid.zbin_global_kid \
        --replicas-dir /path/to/replicas \
        --train-slices-csv /path/to/slice_cache/train.csv \
        --train-dist-csv docs/train_analysis/train_zbin_distribution.csv \
        --val-slices-csv /path/to/slice_cache/val.csv \
        --val-dist-csv docs/val_analysis/val_zbin_distribution.csv \
        --test-slices-csv /path/to/slice_cache/test.csv \
        --test-dist-csv docs/test_analysis/test_zbin_distribution.csv \
        --output-dir /path/to/quality_report \
        --subset-size 1000 \
        --num-subsets 100 \
        --merge-replicas 5

    python -m src.diffusion.scripts.kid.zbin_global_kid \
        --replicas-dir /media/hddb/mario/data/epilepsy/synth/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/replicas \
        --train-slices-csv /media/hddb/mario/data/epilepsy/real/slice_cache/train.csv \
        --train-dist-csv /home/mariopascual/projects/diffusion/js-ddpm-epilepsy/docs/train_analysis/train_zbin_distribution.csv \
        --val-slices-csv /media/hddb/mario/data/epilepsy/real/slice_cache/slice_cache/val.csv \
        --val-dist-csv /home/mariopascual/projects/diffusion/js-ddpm-epilepsy/docs/val_analysis/val_zbin_distribution.csv \
        --output-dir /media/hddb/mario/data/epilepsy/synth/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/kid_quality \
        --subset-size 1000 \
        --num-subsets 100 \
        --merge-replicas 1
        --test-slices-csv /media/hddb/mario/data/epilepsy/real/slice_cache/slice_cache/test.csv \
        --test-dist-csv /home/mariopascual/projects/diffusion/js-ddpm-epilepsy/docs/test_analysis/test_zbin_distribution.csv \

    The real image set will be the union of all provided splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .kid import compute_global_kid, compute_zbin_kid, load_replica, load_test_slices


def load_multi_split_slices(
    split_configs: list[tuple[Path, str]],
    valid_zbins: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and merge images from multiple data splits.

    Args:
        split_configs: List of (slices_csv_path, split_name) tuples.
            Each CSV should have columns: filepath, z_bin, etc.
        valid_zbins: List of valid z-bins to include (default: 0-29).

    Returns:
        images: (N, H, W) float32 array in [-1, 1] (merged from all splits)
        zbins: (N,) int array (merged from all splits)
        splits_used: List of split names that were loaded
    """
    if valid_zbins is None:
        valid_zbins = list(range(30))

    all_images = []
    all_zbins = []
    splits_used = []

    for csv_path, split_name in split_configs:
        print(f"  Loading {split_name} split from {csv_path}...")
        images, zbins = load_test_slices(csv_path, valid_zbins=valid_zbins)
        all_images.append(images)
        all_zbins.append(zbins)
        splits_used.append(split_name)
        print(f"    -> {len(images)} slices from {split_name}")

    # Concatenate all splits
    merged_images = np.concatenate(all_images, axis=0)
    merged_zbins = np.concatenate(all_zbins, axis=0)

    print(f"  Total: {len(merged_images)} slices from {len(splits_used)} splits ({', '.join(splits_used)})")

    return merged_images, merged_zbins, splits_used


def compute_all_replicas_global(
    replicas_dir: Path,
    test_images: np.ndarray,
    subset_size: int,
    num_subsets: int,
    degree: int,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """Compute global KID for all replicas.

    Args:
        replicas_dir: Directory containing replica_{ID:03d}.npz files
        test_images: (N, H, W) test images
        subset_size: KID subset size
        num_subsets: Number of KID subsets
        degree: Polynomial kernel degree
        batch_size: Batch size for feature extraction
        device: cuda or cpu

    Returns:
        DataFrame with columns:
            - replica_id
            - kid_global
            - kid_global_std
            - n_real
            - n_synth
            - feature_extractor
            - feature_layer
            - kid_kernel
            - subset_size
            - num_subsets
    """
    # Find all replica files
    replica_files = sorted(replicas_dir.glob("replica_*.npz"))
    print(f"\nFound {len(replica_files)} replica files")

    results = []

    for replica_file in tqdm(replica_files, desc="Computing global KID"):
        # Extract replica_id from filename
        replica_id = int(replica_file.stem.split("_")[-1])

        # Load replica
        synth_images, _ = load_replica(replica_file)

        # Compute global KID
        kid_result = compute_global_kid(
            test_images,
            synth_images,
            subset_size=subset_size,
            num_subsets=num_subsets,
            degree=degree,
            batch_size=batch_size,
            device=device,
        )

        # Store results
        results.append(
            {
                "replica_id": replica_id,
                "kid_global": kid_result["kid_mean"],
                "kid_global_std": kid_result["kid_std"],
                "n_real": kid_result["n_real"],
                "n_synth": kid_result["n_synth"],
                "feature_extractor": "inceptionv3",
                "feature_layer": "2048",
                "kid_kernel": f"polynomial_d{degree}",
                "subset_size": subset_size,
                "num_subsets": num_subsets,
            }
        )

        print(
            f"[{replica_id+1:2d}/{len(replica_files)}] "
            f"Replica {replica_id}: Global KID = {kid_result['kid_mean']:.6f} "
            f"± {kid_result['kid_std']:.6f}"
        )

    return pd.DataFrame(results)


def compute_all_replicas_zbin_merged(
    replicas_dir: Path,
    test_images: np.ndarray,
    test_zbins: np.ndarray,
    valid_zbins: list[int],
    merge_replicas: int,
    subset_size: int,
    num_subsets: int,
    degree: int,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """Compute per-zbin KID for replica groups (merged replicas).

    Args:
        replicas_dir: Directory containing replica_{ID:03d}.npz files
        test_images: (N, H, W) test images
        test_zbins: (N,) test z-bins
        valid_zbins: List of z-bins to process (0-29)
        merge_replicas: Number of replicas to merge per group
        subset_size: KID subset size
        num_subsets: Number of KID subsets
        degree: Polynomial kernel degree
        batch_size: Batch size for feature extraction
        device: cuda or cpu

    Returns:
        DataFrame with columns:
            - replica_group_id
            - replica_ids (comma-separated list)
            - zbin
            - kid_zbin
            - kid_zbin_std
            - n_real_zbin
            - n_synth_zbin
            - kid_rest
            - kid_rest_std
            - n_real_rest
            - n_synth_rest
    """
    # Find all replica files
    replica_files = sorted(replicas_dir.glob("replica_*.npz"))
    n_replicas = len(replica_files)

    # Create groups of replicas
    groups = []
    for i in range(0, n_replicas, merge_replicas):
        group = replica_files[i : i + merge_replicas]
        groups.append(group)

    print(f"\nMerging {n_replicas} replicas into {len(groups)} groups of ~{merge_replicas} replicas each")

    results = []

    for group_id, replica_group in enumerate(tqdm(groups, desc="Processing replica groups")):
        # Extract replica IDs
        replica_ids = [int(rf.stem.split("_")[-1]) for rf in replica_group]
        replica_ids_str = ",".join(map(str, replica_ids))

        # Load and merge all replicas in this group
        all_synth_images = []
        all_synth_zbins = []

        for replica_file in replica_group:
            synth_images, synth_zbins = load_replica(replica_file)
            all_synth_images.append(synth_images)
            all_synth_zbins.append(synth_zbins)

        # Concatenate all images and zbins
        merged_synth_images = np.concatenate(all_synth_images, axis=0)
        merged_synth_zbins = np.concatenate(all_synth_zbins, axis=0)

        print(
            f"  Group {group_id} (replicas {replica_ids_str}): "
            f"{len(merged_synth_images)} merged samples"
        )

        # Process each z-bin
        for zbin in tqdm(valid_zbins, desc=f"Group {group_id} z-bins", leave=False):
            # Compute KID for this bin vs rest
            kid_result = compute_zbin_kid(
                test_images,
                test_zbins,
                merged_synth_images,
                merged_synth_zbins,
                target_zbin=zbin,
                subset_size=subset_size,
                num_subsets=num_subsets,
                degree=degree,
                batch_size=batch_size,
                device=device,
            )

            # Store results
            results.append(
                {
                    "replica_group_id": group_id,
                    "replica_ids": replica_ids_str,
                    "zbin": zbin,
                    "kid_zbin": kid_result["kid_zbin_mean"],
                    "kid_zbin_std": kid_result["kid_zbin_std"],
                    "n_real_zbin": kid_result["n_real_zbin"],
                    "n_synth_zbin": kid_result["n_synth_zbin"],
                    "kid_rest": kid_result["kid_rest_mean"],
                    "kid_rest_std": kid_result["kid_rest_std"],
                    "n_real_rest": kid_result["n_real_rest"],
                    "n_synth_rest": kid_result["n_synth_rest"],
                }
            )

        print(f"  Group {group_id}: Processed {len(valid_zbins)} z-bins")

    return pd.DataFrame(results)


def compute_all_replicas_zbin(
    replicas_dir: Path,
    test_images: np.ndarray,
    test_zbins: np.ndarray,
    valid_zbins: list[int],
    subset_size: int,
    num_subsets: int,
    degree: int,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """Compute per-zbin KID for all replicas.

    Args:
        replicas_dir: Directory containing replica_{ID:03d}.npz files
        test_images: (N, H, W) test images
        test_zbins: (N,) test z-bins
        valid_zbins: List of z-bins to process (0-29)
        subset_size: KID subset size
        num_subsets: Number of KID subsets
        degree: Polynomial kernel degree
        batch_size: Batch size for feature extraction
        device: cuda or cpu

    Returns:
        DataFrame with columns:
            - replica_id
            - zbin
            - kid_zbin
            - kid_zbin_std
            - n_real_zbin
            - n_synth_zbin
            - kid_rest
            - kid_rest_std
            - n_real_rest
            - n_synth_rest
    """
    # Find all replica files
    replica_files = sorted(replicas_dir.glob("replica_*.npz"))

    results = []

    for replica_file in tqdm(replica_files, desc="Computing per-zbin KID"):
        # Extract replica_id from filename
        replica_id = int(replica_file.stem.split("_")[-1])

        # Load replica
        synth_images, synth_zbins = load_replica(replica_file)

        # Process each z-bin
        for zbin in tqdm(valid_zbins, desc=f"Replica {replica_id} z-bins", leave=False):
            # Compute KID for this bin vs rest
            kid_result = compute_zbin_kid(
                test_images,
                test_zbins,
                synth_images,
                synth_zbins,
                target_zbin=zbin,
                subset_size=subset_size,
                num_subsets=num_subsets,
                degree=degree,
                batch_size=batch_size,
                device=device,
            )

            # Store results
            results.append(
                {
                    "replica_id": replica_id,
                    "zbin": zbin,
                    "kid_zbin": kid_result["kid_zbin_mean"],
                    "kid_zbin_std": kid_result["kid_zbin_std"],
                    "n_real_zbin": kid_result["n_real_zbin"],
                    "n_synth_zbin": kid_result["n_synth_zbin"],
                    "kid_rest": kid_result["kid_rest_mean"],
                    "kid_rest_std": kid_result["kid_rest_std"],
                    "n_real_rest": kid_result["n_real_rest"],
                    "n_synth_rest": kid_result["n_synth_rest"],
                }
            )

        print(f"[{replica_id+1}/{len(replica_files)}] Replica {replica_id}: Processed {len(valid_zbins)} z-bins")

    return pd.DataFrame(results)


def compute_bin_vs_rest_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add statistical analysis columns to zbin DataFrame.

    For each zbin b:
        1. Extract delta_r,b = kid_zbin - kid_rest across replicas/groups
        2. Wilcoxon signed-rank test: H0: median(delta) = 0
        3. Store p-value

    Then apply FDR correction across all zbins.

    Args:
        df: DataFrame from compute_all_replicas_zbin or compute_all_replicas_zbin_merged

    Returns:
        DataFrame with added columns:
            - delta_kid
            - p_value
            - q_value_fdr
            - signif_code
    """
    # Compute delta_kid for each row
    df["delta_kid"] = df["kid_zbin"] - df["kid_rest"]

    # Get unique z-bins
    zbins = sorted(df["zbin"].unique())

    # Determine if we're using merged replicas
    n_samples_per_bin = len(df[df["zbin"] == zbins[0]])
    print(f"\nPerforming statistical tests for {len(zbins)} z-bins (n={n_samples_per_bin} per bin)...")

    # Compute p-values for each z-bin
    p_values = []
    zbin_ids = []

    for zbin in zbins:
        # Get deltas across all replicas/groups for this bin
        deltas = df[df["zbin"] == zbin]["delta_kid"].values

        # Filter out NaN values
        deltas = deltas[~np.isnan(deltas)]

        if len(deltas) < 5:
            # Not enough data for Wilcoxon test
            p_values.append(np.nan)
            print(f"  Bin {zbin:2d}: Insufficient data (n={len(deltas)})")
        else:
            # Wilcoxon signed-rank test (one-sample, test against 0)
            try:
                stat, p_val = wilcoxon(deltas, zero_method="wilcox")
                p_values.append(p_val)
                print(f"  Bin {zbin:2d}: p={p_val:.4f}, median(delta)={np.median(deltas):.6f}")
            except ValueError as e:
                # Handle edge cases (e.g., all zeros)
                print(f"  Bin {zbin:2d}: Wilcoxon test failed ({e})")
                p_values.append(np.nan)

        zbin_ids.append(zbin)

    # Apply FDR correction (Benjamini-Hochberg)
    # Filter out NaN p-values for FDR correction
    valid_mask = ~np.isnan(p_values)
    valid_p_values = np.array(p_values)[valid_mask]

    if len(valid_p_values) > 0:
        # Perform FDR correction
        _, q_values_valid, _, _ = multipletests(valid_p_values, method="fdr_bh")

        # Create full q_values array with NaN for invalid entries
        q_values = np.full(len(p_values), np.nan)
        q_values[valid_mask] = q_values_valid
    else:
        q_values = np.full(len(p_values), np.nan)

    # Create mapping: zbin -> (p_value, q_value)
    stat_map = {}
    for zbin, p_val, q_val in zip(zbin_ids, p_values, q_values):
        stat_map[zbin] = {"p_value": p_val, "q_value_fdr": q_val}

    # Add columns to dataframe
    df["p_value"] = df["zbin"].map(lambda z: stat_map[z]["p_value"])
    df["q_value_fdr"] = df["zbin"].map(lambda z: stat_map[z]["q_value_fdr"])

    # Assign significance codes
    df["signif_code"] = df["q_value_fdr"].apply(assign_significance_code)

    return df


def assign_significance_code(q_value: float) -> str:
    """Assign significance code based on FDR-corrected q-value.

    Args:
        q_value: FDR-corrected q-value

    Returns:
        Significance code: '', '*', '**', '***'
    """
    if np.isnan(q_value):
        return ""
    elif q_value < 0.001:
        return "***"
    elif q_value < 0.01:
        return "**"
    elif q_value < 0.05:
        return "*"
    else:
        return ""


def main(args):
    """Main CLI entry point.

    Steps:
        1. Load real slices (from one or more splits: train/val/test)
        2. Discover all replica NPZ files
        3. Compute global KID for each replica → save CSV
        4. Compute zbin KID for each replica
        5. Perform statistical analysis
        6. Save zbin CSV with statistics
        7. Print summary report
    """
    # Convert paths
    replicas_dir = Path(args.replicas_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define valid z-bins (0-29)
    valid_zbins = list(range(30))

    # Build list of splits to load
    split_configs: list[tuple[Path, str]] = []

    if args.train_slices_csv:
        split_configs.append((Path(args.train_slices_csv), "train"))
    if args.val_slices_csv:
        split_configs.append((Path(args.val_slices_csv), "val"))
    if args.test_slices_csv:
        split_configs.append((Path(args.test_slices_csv), "test"))

    # Validate that at least one split is provided
    if not split_configs:
        raise ValueError(
            "At least one split must be provided. Use --train-slices-csv, "
            "--val-slices-csv, or --test-slices-csv."
        )

    # Step 1: Load real slices (from one or more splits)
    print("=" * 80)
    if len(split_configs) == 1:
        print(f"STEP 1: Loading {split_configs[0][1]} slices")
    else:
        split_names = [s[1] for s in split_configs]
        print(f"STEP 1: Loading slices from {len(split_configs)} splits ({', '.join(split_names)})")
    print("=" * 80)

    real_images, real_zbins, splits_used = load_multi_split_slices(
        split_configs, valid_zbins=valid_zbins
    )
    print(f"Loaded {len(real_images)} real slices total")

    # Step 2: Compute global KID
    print("\n" + "=" * 80)
    print("STEP 2: Computing global KID for all replicas")
    print("=" * 80)
    df_global = compute_all_replicas_global(
        replicas_dir,
        real_images,
        subset_size=args.subset_size,
        num_subsets=args.num_subsets,
        degree=args.degree,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Add metadata about real images source
    df_global["real_splits"] = ",".join(splits_used)
    df_global["n_real_total"] = len(real_images)

    # Save global CSV
    global_csv_path = output_dir / "kid_replica_global.csv"
    df_global.to_csv(global_csv_path, index=False)
    print(f"\nSaved: {global_csv_path} ({len(df_global)} rows)")

    # Step 3: Compute per-zbin KID
    print("\n" + "=" * 80)
    if args.merge_replicas > 1:
        print(f"STEP 3: Computing per-zbin KID with merged replicas (merge={args.merge_replicas})")
    else:
        print("STEP 3: Computing per-zbin KID for all replicas (no merging)")
    print("=" * 80)

    if args.merge_replicas > 1:
        df_zbin = compute_all_replicas_zbin_merged(
            replicas_dir,
            real_images,
            real_zbins,
            valid_zbins=valid_zbins,
            merge_replicas=args.merge_replicas,
            subset_size=args.subset_size,
            num_subsets=args.num_subsets,
            degree=args.degree,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        df_zbin = compute_all_replicas_zbin(
            replicas_dir,
            real_images,
            real_zbins,
            valid_zbins=valid_zbins,
            subset_size=args.subset_size,
            num_subsets=args.num_subsets,
            degree=args.degree,
            batch_size=args.batch_size,
            device=args.device,
        )

    # Add metadata about real images source
    df_zbin["real_splits"] = ",".join(splits_used)

    # Step 4: Perform statistical analysis
    print("\n" + "=" * 80)
    print("STEP 4: Performing statistical analysis")
    print("=" * 80)
    df_zbin = compute_bin_vs_rest_statistics(df_zbin)

    # Save zbin CSV (with suffix if merged)
    if args.merge_replicas > 1:
        zbin_csv_path = output_dir / f"kid_replica_zbin_merged{args.merge_replicas}.csv"
    else:
        zbin_csv_path = output_dir / "kid_replica_zbin.csv"
    df_zbin.to_csv(zbin_csv_path, index=False)
    print(f"\nSaved: {zbin_csv_path} ({len(df_zbin)} rows)")

    # Step 5: Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nGlobal KID (mean across {len(df_global)} replicas):")
    print(f"  Mean: {df_global['kid_global'].mean():.6f}")
    print(f"  Std:  {df_global['kid_global'].std():.6f}")
    print(f"  Min:  {df_global['kid_global'].min():.6f}")
    print(f"  Max:  {df_global['kid_global'].max():.6f}")

    print(f"\nPer-zbin KID statistics:")
    # Count significant bins
    signif_counts = df_zbin.groupby("signif_code").size()
    print(f"  Bins with *** (q<0.001): {signif_counts.get('***', 0)}")
    print(f"  Bins with **  (q<0.01):  {signif_counts.get('**', 0)}")
    print(f"  Bins with *   (q<0.05):  {signif_counts.get('*', 0)}")
    print(f"  Bins with no significance: {signif_counts.get('', 0)}")

    # Top 5 bins with largest |delta_kid|
    unit = "groups" if args.merge_replicas > 1 else "replicas"
    print(f"\nTop 5 bins with largest |delta_kid| (averaged across {unit}):")
    avg_delta = df_zbin.groupby("zbin")["delta_kid"].mean().abs().sort_values(ascending=False)
    for i, (zbin, delta) in enumerate(avg_delta.head(5).items()):
        signif = df_zbin[df_zbin["zbin"] == zbin]["signif_code"].iloc[0]
        print(f"  {i+1}. Bin {zbin:2d}: |delta|={delta:.6f} {signif}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute global and per-zbin KID metrics for synthetic replicas. "
        "Supports comparing against one or more data splits (train/val/test).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test-only (backwards compatible):
  %(prog)s --replicas-dir /path/to/replicas --test-slices-csv test.csv --test-dist-csv test_dist.csv --output-dir /output

  # Multi-split (for uniform replicas):
  %(prog)s --replicas-dir /path/to/replicas --train-slices-csv train.csv --val-slices-csv val.csv --test-slices-csv test.csv --output-dir /output
""",
    )
    parser.add_argument(
        "--replicas-dir",
        type=str,
        required=True,
        help="Directory containing replica_{ID:03d}.npz files",
    )

    # Split arguments (at least one required)
    split_group = parser.add_argument_group(
        "Data splits",
        "Provide at least one split. Multiple splits will be merged as the real image set.",
    )
    split_group.add_argument(
        "--train-slices-csv",
        type=str,
        default=None,
        help="Path to train.csv file (optional)",
    )
    split_group.add_argument(
        "--train-dist-csv",
        type=str,
        default=None,
        help="Path to train_zbin_distribution.csv (for reference, not used in computation)",
    )
    split_group.add_argument(
        "--val-slices-csv",
        type=str,
        default=None,
        help="Path to val.csv file (optional)",
    )
    split_group.add_argument(
        "--val-dist-csv",
        type=str,
        default=None,
        help="Path to val_zbin_distribution.csv (for reference, not used in computation)",
    )
    split_group.add_argument(
        "--test-slices-csv",
        type=str,
        default=None,
        help="Path to test.csv file (optional, but at least one split is required)",
    )
    split_group.add_argument(
        "--test-dist-csv",
        type=str,
        default=None,
        help="Path to test_zbin_distribution.csv (for reference, not used in computation)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1000,
        help="KID subset size (default: 1000)",
    )
    parser.add_argument(
        "--num-subsets",
        type=int,
        default=100,
        help="Number of KID subsets (default: 100)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Polynomial kernel degree (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu, default: cuda)",
    )
    parser.add_argument(
        "--merge-replicas",
        type=int,
        default=1,
        help="Number of replicas to merge for per-zbin KID (default: 1 = no merging). "
        "Merging increases sample size per bin for more stable KID estimates. "
        "Example: --merge-replicas 5 creates groups of 5 replicas each.",
    )

    args = parser.parse_args()
    main(args)
