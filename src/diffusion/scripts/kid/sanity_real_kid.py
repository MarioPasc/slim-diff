"""CLI for computing KID between train and test sets of real images.

This script computes a baseline KID by comparing the test set against random
subsets of the training set. It divides the train set into subsets of the same
size as the test set, computes KID for each subset, and outputs the mean ± std.

This provides a baseline for evaluating synthetic data quality - the KID between
synthetic and test data should ideally be similar to the KID between train and test.

Usage:
    python -m src.diffusion.scripts.kid.sanity_real_kid \
        --train-slices-csv /media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache/train.csv \
        --test-slices-csv /media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache/test.csv \
        --output-dir /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/kid_quality/real_vs_real \
        --num-subsets 30 \
        --subset-size 1000 \
        --degree 3 \
        --batch-size 32 \
        --device cuda

    python -m src.diffusion.scripts.kid.sanity_real_kid \
        --train-slices-csv /media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache/train.csv \
        --test-slices-csv /media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache/test.csv \
        --output-dir /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/kid_quality/real_vs_real \
        --num-subsets 30 \
        --subset-size 250 \
        --degree 3 \
        --batch-size 32 \
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .kid import compute_global_kid, load_test_slices


def compute_train_test_kid_subsets(
    train_images: np.ndarray,
    test_images: np.ndarray,
    num_subsets: int,
    kid_subset_size: int,
    kid_num_subsets: int,
    degree: int,
    batch_size: int,
    device: str,
    seed: int = 42,
) -> tuple[float, float, list[float]]:
    """Compute KID between test set and random train subsets.

    Divides the train set into random subsets of the same size as the test set,
    computes KID between each train subset and the test set, and returns statistics.

    Args:
        train_images: (N_train, H, W) float32 array in [-1, 1]
        test_images: (N_test, H, W) float32 array in [-1, 1]
        num_subsets: Number of train subsets to sample
        kid_subset_size: KID subset size parameter
        kid_num_subsets: Number of KID subsets for internal computation
        degree: Polynomial kernel degree
        batch_size: Batch size for feature extraction
        device: cuda or cpu
        seed: Random seed for reproducibility

    Returns:
        mean_kid: Mean KID across all train subsets
        std_kid: Standard deviation of KID across all train subsets
        all_kids: List of individual KID values
    """
    rng = np.random.default_rng(seed)
    test_size = len(test_images)
    n_train = len(train_images)

    print(f"\nComputing KID between test set ({test_size} samples) and train subsets...")
    print(f"  Train set size: {n_train}")
    print(f"  Test set size: {test_size}")
    print(f"  Number of subsets: {num_subsets}")
    print(f"  Subset size: {test_size} (matching test set)")

    if n_train < test_size:
        print(f"Warning: Train set ({n_train}) is smaller than test set ({test_size}).")
        print("Using full train set for each subset.")

    all_kids = []

    for i in tqdm(range(num_subsets), desc="Computing train-test KID"):
        # Sample a random subset of train images with same size as test set
        if n_train >= test_size:
            indices = rng.choice(n_train, size=test_size, replace=False)
        else:
            # If train is smaller, sample with replacement
            indices = rng.choice(n_train, size=test_size, replace=True)

        train_subset = train_images[indices]

        # Compute KID between this train subset and the test set
        kid_result = compute_global_kid(
            real_images=test_images,
            synth_images=train_subset,
            subset_size=kid_subset_size,
            num_subsets=kid_num_subsets,
            degree=degree,
            batch_size=batch_size,
            device=device,
        )

        kid_value = kid_result["kid_mean"]
        all_kids.append(kid_value)

        print(f"  Subset {i+1:3d}/{num_subsets}: KID = {kid_value:.6f}")

    mean_kid = float(np.mean(all_kids))
    std_kid = float(np.std(all_kids))

    return mean_kid, std_kid, all_kids


def main(args):
    """Main CLI entry point."""
    # Convert paths
    train_csv = Path(args.train_slices_csv)
    test_csv = Path(args.test_slices_csv)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define valid z-bins (0-29)
    valid_zbins = list(range(30))

    # Step 1: Load real slices
    print("=" * 80)
    print("STEP 1: Loading real slices")
    print("=" * 80)

    print("\nLoading train slices...")
    train_images, train_zbins = load_test_slices(train_csv, valid_zbins=valid_zbins)
    print(f"Loaded {len(train_images)} train slices")

    print("\nLoading test slices...")
    test_images, test_zbins = load_test_slices(test_csv, valid_zbins=valid_zbins)
    print(f"Loaded {len(test_images)} test slices")

    # Step 2: Compute KID between train subsets and test set
    print("\n" + "=" * 80)
    print("STEP 2: Computing train-test KID across random subsets")
    print("=" * 80)

    mean_kid, std_kid, all_kids = compute_train_test_kid_subsets(
        train_images=train_images,
        test_images=test_images,
        num_subsets=args.num_subsets,
        kid_subset_size=args.subset_size,
        kid_num_subsets=args.kid_num_subsets,
        degree=args.degree,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )

    # Step 3: Save results
    print("\n" + "=" * 80)
    print("STEP 3: Saving results")
    print("=" * 80)

    # Create results DataFrame
    df_results = pd.DataFrame({
        "subset_id": list(range(len(all_kids))),
        "kid_train_test": all_kids,
    })

    # Add metadata
    df_results["n_train"] = len(train_images)
    df_results["n_test"] = len(test_images)
    df_results["subset_size"] = len(test_images)
    df_results["kid_subset_size"] = args.subset_size
    df_results["kid_num_subsets"] = args.kid_num_subsets
    df_results["degree"] = args.degree
    df_results["seed"] = args.seed

    # Save CSV
    csv_path = output_dir / "kid_real_train_test.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save summary JSON
    import json
    summary = {
        "mean_kid_train_test": mean_kid,
        "std_kid_train_test": std_kid,
        "n_subsets": len(all_kids),
        "n_train": len(train_images),
        "n_test": len(test_images),
        "subset_size": len(test_images),
        "kid_subset_size": args.subset_size,
        "kid_num_subsets": args.kid_num_subsets,
        "degree": args.degree,
        "seed": args.seed,
    }
    json_path = output_dir / "kid_real_train_test_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    # Step 4: Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nReal Train-Test KID (baseline):")
    print(f"  Mean: {mean_kid:.6f}")
    print(f"  Std:  {std_kid:.6f}")
    print(f"  Min:  {min(all_kids):.6f}")
    print(f"  Max:  {max(all_kids):.6f}")
    print(f"  N subsets: {len(all_kids)}")
    print(f"\nUse these values with plot_kid_results.py:")
    print(f"  --mean-real-kid-train-test {mean_kid:.6f}")
    print(f"  --std-real-kid-train-test {std_kid:.6f}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute KID between train and test sets of real images to establish baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script computes a baseline KID by comparing the test set against random
subsets of the training set. The resulting mean ± std can be used as a reference
when evaluating synthetic data quality.

Examples:
    python -m src.diffusion.scripts.kid.sanity_real_kid \\
        --train-slices-csv /path/to/train.csv \\
        --test-slices-csv /path/to/test.csv \\
        --output-dir /path/to/output \\
        --num-subsets 30
""",
    )
    parser.add_argument(
        "--train-slices-csv",
        type=str,
        required=True,
        help="Path to train.csv file with slice filepaths",
    )
    parser.add_argument(
        "--test-slices-csv",
        type=str,
        required=True,
        help="Path to test.csv file with slice filepaths",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-subsets",
        type=int,
        default=30,
        help="Number of random train subsets to sample (default: 30)",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=250,
        help="KID subset size for internal computation (default: 250)",
    )
    parser.add_argument(
        "--kid-num-subsets",
        type=int,
        default=100,
        help="Number of KID subsets for internal computation (default: 100)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    main(args)
