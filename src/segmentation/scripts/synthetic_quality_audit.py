"""Synthetic image quality audit for segmentation.

Compares synthetic images/masks to real data beyond KID, focusing on 
properties that affect segmentation performance.

Usage:
    python -m src.segmentation.scripts.synthetic_quality_audit \
        --real_cache /path/to/slice_cache \
        --synthetic_replica /path/to/replica.npz \
        --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.stats import ks_2samp, wasserstein_distance
from skimage import measure
from tqdm import tqdm

logger = logging.getLogger(__name__)

# =============================================================================
# Mask Analysis Functions
# =============================================================================

def compute_mask_statistics(mask: np.ndarray, threshold: float = 0.0) -> dict[str, Any]:
    """Compute statistics for a single mask.
    
    Args:
        mask: 2D mask array in [-1, 1] or [0, 1] range
        threshold: Binarization threshold
        
    Returns:
        Dictionary of mask statistics
    """
    # Ensure float32 for scipy compatibility
    mask = np.asarray(mask, dtype=np.float32)
    
    # Binarize (keep as bool for bitwise ops)
    binary = mask > threshold
    binary_float = binary.astype(np.float32)
    
    # Basic stats
    lesion_area = binary_float.sum()
    lesion_ratio = lesion_area / binary.size
    
    # Connected components
    labeled, n_components = ndimage.label(binary)
    
    # Component sizes
    component_sizes = []
    if n_components > 0:
        for i in range(1, n_components + 1):
            component_sizes.append((labeled == i).sum())
    
    # Raw mask stats (before binarization)
    raw_mean = mask.mean()
    raw_std = mask.std()
    raw_min = mask.min()
    raw_max = mask.max()
    
    # Edge sharpness (gradient magnitude at lesion boundaries)
    edge_sharpness = 0.0
    if lesion_area > 0:
        grad_y, grad_x = np.gradient(mask)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        # Find boundary pixels (use boolean arrays for bitwise ops)
        dilated = ndimage.binary_dilation(binary)
        boundary = dilated & ~binary
        if boundary.sum() > 0:
            edge_sharpness = grad_mag[boundary].mean()
    
    return {
        "lesion_area": lesion_area,
        "lesion_ratio": lesion_ratio,
        "n_components": n_components,
        "component_sizes": component_sizes,
        "largest_component": max(component_sizes) if component_sizes else 0,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "edge_sharpness": edge_sharpness,
    }


def compute_lesion_shape_features(mask: np.ndarray, threshold: float = 0.0) -> dict[str, Any]:
    """Compute shape features for lesions in a mask.
    
    Args:
        mask: 2D mask array
        threshold: Binarization threshold
        
    Returns:
        Dictionary of shape features
    """
    binary = (mask > threshold).astype(np.uint8)
    
    # Find contours using skimage
    contours = measure.find_contours(binary, 0.5)
    
    if not contours:
        return {
            "circularity": 0.0,
            "solidity": 0.0,
            "eccentricity": 0.0,
            "perimeter": 0.0,
        }
    
    # Use largest contour
    largest_contour = max(contours, key=len)
    
    # Get region properties
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    
    if not regions:
        return {
            "circularity": 0.0,
            "solidity": 0.0,
            "eccentricity": 0.0,
            "perimeter": 0.0,
        }
    
    # Use largest region
    largest_region = max(regions, key=lambda r: r.area)
    
    area = largest_region.area
    perimeter = largest_region.perimeter
    
    # Circularity: 4π * area / perimeter²
    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-8) if perimeter > 0 else 0
    
    return {
        "circularity": circularity,
        "solidity": largest_region.solidity,
        "eccentricity": largest_region.eccentricity,
        "perimeter": perimeter,
    }


# =============================================================================
# Image Analysis Functions
# =============================================================================

def compute_image_statistics(image: np.ndarray) -> dict[str, Any]:
    """Compute statistics for a single image.
    
    Args:
        image: 2D image array
        
    Returns:
        Dictionary of image statistics
    """
    return {
        "mean": image.mean(),
        "std": image.std(),
        "min": image.min(),
        "max": image.max(),
        "p5": np.percentile(image, 5),
        "p95": np.percentile(image, 95),
        "dynamic_range": image.max() - image.min(),
    }


def compute_texture_features(image: np.ndarray) -> dict[str, Any]:
    """Compute texture features for an image.
    
    Args:
        image: 2D image array
        
    Returns:
        Dictionary of texture features
    """
    # Ensure float32 for scipy compatibility (float16 not supported)
    image = image.astype(np.float32)
    
    # Gradient statistics (edge content)
    grad_y, grad_x = np.gradient(image)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Laplacian (high-frequency content)
    laplacian = ndimage.laplace(image)
    
    return {
        "gradient_mean": grad_mag.mean(),
        "gradient_std": grad_mag.std(),
        "laplacian_mean": np.abs(laplacian).mean(),
        "laplacian_std": laplacian.std(),
    }


# =============================================================================
# Data Loading
# =============================================================================

def load_real_data(cache_dir: Path, split: str = "train") -> dict[str, list]:
    """Load real data from slice cache.
    
    Args:
        cache_dir: Path to slice_cache
        split: Which split to load ('train', 'val', 'test')
        
    Returns:
        Dictionary with images, masks, and metadata grouped by zbin
    """
    csv_path = cache_dir / f"{split}.csv"
    
    data_by_zbin = defaultdict(lambda: {
        "images": [], "masks": [], "has_lesion": [], "subject_ids": []
    })
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for row in tqdm(rows, desc=f"Loading real {split}"):
        npz_path = cache_dir / row["filepath"]
        npz_data = np.load(npz_path)
        
        zbin = int(row["z_bin"])
        has_lesion = row["has_lesion"].lower() == "true"
        
        data_by_zbin[zbin]["images"].append(npz_data["image"])
        data_by_zbin[zbin]["masks"].append(npz_data["mask"])
        data_by_zbin[zbin]["has_lesion"].append(has_lesion)
        data_by_zbin[zbin]["subject_ids"].append(row["subject_id"])
    
    return dict(data_by_zbin)


def load_synthetic_data(replica_path: Path) -> dict[str, list]:
    """Load synthetic data from replica NPZ.
    
    Args:
        replica_path: Path to replica NPZ file
        
    Returns:
        Dictionary with images, masks, and metadata grouped by zbin
    """
    data = np.load(replica_path, allow_pickle=True)
    
    images = data["images"]
    masks = data["masks"]
    zbins = data["zbin"]
    lesion_present = data["lesion_present"]
    
    data_by_zbin = defaultdict(lambda: {
        "images": [], "masks": [], "lesion_label": [], "actual_lesion": []
    })
    
    for i in tqdm(range(len(images)), desc="Loading synthetic"):
        zbin = int(zbins[i])
        label = int(lesion_present[i])
        
        # Check if mask actually contains lesion
        mask_binary = (masks[i] > 0.0).astype(float)
        actual_lesion = mask_binary.sum() > 10  # At least 10 pixels
        
        data_by_zbin[zbin]["images"].append(images[i])
        data_by_zbin[zbin]["masks"].append(masks[i])
        data_by_zbin[zbin]["lesion_label"].append(label)
        data_by_zbin[zbin]["actual_lesion"].append(actual_lesion)
    
    return dict(data_by_zbin)


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_distributions(
    real_values: np.ndarray,
    synthetic_values: np.ndarray,
    name: str,
) -> dict[str, float]:
    """Compare two distributions using statistical tests.
    
    Args:
        real_values: Real data values
        synthetic_values: Synthetic data values
        name: Name of the metric
        
    Returns:
        Dictionary with comparison metrics
    """
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = ks_2samp(real_values, synthetic_values)
    
    # Wasserstein distance (Earth Mover's Distance)
    wasserstein = wasserstein_distance(real_values, synthetic_values)
    
    # Simple statistics comparison
    real_mean, synth_mean = real_values.mean(), synthetic_values.mean()
    real_std, synth_std = real_values.std(), synthetic_values.std()
    
    return {
        f"{name}_ks_stat": ks_stat,
        f"{name}_ks_pval": ks_pval,
        f"{name}_wasserstein": wasserstein,
        f"{name}_mean_diff": abs(real_mean - synth_mean),
        f"{name}_std_diff": abs(real_std - synth_std),
    }


def audit_label_accuracy(synthetic_data: dict) -> dict[str, Any]:
    """Check if lesion labels match actual mask content.
    
    Args:
        synthetic_data: Loaded synthetic data by zbin
        
    Returns:
        Dictionary with label accuracy statistics
    """
    total_lesion_label = 0
    total_no_lesion_label = 0
    false_positives = 0  # Label says lesion, mask is empty
    false_negatives = 0  # Label says no lesion, mask has content
    
    details_by_zbin = {}
    
    for zbin, data in synthetic_data.items():
        labels = np.array(data["lesion_label"])
        actual = np.array(data["actual_lesion"])
        
        zbin_lesion_label = labels.sum()
        zbin_no_lesion_label = (~labels.astype(bool)).sum()
        
        # False positives: label=1, actual=0
        zbin_fp = ((labels == 1) & ~actual).sum()
        # False negatives: label=0, actual=1
        zbin_fn = ((labels == 0) & actual).sum()
        
        total_lesion_label += zbin_lesion_label
        total_no_lesion_label += zbin_no_lesion_label
        false_positives += zbin_fp
        false_negatives += zbin_fn
        
        details_by_zbin[zbin] = {
            "n_samples": len(labels),
            "lesion_labels": int(zbin_lesion_label),
            "actual_lesions": int(actual.sum()),
            "false_positives": int(zbin_fp),
            "false_negatives": int(zbin_fn),
        }
    
    total = total_lesion_label + total_no_lesion_label
    
    return {
        "total_samples": total,
        "lesion_label_count": total_lesion_label,
        "no_lesion_label_count": total_no_lesion_label,
        "false_positive_count": false_positives,
        "false_negative_count": false_negatives,
        "false_positive_rate": false_positives / max(total_lesion_label, 1),
        "false_negative_rate": false_negatives / max(total_no_lesion_label, 1),
        "label_accuracy": 1 - (false_positives + false_negatives) / max(total, 1),
        "by_zbin": details_by_zbin,
    }


def run_full_audit(
    real_cache_dir: Path,
    synthetic_replica_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run full synthetic quality audit.
    
    Args:
        real_cache_dir: Path to slice_cache
        synthetic_replica_path: Path to synthetic replica NPZ
        output_dir: Directory for output files
        
    Returns:
        Dictionary with all audit results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading real data...")
    real_train = load_real_data(real_cache_dir, "train")
    real_val = load_real_data(real_cache_dir, "val")
    
    # Merge train and val for comparison
    real_data = defaultdict(lambda: {"images": [], "masks": [], "has_lesion": []})
    for zbin, data in {**real_train, **real_val}.items():
        real_data[zbin]["images"].extend(data["images"])
        real_data[zbin]["masks"].extend(data["masks"])
        real_data[zbin]["has_lesion"].extend(data["has_lesion"])
    real_data = dict(real_data)
    
    logger.info("Loading synthetic data...")
    synthetic_data = load_synthetic_data(synthetic_replica_path)
    
    results = {
        "replica_path": str(synthetic_replica_path),
        "zbins_analyzed": [],
        "label_audit": None,
        "comparisons": {},
    }
    
    # 1. Label accuracy audit
    logger.info("Auditing label accuracy...")
    results["label_audit"] = audit_label_accuracy(synthetic_data)
    
    # Print critical finding
    label_audit = results["label_audit"]
    logger.warning(
        f"LABEL ACCURACY AUDIT:\n"
        f"  Total samples: {label_audit['total_samples']}\n"
        f"  False positives (label=lesion, mask=empty): {label_audit['false_positive_count']} "
        f"({label_audit['false_positive_rate']:.1%})\n"
        f"  False negatives (label=no-lesion, mask=has-content): {label_audit['false_negative_count']} "
        f"({label_audit['false_negative_rate']:.1%})\n"
        f"  Overall label accuracy: {label_audit['label_accuracy']:.1%}"
    )
    
    # 2. Per-zbin comparisons
    logger.info("Computing per-zbin statistics...")
    
    all_zbins = sorted(set(real_data.keys()) & set(synthetic_data.keys()))
    results["zbins_analyzed"] = all_zbins
    
    # Collect all statistics for comparison
    real_stats = {"mask": [], "image": [], "shape": [], "texture": []}
    synth_stats = {"mask": [], "image": [], "shape": [], "texture": []}
    
    for zbin in tqdm(all_zbins, desc="Analyzing zbins"):
        real_zbin = real_data.get(zbin, {"images": [], "masks": []})
        synth_zbin = synthetic_data.get(zbin, {"images": [], "masks": []})
        
        # Analyze real data
        for img, mask in zip(real_zbin["images"], real_zbin["masks"]):
            real_stats["mask"].append(compute_mask_statistics(mask))
            real_stats["image"].append(compute_image_statistics(img))
            if mask.max() > 0:  # Only compute shape for masks with content
                real_stats["shape"].append(compute_lesion_shape_features(mask))
            real_stats["texture"].append(compute_texture_features(img))
        
        # Analyze synthetic data
        for img, mask in zip(synth_zbin["images"], synth_zbin["masks"]):
            synth_stats["mask"].append(compute_mask_statistics(mask))
            synth_stats["image"].append(compute_image_statistics(img))
            if mask.max() > 0:
                synth_stats["shape"].append(compute_lesion_shape_features(mask))
            synth_stats["texture"].append(compute_texture_features(img))
    
    # 3. Distribution comparisons
    logger.info("Comparing distributions...")
    
    # Mask statistics comparison
    for key in ["lesion_area", "edge_sharpness", "raw_std", "n_components"]:
        real_vals = np.array([s[key] for s in real_stats["mask"]])
        synth_vals = np.array([s[key] for s in synth_stats["mask"]])
        results["comparisons"][f"mask_{key}"] = compare_distributions(
            real_vals, synth_vals, f"mask_{key}"
        )
    
    # Shape statistics comparison (only for samples with lesions)
    if real_stats["shape"] and synth_stats["shape"]:
        for key in ["circularity", "solidity", "eccentricity"]:
            real_vals = np.array([s[key] for s in real_stats["shape"]])
            synth_vals = np.array([s[key] for s in synth_stats["shape"]])
            results["comparisons"][f"shape_{key}"] = compare_distributions(
                real_vals, synth_vals, f"shape_{key}"
            )
    
    # Image statistics comparison
    for key in ["mean", "std", "dynamic_range"]:
        real_vals = np.array([s[key] for s in real_stats["image"]])
        synth_vals = np.array([s[key] for s in synth_stats["image"]])
        results["comparisons"][f"image_{key}"] = compare_distributions(
            real_vals, synth_vals, f"image_{key}"
        )
    
    # Texture statistics comparison
    for key in ["gradient_mean", "laplacian_mean"]:
        real_vals = np.array([s[key] for s in real_stats["texture"]])
        synth_vals = np.array([s[key] for s in synth_stats["texture"]])
        results["comparisons"][f"texture_{key}"] = compare_distributions(
            real_vals, synth_vals, f"texture_{key}"
        )
    
    # 4. Generate visualizations
    logger.info("Generating visualizations...")
    generate_audit_plots(real_stats, synth_stats, results, output_dir)
    
    # 5. Save results
    import json
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    results_json = convert_to_serializable(results)
    
    with open(output_dir / "audit_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Audit complete. Results saved to {output_dir}")
    
    return results


def generate_audit_plots(
    real_stats: dict,
    synth_stats: dict,
    results: dict,
    output_dir: Path,
):
    """Generate visualization plots for the audit.
    
    Args:
        real_stats: Statistics from real data
        synth_stats: Statistics from synthetic data
        results: Audit results dictionary
        output_dir: Output directory for plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Lesion area distribution
    ax = axes[0, 0]
    real_areas = [s["lesion_area"] for s in real_stats["mask"]]
    synth_areas = [s["lesion_area"] for s in synth_stats["mask"]]
    ax.hist(real_areas, bins=50, alpha=0.5, label="Real", density=True)
    ax.hist(synth_areas, bins=50, alpha=0.5, label="Synthetic", density=True)
    ax.set_xlabel("Lesion Area (pixels)")
    ax.set_ylabel("Density")
    ax.set_title("Lesion Area Distribution")
    ax.legend()
    
    # 2. Edge sharpness distribution
    ax = axes[0, 1]
    real_edges = [s["edge_sharpness"] for s in real_stats["mask"] if s["edge_sharpness"] > 0]
    synth_edges = [s["edge_sharpness"] for s in synth_stats["mask"] if s["edge_sharpness"] > 0]
    if real_edges and synth_edges:
        ax.hist(real_edges, bins=50, alpha=0.5, label="Real", density=True)
        ax.hist(synth_edges, bins=50, alpha=0.5, label="Synthetic", density=True)
    ax.set_xlabel("Edge Sharpness")
    ax.set_ylabel("Density")
    ax.set_title("Mask Edge Sharpness (Lesion Samples)")
    ax.legend()
    
    # 3. Number of connected components
    ax = axes[0, 2]
    real_comps = [s["n_components"] for s in real_stats["mask"]]
    synth_comps = [s["n_components"] for s in synth_stats["mask"]]
    bins = np.arange(0, max(max(real_comps), max(synth_comps)) + 2) - 0.5
    ax.hist(real_comps, bins=bins, alpha=0.5, label="Real", density=True)
    ax.hist(synth_comps, bins=bins, alpha=0.5, label="Synthetic", density=True)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Density")
    ax.set_title("Connected Components in Mask")
    ax.legend()
    
    # 4. Image intensity distribution
    ax = axes[1, 0]
    real_means = [s["mean"] for s in real_stats["image"]]
    synth_means = [s["mean"] for s in synth_stats["image"]]
    ax.hist(real_means, bins=50, alpha=0.5, label="Real", density=True)
    ax.hist(synth_means, bins=50, alpha=0.5, label="Synthetic", density=True)
    ax.set_xlabel("Image Mean Intensity")
    ax.set_ylabel("Density")
    ax.set_title("Image Intensity Distribution")
    ax.legend()
    
    # 5. Texture (gradient) comparison
    ax = axes[1, 1]
    real_grad = [s["gradient_mean"] for s in real_stats["texture"]]
    synth_grad = [s["gradient_mean"] for s in synth_stats["texture"]]
    ax.hist(real_grad, bins=50, alpha=0.5, label="Real", density=True)
    ax.hist(synth_grad, bins=50, alpha=0.5, label="Synthetic", density=True)
    ax.set_xlabel("Mean Gradient Magnitude")
    ax.set_ylabel("Density")
    ax.set_title("Image Texture (Edge Content)")
    ax.legend()
    
    # 6. Label accuracy by zbin
    ax = axes[1, 2]
    if results["label_audit"] and results["label_audit"]["by_zbin"]:
        zbins = sorted(results["label_audit"]["by_zbin"].keys())
        fp_rates = []
        fn_rates = []
        for zbin in zbins:
            zbin_data = results["label_audit"]["by_zbin"][zbin]
            fp_rate = zbin_data["false_positives"] / max(zbin_data["lesion_labels"], 1)
            fn_rate = zbin_data["false_negatives"] / max(zbin_data["n_samples"] - zbin_data["lesion_labels"], 1)
            fp_rates.append(fp_rate)
            fn_rates.append(fn_rate)
        
        x = np.arange(len(zbins))
        width = 0.35
        ax.bar(x - width/2, fp_rates, width, label="False Positive Rate", alpha=0.7)
        ax.bar(x + width/2, fn_rates, width, label="False Negative Rate", alpha=0.7)
        ax.set_xlabel("Z-bin")
        ax.set_ylabel("Error Rate")
        ax.set_title("Label-Mask Mismatch by Z-bin")
        ax.legend()
        ax.set_xticks(x[::5])
        ax.set_xticklabels([zbins[i] for i in range(0, len(zbins), 5)])
    
    plt.tight_layout()
    plt.savefig(output_dir / "audit_distributions.png", dpi=150)
    plt.close()
    
    # Additional plot: Shape features comparison
    if real_stats["shape"] and synth_stats["shape"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, key in enumerate(["circularity", "solidity", "eccentricity"]):
            ax = axes[idx]
            real_vals = [s[key] for s in real_stats["shape"]]
            synth_vals = [s[key] for s in synth_stats["shape"]]
            ax.hist(real_vals, bins=30, alpha=0.5, label="Real", density=True)
            ax.hist(synth_vals, bins=30, alpha=0.5, label="Synthetic", density=True)
            ax.set_xlabel(key.capitalize())
            ax.set_ylabel("Density")
            ax.set_title(f"Lesion {key.capitalize()}")
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "audit_shape_features.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Synthetic image quality audit")
    parser.add_argument("--real_cache", type=Path, required=True, help="Path to slice_cache")
    parser.add_argument("--synthetic_replica", type=Path, required=True, help="Path to replica NPZ")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    run_full_audit(args.real_cache, args.synthetic_replica, args.output_dir)


if __name__ == "__main__":
    main()
