"""Confusion-matrix-stratified XAI analysis for real vs. synthetic classification.

This module performs explainable AI analysis stratified by confusion matrix
categories (TP, TN, FP, FN) to understand:

1. Why did the classifier correctly identify synthetic samples (TN)?
   - What discriminative features does it focus on?

2. Why did the classifier fail to identify some synthetic samples (FP)?
   - These represent the highest-quality synthetic samples.
   - What makes them indistinguishable from real?

The key insight is that comparing FP (synthetic classified as real) with TN
(synthetic correctly classified) reveals what artifacts the classifier detects
and which synthetic samples successfully avoid them.

Reference methodology:
    - Selvaraju et al., "Grad-CAM" (ICCV 2017)
    - Sundararajan et al., "Integrated Gradients" (ICML 2017)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.classification.diagnostics.utils import (
    discover_checkpoint,
    determine_in_channels,
    ensure_output_dir,
    load_model_from_checkpoint,
    save_result_json,
)
from src.classification.diagnostics.xai.gradcam import GradCAM, GradCAMResult
from src.classification.diagnostics.xai.aggregation import (
    aggregate_heatmaps,
    compute_attention_difference,
    radial_attention_profile,
)
from src.classification.evaluation.confusion_samples import (
    ConfusionMatrixSamples,
    load_all_fold_confusion_samples,
    aggregate_confusion_samples,
    discover_results_dir,
)

logger = logging.getLogger(__name__)


@dataclass
class CategoryAnalysisResult:
    """XAI analysis results for one confusion matrix category.

    Attributes:
        category: Category name ("TP", "TN", "FP", "FN").
        n_samples: Number of samples in this category.
        sample_indices: Original indices for samples in this category.
        z_bins: Z-bin values for samples.
        probs: Classifier probabilities for samples.
        mean_gradcam: Mean GradCAM heatmap across samples (H, W).
        std_gradcam: Std GradCAM heatmap across samples (H, W).
        gradcam_results: Individual GradCAM results per sample.
        channel_contributions: Dict with image_fraction and mask_fraction.
        feature_centroid: Mean GAP feature vector (128-dim).
        feature_std: Std of GAP features (128-dim).
        radial_profile: Radial attention profile from heatmap center.
        per_zbin_counts: Dict mapping z_bin to sample count.
    """

    category: str
    n_samples: int
    sample_indices: list[int] = field(default_factory=list)
    z_bins: list[int] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    mean_gradcam: Optional[np.ndarray] = None
    std_gradcam: Optional[np.ndarray] = None
    gradcam_results: list[GradCAMResult] = field(default_factory=list)
    channel_contributions: dict = field(default_factory=dict)
    feature_centroid: Optional[np.ndarray] = None
    feature_std: Optional[np.ndarray] = None
    radial_profile: Optional[np.ndarray] = None
    per_zbin_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class CategoryComparison:
    """Statistical comparison between two confusion categories.

    Attributes:
        cat_a: First category name.
        cat_b: Second category name.
        attention_difference: Signed difference map (cat_b - cat_a).
        abs_attention_difference: Absolute difference map.
        cosine_similarity: Cosine similarity of mean heatmaps.
        spatial_correlation: Pearson correlation of mean heatmaps.
        feature_distance: Euclidean distance between feature centroids.
        ks_statistic: KS test statistic on heatmap value distributions.
        ks_pvalue: KS test p-value.
        channel_contribution_delta: Difference in image channel contribution.
    """

    cat_a: str
    cat_b: str
    attention_difference: Optional[np.ndarray] = None
    abs_attention_difference: Optional[np.ndarray] = None
    cosine_similarity: float = 0.0
    spatial_correlation: float = 0.0
    spatial_correlation_pvalue: float = 1.0
    feature_distance: float = 0.0
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    channel_contribution_delta: float = 0.0


@dataclass
class ConfusionStratifiedResults:
    """Complete confusion-stratified XAI analysis results.

    Attributes:
        experiment_name: Name of the experiment analyzed.
        categories: Dict mapping category name to CategoryAnalysisResult.
        comparisons: Dict mapping comparison name to CategoryComparison.
        fp_rate: False positive rate (synthetic classified as real).
        fn_rate: False negative rate (real classified as synthetic).
        all_features: GAP features for all samples (N, 128).
        all_labels: Category labels for all samples (N,).
        tsne_embedding: t-SNE 2D embedding of features (N, 2).
        pca_embedding: PCA 2D embedding of features (N, 2).
    """

    experiment_name: str = ""
    categories: dict[str, CategoryAnalysisResult] = field(default_factory=dict)
    comparisons: dict[str, CategoryComparison] = field(default_factory=dict)
    fp_rate: float = 0.0
    fn_rate: float = 0.0
    all_features: Optional[np.ndarray] = None
    all_labels: Optional[np.ndarray] = None
    tsne_embedding: Optional[np.ndarray] = None
    pca_embedding: Optional[np.ndarray] = None


def _load_samples_by_category(
    confusion_samples: ConfusionMatrixSamples,
    patches_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[int]]]:
    """Load patch data for each confusion matrix category.

    Args:
        confusion_samples: ConfusionMatrixSamples with sample references.
        patches_dir: Directory containing real_patches.npz and synthetic_patches.npz.

    Returns:
        Dict mapping category name to (patches, z_bins, original_indices) tuple.
        patches shape: (N, 2, H, W), z_bins shape: (N,).
    """
    patches_dir = Path(patches_dir)

    # Load all patches
    real_data = np.load(patches_dir / "real_patches.npz", allow_pickle=True)
    real_patches = real_data["patches"]

    synth_path = patches_dir / "synthetic_patches.npz"
    if synth_path.exists():
        synth_data = np.load(synth_path, allow_pickle=True)
        synth_patches = synth_data["patches"]
    else:
        synth_patches = np.empty((0, 2, 0, 0), dtype=np.float32)

    result = {}

    for cat_name, samples in [
        ("TP", confusion_samples.true_positives),
        ("TN", confusion_samples.true_negatives),
        ("FP", confusion_samples.false_positives),
        ("FN", confusion_samples.false_negatives),
    ]:
        if not samples:
            result[cat_name] = (np.empty((0, 2, 0, 0), dtype=np.float32), np.array([]), [])
            continue

        patches_list = []
        zbins_list = []
        indices_list = []

        for ref in samples:
            if ref.is_real:
                if ref.original_idx < len(real_patches):
                    patches_list.append(real_patches[ref.original_idx])
            else:
                if ref.original_idx < len(synth_patches):
                    patches_list.append(synth_patches[ref.original_idx])
            zbins_list.append(ref.z_bin)
            indices_list.append(ref.original_idx)

        if patches_list:
            result[cat_name] = (
                np.stack(patches_list, axis=0),
                np.array(zbins_list),
                indices_list,
            )
        else:
            result[cat_name] = (np.empty((0, 2, 0, 0), dtype=np.float32), np.array([]), [])

    return result


def _compute_gradcam_for_category(
    model: nn.Module,
    patches: np.ndarray,
    z_bins: np.ndarray,
    category: str,
    device: str,
) -> list[GradCAMResult]:
    """Compute GradCAM heatmaps for samples in a category.

    Args:
        model: Trained classifier model.
        patches: Patches array (N, 2, H, W).
        z_bins: Z-bin indices (N,).
        category: Category name for logging.
        device: Torch device.

    Returns:
        List of GradCAMResult for each sample.
    """
    if len(patches) == 0:
        return []

    # Get target layer (last Conv2d in backbone)
    target_layer = model.conv_layers[-1].block[0]
    gradcam = GradCAM(model, target_layer)

    # Determine label based on category
    # TP/FN are real (label=0), TN/FP are synthetic (label=1)
    label = 0 if category in ("TP", "FN") else 1
    labels = np.full(len(patches), label, dtype=np.int32)

    try:
        inputs = torch.from_numpy(patches).float()
        results = gradcam.compute_batch(
            inputs,
            labels=labels,
            z_bins=z_bins,
            target_class=1,  # Always compute w.r.t. synthetic class
        )
        logger.info(f"Computed GradCAM for {len(results)} {category} samples")
        return results
    finally:
        gradcam.remove_hooks()


def _extract_gap_features(
    model: nn.Module,
    patches: np.ndarray,
    device: str,
) -> np.ndarray:
    """Extract GAP (Global Average Pooling) layer features.

    Args:
        model: Trained classifier with a global_pool layer.
        patches: Patches array (N, 2, H, W).
        device: Torch device.

    Returns:
        Feature array (N, feature_dim).
    """
    if len(patches) == 0:
        return np.empty((0, 128), dtype=np.float32)

    model.eval()
    features_list = []

    # Register hook to capture GAP output
    gap_output = []

    def hook(module, input, output):
        gap_output.append(output.detach())

    handle = model.global_pool.register_forward_hook(hook)

    try:
        batch_size = 32
        for i in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[i:i + batch_size]).float().to(device)
            gap_output.clear()
            with torch.no_grad():
                model(batch)
            if gap_output:
                # GAP output is (B, C, 1, 1), flatten to (B, C)
                feat = gap_output[0].squeeze(-1).squeeze(-1).cpu().numpy()
                features_list.append(feat)
    finally:
        handle.remove()

    if features_list:
        return np.concatenate(features_list, axis=0)
    return np.empty((0, 128), dtype=np.float32)


def _compute_channel_contributions(
    model: nn.Module,
    patches: np.ndarray,
    device: str,
) -> dict:
    """Compute channel contribution via input gradients.

    Args:
        model: Trained classifier.
        patches: Patches array (N, 2, H, W).
        device: Torch device.

    Returns:
        Dict with 'image_fraction', 'mask_fraction', 'image_magnitude', 'mask_magnitude'.
    """
    if len(patches) == 0:
        return {"image_fraction": 0.5, "mask_fraction": 0.5}

    model.eval()
    image_grads = []
    mask_grads = []

    for i in range(len(patches)):
        x = torch.from_numpy(patches[i:i + 1]).float().to(device).requires_grad_(True)
        logit = model(x).squeeze()
        model.zero_grad()
        logit.backward()

        grad = x.grad.detach().cpu().numpy()  # (1, 2, H, W)
        image_grads.append(np.abs(grad[0, 0]).mean())
        mask_grads.append(np.abs(grad[0, 1]).mean())

    image_mag = np.mean(image_grads)
    mask_mag = np.mean(mask_grads)
    total = image_mag + mask_mag + 1e-8

    return {
        "image_fraction": float(image_mag / total),
        "mask_fraction": float(mask_mag / total),
        "image_magnitude": float(image_mag),
        "mask_magnitude": float(mask_mag),
    }


def _compare_categories(
    cat_a: CategoryAnalysisResult,
    cat_b: CategoryAnalysisResult,
) -> CategoryComparison:
    """Compute statistical comparison between two categories.

    Args:
        cat_a: First category analysis result.
        cat_b: Second category analysis result.

    Returns:
        CategoryComparison with statistical metrics.
    """
    comp = CategoryComparison(cat_a=cat_a.category, cat_b=cat_b.category)

    # Attention difference
    if cat_a.mean_gradcam is not None and cat_b.mean_gradcam is not None:
        if cat_a.mean_gradcam.shape == cat_b.mean_gradcam.shape:
            comp.attention_difference = cat_b.mean_gradcam - cat_a.mean_gradcam
            comp.abs_attention_difference = np.abs(comp.attention_difference)

            # Cosine similarity
            a_flat = cat_a.mean_gradcam.flatten()
            b_flat = cat_b.mean_gradcam.flatten()
            norm_a = np.linalg.norm(a_flat)
            norm_b = np.linalg.norm(b_flat)
            if norm_a > 1e-8 and norm_b > 1e-8:
                comp.cosine_similarity = float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

            # Spatial correlation
            if a_flat.std() > 1e-8 and b_flat.std() > 1e-8:
                corr, pval = stats.pearsonr(a_flat, b_flat)
                comp.spatial_correlation = float(corr)
                comp.spatial_correlation_pvalue = float(pval)

    # Feature distance
    if cat_a.feature_centroid is not None and cat_b.feature_centroid is not None:
        comp.feature_distance = float(np.linalg.norm(
            cat_b.feature_centroid - cat_a.feature_centroid
        ))

    # KS test on heatmap values
    if cat_a.gradcam_results and cat_b.gradcam_results:
        a_vals = np.concatenate([r.heatmap.flatten() for r in cat_a.gradcam_results])
        b_vals = np.concatenate([r.heatmap.flatten() for r in cat_b.gradcam_results])
        if len(a_vals) > 0 and len(b_vals) > 0:
            ks_stat, ks_pval = stats.ks_2samp(a_vals, b_vals)
            comp.ks_statistic = float(ks_stat)
            comp.ks_pvalue = float(ks_pval)

    # Channel contribution delta
    if cat_a.channel_contributions and cat_b.channel_contributions:
        a_img = cat_a.channel_contributions.get("image_fraction", 0.5)
        b_img = cat_b.channel_contributions.get("image_fraction", 0.5)
        comp.channel_contribution_delta = float(b_img - a_img)

    return comp


def run_confusion_stratified_analysis(
    cfg: DictConfig,
    experiment_name: str,
    device: str = "cuda",
    input_mode: str = "joint",
) -> ConfusionStratifiedResults:
    """Run confusion-matrix-stratified XAI analysis.

    Main entry point for the analysis. Loads confusion samples from classification
    results, computes GradCAM, channel contributions, and GAP features for each
    category, then performs pairwise comparisons.

    Args:
        cfg: Master configuration.
        experiment_name: Name of the experiment to analyze.
        device: Torch device string.
        input_mode: Input mode used in classification.

    Returns:
        ConfusionStratifiedResults with full analysis.
    """
    # Use diagnostics config paths
    classification_results_dir = Path(cfg.data.classification_results_dir)
    patches_base_dir = Path(cfg.data.patches_base_dir)
    checkpoints_base_dir = Path(cfg.data.checkpoints_base_dir)
    output_base_dir = Path(cfg.output.base_dir)

    # Discover results directory (handles suffixes like _dithered, _fullimg)
    results_dir = discover_results_dir(classification_results_dir, experiment_name, input_mode)

    if results_dir is None:
        logger.warning(
            f"No results directory found for {experiment_name}/{input_mode}* "
            f"in {classification_results_dir}"
        )
        return ConfusionStratifiedResults(experiment_name=experiment_name)

    logger.info(f"Using results directory: {results_dir}")

    # Detect if using full images based on results directory name
    use_full_images = "_fullimg" in results_dir.name
    if use_full_images:
        # Switch to full_images directory instead of patches
        patches_base_dir = patches_base_dir.parent / "full_images"
        logger.info(f"Detected full-image mode, using: {patches_base_dir}")

    # Load confusion samples from all folds (auto-discover available folds)
    fold_samples = load_all_fold_confusion_samples(results_dir, n_folds=None)

    if not fold_samples:
        logger.warning(f"No confusion samples found in {results_dir}")
        return ConfusionStratifiedResults(experiment_name=experiment_name)

    logger.info(f"Loaded {len(fold_samples)} fold(s) of confusion samples")

    # Use ensemble aggregation when there's a held-out test set
    # (all folds evaluate the same samples with different models)
    use_ensemble = cfg.xai.get("confusion_stratified", {}).get("use_ensemble", True)

    # Aggregate across folds
    aggregated_samples = aggregate_confusion_samples(fold_samples, ensemble=use_ensemble)
    logger.info(
        f"{'Ensembled' if use_ensemble else 'Aggregated'} confusion samples: "
        f"TP={aggregated_samples.n_tp}, TN={aggregated_samples.n_tn}, "
        f"FP={aggregated_samples.n_fp}, FN={aggregated_samples.n_fn}"
    )

    # Load patches for each category
    patches_dir = patches_base_dir / experiment_name
    samples_by_category = _load_samples_by_category(aggregated_samples, patches_dir)

    # Load best model (use fold 0 checkpoint)
    in_channels = determine_in_channels(input_mode)
    ckpt_path = discover_checkpoint(checkpoints_base_dir, experiment_name, fold_idx=0, input_mode=input_mode)
    if ckpt_path is None:
        logger.error(f"No checkpoint found for {experiment_name}")
        return ConfusionStratifiedResults(experiment_name=experiment_name)

    model = load_model_from_checkpoint(ckpt_path, in_channels, device)
    logger.info(f"Loaded model from {ckpt_path}")

    # Analyze each category
    category_results = {}
    all_features = []
    all_labels = []

    for cat_name in ["TP", "TN", "FP", "FN"]:
        patches, z_bins, indices = samples_by_category[cat_name]

        if len(patches) == 0:
            category_results[cat_name] = CategoryAnalysisResult(
                category=cat_name, n_samples=0
            )
            continue

        logger.info(f"Analyzing category {cat_name} ({len(patches)} samples)...")

        # GradCAM
        gradcam_results = _compute_gradcam_for_category(
            model, patches, z_bins, cat_name, device
        )

        # Aggregate heatmaps
        if gradcam_results:
            heatmaps = np.stack([r.heatmap for r in gradcam_results], axis=0)
            mean_gradcam = heatmaps.mean(axis=0)
            std_gradcam = heatmaps.std(axis=0)
            radial_prof = radial_attention_profile(mean_gradcam, n_bins=20)
        else:
            mean_gradcam = None
            std_gradcam = None
            radial_prof = None

        # Channel contributions
        channel_contrib = _compute_channel_contributions(model, patches, device)

        # GAP features
        features = _extract_gap_features(model, patches, device)
        feature_centroid = features.mean(axis=0) if len(features) > 0 else None
        feature_std = features.std(axis=0) if len(features) > 0 else None

        # Store for t-SNE
        if len(features) > 0:
            all_features.append(features)
            all_labels.extend([cat_name] * len(features))

        # Per-z-bin counts
        unique_zbins, counts = np.unique(z_bins, return_counts=True)
        per_zbin_counts = dict(zip(unique_zbins.astype(int).tolist(), counts.tolist()))

        # Get probabilities from confusion samples
        cat_samples = {
            "TP": aggregated_samples.true_positives,
            "TN": aggregated_samples.true_negatives,
            "FP": aggregated_samples.false_positives,
            "FN": aggregated_samples.false_negatives,
        }[cat_name]
        probs = [s.prob for s in cat_samples]

        category_results[cat_name] = CategoryAnalysisResult(
            category=cat_name,
            n_samples=len(patches),
            sample_indices=indices,
            z_bins=z_bins.tolist(),
            probs=probs,
            mean_gradcam=mean_gradcam,
            std_gradcam=std_gradcam,
            gradcam_results=gradcam_results,
            channel_contributions=channel_contrib,
            feature_centroid=feature_centroid,
            feature_std=feature_std,
            radial_profile=radial_prof,
            per_zbin_counts=per_zbin_counts,
        )

    # Pairwise comparisons
    comparisons = {}

    # Primary comparison: FP vs TN (what makes FP indistinguishable)
    if category_results["FP"].n_samples > 0 and category_results["TN"].n_samples > 0:
        comparisons["FP_vs_TN"] = _compare_categories(
            category_results["TN"], category_results["FP"]
        )
        logger.info(
            f"FP vs TN comparison: feature_distance={comparisons['FP_vs_TN'].feature_distance:.4f}, "
            f"cosine_similarity={comparisons['FP_vs_TN'].cosine_similarity:.4f}"
        )

    # Secondary comparison: FN vs TP (unusual real samples)
    if category_results["FN"].n_samples > 0 and category_results["TP"].n_samples > 0:
        comparisons["FN_vs_TP"] = _compare_categories(
            category_results["TP"], category_results["FN"]
        )

    # Compute t-SNE and PCA embeddings
    tsne_embedding = None
    pca_embedding = None
    if all_features:
        all_features_arr = np.concatenate(all_features, axis=0)
        all_labels_arr = np.array(all_labels)

        if len(all_features_arr) >= 4:
            # PCA
            pca = PCA(n_components=2, random_state=42)
            pca_embedding = pca.fit_transform(all_features_arr)

            # t-SNE (only if enough samples)
            if len(all_features_arr) >= 30:
                perplexity = min(30, len(all_features_arr) // 3)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                tsne_embedding = tsne.fit_transform(all_features_arr)
    else:
        all_features_arr = None
        all_labels_arr = None

    # Build final results
    results = ConfusionStratifiedResults(
        experiment_name=experiment_name,
        categories=category_results,
        comparisons=comparisons,
        fp_rate=aggregated_samples.fp_rate,
        fn_rate=aggregated_samples.fn_rate,
        all_features=all_features_arr,
        all_labels=all_labels_arr,
        tsne_embedding=tsne_embedding,
        pca_embedding=pca_embedding,
    )

    # Save results
    output_dir = ensure_output_dir(output_base_dir, experiment_name, "confusion_stratified")
    _save_results(results, output_dir)

    logger.info(f"Confusion-stratified analysis complete for {experiment_name}")
    return results


def _save_results(results: ConfusionStratifiedResults, output_dir: Path) -> None:
    """Save analysis results to JSON.

    Args:
        results: ConfusionStratifiedResults to save.
        output_dir: Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON (without large arrays)
    summary = {
        "experiment_name": results.experiment_name,
        "fp_rate": results.fp_rate,
        "fn_rate": results.fn_rate,
        "categories": {},
        "comparisons": {},
    }

    for cat_name, cat_result in results.categories.items():
        summary["categories"][cat_name] = {
            "n_samples": cat_result.n_samples,
            "channel_contributions": cat_result.channel_contributions,
            "per_zbin_counts": cat_result.per_zbin_counts,
            "mean_prob": float(np.mean(cat_result.probs)) if cat_result.probs else 0.0,
            "std_prob": float(np.std(cat_result.probs)) if cat_result.probs else 0.0,
        }

    for comp_name, comp in results.comparisons.items():
        summary["comparisons"][comp_name] = {
            "cat_a": comp.cat_a,
            "cat_b": comp.cat_b,
            "cosine_similarity": comp.cosine_similarity,
            "spatial_correlation": comp.spatial_correlation,
            "spatial_correlation_pvalue": comp.spatial_correlation_pvalue,
            "feature_distance": comp.feature_distance,
            "ks_statistic": comp.ks_statistic,
            "ks_pvalue": comp.ks_pvalue,
            "channel_contribution_delta": comp.channel_contribution_delta,
        }

    save_result_json(summary, output_dir / "confusion_stratified_summary.json")

    # Save heatmaps as NPZ for visualization
    heatmap_data = {}
    for cat_name, cat_result in results.categories.items():
        if cat_result.mean_gradcam is not None:
            heatmap_data[f"{cat_name}_mean_gradcam"] = cat_result.mean_gradcam
            heatmap_data[f"{cat_name}_std_gradcam"] = cat_result.std_gradcam
        if cat_result.radial_profile is not None:
            heatmap_data[f"{cat_name}_radial_profile"] = cat_result.radial_profile

    for comp_name, comp in results.comparisons.items():
        if comp.attention_difference is not None:
            heatmap_data[f"{comp_name}_attention_diff"] = comp.attention_difference

    if heatmap_data:
        np.savez_compressed(output_dir / "confusion_stratified_heatmaps.npz", **heatmap_data)

    # Save feature embeddings if available
    if results.all_features is not None:
        embedding_data = {
            "features": results.all_features,
            "labels": results.all_labels,
        }
        if results.tsne_embedding is not None:
            embedding_data["tsne"] = results.tsne_embedding
        if results.pca_embedding is not None:
            embedding_data["pca"] = results.pca_embedding
        np.savez_compressed(output_dir / "confusion_stratified_embeddings.npz", **embedding_data)

    logger.info(f"Saved confusion-stratified results to {output_dir}")
