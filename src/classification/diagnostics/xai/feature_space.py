"""Feature space analysis for real vs. synthetic classifier.

Analyzes the classifier's learned representation (128-dim GAP features) to
understand the geometry of real vs. synthetic discrimination. Provides Fisher
discriminant ranking, PCA/t-SNE visualization, and spatial activation maps
for top discriminative features.

Scientific basis:
    The Global Average Pooling (GAP) features represent the model's learned
    embedding before classification. Fisher's Linear Discriminant Ratio per
    dimension identifies which features carry the most class information.
    This follows standard representation analysis methodology.

    References:
        Yosinski, J., et al. (2015). "Understanding Neural Networks Through
        Deep Visualization." ICML 2015 Deep Learning Workshop.

        Fisher, R. A. (1936). "The Use of Multiple Measurements in Taxonomic
        Problems." Annals of Eugenics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from src.classification.diagnostics.utils import (
    discover_checkpoint,
    ensure_output_dir,
    load_model_from_checkpoint,
    load_patches,
    save_csv,
    save_figure,
    save_result_json,
)

logger = logging.getLogger(__name__)


def _extract_gap_features(
    model: torch.nn.Module,
    patches: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract GAP (Global Average Pooling) features from the model.

    Hooks into the model's global_pool layer to extract the 128-dim
    feature vector before the FC classifier head.

    Args:
        model: Classifier model in eval mode.
        patches: Input patches of shape (N, C, H, W).
        device: Torch device string.
        batch_size: Batch size for inference.

    Returns:
        Feature array of shape (N, feature_dim).
    """
    features = []
    hook_output = {}

    def hook_fn(module, input, output):
        # GAP output: (B, C, 1, 1) -> squeeze to (B, C)
        hook_output["features"] = output.squeeze(-1).squeeze(-1).detach().cpu().numpy()

    # Register hook on global_pool
    handle = model.global_pool.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[start:start+batch_size]).float().to(device)
            _ = model(batch)
            features.append(hook_output["features"].copy())

    handle.remove()
    return np.concatenate(features, axis=0)


def _compute_fisher_discriminant(
    features: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute Fisher's Linear Discriminant Ratio per feature dimension.

    FDR_i = (μ_0_i - μ_1_i)² / (σ²_0_i + σ²_1_i)

    Higher ratio means more discriminative feature.

    Args:
        features: Feature array (N, D).
        labels: Binary labels (N,).

    Returns:
        Array of Fisher ratios (D,).
    """
    real_mask = labels == 0
    synth_mask = labels == 1

    mu_real = features[real_mask].mean(axis=0)
    mu_synth = features[synth_mask].mean(axis=0)
    var_real = features[real_mask].var(axis=0)
    var_synth = features[synth_mask].var(axis=0)

    denominator = var_real + var_synth + 1e-12
    fisher_ratio = (mu_real - mu_synth) ** 2 / denominator
    return fisher_ratio


def _compute_feature_spatial_maps(
    model: torch.nn.Module,
    patches: np.ndarray,
    top_feature_indices: list[int],
    device: str,
    n_samples: int = 20,
) -> dict[int, np.ndarray]:
    """Compute gradient of top features w.r.t. input to visualize spatial patterns.

    For each top discriminative feature dimension, compute the mean
    |∂feature_d/∂input| across samples to see what spatial pattern
    activates that feature.

    Args:
        model: Classifier model.
        patches: Input patches (N, C, H, W).
        top_feature_indices: List of feature dimension indices.
        device: Torch device string.
        n_samples: Number of samples to average over.

    Returns:
        Dict mapping feature_idx to spatial map (H, W).
    """
    model.eval()
    spatial_maps = {idx: [] for idx in top_feature_indices}

    hook_output = {}

    def hook_fn(module, input, output):
        hook_output["features"] = output.squeeze(-1).squeeze(-1)

    handle = model.global_pool.register_forward_hook(hook_fn)

    # Subsample
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(patches), min(n_samples, len(patches)), replace=False)

    for i in sample_indices:
        x = torch.from_numpy(patches[i:i+1]).float().to(device).requires_grad_(True)
        _ = model(x)
        feats = hook_output["features"]  # (1, D)

        for feat_idx in top_feature_indices:
            model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
            feats[0, feat_idx].backward(retain_graph=True)
            grad = x.grad.detach().cpu().numpy()[0]  # (C, H, W)
            spatial_map = np.abs(grad).sum(axis=0)  # (H, W) - sum across channels
            spatial_maps[feat_idx].append(spatial_map)

    handle.remove()

    # Average across samples
    result = {}
    for idx in top_feature_indices:
        if spatial_maps[idx]:
            result[idx] = np.mean(spatial_maps[idx], axis=0)
    return result


def _plot_feature_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    zbins: np.ndarray,
    fisher_ratios: np.ndarray,
    pca_result: dict,
    tsne_embedding: np.ndarray | None,
    output_dir: Path,
) -> None:
    """Generate feature space visualizations."""
    # Plot 1: Feature importance bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    top_k = min(20, len(fisher_ratios))
    sorted_idx = np.argsort(fisher_ratios)[::-1][:top_k]
    ax.bar(range(top_k), fisher_ratios[sorted_idx], color="#2196F3")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"F{i}" for i in sorted_idx], rotation=45)
    ax.set_xlabel("Feature dimension")
    ax.set_ylabel("Fisher Discriminant Ratio")
    ax.set_title("Top Discriminative Features (GAP Layer)")
    plt.tight_layout()
    save_figure(fig, output_dir, "feature_importance_bars")
    plt.close(fig)

    # Plot 2: PCA variance explained
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    var_exp = pca_result["variance_explained"]
    cumvar = np.cumsum(var_exp)
    axes[0].bar(range(len(var_exp)), var_exp, color="#4CAF50", alpha=0.7)
    axes[0].plot(range(len(var_exp)), cumvar, "r-o", markersize=4)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained")
    axes[0].set_title("PCA Variance Explained")
    axes[0].axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95%")
    axes[0].legend()

    # PCA scatter (first 2 components)
    pca_coords = pca_result["coordinates"]
    real_mask = labels == 0
    synth_mask = labels == 1
    axes[1].scatter(pca_coords[real_mask, 0], pca_coords[real_mask, 1],
                    c="blue", alpha=0.5, s=15, label="Real")
    axes[1].scatter(pca_coords[synth_mask, 0], pca_coords[synth_mask, 1],
                    c="red", alpha=0.5, s=15, label="Synthetic")
    axes[1].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    axes[1].set_title("PCA Projection")
    axes[1].legend()

    plt.tight_layout()
    save_figure(fig, output_dir, "pca_variance")
    plt.close(fig)

    # Plot 3: t-SNE embedding
    if tsne_embedding is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Colored by class
        axes[0].scatter(tsne_embedding[real_mask, 0], tsne_embedding[real_mask, 1],
                        c="blue", alpha=0.5, s=15, label="Real")
        axes[0].scatter(tsne_embedding[synth_mask, 0], tsne_embedding[synth_mask, 1],
                        c="red", alpha=0.5, s=15, label="Synthetic")
        axes[0].set_title("t-SNE by Class")
        axes[0].legend()

        # Colored by z-bin
        scatter = axes[1].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1],
                                  c=zbins, cmap="viridis", alpha=0.5, s=15)
        plt.colorbar(scatter, ax=axes[1], label="Z-bin")
        axes[1].set_title("t-SNE by Z-bin")

        for ax in axes:
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

        plt.tight_layout()
        save_figure(fig, output_dir, "tsne_embedding")
        plt.close(fig)


def _plot_top_feature_maps(
    spatial_maps: dict[int, np.ndarray],
    fisher_ratios: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot spatial activation maps for top discriminative features."""
    if not spatial_maps:
        return

    n_features = len(spatial_maps)
    fig, axes = plt.subplots(1, n_features, figsize=(3 * n_features, 3))
    if n_features == 1:
        axes = [axes]

    for ax, (feat_idx, smap) in zip(axes, spatial_maps.items()):
        im = ax.imshow(smap, cmap="hot", aspect="equal")
        ax.set_title(f"F{feat_idx}\n(FDR={fisher_ratios[feat_idx]:.2f})", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Spatial Patterns of Top Discriminative Features", fontsize=11)
    plt.tight_layout()
    save_figure(fig, output_dir, "top_feature_maps")
    plt.close(fig)


def run_feature_space_analysis(
    cfg: DictConfig,
    experiment_name: str,
    device: str = "cuda",
) -> dict:
    """Run feature space analysis on classifier's learned representation.

    Extracts GAP features, computes Fisher discriminant ratios, runs
    PCA/t-SNE dimensionality reduction, and generates spatial activation
    maps for top discriminative features.

    Args:
        cfg: Diagnostics configuration.
        experiment_name: Experiment to analyze.
        device: Torch device string.

    Returns:
        Dictionary with feature space analysis results.
    """
    xai_cfg = cfg.get("xai", {}).get("feature_space", {})
    pca_components = xai_cfg.get("pca_components", 10)
    tsne_perplexity = xai_cfg.get("tsne_perplexity", 30)
    tsne_seed = xai_cfg.get("tsne_seed", 42)
    top_features = xai_cfg.get("top_features", 5)

    patches_base_dir = Path(cfg.data.patches_base_dir)
    checkpoints_base_dir = Path(cfg.data.checkpoints_base_dir)
    output_base_dir = Path(cfg.output.base_dir)
    n_folds = cfg.dithering.reclassification.n_folds

    # Load patches
    real_patches, synth_patches, real_zbins, synth_zbins = load_patches(
        patches_base_dir, experiment_name
    )
    all_patches = np.concatenate([real_patches, synth_patches], axis=0)
    all_labels = np.concatenate([
        np.zeros(len(real_patches), dtype=np.int32),
        np.ones(len(synth_patches), dtype=np.int32),
    ])
    all_zbins = np.concatenate([real_zbins, synth_zbins])

    # Extract features from all folds and average
    all_features = []
    folds_processed = 0

    for fold_idx in range(n_folds):
        ckpt_path = discover_checkpoint(checkpoints_base_dir, experiment_name, fold_idx, input_mode="joint")
        if ckpt_path is None:
            continue

        logger.info(f"  Fold {fold_idx}: extracting GAP features")
        model = load_model_from_checkpoint(ckpt_path, in_channels=2, device=device)
        features = _extract_gap_features(model, all_patches, device)
        all_features.append(features)
        folds_processed += 1

        # Keep last model for spatial map computation
        last_model = model

    if folds_processed == 0:
        logger.error("No checkpoints found")
        return {"error": "no_checkpoints"}

    # Average features across folds
    features = np.mean(all_features, axis=0)  # (N, D)
    feature_dim = features.shape[1]
    logger.info(f"Extracted features: shape={features.shape}")

    # Fisher Discriminant Ratio
    fisher_ratios = _compute_fisher_discriminant(features, all_labels)
    top_indices = np.argsort(fisher_ratios)[::-1][:top_features]

    # Per-dimension t-tests with FDR correction
    from statsmodels.stats.multitest import multipletests
    real_features = features[all_labels == 0]
    synth_features = features[all_labels == 1]

    p_values = np.zeros(feature_dim)
    t_stats = np.zeros(feature_dim)
    for d in range(feature_dim):
        t_stat, p_val = stats.ttest_ind(real_features[:, d], synth_features[:, d])
        t_stats[d] = t_stat
        p_values[d] = p_val

    _, p_corrected, _, _ = multipletests(p_values, method="fdr_bh")
    n_significant = int((p_corrected < 0.05).sum())

    # PCA
    n_components = min(pca_components, feature_dim, len(features))
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(features)
    variance_explained = pca.explained_variance_ratio_

    # t-SNE
    tsne_embedding = None
    tsne_silhouette = None
    if len(features) > 10:
        perplexity = min(tsne_perplexity, len(features) // 4)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=tsne_seed)
        tsne_embedding = tsne.fit_transform(features)

        # Silhouette score for cluster separability
        if len(np.unique(all_labels)) > 1:
            tsne_silhouette = float(silhouette_score(tsne_embedding, all_labels))

    # Inter-class metrics
    mean_real = real_features.mean(axis=0)
    mean_synth = synth_features.mean(axis=0)
    cosine_dist = float(1.0 - np.dot(mean_real, mean_synth) / (
        np.linalg.norm(mean_real) * np.linalg.norm(mean_synth) + 1e-12
    ))
    euclidean_dist = float(np.linalg.norm(mean_real - mean_synth))

    # Top-k feature spatial maps
    spatial_maps = _compute_feature_spatial_maps(
        last_model, all_patches, top_indices.tolist(), device, n_samples=20
    )
    del last_model
    torch.cuda.empty_cache() if "cuda" in device else None

    # Compile results
    results = {
        "experiment": experiment_name,
        "n_folds": folds_processed,
        "feature_dim": feature_dim,
        "n_samples": len(features),
        "fisher_discriminant": {
            "top_indices": top_indices.tolist(),
            "top_ratios": fisher_ratios[top_indices].tolist(),
            "mean_ratio": float(fisher_ratios.mean()),
            "max_ratio": float(fisher_ratios.max()),
        },
        "statistical_tests": {
            "n_significant_fdr": n_significant,
            "fraction_significant": float(n_significant / feature_dim),
            "min_p_corrected": float(p_corrected.min()),
        },
        "pca": {
            "n_components": n_components,
            "variance_explained": variance_explained.tolist(),
            "cumulative_3d": float(variance_explained[:3].sum()) if len(variance_explained) >= 3 else float(variance_explained.sum()),
        },
        "cluster_metrics": {
            "tsne_silhouette": tsne_silhouette,
            "inter_class_cosine_distance": cosine_dist,
            "inter_class_euclidean_distance": euclidean_dist,
        },
    }

    # Save results
    output_dir = ensure_output_dir(output_base_dir, experiment_name, "feature_space")
    save_result_json(results, output_dir / "feature_space_results.json")

    # Save raw features
    np.savez_compressed(
        output_dir / "features.npz",
        features=features,
        labels=all_labels,
        zbins=all_zbins,
        fisher_ratios=fisher_ratios,
        pca_coords=pca_coords,
        tsne_embedding=tsne_embedding if tsne_embedding is not None else np.array([]),
    )

    # CSV for cross-experiment
    csv_row = {
        "experiment": experiment_name,
        "fisher_max": float(fisher_ratios.max()),
        "fisher_mean": float(fisher_ratios.mean()),
        "n_significant_features": n_significant,
        "pca_cumvar_3d": results["pca"]["cumulative_3d"],
        "tsne_silhouette": tsne_silhouette,
        "cosine_distance": cosine_dist,
        "euclidean_distance": euclidean_dist,
    }
    save_csv(pd.DataFrame([csv_row]), output_dir / "feature_space_summary.csv")

    # Feature importance CSV
    importance_df = pd.DataFrame({
        "feature_idx": range(feature_dim),
        "fisher_ratio": fisher_ratios,
        "t_statistic": t_stats,
        "p_value_corrected": p_corrected,
        "significant": p_corrected < 0.05,
    }).sort_values("fisher_ratio", ascending=False)
    save_csv(importance_df, output_dir / "feature_importance.csv")

    # Visualizations
    pca_result = {
        "variance_explained": variance_explained,
        "coordinates": pca_coords,
    }
    _plot_feature_analysis(
        features, all_labels, all_zbins, fisher_ratios,
        pca_result, tsne_embedding, output_dir,
    )
    _plot_top_feature_maps(spatial_maps, fisher_ratios, output_dir)

    logger.info(
        f"Feature space analysis: dim={feature_dim}, "
        f"n_significant={n_significant}/{feature_dim}, "
        f"silhouette={tsne_silhouette:.3f}" if tsne_silhouette else
        f"Feature space analysis: dim={feature_dim}, n_significant={n_significant}/{feature_dim}"
    )
    return results
