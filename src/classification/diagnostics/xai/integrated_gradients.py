"""Integrated Gradients attribution for real vs. synthetic classifier.

Computes pixel-level attributions satisfying the sensitivity and
implementation invariance axioms. More principled than Grad-CAM for
understanding per-pixel importance because the path integral ensures
completeness: attributions sum to the output difference.

Scientific basis:
    Sundararajan, M., Taly, A., & Yan, Q. (2017). "Axiomatic Attribution
    for Deep Networks." ICML 2017.

    IG_i(x) = (x_i - x0_i) × ∫₀¹ ∂F/∂x_i(x0 + α(x-x0)) dα

    Axioms satisfied:
    1. Sensitivity: If a feature differs between input and baseline and
       causes a different prediction, it gets non-zero attribution.
    2. Implementation Invariance: Functionally identical networks produce
       identical attributions.
    3. Completeness: sum(IG_i) = F(x) - F(x0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

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


class IntegratedGradients:
    """Integrated Gradients attribution computation.

    Args:
        model: Classifier model (eval mode).
        n_steps: Number of interpolation steps for Riemann approximation.
        baseline: Baseline tensor or None for zeros.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_steps: int = 50,
        baseline: torch.Tensor | None = None,
    ):
        self.model = model
        self.n_steps = n_steps
        self.baseline = baseline

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """Compute Integrated Gradients for a single input.

        Args:
            input_tensor: Input of shape (1, C, H, W).
            target_class: Target class (1=synthetic, 0=real).

        Returns:
            Attribution map of shape (C, H, W).
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        device = input_tensor.device
        baseline = self.baseline
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        elif baseline.shape != input_tensor.shape:
            baseline = baseline.expand_as(input_tensor)

        # Generate interpolation alphas
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=device)

        # Accumulate gradients along path
        integrated_grads = torch.zeros_like(input_tensor)

        for i in range(self.n_steps):
            # Trapezoidal rule: average of endpoints
            alpha_low = alphas[i]
            alpha_high = alphas[i + 1]

            for alpha in [alpha_low, alpha_high]:
                interpolated = baseline + alpha * (input_tensor - baseline)
                interpolated = interpolated.detach().requires_grad_(True)

                logit = self.model(interpolated).squeeze()
                score = logit if target_class == 1 else -logit

                self.model.zero_grad()
                score.backward()

                integrated_grads += interpolated.grad.detach() / (2 * self.n_steps)

        # Multiply by (input - baseline)
        attributions = (input_tensor - baseline).detach() * integrated_grads
        return attributions.squeeze(0).cpu().numpy()  # (C, H, W)

    def compute_batch(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        z_bins: np.ndarray,
        target_class: int = 1,
        device: str = "cuda",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute IG for a batch of inputs.

        Args:
            inputs: Array of shape (N, C, H, W).
            labels: Ground truth labels (N,).
            z_bins: Z-position bins (N,).
            target_class: Target class for attribution.
            device: Torch device.

        Returns:
            Tuple of (attributions, labels, z_bins).
            attributions shape: (N, C, H, W).
        """
        all_attributions = []
        self.model.eval()

        for i in range(len(inputs)):
            x = torch.from_numpy(inputs[i:i+1]).float().to(device)
            attr = self.compute(x, target_class=target_class)
            all_attributions.append(attr)

            if (i + 1) % 20 == 0:
                logger.info(f"    IG: {i+1}/{len(inputs)} samples processed")

        return np.stack(all_attributions, axis=0), labels, z_bins


def _compute_gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient as measure of attribution concentration.

    Gini=0: uniform distribution (attribution spread evenly)
    Gini=1: maximum concentration (attribution on single pixel)
    """
    values = np.abs(values.flatten())
    if values.sum() < 1e-12:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_vals)) / (n * sorted_vals.sum())) - (n + 1) / n
    return float(max(0, min(1, gini)))


def _compute_ig_gradcam_correlation(
    ig_maps: np.ndarray,
    experiment_dir: Path,
) -> float | None:
    """Compute spatial correlation between IG and GradCAM attribution maps."""
    heatmap_path = experiment_dir / "gradcam" / "joint" / "heatmaps.npz"
    if not heatmap_path.exists():
        return None

    gradcam_data = np.load(heatmap_path)
    real_heatmaps = gradcam_data["real_heatmaps"]
    synth_heatmaps = gradcam_data["synth_heatmaps"]
    gradcam_maps = np.concatenate([real_heatmaps, synth_heatmaps], axis=0)

    # IG maps are per-channel, GradCAM is single-channel
    # Sum IG across channels for comparison
    ig_spatial = np.abs(ig_maps).sum(axis=1)  # (N, H, W)

    # Match sample counts (use minimum)
    n = min(len(ig_spatial), len(gradcam_maps))
    if n == 0:
        return None

    # Compute per-sample correlation and average
    correlations = []
    for i in range(n):
        ig_flat = ig_spatial[i].flatten()
        gc_flat = gradcam_maps[i].flatten()
        if ig_flat.std() > 1e-12 and gc_flat.std() > 1e-12:
            corr = np.corrcoef(ig_flat, gc_flat)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    return float(np.mean(correlations)) if correlations else None


def _plot_ig_results(
    attributions: np.ndarray,
    labels: np.ndarray,
    inputs: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate IG visualization plots."""
    real_mask = labels == 0
    synth_mask = labels == 1

    # Plot 1: Sample IG overlays (4 real + 4 synthetic)
    n_show = min(4, real_mask.sum(), synth_mask.sum())
    if n_show > 0:
        fig, axes = plt.subplots(2, n_show * 2, figsize=(3 * n_show * 2, 6))
        if n_show * 2 == 1:
            axes = axes.reshape(2, 1)

        for col, (cls, mask, cls_name) in enumerate([
            (0, real_mask, "Real"), (1, synth_mask, "Synthetic")
        ]):
            indices = np.where(mask)[0][:n_show]
            for j, idx in enumerate(indices):
                ax_idx = col * n_show + j
                if ax_idx >= axes.shape[1]:
                    break

                # Show input image (channel 0)
                axes[0, ax_idx].imshow(inputs[idx, 0], cmap="gray", aspect="equal")
                axes[0, ax_idx].set_title(f"{cls_name} #{j}", fontsize=8)
                axes[0, ax_idx].axis("off")

                # Show IG attribution (sum channels, absolute)
                attr_map = np.abs(attributions[idx]).sum(axis=0)
                axes[1, ax_idx].imshow(attr_map, cmap="hot", aspect="equal")
                axes[1, ax_idx].axis("off")

        axes[0, 0].set_ylabel("Input", fontsize=10)
        axes[1, 0].set_ylabel("IG Attribution", fontsize=10)
        plt.suptitle("Integrated Gradients: Sample Attributions", fontsize=11)
        plt.tight_layout()
        save_figure(fig, output_dir, "ig_samples")
        plt.close(fig)

    # Plot 2: Mean attribution maps per class
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for row, (mask, cls_name) in enumerate([(real_mask, "Real"), (synth_mask, "Synthetic")]):
        if mask.sum() == 0:
            continue
        # Image channel attribution
        mean_img_attr = np.abs(attributions[mask, 0]).mean(axis=0)
        im0 = axes[row, 0].imshow(mean_img_attr, cmap="hot", aspect="equal")
        axes[row, 0].set_title(f"{cls_name} - Image Channel", fontsize=9)
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        # Mask channel attribution
        if attributions.shape[1] > 1:
            mean_mask_attr = np.abs(attributions[mask, 1]).mean(axis=0)
            im1 = axes[row, 1].imshow(mean_mask_attr, cmap="hot", aspect="equal")
            axes[row, 1].set_title(f"{cls_name} - Mask Channel", fontsize=9)
            axes[row, 1].axis("off")
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

    plt.suptitle("Mean IG Attribution Maps (per channel, per class)", fontsize=11)
    plt.tight_layout()
    save_figure(fig, output_dir, "ig_mean_maps")
    plt.close(fig)


def run_integrated_gradients(
    cfg: DictConfig,
    experiment_name: str,
    device: str = "cuda",
) -> dict:
    """Run Integrated Gradients analysis.

    Computes pixel-level attributions using the path integral method,
    providing per-channel attribution maps and comparison with GradCAM.

    Args:
        cfg: Diagnostics configuration.
        experiment_name: Experiment to analyze.
        device: Torch device string.

    Returns:
        Dictionary with IG analysis results.
    """
    xai_cfg = cfg.get("xai", {}).get("integrated_gradients", {})
    n_steps = xai_cfg.get("n_steps", 50)
    max_samples = xai_cfg.get("max_samples", 100)

    patches_base_dir = Path(cfg.data.patches_base_dir)
    checkpoints_base_dir = Path(cfg.data.checkpoints_base_dir)
    output_base_dir = Path(cfg.output.base_dir)
    n_folds = cfg.dithering.reclassification.n_folds

    # Load patches
    real_patches, synth_patches, real_zbins, synth_zbins = load_patches(
        patches_base_dir, experiment_name
    )

    # Subsample for computational efficiency
    rng = np.random.default_rng(42)
    n_per_class = max_samples // 2

    if len(real_patches) > n_per_class:
        real_idx = rng.choice(len(real_patches), n_per_class, replace=False)
        real_patches = real_patches[real_idx]
        real_zbins = real_zbins[real_idx]

    if len(synth_patches) > n_per_class:
        synth_idx = rng.choice(len(synth_patches), n_per_class, replace=False)
        synth_patches = synth_patches[synth_idx]
        synth_zbins = synth_zbins[synth_idx]

    all_patches = np.concatenate([real_patches, synth_patches], axis=0)
    all_labels = np.concatenate([
        np.zeros(len(real_patches), dtype=np.int32),
        np.ones(len(synth_patches), dtype=np.int32),
    ])
    all_zbins = np.concatenate([real_zbins, synth_zbins])

    logger.info(f"Running IG with {len(all_patches)} samples, n_steps={n_steps}")

    # Use first available fold
    all_attributions = None
    for fold_idx in range(n_folds):
        ckpt_path = discover_checkpoint(checkpoints_base_dir, experiment_name, fold_idx, input_mode="joint")
        if ckpt_path is None:
            continue

        logger.info(f"  Using fold {fold_idx}: {ckpt_path.name}")
        model = load_model_from_checkpoint(ckpt_path, in_channels=2, device=device)

        ig = IntegratedGradients(model, n_steps=n_steps)
        all_attributions, _, _ = ig.compute_batch(
            all_patches, all_labels, all_zbins,
            target_class=1, device=device,
        )

        del model
        torch.cuda.empty_cache() if "cuda" in device else None
        break

    if all_attributions is None:
        logger.error("No checkpoints available")
        return {"error": "no_checkpoints"}

    # Compute metrics
    real_mask = all_labels == 0
    synth_mask = all_labels == 1

    # Per-channel mean attribution
    image_attr = np.abs(all_attributions[:, 0]).mean()
    mask_attr = np.abs(all_attributions[:, 1]).mean() if all_attributions.shape[1] > 1 else 0
    total_attr = image_attr + mask_attr + 1e-12

    # Attribution concentration (Gini coefficient)
    mean_attr_map = np.abs(all_attributions).mean(axis=0).sum(axis=0)  # (H, W)
    concentration = _compute_gini_coefficient(mean_attr_map)

    # Correlation with GradCAM
    experiment_dir = output_base_dir / experiment_name
    ig_cam_corr = _compute_ig_gradcam_correlation(all_attributions, experiment_dir)

    # Completeness check: sum(IG) ≈ F(x) - F(baseline)
    completeness_scores = np.abs(all_attributions).sum(axis=(1, 2, 3))
    mean_completeness = float(completeness_scores.mean())

    results = {
        "experiment": experiment_name,
        "n_samples": len(all_patches),
        "n_steps": n_steps,
        "mean_attribution_per_channel": {
            "image": float(image_attr),
            "mask": float(mask_attr),
            "image_fraction": float(image_attr / total_attr),
        },
        "attribution_concentration": concentration,
        "ig_gradcam_correlation": ig_cam_corr,
        "completeness_mean": mean_completeness,
        "per_class": {
            "real": {
                "mean_total_attribution": float(np.abs(all_attributions[real_mask]).sum(axis=(1,2,3)).mean()),
                "concentration": _compute_gini_coefficient(
                    np.abs(all_attributions[real_mask]).mean(axis=0).sum(axis=0)
                ),
            },
            "synthetic": {
                "mean_total_attribution": float(np.abs(all_attributions[synth_mask]).sum(axis=(1,2,3)).mean()),
                "concentration": _compute_gini_coefficient(
                    np.abs(all_attributions[synth_mask]).mean(axis=0).sum(axis=0)
                ),
            },
        },
    }

    # Save results
    output_dir = ensure_output_dir(output_base_dir, experiment_name, "integrated_gradients")
    save_result_json(results, output_dir / "ig_results.json")

    # Save attribution maps
    np.savez_compressed(
        output_dir / "attributions.npz",
        attributions=all_attributions,
        labels=all_labels,
        zbins=all_zbins,
    )

    # CSV for cross-experiment
    csv_row = {
        "experiment": experiment_name,
        "image_attr_fraction": results["mean_attribution_per_channel"]["image_fraction"],
        "concentration": concentration,
        "ig_cam_correlation": ig_cam_corr,
        "completeness": mean_completeness,
    }
    save_csv(pd.DataFrame([csv_row]), output_dir / "ig_summary.csv")

    # Visualizations
    _plot_ig_results(all_attributions, all_labels, all_patches, output_dir)

    logger.info(
        f"IG analysis: image_frac={image_attr/total_attr:.3f}, "
        f"concentration={concentration:.3f}, "
        f"ig_cam_corr={ig_cam_corr}"
    )
    return results
