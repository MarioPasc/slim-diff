"""Channel decomposition analysis for real vs. synthetic classifier.

Decomposes the classifier's decision into image-channel vs mask-channel
contributions using input gradients and ablation tests. Answers the question:
"Is the classifier detecting image artifacts or mask artifacts?"

Scientific basis:
    Input gradients ∂logit/∂x decompose naturally into per-channel attribution
    when the input is multi-channel. The gradient magnitude ||∂logit/∂x_ch||
    quantifies each channel's contribution to the decision. Ablation (zeroing
    one channel) provides complementary causal evidence.

    Reference: Simonyan, K., Vedaldi, A., & Zisserman, A. (2014).
    "Deep Inside Convolutional Networks: Visualising Image Classification
    Models and Saliency Maps." ICLR 2014 Workshop.
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


def _compute_input_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: str,
) -> np.ndarray:
    """Compute input gradients for a batch of samples.

    Args:
        model: Classifier in eval mode.
        inputs: Tensor of shape (N, C, H, W).
        device: Torch device string.

    Returns:
        Gradient array of shape (N, C, H, W).
    """
    model.eval()
    all_grads = []

    for i in range(len(inputs)):
        x = inputs[i:i+1].to(device).requires_grad_(True)
        logit = model(x).squeeze()
        model.zero_grad()
        logit.backward()
        all_grads.append(x.grad.detach().cpu().numpy())

    return np.concatenate(all_grads, axis=0)


def _compute_ablation(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    ablation_value: float,
    device: str,
) -> dict[str, np.ndarray]:
    """Compute predictions with each channel ablated.

    Args:
        model: Classifier in eval mode.
        inputs: Tensor of shape (N, 2, H, W) - joint mode only.
        ablation_value: Value to fill ablated channel with.
        device: Torch device string.

    Returns:
        Dict with 'full', 'image_ablated', 'mask_ablated' logit arrays.
    """
    model.eval()
    results = {"full": [], "image_ablated": [], "mask_ablated": []}

    with torch.no_grad():
        batch_size = 32
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start+batch_size].to(device)

            # Full prediction
            logits_full = model(batch).squeeze(-1)
            results["full"].append(logits_full.cpu().numpy())

            # Ablate image channel (ch 0)
            ablated_img = batch.clone()
            ablated_img[:, 0, :, :] = ablation_value
            logits_img_abl = model(ablated_img).squeeze(-1)
            results["image_ablated"].append(logits_img_abl.cpu().numpy())

            # Ablate mask channel (ch 1)
            ablated_mask = batch.clone()
            ablated_mask[:, 1, :, :] = ablation_value
            logits_mask_abl = model(ablated_mask).squeeze(-1)
            results["mask_ablated"].append(logits_mask_abl.cpu().numpy())

    return {k: np.concatenate(v) for k, v in results.items()}


def _plot_channel_contributions(
    results: dict,
    output_dir: Path,
) -> None:
    """Generate visualization of channel contribution analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Gradient-based contributions per class
    classes = ["Real", "Synthetic"]
    image_fracs = [
        results["per_class"]["real"]["gradient"]["image_fraction"],
        results["per_class"]["synthetic"]["gradient"]["image_fraction"],
    ]
    mask_fracs = [
        results["per_class"]["real"]["gradient"]["mask_fraction"],
        results["per_class"]["synthetic"]["gradient"]["mask_fraction"],
    ]

    x = np.arange(len(classes))
    width = 0.35
    axes[0].bar(x - width/2, image_fracs, width, label="Image channel", color="#2196F3")
    axes[0].bar(x + width/2, mask_fracs, width, label="Mask channel", color="#FF9800")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes)
    axes[0].set_ylabel("Gradient magnitude fraction")
    axes[0].set_title("Gradient-based Channel Contribution")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Panel 2: Ablation-based delta logit per class
    image_deltas = [
        results["per_class"]["real"]["ablation"]["image_ablation_delta"],
        results["per_class"]["synthetic"]["ablation"]["image_ablation_delta"],
    ]
    mask_deltas = [
        results["per_class"]["real"]["ablation"]["mask_ablation_delta"],
        results["per_class"]["synthetic"]["ablation"]["mask_ablation_delta"],
    ]

    axes[1].bar(x - width/2, image_deltas, width, label="Image ablated", color="#2196F3")
    axes[1].bar(x + width/2, mask_deltas, width, label="Mask ablated", color="#FF9800")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes)
    axes[1].set_ylabel("|Δ logit| when channel ablated")
    axes[1].set_title("Ablation Impact (higher = more important)")
    axes[1].legend()

    # Panel 3: Per-z-bin gradient contribution (image channel)
    if results.get("per_zbin"):
        zbins = sorted(results["per_zbin"].keys(), key=int)
        img_fracs_zbin = [
            results["per_zbin"][z]["gradient"]["image_fraction"] for z in zbins
        ]
        axes[2].bar(range(len(zbins)), img_fracs_zbin, color="#2196F3", alpha=0.7)
        axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("Z-bin index")
        axes[2].set_ylabel("Image channel fraction")
        axes[2].set_title("Image Contribution by Z-bin")
        # Only show every Nth tick if too many
        if len(zbins) > 15:
            step = max(1, len(zbins) // 10)
            axes[2].set_xticks(range(0, len(zbins), step))
            axes[2].set_xticklabels([zbins[i] for i in range(0, len(zbins), step)])
        else:
            axes[2].set_xticks(range(len(zbins)))
            axes[2].set_xticklabels(zbins)
    else:
        axes[2].text(0.5, 0.5, "No per-zbin data", ha="center", va="center",
                     transform=axes[2].transAxes)

    plt.tight_layout()
    save_figure(fig, output_dir, "channel_contributions")
    plt.close(fig)


def _plot_ablation_distributions(
    ablation_results: dict,
    labels: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot distribution of logit changes under ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    full = ablation_results["full"]
    img_abl = ablation_results["image_ablated"]
    mask_abl = ablation_results["mask_ablated"]

    delta_img = np.abs(full - img_abl)
    delta_mask = np.abs(full - mask_abl)

    for ax, class_idx, class_name in [(axes[0], 0, "Real"), (axes[1], 1, "Synthetic")]:
        mask = labels == class_idx
        ax.hist(delta_img[mask], bins=30, alpha=0.6, label="Image ablated", color="#2196F3")
        ax.hist(delta_mask[mask], bins=30, alpha=0.6, label="Mask ablated", color="#FF9800")
        ax.set_xlabel("|Δ logit|")
        ax.set_ylabel("Count")
        ax.set_title(f"Ablation Impact ({class_name})")
        ax.legend()

    plt.tight_layout()
    save_figure(fig, output_dir, "ablation_distributions")
    plt.close(fig)


def run_channel_decomposition(
    cfg: DictConfig,
    experiment_name: str,
    device: str = "cuda",
) -> dict:
    """Run channel decomposition analysis across all folds.

    Computes input gradients and ablation tests to determine whether
    the classifier relies more on the image channel or the mask channel
    for its real vs. synthetic discrimination.

    Args:
        cfg: Diagnostics configuration.
        experiment_name: Experiment to analyze.
        device: Torch device string.

    Returns:
        Dictionary with channel contribution metrics.
    """
    n_folds = cfg.dithering.reclassification.n_folds
    patches_base_dir = Path(cfg.data.patches_base_dir)
    checkpoints_base_dir = Path(cfg.data.checkpoints_base_dir)
    output_base_dir = Path(cfg.output.base_dir)
    ablation_value = cfg.get("xai", {}).get("channel_decomposition", {}).get(
        "ablation_baseline", -1.0
    )

    # Load patches (joint mode only - need both channels)
    real_patches, synth_patches, real_zbins, synth_zbins = load_patches(
        patches_base_dir, experiment_name
    )

    # Combine for unified processing
    all_patches = np.concatenate([real_patches, synth_patches], axis=0)
    all_labels = np.concatenate([
        np.zeros(len(real_patches), dtype=np.int32),
        np.ones(len(synth_patches), dtype=np.int32),
    ])
    all_zbins = np.concatenate([real_zbins, synth_zbins])

    # Accumulate gradient magnitudes across folds
    all_grad_image_mag = []
    all_grad_mask_mag = []
    all_ablation_results = {"full": [], "image_ablated": [], "mask_ablated": []}
    folds_processed = 0

    for fold_idx in range(n_folds):
        ckpt_path = discover_checkpoint(
            checkpoints_base_dir, experiment_name, fold_idx
        )
        if ckpt_path is None:
            logger.warning(f"Checkpoint not found for fold {fold_idx}")
            continue

        logger.info(f"  Fold {fold_idx}: {ckpt_path.name}")
        model = load_model_from_checkpoint(ckpt_path, in_channels=2, device=device)

        # Compute input gradients
        inputs = torch.from_numpy(all_patches).float()
        grads = _compute_input_gradients(model, inputs, device)

        # Per-channel gradient magnitude
        grad_image_mag = np.abs(grads[:, 0, :, :]).mean(axis=(1, 2))  # (N,)
        grad_mask_mag = np.abs(grads[:, 1, :, :]).mean(axis=(1, 2))  # (N,)
        all_grad_image_mag.append(grad_image_mag)
        all_grad_mask_mag.append(grad_mask_mag)

        # Ablation test
        ablation = _compute_ablation(model, inputs, ablation_value, device)
        for key in all_ablation_results:
            all_ablation_results[key].append(ablation[key])

        folds_processed += 1
        del model
        torch.cuda.empty_cache() if "cuda" in device else None

    if folds_processed == 0:
        logger.error("No checkpoints found for any fold")
        return {"error": "no_checkpoints"}

    # Average across folds
    grad_image = np.mean(all_grad_image_mag, axis=0)  # (N,)
    grad_mask = np.mean(all_grad_mask_mag, axis=0)  # (N,)
    total_grad = grad_image + grad_mask + 1e-12
    image_fraction = grad_image / total_grad
    mask_fraction = grad_mask / total_grad

    ablation_full = np.mean(all_ablation_results["full"], axis=0)
    ablation_img = np.mean(all_ablation_results["image_ablated"], axis=0)
    ablation_mask = np.mean(all_ablation_results["mask_ablated"], axis=0)
    delta_image = np.abs(ablation_full - ablation_img)
    delta_mask = np.abs(ablation_full - ablation_mask)

    # Compute per-class statistics
    def _class_stats(mask):
        return {
            "gradient": {
                "image_magnitude": float(grad_image[mask].mean()),
                "mask_magnitude": float(grad_mask[mask].mean()),
                "image_fraction": float(image_fraction[mask].mean()),
                "mask_fraction": float(mask_fraction[mask].mean()),
            },
            "ablation": {
                "image_ablation_delta": float(delta_image[mask].mean()),
                "mask_ablation_delta": float(delta_mask[mask].mean()),
            },
        }

    real_mask = all_labels == 0
    synth_mask = all_labels == 1

    results = {
        "experiment": experiment_name,
        "n_folds": folds_processed,
        "n_real": int(real_mask.sum()),
        "n_synth": int(synth_mask.sum()),
        "ablation_baseline": ablation_value,
        "overall": {
            "gradient": {
                "image_magnitude_mean": float(grad_image.mean()),
                "mask_magnitude_mean": float(grad_mask.mean()),
                "image_fraction": float(image_fraction.mean()),
                "mask_fraction": float(mask_fraction.mean()),
            },
            "ablation": {
                "image_ablation_delta_mean": float(delta_image.mean()),
                "mask_ablation_delta_mean": float(delta_mask.mean()),
            },
            "dominant_channel": "image" if image_fraction.mean() > 0.5 else "mask",
        },
        "per_class": {
            "real": _class_stats(real_mask),
            "synthetic": _class_stats(synth_mask),
        },
    }

    # Per-z-bin breakdown
    unique_zbins = np.unique(all_zbins)
    per_zbin = {}
    for zb in unique_zbins:
        zb_mask = all_zbins == zb
        if zb_mask.sum() < 3:
            continue
        per_zbin[str(int(zb))] = {
            "gradient": {
                "image_fraction": float(image_fraction[zb_mask].mean()),
                "mask_fraction": float(mask_fraction[zb_mask].mean()),
            },
            "ablation": {
                "image_ablation_delta": float(delta_image[zb_mask].mean()),
                "mask_ablation_delta": float(delta_mask[zb_mask].mean()),
            },
            "n_samples": int(zb_mask.sum()),
        }
    results["per_zbin"] = per_zbin

    # Save results
    output_dir = ensure_output_dir(output_base_dir, experiment_name, "channel_decomposition")
    save_result_json(results, output_dir / "channel_decomposition_results.json")

    # CSV for cross-experiment comparison
    csv_rows = [{
        "experiment": experiment_name,
        "image_grad_fraction": results["overall"]["gradient"]["image_fraction"],
        "mask_grad_fraction": results["overall"]["gradient"]["mask_fraction"],
        "image_ablation_delta": results["overall"]["ablation"]["image_ablation_delta_mean"],
        "mask_ablation_delta": results["overall"]["ablation"]["mask_ablation_delta_mean"],
        "dominant_channel": results["overall"]["dominant_channel"],
        "image_frac_real": results["per_class"]["real"]["gradient"]["image_fraction"],
        "image_frac_synth": results["per_class"]["synthetic"]["gradient"]["image_fraction"],
    }]
    save_csv(pd.DataFrame(csv_rows), output_dir / "channel_decomposition_summary.csv")

    # Visualizations
    _plot_channel_contributions(results, output_dir)
    _plot_ablation_distributions(
        {"full": ablation_full, "image_ablated": ablation_img, "mask_ablated": ablation_mask},
        all_labels, output_dir,
    )

    logger.info(
        f"Channel decomposition complete: "
        f"image={results['overall']['gradient']['image_fraction']:.3f}, "
        f"mask={results['overall']['gradient']['mask_fraction']:.3f}, "
        f"dominant={results['overall']['dominant_channel']}"
    )
    return results
