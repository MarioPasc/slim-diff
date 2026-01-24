"""Grad-CAM implementation for real vs. synthetic classifier analysis.

Implements Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)
to visualize which spatial regions the SimpleCNNClassifier uses to distinguish
real from synthetic MRI patches.

Reference:
    Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
    (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization. ICCV 2017.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.classification.diagnostics.utils import (
    ensure_output_dir,
    load_patches,
    save_csv,
    save_result_json,
)
from src.classification.diagnostics.xai.aggregation import (
    aggregate_heatmaps,
    compute_attention_difference,
    plot_gradcam_results,
)
from src.classification.training.lit_module import ClassificationLightningModule

logger = logging.getLogger(__name__)


@dataclass
class GradCAMResult:
    """Result of a single Grad-CAM computation.

    Attributes:
        heatmap: Normalized activation heatmap of shape (H, W), values in [0, 1].
        prediction: Model sigmoid probability for the target class.
        label: Ground truth label (0 = real, 1 = synthetic).
        z_bin: Axial z-position bin of the input patch.
    """

    heatmap: np.ndarray
    prediction: float
    label: int
    z_bin: int


class GradCAM:
    """Grad-CAM computation for convolutional classifiers.

    Registers forward and backward hooks on the target convolutional layer to
    capture activations and gradients. Computes class-discriminative saliency
    maps by weighting feature maps with their gradient-derived importance.

    Args:
        model: The classifier model (expected to be a SimpleCNNClassifier).
        target_layer: The convolutional layer to compute Grad-CAM for.
            Typically ``model.conv_layers[-1].block[0]`` (last Conv2d).

    Example:
        >>> from src.classification.models.simple_cnn import SimpleCNNClassifier
        >>> model = SimpleCNNClassifier(in_channels=2)
        >>> target_layer = model.conv_layers[-1].block[0]
        >>> gradcam = GradCAM(model, target_layer)
        >>> heatmap = gradcam.compute(input_tensor)
        >>> gradcam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Forward hook: store activations from the target layer."""
        self._activations = output.detach()

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Backward hook: store gradients flowing into the target layer."""
        self._gradients = grad_output[0].detach()

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single input.

        Args:
            input_tensor: Input tensor of shape (1, C, H, W).
            target_class: Class index for gradient computation. For binary
                classification with a single logit output, use None (default)
                to compute w.r.t. the raw logit, 1 for synthetic class
                (positive logit direction), or 0 for real class (negative
                logit direction).

        Returns:
            Heatmap of shape (H, W) with values in [0, 1], upsampled to
            match the input spatial dimensions.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)  # (1, 1)
        logit = logits.squeeze()

        # For binary classification: compute gradient w.r.t. the logit
        # target_class=1 (synthetic): maximize logit -> gradient of logit
        # target_class=0 (real): maximize negative logit -> gradient of -logit
        if target_class == 0:
            score = -logit
        else:
            score = logit

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # Retrieve cached activations and gradients
        activations = self._activations  # (1, C_feat, h, w)
        gradients = self._gradients  # (1, C_feat, h, w)

        if activations is None or gradients is None:
            raise RuntimeError(
                "Activations or gradients were not captured. "
                "Ensure the target layer is part of the forward pass."
            )

        # Global average pooling of gradients -> channel importance weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C_feat, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Upsample to input spatial dimensions
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(
            cam, size=(input_h, input_w), mode="bilinear", align_corners=False
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def compute_batch(
        self,
        inputs: torch.Tensor,
        labels: np.ndarray,
        z_bins: np.ndarray,
        target_class: int = 1,
    ) -> list[GradCAMResult]:
        """Compute Grad-CAM for a batch of inputs.

        Processes each sample individually (required for per-sample gradients)
        and packages results with metadata.

        Args:
            inputs: Batch tensor of shape (N, C, H, W).
            labels: Ground truth labels array of shape (N,).
            z_bins: Z-position bins array of shape (N,).
            target_class: Target class for Grad-CAM (default: 1 = synthetic).

        Returns:
            List of GradCAMResult instances, one per input sample.
        """
        device = next(self.model.parameters()).device
        results: list[GradCAMResult] = []

        for i in range(len(inputs)):
            input_tensor = inputs[i : i + 1].to(device)

            # Compute heatmap
            heatmap = self.compute(input_tensor, target_class=target_class)

            # Get prediction probability
            with torch.no_grad():
                logit = self.model(input_tensor).squeeze()
                prob = torch.sigmoid(logit).item()

            results.append(
                GradCAMResult(
                    heatmap=heatmap,
                    prediction=prob,
                    label=int(labels[i]),
                    z_bin=int(z_bins[i]),
                )
            )

        return results

    def remove_hooks(self) -> None:
        """Remove registered forward and backward hooks.

        Must be called when done with Grad-CAM to avoid memory leaks and
        interference with subsequent model operations.
        """
        self._forward_hook.remove()
        self._backward_hook.remove()
        self._activations = None
        self._gradients = None
        logger.debug("Grad-CAM hooks removed.")


def _load_model_from_checkpoint(
    ckpt_path: Path,
    in_channels: int,
    device: str,
) -> nn.Module:
    """Load a trained classifier from a checkpoint.

    Creates a minimal compatible config for ClassificationLightningModule
    and loads the model weights from the checkpoint.

    Args:
        ckpt_path: Path to the .ckpt file.
        in_channels: Number of input channels for the model.
        device: Target device string.

    Returns:
        The loaded model in eval mode on the specified device.
    """
    from omegaconf import OmegaConf

    # Create minimal config compatible with ClassificationLightningModule
    # The model factory resolves relative config_path from src/classification/config/
    train_cfg = OmegaConf.create({
        "training": {
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "max_epochs": 50,
            "scheduler": {"type": "reduce_on_plateau", "factor": 0.5, "patience": 5, "min_lr": 1e-6},
            "early_stopping": {"monitor": "val/auc", "patience": 10, "min_delta": 0.001},
        },
        "model": {"config_path": "models/simple_cnn.yaml"},
    })

    lit_module = ClassificationLightningModule.load_from_checkpoint(
        str(ckpt_path),
        cfg=train_cfg,
        in_channels=in_channels,
        map_location=device,
    )
    lit_module.eval()
    model = lit_module.model.to(device)
    model.eval()
    return model


def _discover_checkpoint(
    checkpoints_base_dir: Path,
    experiment_name: str,
    fold_idx: int,
) -> Path | None:
    """Discover the best checkpoint file for a given experiment and fold.

    Searches all subdirectories under the experiment's checkpoint directory
    for fold checkpoint files. Handles versioned checkpoints (e.g., -v1, -v2)
    by preferring the base name without version suffix.

    Args:
        checkpoints_base_dir: Base directory containing experiment subdirs.
        experiment_name: Experiment name (e.g., 'epsilon_lp_1.5').
        fold_idx: Fold index to find checkpoint for.

    Returns:
        Path to the best checkpoint file, or None if not found.
    """
    exp_dir = checkpoints_base_dir / experiment_name
    if not exp_dir.exists():
        return None

    # Find all subdirectories that contain checkpoints
    base_name = f"fold{fold_idx}_best.ckpt"
    candidates: list[Path] = []

    for subdir in sorted(exp_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Prefer the base checkpoint (no version suffix)
        base_ckpt = subdir / base_name
        if base_ckpt.exists():
            candidates.append(base_ckpt)
        else:
            # Look for versioned checkpoints and take the latest
            versioned = sorted(
                subdir.glob(f"fold{fold_idx}_best-v*.ckpt"),
                key=lambda p: p.name,
            )
            if versioned:
                candidates.append(versioned[-1])

    if not candidates:
        return None

    # If multiple subdirectories have checkpoints, prefer the most recent
    # (by modification time)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _determine_in_channels(mode: str) -> int:
    """Map input mode string to channel count.

    Args:
        mode: One of 'joint', 'image_only', 'mask_only'.

    Returns:
        Number of input channels (2 for joint, 1 otherwise).
    """
    if mode == "joint":
        return 2
    elif mode in ("image_only", "mask_only"):
        return 1
    else:
        raise ValueError(f"Unknown input mode: {mode}")


def _select_patches_by_mode(
    patches: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Select channels from patches based on input mode.

    Args:
        patches: Array of shape (N, 2, H, W).
        mode: One of 'joint', 'image_only', 'mask_only'.

    Returns:
        Array of shape (N, C, H, W) where C depends on mode.
    """
    if mode == "joint":
        return patches
    elif mode == "image_only":
        return patches[:, 0:1, :, :]
    elif mode == "mask_only":
        return patches[:, 1:2, :, :]
    else:
        raise ValueError(f"Unknown input mode: {mode}")


def run_gradcam_analysis(
    cfg: DictConfig,
    experiment_name: str,
    device: str = "cuda",
) -> dict:
    """Run full Grad-CAM analysis across all folds and input modes.

    This is the main entry point for Grad-CAM-based interpretability analysis.
    For each fold and input mode:
      1. Loads the pre-trained classifier checkpoint.
      2. Loads the corresponding validation patches.
      3. Computes Grad-CAM heatmaps for all validation samples.
      4. Aggregates results by class and z-bin.
      5. Generates visualization figures.
      6. Saves numerical results to JSON.

    Args:
        cfg: Master DictConfig (from classification_task.yaml), expected to have:
            - cfg.data.checkpoints_base_dir: Directory with trained checkpoints.
            - cfg.data.patches_base_dir: Directory with extracted patches.
            - cfg.data.kfold.n_folds: Number of cross-validation folds.
            - cfg.data.input_modes: List of modes to analyze.
            - cfg.output.base_dir: Base output directory.
        experiment_name: Name of the experiment to analyze (e.g., 'velocity_lp_1.5').
        device: Torch device string ('cuda' or 'cpu').

    Returns:
        Dictionary with analysis results keyed by input mode, containing
        per-fold and aggregated metrics.
    """
    n_folds = cfg.dithering.reclassification.n_folds
    input_modes = cfg.gradcam.input_modes
    patches_base_dir = Path(cfg.data.patches_base_dir)
    checkpoints_base_dir = Path(cfg.data.checkpoints_base_dir)
    output_base_dir = Path(cfg.output.base_dir)

    # Load patches (shared across folds, mode-independent at this stage)
    real_patches, synth_patches, real_zbins, synth_zbins = load_patches(
        patches_base_dir, experiment_name
    )

    all_mode_results: dict[str, dict] = {}

    for mode in input_modes:
        logger.info(f"Running Grad-CAM analysis: experiment={experiment_name}, mode={mode}")

        in_channels = _determine_in_channels(mode)
        real_mode_patches = _select_patches_by_mode(real_patches, mode)
        synth_mode_patches = _select_patches_by_mode(synth_patches, mode)

        fold_results: list[GradCAMResult] = []

        for fold_idx in range(n_folds):
            # Locate checkpoint (auto-discover subdirectory and version)
            ckpt_path = _discover_checkpoint(
                checkpoints_base_dir, experiment_name, fold_idx
            )
            if ckpt_path is None:
                logger.warning(
                    f"Checkpoint not found for fold {fold_idx} in "
                    f"{checkpoints_base_dir / experiment_name}"
                )
                continue

            logger.info(f"  Fold {fold_idx}: loading {ckpt_path.relative_to(checkpoints_base_dir)}")

            # Load model
            model = _load_model_from_checkpoint(ckpt_path, in_channels, device)

            # Determine target layer (last Conv2d in the CNN backbone)
            target_layer = model.conv_layers[-1].block[0]

            # Initialize Grad-CAM
            gradcam = GradCAM(model, target_layer)

            try:
                # Compute Grad-CAM for real patches (label=0)
                real_tensor = torch.from_numpy(real_mode_patches).float()
                real_results = gradcam.compute_batch(
                    real_tensor,
                    labels=np.zeros(len(real_mode_patches), dtype=np.int32),
                    z_bins=real_zbins,
                    target_class=1,
                )
                fold_results.extend(real_results)

                # Compute Grad-CAM for synthetic patches (label=1)
                synth_tensor = torch.from_numpy(synth_mode_patches).float()
                synth_results = gradcam.compute_batch(
                    synth_tensor,
                    labels=np.ones(len(synth_mode_patches), dtype=np.int32),
                    z_bins=synth_zbins,
                    target_class=1,
                )
                fold_results.extend(synth_results)
            finally:
                gradcam.remove_hooks()

            logger.info(
                f"  Fold {fold_idx}: processed {len(real_mode_patches)} real + "
                f"{len(synth_mode_patches)} synthetic samples"
            )

        if not fold_results:
            logger.warning(f"No results for mode={mode}, skipping aggregation.")
            continue

        # Aggregate heatmaps
        aggregated = aggregate_heatmaps(fold_results, group_by="both")

        # Compute attention difference between real and synthetic classes
        real_agg = aggregate_heatmaps(
            [r for r in fold_results if r.label == 0], group_by="class"
        )
        synth_agg = aggregate_heatmaps(
            [r for r in fold_results if r.label == 1], group_by="class"
        )

        attention_diff = {}
        if "class_0" in real_agg and "class_1" in synth_agg:
            attention_diff = compute_attention_difference(
                real_agg["class_0"], synth_agg["class_1"]
            )

        # Set up output directory
        output_dir = ensure_output_dir(output_base_dir, experiment_name, f"gradcam/{mode}")

        # Generate visualizations
        plot_gradcam_results(aggregated, fold_results, output_dir, cfg)

        # Compile summary statistics
        mode_summary = {
            "experiment": experiment_name,
            "mode": mode,
            "n_folds_processed": sum(
                1
                for fi in range(n_folds)
                if _discover_checkpoint(checkpoints_base_dir, experiment_name, fi)
                is not None
            ),
            "n_real_samples": sum(1 for r in fold_results if r.label == 0),
            "n_synth_samples": sum(1 for r in fold_results if r.label == 1),
            "mean_pred_real": float(
                np.mean([r.prediction for r in fold_results if r.label == 0])
            ),
            "mean_pred_synth": float(
                np.mean([r.prediction for r in fold_results if r.label == 1])
            ),
            "attention_difference": attention_diff,
        }

        # Save results
        save_result_json(mode_summary, output_dir / "gradcam_summary.json")

        # Save CSV: per-sample attention statistics for inter-experiment analysis
        csv_rows = []
        for r in fold_results:
            csv_rows.append({
                "experiment": experiment_name,
                "mode": mode,
                "label": r.label,
                "z_bin": r.z_bin,
                "prediction": r.prediction,
                "heatmap_mean": float(r.heatmap.mean()),
                "heatmap_max": float(r.heatmap.max()),
                "heatmap_std": float(r.heatmap.std()),
                # Center vs periphery attention ratio
                "center_attention": float(
                    r.heatmap[
                        r.heatmap.shape[0]//4:3*r.heatmap.shape[0]//4,
                        r.heatmap.shape[1]//4:3*r.heatmap.shape[1]//4,
                    ].mean()
                ),
                "periphery_attention": float(
                    np.mean([
                        r.heatmap[:r.heatmap.shape[0]//4, :].mean(),
                        r.heatmap[3*r.heatmap.shape[0]//4:, :].mean(),
                        r.heatmap[:, :r.heatmap.shape[1]//4].mean(),
                        r.heatmap[:, 3*r.heatmap.shape[1]//4:].mean(),
                    ])
                ),
            })
        save_csv(pd.DataFrame(csv_rows), output_dir / "gradcam_samples.csv")

        # Save raw heatmaps for further analysis
        heatmap_data = {
            "real_heatmaps": np.stack(
                [r.heatmap for r in fold_results if r.label == 0]
            ),
            "synth_heatmaps": np.stack(
                [r.heatmap for r in fold_results if r.label == 1]
            ),
            "real_zbins": np.array(
                [r.z_bin for r in fold_results if r.label == 0]
            ),
            "synth_zbins": np.array(
                [r.z_bin for r in fold_results if r.label == 1]
            ),
        }
        np.savez_compressed(output_dir / "heatmaps.npz", **heatmap_data)
        logger.info(f"Saved heatmap arrays to {output_dir / 'heatmaps.npz'}")

        all_mode_results[mode] = mode_summary

    return all_mode_results
