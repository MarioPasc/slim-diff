"""Callback for visualizing segmentation predictions during training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

# Use non-interactive backend for headless environments
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class SegmentationVisualizationCallback(Callback):
    """Visualize segmentation predictions during validation.

    Creates overlay images showing:
    - Original image (grayscale)
    - Ground truth mask (green overlay)
    - Predicted mask (red overlay)

    Samples are selected based on the configured strategy:
    - 'random': Random samples
    - 'worst': Samples with lowest Dice scores
    - 'best': Samples with highest Dice scores
    - 'mixed': Mix of best, worst, and random
    """

    def __init__(self, cfg: DictConfig, fold_idx: int):
        """Initialize callback.

        Args:
            cfg: Configuration containing visualization settings
            fold_idx: Current fold index
        """
        super().__init__()
        self.cfg = cfg
        self.fold_idx = fold_idx

        # Extract visualization config
        viz_cfg = cfg.visualization
        self.enabled = viz_cfg.enabled
        self.every_n_epochs = viz_cfg.every_n_epochs
        self.n_samples = viz_cfg.n_samples
        self.selection_strategy: Literal["random", "worst", "best", "mixed"] = viz_cfg.selection_strategy
        self.save_png = viz_cfg.save_png
        self.log_to_wandb = viz_cfg.log_to_wandb

        # Overlay settings
        self.overlay_enabled = viz_cfg.overlay.enabled
        self.alpha = viz_cfg.overlay.alpha
        self.pred_color = np.array(viz_cfg.overlay.pred_color) / 255.0  # Red
        self.gt_color = np.array(viz_cfg.overlay.gt_color) / 255.0  # Green

        # Create output directory
        self.viz_dir = Path(cfg.experiment.output_dir) / "visualizations"
        if self.enabled and self.save_png:
            self.viz_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Visualization output directory: {self.viz_dir}")

        # Storage for current epoch's validation results
        self.val_images = []
        self.val_masks = []
        self.val_preds = []
        self.val_dice_scores = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Collect validation batch results for visualization.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            outputs: Validation step outputs
            batch: Current batch
            batch_idx: Batch index
            dataloader_idx: Dataloader index
        """
        # Skip if disabled or during sanity check
        if not self.enabled or trainer.sanity_checking:
            return

        # Only visualize every N epochs
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Get images and masks from batch
        images = batch["image"]  # (B, 1, H, W)
        masks = batch["mask"]    # (B, 1, H, W)

        # Get predictions from module
        with torch.no_grad():
            preds_logits = pl_module(images)  # (B, 1, H, W)
            preds_prob = torch.sigmoid(preds_logits)
            preds_binary = (preds_prob > 0.5).float()

        # Compute Dice scores for each sample in batch
        for i in range(images.shape[0]):
            pred_i = preds_binary[i:i+1]
            mask_i = masks[i:i+1]

            # Compute Dice score
            dice_i = self._compute_dice(pred_i, mask_i)

            # Store results (move to CPU and convert to numpy)
            self.val_images.append(images[i].cpu().numpy())
            self.val_masks.append(masks[i].cpu().numpy())
            self.val_preds.append(pred_i[0].cpu().numpy())
            self.val_dice_scores.append(dice_i.item())

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Create and save visualizations at validation epoch end.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        # Skip if disabled, during sanity check, or not visualization epoch
        if not self.enabled or trainer.sanity_checking:
            return

        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Check if we have collected samples
        if len(self.val_images) == 0:
            logger.warning("No validation samples collected for visualization")
            return

        epoch = trainer.current_epoch

        # Select samples based on strategy
        indices = self._select_samples()

        # Create visualizations
        fig = self._create_visualization_figure(indices, epoch)

        # Save to file
        if self.save_png:
            filename = f"fold_{self.fold_idx}_epoch_{epoch:04d}.png"
            save_path = self.viz_dir / filename
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved visualization: {save_path}")

        # Log to W&B if enabled
        if self.log_to_wandb and trainer.logger is not None:
            try:
                import wandb
                trainer.logger.experiment.log({
                    f"visualizations/fold_{self.fold_idx}": wandb.Image(fig),
                    "epoch": epoch,
                })
            except Exception as e:
                logger.warning(f"Failed to log visualization to W&B: {e}")

        plt.close(fig)

        # Clear collected samples
        self.val_images = []
        self.val_masks = []
        self.val_preds = []
        self.val_dice_scores = []

    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Compute Dice score.

        Args:
            pred: Predicted binary mask (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
            smooth: Smoothing factor

        Returns:
            Dice score
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice

    def _select_samples(self) -> list[int]:
        """Select sample indices based on configured strategy.

        Returns:
            List of sample indices to visualize
        """
        n_available = len(self.val_images)
        n_to_select = min(self.n_samples, n_available)

        dice_scores = np.array(self.val_dice_scores)

        if self.selection_strategy == "random":
            indices = np.random.choice(n_available, size=n_to_select, replace=False)

        elif self.selection_strategy == "worst":
            indices = np.argsort(dice_scores)[:n_to_select]

        elif self.selection_strategy == "best":
            indices = np.argsort(dice_scores)[-n_to_select:][::-1]

        elif self.selection_strategy == "mixed":
            # Mix: 1/3 best, 1/3 worst, 1/3 random
            n_each = max(1, n_to_select // 3)

            # Best samples
            best_indices = np.argsort(dice_scores)[-n_each:]

            # Worst samples
            worst_indices = np.argsort(dice_scores)[:n_each]

            # Random samples (excluding best and worst)
            all_indices = set(range(n_available))
            used_indices = set(best_indices.tolist() + worst_indices.tolist())
            available_for_random = list(all_indices - used_indices)

            n_random = n_to_select - len(best_indices) - len(worst_indices)
            if len(available_for_random) >= n_random:
                random_indices = np.random.choice(available_for_random, size=n_random, replace=False)
            else:
                random_indices = np.array(available_for_random)

            # Combine
            indices = np.concatenate([worst_indices, random_indices, best_indices])

        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

        return indices.tolist()

    def _create_visualization_figure(self, indices: list[int], epoch: int) -> plt.Figure:
        """Create visualization figure with selected samples.

        Args:
            indices: Sample indices to visualize
            epoch: Current epoch number

        Returns:
            Matplotlib figure
        """
        n_samples = len(indices)

        # Create figure with 3 columns: Image, Ground Truth, Prediction
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

        # Handle single sample case
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            f"Fold {self.fold_idx} - Epoch {epoch} - Strategy: {self.selection_strategy}",
            fontsize=14,
            fontweight='bold'
        )

        for row_idx, sample_idx in enumerate(indices):
            image = self.val_images[sample_idx][0]  # (H, W)
            mask = self.val_masks[sample_idx][0]    # (H, W)
            pred = self.val_preds[sample_idx][0]    # (H, W)
            dice_score = self.val_dice_scores[sample_idx]

            # Normalize image to [0, 1] for visualization
            image_norm = self._normalize_image(image)

            # Create RGB images for overlay
            if self.overlay_enabled:
                # Image with ground truth overlay (green)
                gt_overlay = self._create_overlay(image_norm, mask, self.gt_color)

                # Image with prediction overlay (red)
                pred_overlay = self._create_overlay(image_norm, pred, self.pred_color)
            else:
                gt_overlay = np.stack([image_norm] * 3, axis=-1)
                pred_overlay = np.stack([image_norm] * 3, axis=-1)

            # Plot original image
            axes[row_idx, 0].imshow(image_norm, cmap='gray')
            axes[row_idx, 0].set_title('Input Image')
            axes[row_idx, 0].axis('off')

            # Plot ground truth overlay
            axes[row_idx, 1].imshow(gt_overlay)
            axes[row_idx, 1].set_title(f'Ground Truth\n(has lesion: {mask.max() > 0})')
            axes[row_idx, 1].axis('off')

            # Plot prediction overlay
            axes[row_idx, 2].imshow(pred_overlay)
            axes[row_idx, 2].set_title(f'Prediction\nDice: {dice_score:.4f}')
            axes[row_idx, 2].axis('off')

        plt.tight_layout()
        return fig

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] for visualization.

        Args:
            image: Input image array

        Returns:
            Normalized image
        """
        img_min = image.min()
        img_max = image.max()

        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        else:
            return np.zeros_like(image)

    def _create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: np.ndarray,
    ) -> np.ndarray:
        """Create RGB overlay of mask on image.

        Args:
            image: Grayscale image (H, W) in [0, 1]
            mask: Binary mask (H, W)
            color: RGB color array (3,) in [0, 1]

        Returns:
            RGB image with overlay (H, W, 3)
        """
        # Convert grayscale to RGB
        rgb_image = np.stack([image] * 3, axis=-1)

        # Create colored mask overlay
        mask_binary = mask > 0.5
        mask_rgb = np.zeros_like(rgb_image)
        mask_rgb[mask_binary] = color

        # Blend
        overlay = rgb_image.copy()
        overlay[mask_binary] = (
            (1 - self.alpha) * rgb_image[mask_binary] +
            self.alpha * mask_rgb[mask_binary]
        )

        return overlay
