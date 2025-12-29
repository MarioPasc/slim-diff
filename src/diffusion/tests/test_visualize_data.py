"""Test for visualizing input data (image-mask pairs) fed to the diffusion model.

This test loads data using the same pipeline as training and saves sample
visualizations to inspect the actual input data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from src.diffusion.data.dataset import create_dataloader

logger = logging.getLogger(__name__)


def test_visualize_input_data():
    """Visualize image-mask pairs from the dataset.

    This test:
    1. Loads the config from slurm/jsddpm_baseline/jsddpm_baseline.yaml
    2. Creates a dataloader using the same functions as training
    3. Samples a few batches and saves visualizations

    The visualizations show:
    - Input FLAIR images (normalized to [-1, 1])
    - Lesion masks (binary, mapped to [-1, 1])
    - Overlays of images with masks
    """
    # Load config
    config_path = Path("slurm/jsddpm_baseline/jsddpm_baseline.yaml")
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Skipping test - config file required")
        return

    cfg = OmegaConf.load(config_path)

    # Check if cache exists
    cache_dir = Path(cfg.data.cache_dir)
    if not cache_dir.exists():
        logger.warning(f"Cache directory not found: {cache_dir}")
        logger.info("Skipping test - run cache builder first")
        return

    # Create output directory for visualizations
    output_dir = Path("test_outputs/data_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from cache: {cache_dir}")

    # Create dataloaders for train and val
    for split in ["train", "val"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Visualizing {split} split")
        logger.info(f"{'='*60}")

        try:
            dataloader = create_dataloader(cfg, split=split, shuffle=True)
        except FileNotFoundError as e:
            logger.warning(f"Could not load {split} split: {e}")
            continue

        # Get first batch
        batch = next(iter(dataloader))

        # Extract data
        images = batch["image"]  # (B, 1, H, W)
        masks = batch["mask"]    # (B, 1, H, W)
        tokens = batch["token"]  # (B,)
        metadata = batch["metadata"]

        B = images.shape[0]
        H, W = images.shape[2], images.shape[3]

        logger.info(f"Batch shape: {images.shape}")
        logger.info(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        logger.info(f"Mask range: [{masks.min():.3f}, {masks.max():.3f}]")
        logger.info(f"Tokens: {tokens[:min(5, B)].tolist()}...")

        # Visualize first 8 samples (or less if batch is smaller)
        n_samples = min(8, B)

        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Extract single sample
            img = images[i, 0].cpu().numpy()  # (H, W)
            msk = masks[i, 0].cpu().numpy()   # (H, W)
            token = tokens[i].item()

            # Get metadata
            subject_id = metadata["subject_id"][i]
            z_index = metadata["z_index"][i]
            z_bin = metadata["z_bin"][i]
            has_lesion = metadata["has_lesion"][i]
            source = metadata["source"][i]

            # Convert from [-1, 1] to [0, 1] for visualization
            img_vis = (img + 1) / 2
            msk_vis = (msk + 1) / 2

            # Plot image
            axes[i, 0].imshow(img_vis, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f'Image\n{subject_id}\nz={z_index}')
            axes[i, 0].axis('off')

            # Plot mask
            axes[i, 1].imshow(msk_vis, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Mask\nLesion: {has_lesion}')
            axes[i, 1].axis('off')

            # Plot overlay
            overlay = np.stack([img_vis, img_vis, img_vis], axis=-1)
            # Highlight lesions in red
            lesion_mask = msk_vis > 0.5
            overlay[lesion_mask, 0] = 1.0  # Red channel
            overlay[lesion_mask, 1] = 0.0  # Green channel (reduce)
            overlay[lesion_mask, 2] = 0.0  # Blue channel (reduce)

            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'Overlay\ntoken={token}, z_bin={z_bin}\n{source}')
            axes[i, 2].axis('off')

        plt.tight_layout()

        # Save figure
        output_path = output_dir / f"{split}_samples.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to: {output_path}")
        plt.close()

        # Print statistics
        logger.info(f"\nDataset statistics ({split}):")
        logger.info(f"  Total samples in batch: {B}")
        logger.info(f"  Samples with lesions: {sum(metadata['has_lesion'])}")
        logger.info(f"  Image shape: {H}x{W}")
        logger.info(f"  Unique tokens: {len(torch.unique(tokens))}")
        logger.info(f"  Unique z-bins: {sorted(set(metadata['z_bin']))}")

        # Value range checks
        assert images.min() >= -1.0 - 1e-6, f"Images below -1: {images.min()}"
        assert images.max() <= 1.0 + 1e-6, f"Images above 1: {images.max()}"
        assert masks.min() >= -1.0 - 1e-6, f"Masks below -1: {masks.min()}"
        assert masks.max() <= 1.0 + 1e-6, f"Masks above 1: {masks.max()}"

        logger.info(f"✓ Value ranges are correct for {split} split")

    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Test complete! Check visualizations in: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run test
    test_visualize_input_data()
