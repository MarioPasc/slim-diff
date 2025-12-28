#!/usr/bin/env python
"""Visualize z-binning for a random epilepsy patient.

This script:
1. Loads the JS-DDPM configuration.
2. Selects a random epilepsy patient.
3. Loads and transforms the patient's MRI and lesion mask.
4. Divides the slices into z-bins according to the configuration.
5. Saves visualizations of the slices with lesion overlays in bin-specific folders.
"""

from __future__ import annotations

import argparse
import random
import sys
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

# Add repo root to path to ensure src is importable
# src/diffusion/scripts/visualize_binning.py -> src/diffusion/scripts -> src/diffusion -> src -> repo_root
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.diffusion.data.splits import get_all_subject_infos
from src.diffusion.data.transforms import get_volume_transforms
from src.diffusion.model.embeddings.zpos import quantize_z


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay binary mask on image.
    
    Args:
        image: Grayscale image (H, W), normalized to [0, 1] for display.
        mask: Binary mask (H, W).
        alpha: Transparency of the overlay.
        
    Returns:
        RGB image with overlay.
    """
    # Create RGB image
    img_rgb = np.stack([image, image, image], axis=-1)
    
    # Create red overlay
    overlay = np.zeros_like(img_rgb)
    overlay[..., 0] = 1.0  # Red channel
    
    # Combine
    mask_bool = mask > 0
    img_rgb[mask_bool] = (1 - alpha) * img_rgb[mask_bool] + alpha * overlay[mask_bool]
    
    return img_rgb


def main():
    parser = argparse.ArgumentParser(description="Visualize z-binning for a random epilepsy patient")
    parser.add_argument(
        "--config", 
        type=str, 
        default="slurm/jsddpm_baseline/jsddpm_baseline.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="docs/outputs/binning_visualization",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for subject selection"
    )
    args = parser.parse_args()

    # Set seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return 1
    
    cfg = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")

    # Get configuration parameters
    z_bins = cfg.conditioning.z_bins
    z_range = tuple(cfg.data.slice_sampling.z_range)
    min_z, max_z = z_range
    
    print(f"Configuration:")
    print(f"  z_bins: {z_bins}")
    print(f"  z_range: {z_range}")

    # Get all subjects
    print("Loading subject list...")
    # We need to resolve the root_dir relative to the config or use absolute path
    # The config uses ${data.root_dir}, OmegaConf should resolve it if we had the base config,
    # but here we loaded a specific config. 
    # Let's assume the path in yaml is absolute or we need to fix it.
    # The yaml says: root_dir: "/media/mpascual/Sandisk2TB/research/epilepsy/data"
    # We'll trust OmegaConf to handle variable interpolation if it's within the same file.
    
    try:
        # get_all_subject_infos expects cfg.data.root_dir and dataset names
        # It returns a dict with 'train', 'val', 'test' lists of SubjectInfo
        splits = get_all_subject_infos(cfg)
    except Exception as e:
        print(f"Error loading subject info: {e}")
        # Fallback: try to manually construct if get_all_subject_infos fails due to missing paths
        return 1

    # Flatten all splits to find epilepsy subjects
    all_subjects = []
    for split_name, subjects in splits.items():
        all_subjects.extend(subjects)
    
    # Filter for epilepsy subjects (those with labels)
    epilepsy_subjects = [s for s in all_subjects if s.label_path is not None]
    
    if not epilepsy_subjects:
        print("Error: No epilepsy subjects found (subjects with label_path).")
        return 1
    
    print(f"Found {len(epilepsy_subjects)} epilepsy subjects.")

    # Select random subject
    subject = random.choice(epilepsy_subjects)
    print(f"Selected subject: {subject.subject_id}")
    print(f"  Image: {subject.image_path}")
    print(f"  Label: {subject.label_path}")

    # Prepare output directory
    output_dir = Path(args.output) / subject.subject_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create bin directories
    for i in range(z_bins):
        (output_dir / f"bin_{i:02d}").mkdir(exist_ok=True)

    # Load and transform volume
    print("Loading and transforming volume...")
    transforms = get_volume_transforms(cfg, has_label=True)
    
    data_dict = {
        "image": str(subject.image_path),
        "seg": str(subject.label_path)
    }
    
    try:
        transformed = transforms(data_dict)
    except Exception as e:
        print(f"Error transforming volume: {e}")
        return 1

    image_vol = transformed["image"] # (C, H, W, D) or (H, W, D) depending on transform
    mask_vol = transformed["seg"]

    # Ensure channel dim is handled (monai usually adds channel dim)
    if isinstance(image_vol, torch.Tensor):
        image_vol = image_vol.numpy()
    if isinstance(mask_vol, torch.Tensor):
        mask_vol = mask_vol.numpy()

    # Assuming (C, H, W, D) or (H, W, D). 
    # The transforms usually output (C, H, W, D).
    if image_vol.ndim == 4:
        image_vol = image_vol[0] # Take first channel
    if mask_vol.ndim == 4:
        mask_vol = mask_vol[0]

    # Shape should now be (H, W, D)
    H, W, D = image_vol.shape
    print(f"Volume shape: {image_vol.shape}")

    # Process slices
    print("Processing slices...")
    slices_processed = 0
    slices_saved = 0

    # Normalize image for visualization [0, 1]
    img_min, img_max = image_vol.min(), image_vol.max()
    if img_max > img_min:
        image_vol_norm = (image_vol - img_min) / (img_max - img_min)
    else:
        image_vol_norm = image_vol

    for z in range(D):
        # Check if slice is within z_range
        if z < min_z or z > max_z:
            continue
            
        slices_processed += 1
        
        # Determine bin
        try:
            bin_idx = quantize_z(z, z_range, z_bins)
        except ValueError as e:
            print(f"Skipping slice {z}: {e}")
            continue

        # Get slice data
        # Note: Orientation depends on transforms. Assuming last dim is Z.
        img_slice = image_vol_norm[..., z]
        mask_slice = mask_vol[..., z]

        # Rotate if needed (often medical images are rotated)
        # We'll just save as is, user can check orientation.
        # But typically we want axial view.
        
        # Create visualization
        viz_img = overlay_mask(img_slice, mask_slice)
        
        # Save
        filename = f"slice_{z:03d}_bin_{bin_idx:02d}.png"
        save_path = output_dir / f"bin_{bin_idx:02d}" / filename
        
        plt.imsave(save_path, viz_img)
        slices_saved += 1

    print(f"Done!")
    print(f"  Processed slices (in range): {slices_processed}")
    print(f"  Saved images: {slices_saved}")
    print(f"  Results in: {output_dir}")

if __name__ == "__main__":
    main()
