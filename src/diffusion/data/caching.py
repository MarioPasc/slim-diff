"""Slice-level caching for efficient data loading.

This module provides functionality to:
1. Process 3D volumes into 2D slices
2. Cache slices as .npz files with metadata
3. Generate index CSVs for train/val/test splits
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.diffusion.data.splits import (
    SubjectInfo,
    create_all_splits,
    get_all_subject_infos,
)
from src.diffusion.data.transforms import (
    check_brain_content,
    check_lesion_content,
    extract_axial_slice,
    get_volume_transforms,
)
from src.diffusion.model.embeddings.zpos import quantize_z
from src.diffusion.utils.io import save_sample_npz
from src.diffusion.utils.logging import setup_logger
from src.diffusion.utils.seeding import seed_everything
from src.diffusion.utils.zbin_priors import compute_zbin_priors, save_zbin_priors

logger = logging.getLogger(__name__)


def _should_recompute_priors(
    priors_path: Path,
    z_bins: int,
    zbin_cfg: dict[str, Any],
) -> bool:
    """Check if priors need recomputing due to file missing or param mismatch.

    Args:
        priors_path: Path to priors file.
        z_bins: Expected number of z-bins.
        zbin_cfg: Z-bin prior config dict.

    Returns:
        True if priors should be recomputed.
    """
    if not priors_path.exists():
        return True

    # Load metadata and compare params
    try:
        data = np.load(priors_path, allow_pickle=True)
        metadata = data["metadata"].item()

        # Check z_bins match
        if metadata.get("z_bins") != z_bins:
            logger.info(
                f"Z-bins mismatch: file has {metadata.get('z_bins')}, "
                f"config has {z_bins}. Recomputing priors."
            )
            return True

        # Check key params match
        for key in [
            "prob_threshold",
            "dilate_radius_px",
            "gaussian_sigma_px",
            "n_first_bins",
            "max_components_for_first_bins",
        ]:
            stored = metadata.get(key)
            configured = zbin_cfg.get(key)
            if stored != configured:
                logger.info(
                    f"Parameter mismatch for {key}: "
                    f"file has {stored}, config has {configured}. Recomputing priors."
                )
                return True

        return False
    except Exception as e:
        logger.warning(f"Failed to read priors metadata: {e}. Recomputing.")
        return True


def process_subject(
    subject_info: SubjectInfo,
    cfg: Any,
    output_dir: Path,
    z_bins: int,
) -> list[dict[str, Any]]:
    """Process a single subject and extract all valid slices.

    Args:
        subject_info: Subject information with paths.
        cfg: Configuration.
        output_dir: Directory to save slice .npz files.
        z_bins: Number of z-position bins.

    Returns:
        List of slice metadata dictionaries.
    """
    has_label = subject_info.label_path is not None

    # Get transforms
    transforms = get_volume_transforms(cfg, has_label=has_label)

    # Build data dictionary
    data_dict = {"image": str(subject_info.image_path)}
    if has_label:
        data_dict["seg"] = str(subject_info.label_path)

    try:
        # Apply transforms
        transformed = transforms(data_dict)
    except Exception as e:
        logger.error(f"Failed to transform {subject_info.subject_id}: {e}")
        return []

    # Get tensors
    image_vol = transformed["image"]
    mask_vol = transformed["seg"]

    # Ensure tensors
    if not isinstance(image_vol, torch.Tensor):
        image_vol = torch.from_numpy(image_vol)
    if not isinstance(mask_vol, torch.Tensor):
        mask_vol = torch.from_numpy(mask_vol)

    # Get volume shape (should be 128^3 after transforms)
    spatial_shape = image_vol.shape[1:]  # (H, W, D)
    n_slices = spatial_shape[-1]

    slice_sampling = cfg.data.slice_sampling
    brain_threshold = slice_sampling.brain_threshold
    brain_min_fraction = slice_sampling.brain_min_fraction
    filter_empty = slice_sampling.filter_empty_brain
    z_range = slice_sampling.z_range  # [min_z, max_z]

    slices_metadata = []
    subject_dir = output_dir / "slices"
    subject_dir.mkdir(parents=True, exist_ok=True)

    # Use z_range to filter slices
    min_z, max_z = z_range
    for z_idx in range(min_z, min(max_z + 1, n_slices)):
        # Extract slice
        image_slice = extract_axial_slice(image_vol, z_idx)
        mask_slice = extract_axial_slice(mask_vol, z_idx)

        # Check brain content
        if filter_empty:
            if not check_brain_content(
                image_slice,
                threshold=brain_threshold,
                min_fraction=brain_min_fraction,
            ):
                continue

        # Compute z-bin and class using LOCAL binning within z_range
        z_bin = quantize_z(z_idx, tuple(z_range), z_bins)
        has_lesion = check_lesion_content(mask_slice)
        pathology_class = 1 if has_lesion else 0
        token = z_bin + pathology_class * z_bins

        # Convert to numpy
        if isinstance(image_slice, torch.Tensor):
            image_np = image_slice.numpy()
        else:
            image_np = image_slice
        if isinstance(mask_slice, torch.Tensor):
            mask_np = mask_slice.numpy()
        else:
            mask_np = mask_slice

        # Create metadata
        metadata = {
            "subject_id": subject_info.subject_id,
            "z_index": int(z_idx),
            "z_bin": int(z_bin),
            "pathology_class": int(pathology_class),
            "token": int(token),
            "source": subject_info.source,
            "split": subject_info.split,
            "has_lesion": bool(has_lesion),
        }

        # Create filename
        filename = (
            f"{subject_info.subject_id}_z{z_idx:03d}_"
            f"bin{z_bin:02d}_c{pathology_class}.npz"
        )
        filepath = subject_dir / filename

        # Save slice
        save_sample_npz(filepath, image_np, mask_np, metadata)

        # Add filepath to metadata for CSV
        metadata["filepath"] = str(filepath.relative_to(output_dir))
        slices_metadata.append(metadata)

    return slices_metadata


def write_index_csv(
    metadata_list: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write slice metadata to CSV index file.

    Args:
        metadata_list: List of metadata dictionaries.
        output_path: Path to output CSV file.
    """
    if not metadata_list:
        logger.warning(f"No metadata to write to {output_path}")
        return

    # Get fieldnames from first entry
    fieldnames = list(metadata_list[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)

    logger.info(f"Wrote {len(metadata_list)} entries to {output_path}")


def build_slice_cache(cfg: Any) -> None:
    """Build the complete slice cache from 3D volumes.

    Args:
        cfg: Configuration object.
    """
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    z_bins = cfg.conditioning.z_bins

    logger.info(f"Building slice cache in {cache_dir}")

    # Create splits
    splits = create_all_splits(cfg)
    subject_infos = get_all_subject_infos(cfg, splits)

    # Collect metadata by split
    split_metadata: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Process all subjects
    for split_name, infos in subject_infos.items():
        logger.info(f"Processing {len(infos)} subjects for {split_name} split")

        for info in tqdm(infos, desc=f"Processing {split_name}"):
            slices = process_subject(info, cfg, cache_dir, z_bins)
            split_metadata[split_name].extend(slices)

        logger.info(
            f"{split_name}: {len(split_metadata[split_name])} slices extracted"
        )

    # Write CSV index files
    for split_name, metadata in split_metadata.items():
        csv_path = cache_dir / f"{split_name}.csv"
        write_index_csv(metadata, csv_path)

    # Write combined statistics
    stats = {
        "total_slices": sum(len(m) for m in split_metadata.values()),
        "train_slices": len(split_metadata["train"]),
        "val_slices": len(split_metadata["val"]),
        "test_slices": len(split_metadata["test"]),
        "z_bins": z_bins,
    }

    # Count lesion vs non-lesion slices
    for split_name, metadata in split_metadata.items():
        n_lesion = sum(1 for m in metadata if m["has_lesion"])
        stats[f"{split_name}_lesion_slices"] = n_lesion
        stats[f"{split_name}_empty_slices"] = len(metadata) - n_lesion

    # Save stats
    stats_path = cache_dir / "cache_stats.yaml"
    OmegaConf.save(OmegaConf.create(stats), stats_path)

    logger.info(f"Cache statistics: {stats}")

    # Compute z-bin priors if enabled
    pp_cfg = cfg.get("postprocessing", {})
    zbin_cfg = pp_cfg.get("zbin_priors", {})

    if zbin_cfg.get("enabled", False):
        priors_filename = zbin_cfg.get("priors_filename", "zbin_priors_brain_roi.npz")
        priors_path = cache_dir / priors_filename

        # Check if priors need recomputing
        z_range = tuple(cfg.data.slice_sampling.z_range)
        needs_compute = _should_recompute_priors(priors_path, z_bins, zbin_cfg)

        if needs_compute:
            logger.info("Computing z-bin priors...")
            try:
                result = compute_zbin_priors(
                    cache_dir=cache_dir,
                    z_bins=z_bins,
                    z_range=z_range,
                    prob_threshold=zbin_cfg.get("prob_threshold", 0.20),
                    dilate_radius_px=zbin_cfg.get("dilate_radius_px", 3),
                    gaussian_sigma_px=zbin_cfg.get("gaussian_sigma_px", 0.7),
                    min_component_px=zbin_cfg.get("min_component_px", 500),
                    n_first_bins=zbin_cfg.get("n_first_bins", 0),
                    max_components_for_first_bins=zbin_cfg.get("max_components_for_first_bins", 1),
                    relaxed_threshold_factor=zbin_cfg.get("relaxed_threshold_factor", 0.1),
                )
                save_zbin_priors(result["priors"], result["metadata"], priors_path)
                logger.info(f"Saved z-bin priors to {priors_path}")
            except Exception as e:
                logger.error(f"Failed to compute z-bin priors: {e}")
                logger.warning("Continuing without priors. Post-processing will be disabled.")
        else:
            logger.info(f"Z-bin priors already exist at {priors_path}, skipping computation")

    logger.info(f"Cache build complete. Saved to {cache_dir}")


def main() -> None:
    """CLI entrypoint for cache building."""
    parser = argparse.ArgumentParser(
        description="Build slice-level cache for JS-DDPM training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/diffusion/config/jsddpm.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Override seed if provided
    if args.seed is not None:
        cfg.experiment.seed = args.seed

    # Setup
    setup_logger("jsddpm", level=logging.INFO)
    seed_everything(cfg.experiment.seed)

    # Build cache
    build_slice_cache(cfg)


if __name__ == "__main__":
    main()
