"""Generation runner for JS-DDPM.

CLI entrypoint for generating synthetic samples from a trained model.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.diffusion.model.components.conditioning import get_token_for_condition
from src.diffusion.model.factory import DiffusionSampler, build_inferer, build_model
from src.diffusion.training.lit_modules import JSDDPMLightningModule
from src.diffusion.utils.io import save_sample_npz
from src.diffusion.utils.logging import setup_logger
from src.diffusion.utils.seeding import get_generator, seed_everything

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str,
    cfg: DictConfig,
    device: str = "cuda",
) -> tuple[torch.nn.Module, DiffusionSampler]:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to Lightning checkpoint.
        cfg: Configuration object.
        device: Device to load model on.

    Returns:
        Tuple of (model, sampler).
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load Lightning module
    lit_module = JSDDPMLightningModule.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,
        map_location=device,
    )
    lit_module.eval()
    lit_module.to(device)

    # Create sampler
    sampler = DiffusionSampler(
        model=lit_module.model,
        scheduler=lit_module.inferer,
        cfg=cfg,
        device=device,
    )

    return lit_module.model, sampler


def generate_samples(
    sampler: DiffusionSampler,
    z_bins: list[int],
    classes: list[int],
    n_per_condition: int,
    output_dir: Path,
    cfg: DictConfig,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate synthetic samples for specified conditions.

    Args:
        sampler: DiffusionSampler instance.
        z_bins: List of z-bins to generate.
        classes: List of pathology classes (0=control, 1=lesion).
        n_per_condition: Number of samples per (z_bin, class) pair.
        output_dir: Directory to save samples.
        cfg: Configuration object.
        seed: Random seed.

    Returns:
        List of sample metadata dictionaries.
    """
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    n_bins = cfg.conditioning.z_bins
    all_metadata = []

    # Set generator for reproducibility
    generator = get_generator(seed)

    total = len(z_bins) * len(classes) * n_per_condition
    pbar = tqdm(total=total, desc="Generating samples")

    for z_bin in z_bins:
        for pathology_class in classes:
            token = get_token_for_condition(z_bin, pathology_class, n_bins)

            for sample_idx in range(n_per_condition):
                # Generate sample
                sample = sampler.sample_single(
                    token=token,
                    generator=generator,
                )

                # Split into image and mask
                image = sample[0].cpu().numpy()  # (H, W)
                mask = sample[1].cpu().numpy()   # (H, W)

                # Create metadata
                metadata = {
                    "sample_id": f"z{z_bin:02d}_c{pathology_class}_s{sample_idx:04d}",
                    "z_bin": int(z_bin),
                    "pathology_class": int(pathology_class),
                    "token": int(token),
                    "sample_idx": sample_idx,
                }

                # Save sample
                filename = f"{metadata['sample_id']}.npz"
                filepath = samples_dir / filename

                save_sample_npz(filepath, image, mask, metadata)

                # Add filepath to metadata
                metadata["filepath"] = str(filepath.relative_to(output_dir))
                all_metadata.append(metadata)

                pbar.update(1)

    pbar.close()
    return all_metadata


def write_generation_index(
    metadata_list: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write generation metadata to CSV index.

    Args:
        metadata_list: List of sample metadata.
        output_path: Path to output CSV.
    """
    if not metadata_list:
        logger.warning("No samples to index")
        return

    fieldnames = list(metadata_list[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)

    logger.info(f"Wrote {len(metadata_list)} entries to {output_path}")


def generate(
    cfg: DictConfig,
    checkpoint_path: str,
    output_dir: str,
    z_bins: list[int] | None = None,
    classes: list[int] | None = None,
    n_per_condition: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Main generation function.

    Args:
        cfg: Configuration object.
        checkpoint_path: Path to model checkpoint.
        output_dir: Directory to save generated samples.
        z_bins: Z-bins to generate (default from config or all).
        classes: Pathology classes to generate.
        n_per_condition: Samples per condition.
        seed: Random seed.
        device: Device to use.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    seed_everything(seed)

    # Defaults from config
    gen_cfg = cfg.generation
    if z_bins is None:
        z_bins = gen_cfg.z_bins
        if z_bins is None:
            # Default to all bins, but respect z_range if configured
            z_range = cfg.data.slice_sampling.z_range
            min_z, max_z = z_range
            # Convert z_indices to z_bins
            from src.diffusion.model.embeddings.zpos import quantize_z
            n_bins = cfg.conditioning.z_bins
            z_bins_set = set()
            for z_idx in range(min_z, max_z + 1):
                z_bin = quantize_z(z_idx, 127, n_bins)  # 127 is max_z for 128 slices
                z_bins_set.add(z_bin)
            z_bins = sorted(list(z_bins_set))
    if classes is None:
        classes = list(gen_cfg.classes)
    if n_per_condition is None:
        n_per_condition = gen_cfg.n_per_condition

    logger.info(f"Generating samples:")
    logger.info(f"  Z-bins: {len(z_bins)} values")
    logger.info(f"  Classes: {classes}")
    logger.info(f"  Samples per condition: {n_per_condition}")
    logger.info(f"  Total samples: {len(z_bins) * len(classes) * n_per_condition}")

    # Load model
    model, sampler = load_model_from_checkpoint(checkpoint_path, cfg, device)

    # Generate samples
    metadata = generate_samples(
        sampler,
        z_bins,
        classes,
        n_per_condition,
        output_dir,
        cfg,
        seed,
    )

    # Write index
    write_generation_index(metadata, output_dir / "generated_samples.csv")

    # Save generation config
    gen_config = {
        "checkpoint": checkpoint_path,
        "z_bins": z_bins,
        "classes": classes,
        "n_per_condition": n_per_condition,
        "seed": seed,
        "total_samples": len(metadata),
    }
    OmegaConf.save(OmegaConf.create(gen_config), output_dir / "generation_config.yaml")

    logger.info(f"Generation complete! Saved {len(metadata)} samples to {output_dir}")


def main() -> None:
    """CLI entrypoint for generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic samples from trained JS-DDPM"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/diffusion/config/jsddpm.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--z_bins",
        type=str,
        default=None,
        help="Comma-separated list of z-bins to generate (e.g., '0,12,25,37,49')",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of classes (e.g., '0,1')",
    )
    parser.add_argument(
        "--n_per_condition",
        type=int,
        default=None,
        help="Number of samples per (z_bin, class) condition",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Parse z_bins if provided
    z_bins = None
    if args.z_bins:
        z_bins = [int(x) for x in args.z_bins.split(",")]

    # Parse classes if provided
    classes = None
    if args.classes:
        classes = [int(x) for x in args.classes.split(",")]

    # Setup logging
    setup_logger("jsddpm", level=logging.INFO)

    # Generate
    generate(
        cfg=cfg,
        checkpoint_path=args.ckpt,
        output_dir=args.out_dir,
        z_bins=z_bins,
        classes=classes,
        n_per_condition=args.n_per_condition,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
