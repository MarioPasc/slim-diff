"""Replica-based synthetic image generation for JS-DDPM.

CLI entrypoint for generating deterministic replicas of synthetic datasets.
Supports two modes:

1. CSV mode (default): Generate replicas matching the test set distribution
   from a test_zbin_distribution.csv file.

2. Uniform mode: Generate equal samples for each (zbin, domain) combination,
   yielding a uniform distribution across all conditions.

Each replica is independently generated with per-sample SHA256 seeding for
full reproducibility.

Usage (CSV mode - matches test distribution):
    python -m src.diffusion.training.runners.generate_replicas \
        --config path/to/config.yaml \
        --checkpoint path/to/checkpoint.ckpt \
        --test_dist_csv path/to/test_zbin_distribution.csv \
        --out_dir path/to/output \
        --replica_id 0 \
        --num_replicas 30

Usage (Uniform mode - balanced control vs epilepsy-with-lesion):
    python -m src.diffusion.training.runners.generate_replicas \
        --config path/to/config.yaml \
        --checkpoint path/to/checkpoint.ckpt \
        --uniform_modes_generation \
        --n_samples_per_mode 1000 \
        --out_dir path/to/output \
        --replica_id 0 \
        --num_replicas 30

    This generates 30 zbins × 2 conditions (control + epilepsy-with-lesion) × 1000 samples
    = 60,000 samples per replica, with equal representation of both classes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.diffusion.model.factory import DiffusionSampler
from src.diffusion.training.lit_modules import JSDDPMLightningModule
from src.diffusion.utils.logging import setup_logger
from src.diffusion.utils.seeding import seed_everything
from src.diffusion.utils.zbin_priors import (
    get_anatomical_priors_as_input,
    load_zbin_priors,
)

logger = logging.getLogger(__name__)

# Explicit domain mapping for round-trip serialization
DOMAIN_MAP = {"control": 0, "epilepsy": 1}
DOMAIN_MAP_INV = {0: "control", 1: "epilepsy"}


@dataclass
class ManifestEntry:
    """Single condition entry from test distribution CSV."""

    zbin: int
    lesion_present: int
    domain_str: str
    domain_int: int
    n_slices: int


def generate_sample_seed(
    seed_base: int,
    replica_id: int,
    zbin: int,
    lesion_present: int,
    domain_int: int,
    sample_index: int,
) -> int:
    """Generate deterministic 64-bit seed using SHA256.

    SHA256 guarantees identical output across all runs/environments
    (unlike Python hash() which is salted per process).

    Args:
        seed_base: Base seed for all replicas.
        replica_id: Which replica (0 to num_replicas-1).
        zbin: Z-bin index.
        lesion_present: 0 or 1.
        domain_int: 0 (control) or 1 (epilepsy).
        sample_index: Index within this condition (0 to n_slices-1).

    Returns:
        Positive int64 seed for PyTorch Generator.
    """
    input_tuple = (seed_base, replica_id, zbin, lesion_present, domain_int, sample_index)
    hash_bytes = hashlib.sha256(str(input_tuple).encode("utf-8")).digest()
    seed = int.from_bytes(hash_bytes[:8], byteorder="big", signed=False)
    return seed & 0x7FFFFFFFFFFFFFFF  # Ensure positive int64


def load_test_distribution(csv_path: Path) -> tuple[list[ManifestEntry], int]:
    """Load test distribution CSV and create generation manifest.

    Computes condition count dynamically from the CSV content.

    Args:
        csv_path: Path to test_zbin_distribution.csv.

    Returns:
        Tuple of (manifest_entries, total_samples).
    """
    df = pd.read_csv(csv_path)
    df = df[df["split"] == "test"]

    manifest = []
    total = 0

    for _, row in df.iterrows():
        n_slices = int(row["n_slices"])
        if n_slices > 0:
            domain_str = row["domain"]
            if domain_str not in DOMAIN_MAP:
                raise ValueError(f"Unknown domain '{domain_str}'. Expected 'control' or 'epilepsy'.")

            entry = ManifestEntry(
                zbin=int(row["zbin"]),
                lesion_present=int(row["lesion_present"]),
                domain_str=domain_str,
                domain_int=DOMAIN_MAP[domain_str],
                n_slices=n_slices,
            )
            manifest.append(entry)
            total += n_slices

    n_conditions = len(manifest)
    logger.info(f"Loaded {n_conditions} unique conditions, {total} total samples from {csv_path}")
    return manifest, total


def create_uniform_distribution(
    n_zbins: int,
    n_samples_per_mode: int,
) -> tuple[list[ManifestEntry], int]:
    """Create a uniform distribution manifest for all zbin-domain combinations.

    Generates a manifest with equal samples for each (zbin, condition) pair:
    - Control domain (domain=0): lesion_present=0 (healthy controls)
    - Epilepsy domain (domain=1): lesion_present=1 (epilepsy with visible lesion)

    This produces balanced classes: equal control vs epilepsy-with-lesion samples.
    Non-lesional epilepsy (domain=1, lesion_present=0) is excluded as it doesn't
    represent a positive case for lesion detection.

    Args:
        n_zbins: Number of z-bins (typically 30).
        n_samples_per_mode: Number of samples per (zbin, condition) combination.

    Returns:
        Tuple of (manifest_entries, total_samples).
    """
    manifest = []
    total = 0

    for zbin in range(n_zbins):
        # Control: domain=0, lesion_present=0 (healthy controls have no lesion)
        entry_control = ManifestEntry(
            zbin=zbin,
            lesion_present=0,
            domain_str="control",
            domain_int=DOMAIN_MAP["control"],
            n_slices=n_samples_per_mode,
        )
        manifest.append(entry_control)
        total += n_samples_per_mode

        # Epilepsy with lesion: domain=1, lesion_present=1 (actual positive cases)
        entry_epilepsy = ManifestEntry(
            zbin=zbin,
            lesion_present=1,
            domain_str="epilepsy",
            domain_int=DOMAIN_MAP["epilepsy"],
            n_slices=n_samples_per_mode,
        )
        manifest.append(entry_epilepsy)
        total += n_samples_per_mode

    n_conditions = len(manifest)
    logger.info(
        f"Created uniform distribution: {n_zbins} zbins x 2 conditions "
        f"(control + epilepsy-with-lesion) = {n_conditions} conditions, "
        f"{n_samples_per_mode} samples each, {total} total"
    )
    return manifest, total


def load_model_with_ema(
    checkpoint_path: Path,
    cfg: DictConfig,
    use_ema: bool,
    device: str,
) -> tuple[torch.nn.Module, DiffusionSampler, bool]:
    """Load model with robust EMA detection and fallback.

    Args:
        checkpoint_path: Path to Lightning checkpoint.
        cfg: Configuration object.
        use_ema: Whether to attempt loading EMA weights.
        device: Device to load model on.

    Returns:
        Tuple of (model, sampler, ema_loaded_flag).
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint dict for EMA extraction
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load Lightning module (weights_only=False needed for OmegaConf in checkpoint)
    lit_module = JSDDPMLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        cfg=cfg,
        map_location=device,
        weights_only=False,
    )
    lit_module.eval()
    lit_module.to(device)

    # Robust EMA loading
    ema_loaded = False
    if use_ema:
        # Try top-level ema_state_dict first (export_to_checkpoint=True)
        if "ema_state_dict" in ckpt and isinstance(ckpt["ema_state_dict"], dict):
            lit_module.model.load_state_dict(ckpt["ema_state_dict"], strict=False)
            ema_loaded = True
            logger.info("Loaded EMA weights from checkpoint['ema_state_dict']")

        # Fallback: try callback state (older format)
        elif "callbacks" in ckpt:
            cb_state = ckpt.get("callbacks", {}).get("EMACallback", {})
            if "ema" in cb_state and cb_state["ema"] is not None:
                lit_module.model.load_state_dict(cb_state["ema"], strict=False)
                ema_loaded = True
                logger.info("Loaded EMA weights from callback state")

        if not ema_loaded:
            logger.warning(
                "EMA weights requested but not found in checkpoint. "
                "Using regular model weights. Ensure export_to_checkpoint=True during training."
            )

    # Get anatomical encoder if using cross_attention method
    anatomical_encoder = getattr(lit_module, "_anatomical_encoder", None)

    # Create sampler
    sampler = DiffusionSampler(
        model=lit_module.model,
        scheduler=lit_module.inferer,
        cfg=cfg,
        device=device,
        anatomical_encoder=anatomical_encoder,
    )

    return lit_module.model, sampler, ema_loaded


def validate_inputs(
    cfg: DictConfig,
    checkpoint_path: Path,
    replica_id: int,
    num_replicas: int,
    csv_path: Path | None = None,
    uniform_modes: bool = False,
    n_samples_per_mode: int | None = None,
) -> tuple[list[ManifestEntry], int]:
    """Validate all inputs before generation.

    Supports two modes:
    1. CSV mode (default): Load distribution from test_zbin_distribution.csv
    2. Uniform mode: Generate equal samples for each (zbin, domain) combination

    Args:
        cfg: Configuration object.
        checkpoint_path: Path to checkpoint.
        replica_id: Replica ID.
        num_replicas: Total number of replicas.
        csv_path: Path to distribution CSV (required if not uniform_modes).
        uniform_modes: If True, generate uniform distribution instead of CSV.
        n_samples_per_mode: Samples per (zbin, domain) pair (required if uniform_modes).

    Returns:
        Tuple of (manifest, total_samples).

    Raises:
        FileNotFoundError: If required files don't exist.
        ValueError: If validation fails.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not (0 <= replica_id < num_replicas):
        raise ValueError(f"replica_id {replica_id} not in [0, {num_replicas})")

    config_z_bins = cfg.conditioning.z_bins

    if uniform_modes:
        # Uniform mode: generate equal samples per (zbin, domain)
        if n_samples_per_mode is None or n_samples_per_mode <= 0:
            raise ValueError(
                "--n_samples_per_mode must be a positive integer when using --uniform_modes_generation"
            )
        manifest, total = create_uniform_distribution(config_z_bins, n_samples_per_mode)
    else:
        # CSV mode: load from file
        if csv_path is None:
            raise ValueError("--test_dist_csv is required when not using --uniform_modes_generation")
        if not csv_path.exists():
            raise FileNotFoundError(f"Distribution CSV not found: {csv_path}")

        manifest, total = load_test_distribution(csv_path)

        # Validate manifest is not empty
        if not manifest:
            raise ValueError(
                f"No valid test conditions found in CSV: {csv_path}. "
                "Ensure the CSV contains rows with split='test' and n_slices > 0."
            )

        # Validate z-bins are within valid range
        min_zbin = min(e.zbin for e in manifest)
        max_zbin = max(e.zbin for e in manifest)

        if min_zbin < 0:
            raise ValueError(
                f"CSV contains negative zbin={min_zbin}. "
                "Z-bin values must be non-negative integers in [0, z_bins-1]."
            )

        if max_zbin >= config_z_bins:
            raise ValueError(
                f"CSV contains zbin={max_zbin} but config has z_bins={config_z_bins}. "
                f"Max allowed zbin is {config_z_bins - 1}."
            )

    return manifest, total


def validate_post_save(
    npz_path: Path,
    manifest: list[ManifestEntry],
    expected_total: int,
) -> None:
    """Reload and validate saved output.

    Args:
        npz_path: Path to saved NPZ file.
        manifest: Expected manifest entries.
        expected_total: Expected total sample count.

    Raises:
        ValueError: If validation fails.
    """
    data = np.load(npz_path)

    # Total count
    actual_total = len(data["images"])
    if actual_total != expected_total:
        raise ValueError(f"Expected {expected_total} samples, got {actual_total}")

    # Shape consistency
    required_keys = ["images", "masks", "zbin", "lesion_present", "domain", "seed"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required array '{key}' in NPZ")
        if len(data[key]) != expected_total:
            raise ValueError(f"Array '{key}' has wrong length: {len(data[key])}")

    # Per-condition count validation
    expected_counts: dict[tuple[int, int, int], int] = {
        (e.zbin, e.lesion_present, e.domain_int): e.n_slices for e in manifest
    }

    actual_counts: dict[tuple[int, int, int], int] = {}
    for i in range(actual_total):
        key = (int(data["zbin"][i]), int(data["lesion_present"][i]), int(data["domain"][i]))
        actual_counts[key] = actual_counts.get(key, 0) + 1

    if expected_counts != actual_counts:
        # Find mismatches for detailed error
        for key, expected in expected_counts.items():
            actual = actual_counts.get(key, 0)
            if expected != actual:
                logger.error(f"Condition {key}: expected {expected}, got {actual}")
        raise ValueError("Per-condition counts do not match expected distribution")

    logger.info(f"Post-save validation passed: {actual_total} samples across {len(manifest)} conditions")


def generate_replica(
    sampler: DiffusionSampler,
    manifest: list[ManifestEntry],
    total_samples: int,
    replica_id: int,
    seed_base: int,
    batch_size: int,
    cfg: DictConfig,
    zbin_priors: dict | None,
    device: str,
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    """Generate all samples for one replica with preallocated arrays.

    Args:
        sampler: DiffusionSampler instance.
        manifest: List of condition entries.
        total_samples: Total number of samples to generate.
        replica_id: Replica ID.
        seed_base: Base seed for all replicas.
        batch_size: Batch size for generation.
        cfg: Configuration object.
        zbin_priors: Z-bin priors dict or None.
        device: Device string.
        dtype: Output numpy dtype.

    Returns:
        Dictionary of numpy arrays for NPZ saving.
    """
    # Read image dimensions from config (roi_size is [H, W, D] for 3D, we use H, W)
    roi_size = cfg.data.transforms.roi_size
    H, W = roi_size[0], roi_size[1]
    z_bins = cfg.conditioning.z_bins
    use_anatomical = cfg.model.get("anatomical_conditioning", False)

    # Phase 1: Build sample queue with seeds
    sample_queue: list[dict[str, Any]] = []
    for entry in manifest:
        for k in range(entry.n_slices):
            seed = generate_sample_seed(
                seed_base,
                replica_id,
                entry.zbin,
                entry.lesion_present,
                entry.domain_int,
                k,
            )
            # Token formula: token = zbin + lesion_present * z_bins
            token = entry.zbin + entry.lesion_present * z_bins
            sample_queue.append(
                {
                    "zbin": entry.zbin,
                    "lesion_present": entry.lesion_present,
                    "domain_int": entry.domain_int,
                    "seed": seed,
                    "token": token,
                    "k": k,
                }
            )

    if len(sample_queue) != total_samples:
        raise ValueError(f"Sample queue length {len(sample_queue)} != expected {total_samples}")

    # Phase 2: Preallocate output arrays
    images = np.empty((total_samples, H, W), dtype=dtype)
    masks = np.empty((total_samples, H, W), dtype=dtype)
    zbins_arr = np.empty(total_samples, dtype=np.int32)
    lesion_present_arr = np.empty(total_samples, dtype=np.uint8)
    domains_arr = np.empty(total_samples, dtype=np.uint8)
    tokens_arr = np.empty(total_samples, dtype=np.int32)
    seeds_arr = np.empty(total_samples, dtype=np.int64)
    k_indices_arr = np.empty(total_samples, dtype=np.int32)

    # Phase 3: Batch generation
    idx = 0
    n_batches = (total_samples + batch_size - 1) // batch_size

    for batch_start in tqdm(range(0, total_samples, batch_size), desc=f"Replica {replica_id}", total=n_batches):
        batch = sample_queue[batch_start : batch_start + batch_size]
        B = len(batch)

        # Generate per-sample noise with different seeds
        noise_list = []
        for s in batch:
            gen = torch.Generator(device=device).manual_seed(s["seed"])
            noise = torch.randn((2, H, W), generator=gen, device=device)
            noise_list.append(noise)
        x_T = torch.stack(noise_list, dim=0)  # (B, 2, H, W)

        # Prepare tokens
        token_tensor = torch.tensor([s["token"] for s in batch], device=device, dtype=torch.long)

        # Prepare anatomical mask if needed
        anatomical_mask = None
        if use_anatomical and zbin_priors is not None:
            batch_zbins = [s["zbin"] for s in batch]
            anatomical_mask = get_anatomical_priors_as_input(batch_zbins, zbin_priors, device)

        # Run sampling with pre-generated noise
        samples = sampler.sample(
            tokens=token_tensor,
            shape=(B, 2, H, W),
            anatomical_mask=anatomical_mask,
            x_T=x_T,
        )

        # Store results directly into preallocated arrays
        samples_np = samples.cpu().numpy()
        for i in range(B):
            images[idx] = samples_np[i, 0].astype(dtype)
            masks[idx] = samples_np[i, 1].astype(dtype)
            zbins_arr[idx] = batch[i]["zbin"]
            lesion_present_arr[idx] = batch[i]["lesion_present"]
            domains_arr[idx] = batch[i]["domain_int"]
            tokens_arr[idx] = batch[i]["token"]
            seeds_arr[idx] = batch[i]["seed"]
            k_indices_arr[idx] = batch[i]["k"]
            idx += 1

    return {
        "images": images,
        "masks": masks,
        "zbin": zbins_arr,
        "lesion_present": lesion_present_arr,
        "domain": domains_arr,
        "condition_token": tokens_arr,
        "seed": seeds_arr,
        "k_index": k_indices_arr,
        "replica_id": np.full(total_samples, replica_id, dtype=np.int32),
    }


def save_replica(
    output_data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    out_dir: Path,
    replica_id: int,
    overwrite: bool,
) -> Path:
    """Save replica NPZ and JSON metadata.

    Args:
        output_data: Dictionary of numpy arrays.
        metadata: Generation metadata.
        out_dir: Output directory.
        replica_id: Replica ID.
        overwrite: Whether to overwrite existing files.

    Returns:
        Path to saved NPZ file.

    Raises:
        FileExistsError: If file exists and overwrite=False.
    """
    replicas_dir = out_dir / "replicas"
    replicas_dir.mkdir(parents=True, exist_ok=True)

    npz_path = replicas_dir / f"replica_{replica_id:03d}.npz"
    json_path = replicas_dir / f"replica_{replica_id:03d}_meta.json"

    if npz_path.exists() and not overwrite:
        raise FileExistsError(f"Replica file already exists: {npz_path}. Use --overwrite to replace.")

    # Save NPZ with compression
    np.savez_compressed(npz_path, **output_data)
    logger.info(f"Saved replica NPZ to {npz_path} ({npz_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save JSON metadata
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {json_path}")

    return npz_path


def main() -> None:
    """CLI entrypoint for replica generation."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic replicas of synthetic samples matching test distribution"
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--test_dist_csv",
        type=str,
        required=False,
        default=None,
        help="Path to test_zbin_distribution.csv (required unless --uniform_modes_generation is used)",
    )
    parser.add_argument(
        "--uniform_modes_generation",
        action="store_true",
        help="Generate uniform distribution: equal samples for each (zbin, domain) combination",
    )
    parser.add_argument(
        "--n_samples_per_mode",
        type=int,
        default=None,
        help="Number of samples per (zbin, domain) mode (required with --uniform_modes_generation)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for replicas",
    )
    parser.add_argument(
        "--replica_id",
        type=int,
        required=True,
        help="Replica ID (0 to num_replicas-1)",
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        required=True,
        help="Total number of replicas",
    )

    # Optional arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for generation (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Output dtype (default: float16)",
    )
    parser.add_argument(
        "--seed_base",
        type=int,
        default=42,
        help="Base seed for all replicas (default: 42)",
    )
    parser.add_argument(
        "--use_ema",
        dest="use_ema",
        action="store_true",
        default=True,
        help="Use EMA weights (default)",
    )
    parser.add_argument(
        "--no_ema",
        dest="use_ema",
        action="store_false",
        help="Disable EMA weights",
    )
    parser.add_argument(
        "--override_steps",
        type=int,
        default=None,
        help="Override DDIM inference steps",
    )
    parser.add_argument(
        "--override_eta",
        type=float,
        default=None,
        help="Override DDIM eta",
    )
    parser.add_argument(
        "--override_guidance",
        type=float,
        default=None,
        help="Override CFG guidance scale",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing replica files",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger("jsddpm", level=logging.INFO)

    # Resolve paths
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    csv_path = Path(args.test_dist_csv) if args.test_dist_csv else None
    out_dir = Path(args.out_dir)

    # Load config
    cfg = OmegaConf.load(config_path)

    # Apply overrides
    if args.override_steps is not None:
        cfg.sampler.num_inference_steps = args.override_steps
        logger.info(f"Override: num_inference_steps = {args.override_steps}")
    if args.override_eta is not None:
        cfg.sampler.eta = args.override_eta
        logger.info(f"Override: eta = {args.override_eta}")
    if args.override_guidance is not None:
        cfg.sampler.guidance_scale = args.override_guidance
        logger.info(f"Override: guidance_scale = {args.override_guidance}")

    # Determine EMA usage (--use_ema is default, --no_ema disables it)
    use_ema = args.use_ema

    # Determine dtype
    dtype = np.float16 if args.dtype == "float16" else np.float32

    # Set global seed for any remaining randomness
    seed_everything(args.seed_base)

    logger.info("=" * 60)
    logger.info(f"Replica Generation: ID {args.replica_id} / {args.num_replicas}")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    if args.uniform_modes_generation:
        logger.info(f"Mode: UNIFORM ({args.n_samples_per_mode} samples per zbin-domain)")
    else:
        logger.info(f"Mode: CSV distribution")
        logger.info(f"Distribution CSV: {csv_path}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Use EMA: {use_ema}")
    logger.info(f"Output dtype: {args.dtype}")
    logger.info(f"Seed base: {args.seed_base}")

    # Validate inputs
    manifest, total_samples = validate_inputs(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        replica_id=args.replica_id,
        num_replicas=args.num_replicas,
        csv_path=csv_path,
        uniform_modes=args.uniform_modes_generation,
        n_samples_per_mode=args.n_samples_per_mode,
    )

    logger.info(f"Conditions: {len(manifest)}, Total samples: {total_samples}")

    # Load model with EMA
    model, sampler, ema_loaded = load_model_with_ema(
        checkpoint_path,
        cfg,
        use_ema,
        args.device,
    )

    # Load z-bin priors if needed
    use_anatomical = cfg.model.get("anatomical_conditioning", False)
    zbin_priors = None

    if use_anatomical:
        try:
            cache_dir = Path(cfg.data.cache_dir)
            pp_cfg = cfg.get("postprocessing", {})
            zbin_cfg = pp_cfg.get("zbin_priors", {})
            priors_filename = zbin_cfg.get("priors_filename", "zbin_priors_brain_roi.npz")

            zbin_priors = load_zbin_priors(cache_dir, priors_filename, cfg.conditioning.z_bins)
            logger.info(f"Loaded z-bin priors for anatomical conditioning ({len(zbin_priors)} bins)")
        except Exception as e:
            logger.error(f"Failed to load z-bin priors: {e}")
            raise

    # Generate replica
    logger.info(f"Starting generation of {total_samples} samples...")
    output_data = generate_replica(
        sampler=sampler,
        manifest=manifest,
        total_samples=total_samples,
        replica_id=args.replica_id,
        seed_base=args.seed_base,
        batch_size=args.batch_size,
        cfg=cfg,
        zbin_priors=zbin_priors,
        device=args.device,
        dtype=dtype,
    )

    # Build metadata
    metadata = {
        "replica_id": args.replica_id,
        "seed_base": args.seed_base,
        "n_samples": total_samples,
        "n_conditions": len(manifest),
        "generation_timestamp": datetime.now().isoformat(),
        "generation_mode": "uniform" if args.uniform_modes_generation else "csv",
        "config": {
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
            "z_bins": cfg.conditioning.z_bins,
            "image_size": list(cfg.data.transforms.roi_size[:2]),
            "batch_size": args.batch_size,
            "use_ema": use_ema,
            "ema_loaded": ema_loaded,
            "anatomical_conditioning": use_anatomical,
            "output_dtype": args.dtype,
        },
        "sampler": {
            "type": "DDIM",
            "num_inference_steps": cfg.sampler.num_inference_steps,
            "eta": cfg.sampler.eta,
            "guidance_scale": cfg.sampler.guidance_scale,
        },
        "domain_mapping": DOMAIN_MAP,
        "domain_mapping_inverse": {str(k): v for k, v in DOMAIN_MAP_INV.items()},
    }

    # Add mode-specific metadata
    if args.uniform_modes_generation:
        metadata["uniform_mode"] = {
            "n_samples_per_mode": args.n_samples_per_mode,
            "n_zbins": cfg.conditioning.z_bins,
            "n_domains": len(DOMAIN_MAP),
        }
    else:
        metadata["csv_path"] = str(csv_path) if csv_path else None

    # Save replica
    npz_path = save_replica(
        output_data,
        metadata,
        out_dir,
        args.replica_id,
        args.overwrite,
    )

    # Post-save validation
    validate_post_save(npz_path, manifest, total_samples)

    logger.info("=" * 60)
    logger.info(f"Replica {args.replica_id} complete!")
    logger.info(f"Output: {npz_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
