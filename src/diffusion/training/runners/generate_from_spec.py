"""JSON specification-based generation runner for SLIM-Diff.

CLI entrypoint for generating synthetic samples from a JSON specification file.
This provides reproducible generation with two supported formats:

1. **Detailed format**: Explicit per-zbin specification
2. **Compact format**: Defaults with optional overrides

Example usage:
    slimdiff-generate-spec --spec generation_spec.json
    slimdiff-generate-spec --spec spec.json --validate-only
    slimdiff-generate-spec --spec spec.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.diffusion.model.components.conditioning import get_token_for_condition
from src.diffusion.model.factory import DiffusionSampler
from src.diffusion.training.lit_modules import JSDDPMLightningModule
from src.diffusion.utils.io import save_sample_npz
from src.diffusion.utils.logging import setup_logger
from src.diffusion.utils.seeding import get_generator, seed_everything
from src.diffusion.utils.zbin_priors import (
    apply_zbin_prior_postprocess,
    get_anatomical_priors_as_input,
    load_zbin_priors,
)

logger = logging.getLogger(__name__)


@dataclass
class ZBinSampleSpec:
    """Specification for samples at a single z-bin."""

    zbin: int
    control: int = 0
    lesion: int = 0

    def total_samples(self) -> int:
        """Total samples for this z-bin."""
        return self.control + self.lesion

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {"zbin": self.zbin, "control": self.control, "lesion": self.lesion}


@dataclass
class GenerationSpec:
    """Full generation specification."""

    checkpoint_path: Path
    config_path: Path
    output_dir: Path
    seed: int = 42
    device: str = "cuda"
    zbin_specs: list[ZBinSampleSpec] = field(default_factory=list)

    def total_samples(self) -> int:
        """Total samples across all z-bins."""
        return sum(spec.total_samples() for spec in self.zbin_specs)

    def total_control(self) -> int:
        """Total control samples."""
        return sum(spec.control for spec in self.zbin_specs)

    def total_lesion(self) -> int:
        """Total lesion samples."""
        return sum(spec.lesion for spec in self.zbin_specs)

    def active_zbins(self) -> list[int]:
        """Z-bins with at least one sample requested."""
        return [spec.zbin for spec in self.zbin_specs if spec.total_samples() > 0]


def load_spec(spec_path: Path) -> dict[str, Any]:
    """Load JSON specification file.

    Args:
        spec_path: Path to JSON specification file.

    Returns:
        Parsed JSON dictionary.

    Raises:
        FileNotFoundError: If spec file doesn't exist.
        json.JSONDecodeError: If spec file is invalid JSON.
    """
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")

    with open(spec_path) as f:
        return json.load(f)


def detect_format(spec_dict: dict[str, Any]) -> str:
    """Auto-detect specification format.

    Args:
        spec_dict: Parsed JSON specification.

    Returns:
        Format string: "detailed" or "compact".
    """
    if "format" in spec_dict:
        return spec_dict["format"]

    # Heuristic detection
    if "samples" in spec_dict:
        return "detailed"
    if "defaults" in spec_dict:
        return "compact"

    raise ValueError(
        "Cannot detect specification format. "
        "Please add 'format': 'detailed' or 'format': 'compact' to your spec."
    )


def parse_zbin_range(zbin_str: str | int | list, n_bins: int) -> list[int]:
    """Parse z-bin specification into list of integers.

    Supports:
        - "all": All bins from 0 to n_bins-1
        - "0-29": Range from 0 to 29 (inclusive)
        - "0,5,10,15": Comma-separated list
        - [0, 5, 10]: Python list
        - 5: Single integer

    Args:
        zbin_str: Z-bin specification.
        n_bins: Total number of z-bins in the model.

    Returns:
        List of z-bin indices.
    """
    if isinstance(zbin_str, int):
        return [zbin_str]

    if isinstance(zbin_str, list):
        return [int(z) for z in zbin_str]

    zbin_str = str(zbin_str).strip().lower()

    if zbin_str == "all":
        return list(range(n_bins))

    if "-" in zbin_str and "," not in zbin_str:
        # Range format: "0-29"
        start, end = zbin_str.split("-")
        return list(range(int(start), int(end) + 1))

    if "," in zbin_str:
        # Comma-separated: "0,5,10,15"
        return [int(z.strip()) for z in zbin_str.split(",")]

    # Single integer as string
    return [int(zbin_str)]


def parse_detailed_spec(spec_dict: dict[str, Any]) -> GenerationSpec:
    """Parse detailed format specification.

    Detailed format example:
    ```json
    {
        "checkpoint_path": "/path/to/model.ckpt",
        "config_path": "src/diffusion/config/jsddpm.yaml",
        "output_dir": "./outputs/generation_run_001",
        "seed": 42,
        "device": "cuda",
        "format": "detailed",
        "samples": [
            {"zbin": 0, "control": 10, "lesion": 5},
            {"zbin": 5, "control": 20, "lesion": 10}
        ]
    }
    ```

    Args:
        spec_dict: Parsed JSON specification.

    Returns:
        GenerationSpec object.
    """
    zbin_specs = []
    for sample in spec_dict.get("samples", []):
        zbin_specs.append(
            ZBinSampleSpec(
                zbin=sample["zbin"],
                control=sample.get("control", 0),
                lesion=sample.get("lesion", 0),
            )
        )

    return GenerationSpec(
        checkpoint_path=Path(spec_dict["checkpoint_path"]),
        config_path=Path(spec_dict["config_path"]),
        output_dir=Path(spec_dict["output_dir"]),
        seed=spec_dict.get("seed", 42),
        device=spec_dict.get("device", "cuda"),
        zbin_specs=zbin_specs,
    )


def parse_compact_spec(spec_dict: dict[str, Any], n_bins: int) -> GenerationSpec:
    """Parse compact format specification.

    Compact format example:
    ```json
    {
        "checkpoint_path": "/path/to/model.ckpt",
        "config_path": "src/diffusion/config/jsddpm.yaml",
        "output_dir": "./outputs/generation_run_002",
        "seed": 123,
        "device": "cuda",
        "format": "compact",
        "defaults": {
            "control": 10,
            "lesion": 10,
            "zbins": "all"
        },
        "overrides": [
            {"zbin": 0, "control": 5, "lesion": 2},
            {"zbin": 29, "control": 5, "lesion": 2}
        ]
    }
    ```

    Args:
        spec_dict: Parsed JSON specification.
        n_bins: Total number of z-bins in the model.

    Returns:
        GenerationSpec object.
    """
    defaults = spec_dict.get("defaults", {})
    default_control = defaults.get("control", 0)
    default_lesion = defaults.get("lesion", 0)
    zbins_spec = defaults.get("zbins", "all")

    # Parse z-bins
    zbins = parse_zbin_range(zbins_spec, n_bins)

    # Create default specs for all z-bins
    zbin_spec_map = {
        zbin: ZBinSampleSpec(zbin=zbin, control=default_control, lesion=default_lesion)
        for zbin in zbins
    }

    # Apply overrides
    for override in spec_dict.get("overrides", []):
        zbin = override["zbin"]
        if zbin in zbin_spec_map:
            if "control" in override:
                zbin_spec_map[zbin].control = override["control"]
            if "lesion" in override:
                zbin_spec_map[zbin].lesion = override["lesion"]
        else:
            # Override for a z-bin not in defaults - add it
            zbin_spec_map[zbin] = ZBinSampleSpec(
                zbin=zbin,
                control=override.get("control", default_control),
                lesion=override.get("lesion", default_lesion),
            )

    # Sort by z-bin
    zbin_specs = [zbin_spec_map[zbin] for zbin in sorted(zbin_spec_map.keys())]

    return GenerationSpec(
        checkpoint_path=Path(spec_dict["checkpoint_path"]),
        config_path=Path(spec_dict["config_path"]),
        output_dir=Path(spec_dict["output_dir"]),
        seed=spec_dict.get("seed", 42),
        device=spec_dict.get("device", "cuda"),
        zbin_specs=zbin_specs,
    )


def validate_spec(spec: GenerationSpec, cfg: DictConfig) -> list[str]:
    """Validate specification against config.

    Args:
        spec: Generation specification.
        cfg: Model configuration.

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []

    # Check paths exist
    if not spec.checkpoint_path.exists():
        errors.append(f"Checkpoint not found: {spec.checkpoint_path}")

    if not spec.config_path.exists():
        errors.append(f"Config not found: {spec.config_path}")

    # Check z-bin ranges
    n_bins = cfg.conditioning.z_bins
    for zbin_spec in spec.zbin_specs:
        if zbin_spec.zbin < 0 or zbin_spec.zbin >= n_bins:
            errors.append(
                f"Z-bin {zbin_spec.zbin} out of range [0, {n_bins - 1}]"
            )

        if zbin_spec.control < 0:
            errors.append(f"Z-bin {zbin_spec.zbin}: control count cannot be negative")

        if zbin_spec.lesion < 0:
            errors.append(f"Z-bin {zbin_spec.zbin}: lesion count cannot be negative")

    # Check device
    if spec.device.startswith("cuda") and not torch.cuda.is_available():
        errors.append(f"CUDA device requested but not available: {spec.device}")

    # Check total samples
    if spec.total_samples() == 0:
        errors.append("No samples requested (total is 0)")

    return errors


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
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
        str(checkpoint_path),
        cfg=cfg,
        map_location=device,
        weights_only=False,
    )
    lit_module.eval()
    lit_module.to(device)

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

    return lit_module.model, sampler


def generate_samples_from_spec(
    sampler: DiffusionSampler,
    spec: GenerationSpec,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Generate synthetic samples according to specification.

    Args:
        sampler: DiffusionSampler instance.
        spec: Generation specification.
        cfg: Configuration object.

    Returns:
        List of sample metadata dictionaries.
    """
    samples_dir = spec.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    n_bins = cfg.conditioning.z_bins
    all_metadata = []

    # Check if anatomical conditioning is enabled
    use_anatomical_conditioning = cfg.model.get("anatomical_conditioning", False)

    # Load z-bin priors if enabled
    pp_cfg = cfg.get("postprocessing", {})
    zbin_cfg = pp_cfg.get("zbin_priors", {})
    use_zbin_priors = (
        zbin_cfg.get("enabled", False)
        and "generation" in zbin_cfg.get("apply_to", [])
    )

    zbin_priors = None
    if use_zbin_priors or use_anatomical_conditioning:
        try:
            cache_dir = Path(cfg.data.cache_dir)
            zbin_priors = load_zbin_priors(
                cache_dir,
                zbin_cfg.get("priors_filename", "zbin_priors_brain_roi.npz"),
                n_bins,
            )
            if use_zbin_priors:
                logger.info(f"Loaded z-bin priors for post-processing ({len(zbin_priors)} bins)")
            if use_anatomical_conditioning:
                logger.info(f"Loaded z-bin priors for anatomical conditioning ({len(zbin_priors)} bins)")
        except Exception as e:
            logger.warning(f"Failed to load z-bin priors: {e}. Features disabled.")
            use_zbin_priors = False
            use_anatomical_conditioning = False

    # Set generator for reproducibility
    generator = get_generator(spec.seed)

    # Calculate total samples
    total = spec.total_samples()
    pbar = tqdm(total=total, desc="Generating samples")

    sample_counter = 0

    for zbin_spec in spec.zbin_specs:
        z_bin = zbin_spec.zbin

        # Generate control samples (class 0)
        for sample_idx in range(zbin_spec.control):
            metadata = _generate_single_sample(
                sampler=sampler,
                z_bin=z_bin,
                pathology_class=0,
                sample_idx=sample_idx,
                global_idx=sample_counter,
                samples_dir=samples_dir,
                output_dir=spec.output_dir,
                n_bins=n_bins,
                cfg=cfg,
                generator=generator,
                use_anatomical_conditioning=use_anatomical_conditioning,
                use_zbin_priors=use_zbin_priors,
                zbin_priors=zbin_priors,
                zbin_cfg=zbin_cfg,
            )
            all_metadata.append(metadata)
            sample_counter += 1
            pbar.update(1)

        # Generate lesion samples (class 1)
        for sample_idx in range(zbin_spec.lesion):
            metadata = _generate_single_sample(
                sampler=sampler,
                z_bin=z_bin,
                pathology_class=1,
                sample_idx=sample_idx,
                global_idx=sample_counter,
                samples_dir=samples_dir,
                output_dir=spec.output_dir,
                n_bins=n_bins,
                cfg=cfg,
                generator=generator,
                use_anatomical_conditioning=use_anatomical_conditioning,
                use_zbin_priors=use_zbin_priors,
                zbin_priors=zbin_priors,
                zbin_cfg=zbin_cfg,
            )
            all_metadata.append(metadata)
            sample_counter += 1
            pbar.update(1)

    pbar.close()
    return all_metadata


def _generate_single_sample(
    sampler: DiffusionSampler,
    z_bin: int,
    pathology_class: int,
    sample_idx: int,
    global_idx: int,
    samples_dir: Path,
    output_dir: Path,
    n_bins: int,
    cfg: DictConfig,
    generator: torch.Generator,
    use_anatomical_conditioning: bool,
    use_zbin_priors: bool,
    zbin_priors: dict | None,
    zbin_cfg: dict,
) -> dict[str, Any]:
    """Generate a single sample.

    Args:
        sampler: DiffusionSampler instance.
        z_bin: Z-bin index.
        pathology_class: 0 for control, 1 for lesion.
        sample_idx: Sample index within (z_bin, class) pair.
        global_idx: Global sample index.
        samples_dir: Directory to save samples.
        output_dir: Base output directory.
        n_bins: Number of z-bins.
        cfg: Configuration object.
        generator: Random generator for reproducibility.
        use_anatomical_conditioning: Whether to use anatomical conditioning.
        use_zbin_priors: Whether to apply z-bin prior post-processing.
        zbin_priors: Z-bin priors dictionary.
        zbin_cfg: Z-bin prior configuration.

    Returns:
        Sample metadata dictionary.
    """
    token = get_token_for_condition(z_bin, pathology_class, n_bins)

    # Get anatomical prior if needed
    anatomical_mask = None
    if use_anatomical_conditioning and zbin_priors is not None:
        anatomical_mask = get_anatomical_priors_as_input(
            [z_bin],
            zbin_priors,
            device=sampler.device,
        ).squeeze(0)  # Remove batch dim: (1, 1, H, W) -> (1, H, W)

    # Generate sample
    sample = sampler.sample_single(
        token=token,
        generator=generator,
        anatomical_mask=anatomical_mask,
    )

    # Split into image and mask
    image = sample[0].cpu().numpy()  # (H, W)
    mask = sample[1].cpu().numpy()   # (H, W)

    # Apply z-bin prior post-processing (if enabled)
    if use_zbin_priors and zbin_priors is not None:
        image, mask = apply_zbin_prior_postprocess(
            image, mask, z_bin, zbin_priors,
            zbin_cfg.get("gaussian_sigma_px", 0.7),
            zbin_cfg.get("min_component_px", 500),
            zbin_cfg.get("fallback", "prior"),
            zbin_cfg.get("n_first_bins", 0),
            zbin_cfg.get("max_components_for_first_bins", 1),
            zbin_cfg.get("relaxed_threshold_factor", 0.1),
            zbin_cfg.get("background_value", -1.0),
            zbin_cfg.get("use_prior_directly", False),
        )

    # Create metadata
    class_name = "control" if pathology_class == 0 else "lesion"
    metadata = {
        "sample_id": f"z{z_bin:02d}_{class_name}_s{sample_idx:04d}",
        "global_idx": global_idx,
        "z_bin": int(z_bin),
        "pathology_class": int(pathology_class),
        "class_name": class_name,
        "token": int(token),
        "sample_idx": sample_idx,
    }

    # Save sample
    filename = f"{metadata['sample_id']}.npz"
    filepath = samples_dir / filename

    save_sample_npz(filepath, image, mask, metadata)

    # Add filepath to metadata
    metadata["filepath"] = str(filepath.relative_to(output_dir))
    return metadata


def write_generation_index(
    metadata_list: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write generation metadata to CSV index.

    Args:
        metadata_list: List of sample metadata.
        output_path: Path to output CSV.
    """
    import csv

    if not metadata_list:
        logger.warning("No samples to index")
        return

    fieldnames = list(metadata_list[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)

    logger.info(f"Wrote {len(metadata_list)} entries to {output_path}")


def write_generation_summary(
    spec: GenerationSpec,
    metadata_list: list[dict[str, Any]],
    output_path: Path,
    elapsed_time: float | None = None,
) -> None:
    """Write human-readable generation summary.

    Args:
        spec: Generation specification.
        metadata_list: List of generated sample metadata.
        output_path: Path to output summary file.
        elapsed_time: Time taken for generation in seconds.
    """
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("SLIM-Diff Generation Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Checkpoint: {spec.checkpoint_path}\n")
        f.write(f"Config: {spec.config_path}\n")
        f.write(f"Output directory: {spec.output_dir}\n")
        f.write(f"Seed: {spec.seed}\n")
        f.write(f"Device: {spec.device}\n\n")

        f.write("-" * 40 + "\n")
        f.write("Sample Counts\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {len(metadata_list)}\n")
        f.write(f"Control samples: {spec.total_control()}\n")
        f.write(f"Lesion samples: {spec.total_lesion()}\n")
        f.write(f"Active z-bins: {len(spec.active_zbins())}\n\n")

        if elapsed_time is not None:
            samples_per_sec = len(metadata_list) / elapsed_time if elapsed_time > 0 else 0
            f.write(f"Generation time: {elapsed_time:.1f}s\n")
            f.write(f"Samples/second: {samples_per_sec:.2f}\n\n")

        f.write("-" * 40 + "\n")
        f.write("Per-ZBin Breakdown\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'ZBin':>6} {'Control':>10} {'Lesion':>10} {'Total':>10}\n")
        f.write("-" * 40 + "\n")

        for zbin_spec in spec.zbin_specs:
            if zbin_spec.total_samples() > 0:
                f.write(
                    f"{zbin_spec.zbin:>6} "
                    f"{zbin_spec.control:>10} "
                    f"{zbin_spec.lesion:>10} "
                    f"{zbin_spec.total_samples():>10}\n"
                )

        f.write("-" * 40 + "\n")
        f.write(
            f"{'TOTAL':>6} "
            f"{spec.total_control():>10} "
            f"{spec.total_lesion():>10} "
            f"{spec.total_samples():>10}\n"
        )
        f.write("=" * 60 + "\n")

    logger.info(f"Wrote generation summary to {output_path}")


def print_dry_run(spec: GenerationSpec) -> None:
    """Print dry run information.

    Args:
        spec: Generation specification.
    """
    print("\n" + "=" * 60)
    print("DRY RUN - Generation Plan")
    print("=" * 60)
    print(f"\nCheckpoint: {spec.checkpoint_path}")
    print(f"Config: {spec.config_path}")
    print(f"Output: {spec.output_dir}")
    print(f"Seed: {spec.seed}")
    print(f"Device: {spec.device}")
    print(f"\nTotal samples to generate: {spec.total_samples()}")
    print(f"  - Control: {spec.total_control()}")
    print(f"  - Lesion: {spec.total_lesion()}")
    print(f"\nZ-bins with samples: {len(spec.active_zbins())}")
    print("\nPer-ZBin Plan:")
    print("-" * 40)
    print(f"{'ZBin':>6} {'Control':>10} {'Lesion':>10}")
    print("-" * 40)

    for zbin_spec in spec.zbin_specs:
        if zbin_spec.total_samples() > 0:
            print(f"{zbin_spec.zbin:>6} {zbin_spec.control:>10} {zbin_spec.lesion:>10}")

    print("=" * 60 + "\n")


def visualize_generated_samples(
    output_dir: Path,
    max_samples_per_class: int = 8,
    max_zbins: int = 6,
    figsize: tuple[int, int] | None = None,
    save_path: Path | None = None,
) -> None:
    """Create a collage visualization of generated samples.

    Creates a grid showing image/mask pairs organized by z-bin and class.
    Layout: rows = z-bins (with image+mask sub-rows), columns = [control | lesion]

    Args:
        output_dir: Path to generation output directory.
        max_samples_per_class: Maximum samples to show per class per z-bin.
        max_zbins: Maximum number of z-bins to display.
        figsize: Optional figure size. Auto-calculated if None.
        save_path: Path to save PNG. Defaults to output_dir/samples_collage.png.
    """
    import matplotlib.pyplot as plt

    samples_dir = output_dir / "samples"
    if not samples_dir.exists():
        logger.error(f"Samples directory not found: {samples_dir}")
        return

    # Load all sample files and organize by z-bin and class
    sample_files = sorted(samples_dir.glob("*.npz"))
    if not sample_files:
        logger.error(f"No sample files found in {samples_dir}")
        return

    # Organize samples: {z_bin: {"control": [...], "lesion": [...]}}
    samples_by_zbin: dict[int, dict[str, list[Path]]] = {}
    for f in sample_files:
        data = np.load(f)
        z_bin = int(data["z_bin"])
        class_name = str(data["class_name"])

        if z_bin not in samples_by_zbin:
            samples_by_zbin[z_bin] = {"control": [], "lesion": []}
        samples_by_zbin[z_bin][class_name].append(f)

    # Select z-bins to display (evenly spaced if too many)
    zbins = sorted(samples_by_zbin.keys())
    if len(zbins) > max_zbins:
        indices = np.linspace(0, len(zbins) - 1, max_zbins, dtype=int)
        zbins = [zbins[i] for i in indices]

    # Determine actual number of samples per class (use max found, capped)
    n_control = min(max_samples_per_class, max(
        len(samples_by_zbin[z]["control"]) for z in zbins
    ))
    n_lesion = min(max_samples_per_class, max(
        len(samples_by_zbin[z]["lesion"]) for z in zbins
    ))

    n_zbins = len(zbins)
    # Columns: control images + control masks + separator + lesion images + lesion masks
    n_cols = n_control * 2 + n_lesion * 2  # image+mask for each sample

    # Calculate figure size
    if figsize is None:
        figsize = (max(n_cols * 1.2, 8), n_zbins * 2.5 + 1)

    fig, axes = plt.subplots(
        n_zbins, n_cols,
        figsize=figsize,
        squeeze=False,
    )

    # Remove all axes initially
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for row_idx, z_bin in enumerate(zbins):
        zbin_data = samples_by_zbin[z_bin]
        col_offset = 0

        # Plot control samples (image + mask pairs)
        control_files = zbin_data["control"][:n_control]
        for sample_path in control_files:
            data = np.load(sample_path)
            image = data["image"]
            mask = data["mask"]

            # Normalize image for display
            img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

            # Image
            ax_img = axes[row_idx, col_offset]
            ax_img.imshow(img_norm, cmap="gray", vmin=0, vmax=1)
            ax_img.axis("off")
            if row_idx == 0:
                ax_img.set_title("Img", fontsize=8, color="blue")

            # Mask
            ax_mask = axes[row_idx, col_offset + 1]
            ax_mask.imshow(mask, cmap="gray", vmin=-1, vmax=1)
            ax_mask.axis("off")
            if row_idx == 0:
                ax_mask.set_title("Mask", fontsize=8, color="blue")

            col_offset += 2

        # Fill remaining control columns if needed
        col_offset = n_control * 2

        # Plot lesion samples (image + mask pairs)
        lesion_files = zbin_data["lesion"][:n_lesion]
        for sample_path in lesion_files:
            data = np.load(sample_path)
            image = data["image"]
            mask = data["mask"]

            # Normalize image for display
            img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

            # Image
            ax_img = axes[row_idx, col_offset]
            ax_img.imshow(img_norm, cmap="gray", vmin=0, vmax=1)
            ax_img.axis("off")
            if row_idx == 0:
                ax_img.set_title("Img", fontsize=8, color="red")

            # Mask with lesion overlay
            ax_mask = axes[row_idx, col_offset + 1]
            # Create RGB to highlight lesions in red
            mask_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
            lesion_region = mask > 0
            if lesion_region.any():
                mask_rgb[lesion_region, 0] = 1.0  # Red channel
                mask_rgb[lesion_region, 1] *= 0.3
                mask_rgb[lesion_region, 2] *= 0.3
            ax_mask.imshow(mask_rgb)
            ax_mask.axis("off")
            if row_idx == 0:
                ax_mask.set_title("Mask", fontsize=8, color="red")

            col_offset += 2

        # Add z-bin label on the left side
        axes[row_idx, 0].annotate(
            f"z={z_bin}",
            xy=(-0.3, 0.5),
            xycoords="axes fraction",
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="center",
        )

    # Add section labels
    fig.text(
        0.25, 0.98, "Control",
        ha="center", fontsize=12, fontweight="bold", color="blue",
    )
    fig.text(
        0.75, 0.98, "Lesion",
        ha="center", fontsize=12, fontweight="bold", color="red",
    )

    # Add title
    fig.suptitle(
        f"Generated Samples: {output_dir.name}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Save figure
    if save_path is None:
        save_path = output_dir / "samples_collage.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved samples collage to {save_path}")

    plt.close(fig)


def main() -> None:
    """CLI entrypoint for JSON-based generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic samples from JSON specification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard generation
    slimdiff-generate-spec --spec generation_spec.json

    # Generate and create visualization collage
    slimdiff-generate-spec --spec spec.json --visualize

    # Visualize existing samples (without regenerating)
    slimdiff-generate-spec --spec spec.json --visualize-only

    # Validate specification without generating
    slimdiff-generate-spec --spec spec.json --validate-only

    # Dry run (show plan without generating)
    slimdiff-generate-spec --spec spec.json --dry-run

    # Verbose output
    slimdiff-generate-spec --spec spec.json --verbose

JSON Specification Formats:

1. Detailed format - explicit per-zbin specification:
   {
       "checkpoint_path": "/path/to/model.ckpt",
       "config_path": "src/diffusion/config/jsddpm.yaml",
       "output_dir": "./outputs/run_001",
       "seed": 42,
       "format": "detailed",
       "samples": [
           {"zbin": 0, "control": 10, "lesion": 5},
           {"zbin": 5, "control": 20, "lesion": 10}
       ]
   }

2. Compact format - defaults with optional overrides:
   {
       "checkpoint_path": "/path/to/model.ckpt",
       "config_path": "src/diffusion/config/jsddpm.yaml",
       "output_dir": "./outputs/run_002",
       "seed": 123,
       "format": "compact",
       "defaults": {
           "control": 10,
           "lesion": 10,
           "zbins": "all"
       },
       "overrides": [
           {"zbin": 0, "control": 5, "lesion": 2}
       ]
   }

The "zbins" field supports: "all", ranges like "0-29", or lists like "0,5,10,15".
        """,
    )
    parser.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Path to JSON specification file",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate specification without generating",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show generation plan without actually generating",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create collage visualization after generation",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only create visualization from existing samples (skip generation)",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=8,
        help="Max samples per class (control/lesion) in visualization (default: 8)",
    )
    parser.add_argument(
        "--max-zbins",
        type=int,
        default=6,
        help="Max z-bins to show in visualization (default: 6)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger("slimdiff", level=log_level)

    # Load specification
    spec_path = Path(args.spec)
    logger.info(f"Loading specification from {spec_path}")

    try:
        spec_dict = load_spec(spec_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in specification file: {e}")
        return

    # Load config to get n_bins for compact format parsing
    config_path = Path(spec_dict["config_path"])
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    cfg = OmegaConf.load(config_path)
    n_bins = cfg.conditioning.z_bins

    # Detect and parse format
    spec_format = detect_format(spec_dict)
    logger.info(f"Detected specification format: {spec_format}")

    if spec_format == "detailed":
        spec = parse_detailed_spec(spec_dict)
    elif spec_format == "compact":
        spec = parse_compact_spec(spec_dict, n_bins)
    else:
        logger.error(f"Unknown specification format: {spec_format}")
        return

    # Validate specification
    errors = validate_spec(spec, cfg)
    if errors:
        logger.error("Specification validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return

    logger.info("Specification validated successfully")

    # Handle validate-only mode
    if args.validate_only:
        print("\nSpecification is valid!")
        print(f"  Total samples: {spec.total_samples()}")
        print(f"  Control: {spec.total_control()}")
        print(f"  Lesion: {spec.total_lesion()}")
        print(f"  Active z-bins: {len(spec.active_zbins())}")
        return

    # Handle dry-run mode
    if args.dry_run:
        print_dry_run(spec)
        return

    # Handle visualize-only mode
    if args.visualize_only:
        if not spec.output_dir.exists():
            logger.error(f"Output directory not found: {spec.output_dir}")
            logger.error("Cannot visualize - no samples exist. Run generation first.")
            return
        logger.info(f"Creating visualization from existing samples in {spec.output_dir}")
        visualize_generated_samples(
            output_dir=spec.output_dir,
            max_samples_per_class=args.max_samples_per_class,
            max_zbins=args.max_zbins,
        )
        return

    # Set seed
    seed_everything(spec.seed)

    # Create output directory
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    # Save specification copy
    spec_copy_path = spec.output_dir / "generation_spec.json"
    with open(spec_copy_path, "w") as f:
        json.dump(spec_dict, f, indent=2)
    logger.info(f"Saved specification copy to {spec_copy_path}")

    # Load model
    logger.info("Loading model...")
    model, sampler = load_model_from_checkpoint(spec.checkpoint_path, cfg, spec.device)

    # Generate samples
    logger.info(f"Generating {spec.total_samples()} samples...")
    import time
    start_time = time.time()

    metadata = generate_samples_from_spec(sampler, spec, cfg)

    elapsed_time = time.time() - start_time

    # Write outputs
    write_generation_index(metadata, spec.output_dir / "generated_samples.csv")
    write_generation_summary(
        spec, metadata, spec.output_dir / "generation_summary.txt", elapsed_time
    )

    # Save generation config for reproducibility
    gen_config = {
        "spec_file": str(spec_path),
        "checkpoint": str(spec.checkpoint_path),
        "config": str(spec.config_path),
        "seed": spec.seed,
        "device": spec.device,
        "total_samples": len(metadata),
        "total_control": spec.total_control(),
        "total_lesion": spec.total_lesion(),
        "elapsed_time_seconds": elapsed_time,
    }
    OmegaConf.save(OmegaConf.create(gen_config), spec.output_dir / "generation_config.yaml")

    logger.info(f"Generation complete! Saved {len(metadata)} samples to {spec.output_dir}")
    logger.info(f"Time elapsed: {elapsed_time:.1f}s ({len(metadata)/elapsed_time:.2f} samples/sec)")

    # Create visualization if requested
    if args.visualize:
        logger.info("Creating samples visualization collage...")
        visualize_generated_samples(
            output_dir=spec.output_dir,
            max_samples_per_class=args.max_samples_per_class,
            max_zbins=args.max_zbins,
        )


if __name__ == "__main__":
    main()
