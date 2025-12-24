"""I/O utilities for JS-DDPM.

Provides functions for reading dataset metadata, discovering subjects,
constructing file paths, and saving generated samples.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def read_dataset_json(dataset_path: Path | str) -> dict[str, Any]:
    """Read and parse a dataset.json file.

    Args:
        dataset_path: Path to dataset directory containing dataset.json.

    Returns:
        Parsed JSON content as dictionary.

    Raises:
        FileNotFoundError: If dataset.json does not exist.
        json.JSONDecodeError: If JSON is malformed.
    """
    dataset_path = Path(dataset_path)
    json_path = dataset_path / "dataset.json"

    if not json_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    logger.debug(f"Loaded dataset.json from {json_path}")
    return data


def extract_subject_id(filename: str, prefix: str) -> str | None:
    """Extract subject ID from a filename.

    Args:
        filename: Filename like 'MRIe_001_0000.nii.gz'.
        prefix: Expected prefix like 'MRIe' or 'MRIcontrol'.

    Returns:
        Subject ID like 'MRIe_001' or None if pattern doesn't match.
    """
    # Pattern: {prefix}_{number}_XXXX.nii.gz
    pattern = rf"^({prefix}_\d+)_\d+\.nii\.gz$"
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    return None


def discover_subjects(
    dataset_path: Path | str,
    image_dir: str = "imagesTr",
    prefix: str = "MRIe",
    modality_index: int = 0,
) -> list[str]:
    """Discover unique subject IDs in a dataset directory.

    Args:
        dataset_path: Path to dataset directory.
        image_dir: Subdirectory containing images.
        prefix: Filename prefix for subjects.
        modality_index: Modality index to filter by (0=FLAIR, 1=T1N).

    Returns:
        Sorted list of unique subject IDs.
    """
    dataset_path = Path(dataset_path)
    images_path = dataset_path / image_dir

    if not images_path.exists():
        logger.warning(f"Image directory not found: {images_path}")
        return []

    # Find all files matching the modality index
    modality_suffix = f"_{modality_index:04d}.nii.gz"
    subject_ids = set()

    for f in images_path.iterdir():
        if f.name.endswith(modality_suffix):
            subject_id = extract_subject_id(f.name, prefix)
            if subject_id:
                subject_ids.add(subject_id)

    subjects = sorted(subject_ids)
    logger.info(f"Discovered {len(subjects)} subjects in {images_path}")
    return subjects


def get_image_path(
    dataset_path: Path | str,
    subject_id: str,
    modality_index: int = 0,
    image_dir: str = "imagesTr",
) -> Path:
    """Construct image file path for a subject.

    Args:
        dataset_path: Path to dataset directory.
        subject_id: Subject ID like 'MRIe_001'.
        modality_index: Modality index (0=FLAIR, 1=T1N).
        image_dir: Subdirectory containing images.

    Returns:
        Full path to the image file.
    """
    dataset_path = Path(dataset_path)
    filename = f"{subject_id}_{modality_index:04d}.nii.gz"
    return dataset_path / image_dir / filename


def get_label_path(
    dataset_path: Path | str,
    subject_id: str,
    label_dir: str = "labelsTr",
) -> Path:
    """Construct label file path for a subject.

    Args:
        dataset_path: Path to dataset directory.
        subject_id: Subject ID like 'MRIe_001'.
        label_dir: Subdirectory containing labels.

    Returns:
        Full path to the label file.
    """
    dataset_path = Path(dataset_path)
    filename = f"{subject_id}.nii.gz"
    return dataset_path / label_dir / filename


def save_sample_npz(
    output_path: Path | str,
    image: NDArray[np.float32],
    mask: NDArray[np.float32],
    metadata: dict[str, Any],
) -> None:
    """Save a generated sample as .npz file.

    Args:
        output_path: Path to save the .npz file.
        image: Image array of shape (H, W) or (1, H, W), values in [-1, 1].
        mask: Mask array of shape (H, W) or (1, H, W), values in {-1, 1}.
        metadata: Dictionary of metadata (subject_id, z_index, etc.).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure 2D for storage
    if image.ndim == 3:
        image = image.squeeze(0)
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    np.savez_compressed(
        output_path,
        image=image.astype(np.float32),
        mask=mask.astype(np.float32),
        **metadata,
    )
    logger.debug(f"Saved sample to {output_path}")


def load_sample_npz(npz_path: Path | str) -> dict[str, Any]:
    """Load a sample from .npz file.

    Args:
        npz_path: Path to the .npz file.

    Returns:
        Dictionary with 'image', 'mask', and metadata fields.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    result = {
        "image": data["image"],
        "mask": data["mask"],
    }

    # Load any additional metadata
    for key in data.files:
        if key not in ("image", "mask"):
            result[key] = data[key].item() if data[key].ndim == 0 else data[key]

    return result


def make_grid_filename(
    output_dir: Path | str,
    epoch: int,
    prefix: str = "viz",
) -> Path:
    """Create filename for visualization grid.

    Args:
        output_dir: Output directory.
        epoch: Current epoch number.
        prefix: Filename prefix.

    Returns:
        Full path for the grid image.
    """
    output_dir = Path(output_dir) / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{prefix}_epoch_{epoch:04d}.png"


def parse_subject_prefix(dataset_name: str) -> str:
    """Infer subject filename prefix from dataset name.

    Args:
        dataset_name: Dataset name like 'Dataset210_MRIe_none'.

    Returns:
        Prefix like 'MRIe' or 'MRIcontrol'.
    """
    if "MRIe" in dataset_name and "control" not in dataset_name.lower():
        return "MRIe"
    elif "MRIcontrol" in dataset_name or "control" in dataset_name.lower():
        return "MRIcontrol"
    else:
        # Fallback: extract from dataset name
        parts = dataset_name.split("_")
        if len(parts) >= 2:
            return parts[1]
        return "MRI"
