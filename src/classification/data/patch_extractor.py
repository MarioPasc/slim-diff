"""Lesion-centered patch extraction for real and synthetic data.

Extracts patches around lesion regions from both real (slice cache) and
synthetic (replica NPZ) datasets for training the real-vs-synthetic classifier.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.shared.ablation import ExperimentCoordinate, ExperimentDiscoverer, AblationSpace

logger = logging.getLogger(__name__)


@dataclass
class PatchInfo:
    """Metadata for a single extracted patch."""

    source: Literal["real", "synthetic"]
    z_bin: int
    subject_id: str = ""
    original_filepath: str = ""
    replica_id: int = -1
    sample_idx: int = -1
    bbox: tuple[int, int, int, int] = field(default_factory=lambda: (0, 0, 0, 0))


@dataclass
class PatchCollection:
    """Collection of extracted patches with metadata."""

    patches: np.ndarray  # (N, 2, H, W) — image + mask channels
    z_bins: np.ndarray  # (N,) int
    subject_ids: np.ndarray  # (N,) str
    infos: list[PatchInfo]

    @property
    def n_samples(self) -> int:
        return len(self.patches)


@dataclass
class ExtractionStats:
    """Statistics from patch extraction."""

    n_real: int
    n_synthetic: int
    max_lesion_height: int
    max_lesion_width: int
    patch_size: int
    zbin_distribution_real: dict[int, int]
    zbin_distribution_synthetic: dict[int, int]


class PatchExtractor:
    """Extract lesion-centered patches from real and synthetic data.

    Computes a dynamic (or fixed) patch size, then extracts patches centered
    on lesion centroids from both data sources.

    Args:
        cfg: Master configuration (classification_task.yaml).
        experiment: Experiment to extract from. Can be:
            - ExperimentCoordinate: Direct coordinate object
            - str: Display name (e.g., "sc_0.5__x0_lp_1.5") or legacy name (e.g., "x0_lp_1.5")
            - None: Extract only real patches
    """

    def __init__(
        self,
        cfg: DictConfig,
        experiment: ExperimentCoordinate | str | None = None,
    ) -> None:
        self.cfg = cfg
        self.seed = cfg.experiment.seed
        self.patch_cfg = cfg.data.patch_extraction

        # Real data paths
        self.real_cache_dir = Path(cfg.data.real.cache_dir)
        self.real_slices_dir = self.real_cache_dir / cfg.data.real.slices_subdir
        self.csv_files = cfg.data.real.csv_files

        # Synthetic data paths (per experiment)
        self.experiment_coord = self._resolve_experiment(experiment)
        self.experiment_name = self.experiment_coord.to_display_name() if self.experiment_coord else None

        # Create discoverer for path resolution
        self._discoverer: ExperimentDiscoverer | None = None
        if self.experiment_coord is not None:
            try:
                space = AblationSpace.from_config(cfg)
            except (KeyError, TypeError):
                space = AblationSpace.default()
            self._discoverer = ExperimentDiscoverer(
                base_dir=Path(cfg.data.synthetic.results_base_dir),
                space=space,
                default_self_cond_p=cfg.get("ablation", {}).get("default_self_cond_p", 0.5),
            )

        # Patch settings
        self.padding = self.patch_cfg.padding
        self.min_patch_size = self.patch_cfg.min_patch_size
        self.max_patch_size = self.patch_cfg.max_patch_size
        self.method = self.patch_cfg.method

    def _resolve_experiment(
        self, experiment: ExperimentCoordinate | str | None
    ) -> ExperimentCoordinate | None:
        """Resolve experiment argument to ExperimentCoordinate.

        Args:
            experiment: ExperimentCoordinate, display name, legacy name, or None.

        Returns:
            ExperimentCoordinate or None.
        """
        if experiment is None:
            return None

        if isinstance(experiment, ExperimentCoordinate):
            return experiment

        # Try parsing as display name first (sc_0.5__x0_lp_1.5)
        try:
            return ExperimentCoordinate.from_display_name(experiment)
        except ValueError:
            pass

        # Try parsing as legacy name (x0_lp_1.5)
        default_sc = self.cfg.get("ablation", {}).get("default_self_cond_p", 0.5)
        try:
            return ExperimentCoordinate.from_legacy_name(experiment, default_self_cond_p=default_sc)
        except ValueError as e:
            raise ValueError(f"Cannot parse experiment: {experiment}") from e

    def _get_replicas_dir(self) -> Path:
        """Get the replicas directory for the current experiment."""
        if self._discoverer is None or self.experiment_coord is None:
            raise ValueError("No experiment configured")

        replicas_subdir = self.cfg.data.synthetic.get("replicas_subdir", "replicas")
        return self._discoverer.get_replicas_path(self.experiment_coord, replicas_subdir)

    def _get_replica_ids(self) -> list[int]:
        """Get replica IDs to process."""
        return list(self.cfg.data.synthetic.get("replica_ids", [0, 1, 2, 3, 4]))

    def extract_all(self, output_dir: Path) -> ExtractionStats:
        """Extract patches from both real and synthetic datasets.

        Args:
            output_dir: Directory to save extracted patches.

        Returns:
            Extraction statistics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Scan datasets for lesion samples
        logger.info("Scanning real dataset for lesion samples...")
        real_samples, real_bboxes = self._scan_real_dataset()
        logger.info(f"Found {len(real_samples)} real lesion samples")

        synth_samples: list[dict] = []
        synth_bboxes: list[tuple[int, int, int, int]] = []
        if self.experiment_coord is not None:
            logger.info(f"Scanning synthetic dataset ({self.experiment_name})...")
            synth_samples, synth_bboxes = self._scan_synthetic_dataset()
            logger.info(f"Found {len(synth_samples)} synthetic lesion samples")

        # Step 2: Compute patch size
        all_bboxes = real_bboxes + synth_bboxes
        if not all_bboxes:
            raise ValueError("No lesion samples found in either dataset.")

        if self.method == "dynamic":
            patch_size = self._compute_dynamic_patch_size(all_bboxes)
        else:
            patch_size = self.patch_cfg.fixed_size
        logger.info(f"Using patch size: {patch_size}x{patch_size}")

        # Step 3: Balance synthetic to match real per z-bin
        if synth_samples:
            synth_samples, synth_bboxes = self._balance_per_zbin(
                real_samples, synth_samples, synth_bboxes
            )
            logger.info(f"After balancing: {len(synth_samples)} synthetic samples")

        # Step 4: Extract patches
        logger.info("Extracting real patches...")
        real_collection = self._extract_patches(real_samples, real_bboxes, patch_size, "real")

        synth_collection = None
        if synth_samples:
            logger.info("Extracting synthetic patches...")
            synth_collection = self._extract_patches(synth_samples, synth_bboxes, patch_size, "synthetic")

        # Step 5: Save
        self._save_collection(output_dir / "real_patches.npz", real_collection)
        if synth_collection is not None:
            self._save_collection(output_dir / "synthetic_patches.npz", synth_collection)

        # Compute and save stats
        stats = ExtractionStats(
            n_real=real_collection.n_samples,
            n_synthetic=synth_collection.n_samples if synth_collection else 0,
            max_lesion_height=max(y2 - y1 + 1 for y1, y2, _, _ in all_bboxes),
            max_lesion_width=max(x2 - x1 + 1 for _, _, x1, x2 in all_bboxes),
            patch_size=patch_size,
            zbin_distribution_real=self._zbin_dist(real_collection.infos),
            zbin_distribution_synthetic=self._zbin_dist(synth_collection.infos) if synth_collection else {},
        )
        self._save_stats(output_dir / "extraction_stats.json", stats)

        return stats

    def extract_real_only(self, output_dir: Path) -> PatchCollection:
        """Extract only real patches (for control experiment).

        Args:
            output_dir: Directory to save extracted patches.

        Returns:
            PatchCollection of real lesion patches.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        real_samples, real_bboxes = self._scan_real_dataset()
        if not real_bboxes:
            raise ValueError("No real lesion samples found.")

        if self.method == "dynamic":
            patch_size = self._compute_dynamic_patch_size(real_bboxes)
        else:
            patch_size = self.patch_cfg.fixed_size

        collection = self._extract_patches(real_samples, real_bboxes, patch_size, "real")
        self._save_collection(output_dir / "real_patches.npz", collection)
        return collection

    # -------------------------------------------------------------------------
    # Scanning
    # -------------------------------------------------------------------------

    def _scan_real_dataset(self) -> tuple[list[dict], list[tuple[int, int, int, int]]]:
        """Scan real slice cache for lesion samples."""
        samples: list[dict] = []
        bboxes: list[tuple[int, int, int, int]] = []

        for csv_file in self.csv_files:
            csv_path = self.real_cache_dir / csv_file
            if not csv_path.exists():
                logger.warning(f"CSV not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            lesion_df = df[df["has_lesion"] == True]  # noqa: E712

            for _, row in tqdm(
                lesion_df.iterrows(), total=len(lesion_df), desc=f"Scanning {csv_file}"
            ):
                filepath = self.real_slices_dir / Path(row["filepath"]).name
                if not filepath.exists():
                    continue

                data = np.load(filepath)
                mask = data["mask"]
                binary_mask = (mask > 0).astype(np.uint8)

                if binary_mask.sum() == 0:
                    continue

                bbox = self._compute_bbox(binary_mask)
                bboxes.append(bbox)
                samples.append({
                    "filepath": str(filepath),
                    "z_bin": int(row["z_bin"]),
                    "subject_id": str(row["subject_id"]),
                    "image": data["image"],
                    "mask": mask,
                })

        return samples, bboxes

    def _scan_synthetic_dataset(self) -> tuple[list[dict], list[tuple[int, int, int, int]]]:
        """Scan synthetic replicas for lesion samples."""
        samples: list[dict] = []
        bboxes: list[tuple[int, int, int, int]] = []

        if self.experiment_coord is None:
            return samples, bboxes

        replicas_dir = self._get_replicas_dir()
        replica_ids = self._get_replica_ids()

        for replica_id in tqdm(replica_ids, desc="Scanning replicas"):
            replica_path = replicas_dir / f"replica_{replica_id:03d}.npz"
            if not replica_path.exists():
                logger.warning(f"Replica not found: {replica_path}")
                continue

            data = np.load(replica_path)
            images = data["images"]
            masks = data["masks"]
            zbins = data["zbin"]
            lesion_present = data["lesion_present"]

            lesion_indices = np.where(lesion_present == 1)[0]

            for idx in lesion_indices:
                mask = masks[idx]
                binary_mask = (mask > 0).astype(np.uint8)

                if binary_mask.sum() == 0:
                    continue

                bbox = self._compute_bbox(binary_mask)
                bboxes.append(bbox)
                samples.append({
                    "replica_id": int(replica_id),
                    "sample_idx": int(idx),
                    "z_bin": int(zbins[idx]),
                    "subject_id": f"synth_r{replica_id:03d}_s{idx:04d}",
                    "image": images[idx],
                    "mask": mask,
                })

        return samples, bboxes

    # -------------------------------------------------------------------------
    # Patch computation
    # -------------------------------------------------------------------------

    def _compute_bbox(self, binary_mask: np.ndarray) -> tuple[int, int, int, int]:
        """Compute bounding box of non-zero region."""
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return (int(y1), int(y2), int(x1), int(x2))

    def _compute_dynamic_patch_size(self, bboxes: list[tuple[int, int, int, int]]) -> int:
        """Compute patch size from max lesion dimensions plus padding."""
        max_height = max(y2 - y1 + 1 for y1, y2, _, _ in bboxes)
        max_width = max(x2 - x1 + 1 for _, _, x1, x2 in bboxes)
        max_dim = max(max_height, max_width) + 2 * self.padding
        patch_size = int(np.ceil(max_dim / 8) * 8)  # Round to multiple of 8
        patch_size = max(self.min_patch_size, min(patch_size, self.max_patch_size))
        logger.info(f"Max lesion size: {max_height}x{max_width}, patch size: {patch_size}")
        return patch_size

    def _balance_per_zbin(
        self,
        real_samples: list[dict],
        synth_samples: list[dict],
        synth_bboxes: list[tuple[int, int, int, int]],
    ) -> tuple[list[dict], list[tuple[int, int, int, int]]]:
        """Balance synthetic samples to match real count per z-bin."""
        rng = np.random.default_rng(self.seed)

        # Count real per z-bin
        real_counts: dict[int, int] = {}
        for s in real_samples:
            zb = s["z_bin"]
            real_counts[zb] = real_counts.get(zb, 0) + 1

        # Group synthetic by z-bin
        synth_by_zbin: dict[int, list[int]] = {}
        for i, s in enumerate(synth_samples):
            zb = s["z_bin"]
            synth_by_zbin.setdefault(zb, []).append(i)

        # Sample synthetic to match real per z-bin
        selected_indices: list[int] = []
        for zb, count in real_counts.items():
            available = synth_by_zbin.get(zb, [])
            if not available:
                continue
            n_select = min(count, len(available))
            chosen = rng.choice(available, size=n_select, replace=False)
            selected_indices.extend(chosen.tolist())

        selected_indices.sort()
        balanced_samples = [synth_samples[i] for i in selected_indices]
        balanced_bboxes = [synth_bboxes[i] for i in selected_indices]
        return balanced_samples, balanced_bboxes

    def _extract_patches(
        self,
        samples: list[dict],
        bboxes: list[tuple[int, int, int, int]],
        patch_size: int,
        source: Literal["real", "synthetic"],
    ) -> PatchCollection:
        """Extract centered patches from samples."""
        patches: list[np.ndarray] = []
        infos: list[PatchInfo] = []

        for sample, bbox in tqdm(
            zip(samples, bboxes), total=len(samples), desc=f"Extracting {source}"
        ):
            image = sample["image"]
            mask = sample["mask"]

            # Centroid of lesion bbox
            y1, y2, x1, x2 = bbox
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2

            patch_image = self._extract_centered_patch(image, cy, cx, patch_size)
            patch_mask = self._extract_centered_patch(mask, cy, cx, patch_size)

            patch = np.stack([patch_image, patch_mask], axis=0)  # (2, H, W)
            patches.append(patch)

            infos.append(PatchInfo(
                source=source,
                z_bin=sample["z_bin"],
                subject_id=sample.get("subject_id", ""),
                original_filepath=sample.get("filepath", ""),
                replica_id=sample.get("replica_id", -1),
                sample_idx=sample.get("sample_idx", -1),
                bbox=bbox,
            ))

        patches_array = np.stack(patches, axis=0).astype(np.float32)
        z_bins = np.array([info.z_bin for info in infos], dtype=np.int32)
        subject_ids = np.array([info.subject_id for info in infos])

        return PatchCollection(
            patches=patches_array,
            z_bins=z_bins,
            subject_ids=subject_ids,
            infos=infos,
        )

    def _extract_centered_patch(
        self, image: np.ndarray, cy: int, cx: int, patch_size: int
    ) -> np.ndarray:
        """Extract a square patch centered at (cy, cx), padding at boundaries."""
        h, w = image.shape[:2]
        half = patch_size // 2

        y1 = cy - half
        y2 = cy + half
        x1 = cx - half
        x2 = cx + half

        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - h)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - w)

        y1_c = max(0, y1)
        y2_c = min(h, y2)
        x1_c = max(0, x1)
        x2_c = min(w, x2)

        patch = image[y1_c:y2_c, x1_c:x2_c]

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            patch = np.pad(
                patch,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=-1.0,
            )

        return patch

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------

    def _save_collection(self, filepath: Path, collection: PatchCollection) -> None:
        """Save patch collection to compressed NPZ."""
        np.savez_compressed(
            filepath,
            patches=collection.patches,
            z_bins=collection.z_bins,
            subject_ids=collection.subject_ids,
            sources=np.array([info.source for info in collection.infos]),
            replica_ids=np.array([info.replica_id for info in collection.infos], dtype=np.int32),
            sample_indices=np.array([info.sample_idx for info in collection.infos], dtype=np.int32),
        )
        logger.info(f"Saved {collection.n_samples} patches to {filepath}")

    @staticmethod
    def load_collection(filepath: Path) -> PatchCollection:
        """Load a saved patch collection from NPZ.

        Args:
            filepath: Path to the NPZ file.

        Returns:
            Loaded PatchCollection.
        """
        data = np.load(filepath, allow_pickle=True)
        patches = data["patches"]
        z_bins = data["z_bins"]
        subject_ids = data["subject_ids"]
        sources = data["sources"]
        replica_ids = data["replica_ids"]
        sample_indices = data["sample_indices"]

        infos = []
        for i in range(len(patches)):
            infos.append(PatchInfo(
                source=str(sources[i]),
                z_bin=int(z_bins[i]),
                subject_id=str(subject_ids[i]),
                replica_id=int(replica_ids[i]),
                sample_idx=int(sample_indices[i]),
            ))

        return PatchCollection(
            patches=patches,
            z_bins=z_bins,
            subject_ids=subject_ids,
            infos=infos,
        )

    def _save_stats(self, filepath: Path, stats: ExtractionStats) -> None:
        """Save extraction statistics to JSON."""
        stats_dict = {
            "n_real": stats.n_real,
            "n_synthetic": stats.n_synthetic,
            "max_lesion_height": stats.max_lesion_height,
            "max_lesion_width": stats.max_lesion_width,
            "patch_size": stats.patch_size,
            "zbin_distribution_real": stats.zbin_distribution_real,
            "zbin_distribution_synthetic": stats.zbin_distribution_synthetic,
        }
        with open(filepath, "w") as f:
            json.dump(stats_dict, f, indent=2)
        logger.info(f"Saved stats to {filepath}")

    @staticmethod
    def _zbin_dist(infos: list[PatchInfo]) -> dict[int, int]:
        dist: dict[int, int] = {}
        for info in infos:
            dist[info.z_bin] = dist.get(info.z_bin, 0) + 1
        return dist
