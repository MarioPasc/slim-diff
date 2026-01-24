"""Full-image extraction for real and synthetic data (no patch cropping).

Loads full 160x160 images from the slice cache and replica NPZ files,
assembling 2-channel (image + mask) tensors in the same format as the
patch extractor. This allows classification on full images to rule out
any bias introduced by the patch extraction process.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FullImageExtractor:
    """Extract full 160x160 images from real and synthetic data.

    Produces the same output format as PatchExtractor:
    - real_patches.npz: (N, 2, 160, 160) with z_bins, subject_ids, etc.
    - synthetic_patches.npz: (N, 2, 160, 160) with z_bins, etc.

    The only difference is that images are full-size instead of cropped patches.

    Args:
        cfg: Master configuration (classification_task.yaml).
        experiment_name: Name of the synthetic experiment.
    """

    def __init__(self, cfg: DictConfig, experiment_name: str | None = None) -> None:
        self.cfg = cfg
        self.seed = cfg.experiment.seed
        self.experiment_name = experiment_name

        # Real data paths
        self.real_cache_dir = Path(cfg.data.real.cache_dir)
        self.real_slices_dir = self.real_cache_dir / cfg.data.real.slices_subdir
        self.csv_files = cfg.data.real.csv_files

        # Synthetic experiment config
        self.synthetic_cfg = self._find_experiment_cfg(experiment_name) if experiment_name else None

    def _find_experiment_cfg(self, name: str) -> DictConfig | None:
        """Find experiment config by name."""
        for exp in self.cfg.data.synthetic.experiments:
            if exp.name == name:
                return exp
        raise ValueError(f"Experiment '{name}' not found in config.")

    def extract_all(self, output_dir: Path) -> dict:
        """Extract full images from both real and synthetic datasets.

        Args:
            output_dir: Directory to save extracted images.

        Returns:
            Extraction statistics dict.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Load real lesion samples
        logger.info("Loading real lesion images...")
        real_images, real_zbins, real_subjects = self._load_real_images()
        logger.info(f"Loaded {len(real_images)} real lesion images")

        # Step 2: Load synthetic lesion samples
        synth_images: np.ndarray | None = None
        synth_zbins: np.ndarray | None = None
        if self.synthetic_cfg is not None:
            logger.info(f"Loading synthetic images ({self.experiment_name})...")
            synth_images, synth_zbins = self._load_synthetic_images()
            logger.info(f"Loaded {len(synth_images)} synthetic lesion images")

            # Step 3: Balance per z-bin
            synth_images, synth_zbins = self._balance_per_zbin(
                real_zbins, synth_images, synth_zbins
            )
            logger.info(f"After balancing: {len(synth_images)} synthetic images")

        # Step 4: Save in the same format as patches
        self._save_npz(
            output_dir / "real_patches.npz",
            images=real_images,
            z_bins=real_zbins,
            subject_ids=real_subjects,
            source="real",
        )
        if synth_images is not None:
            self._save_npz(
                output_dir / "synthetic_patches.npz",
                images=synth_images,
                z_bins=synth_zbins,
                subject_ids=np.array([f"synth_{i}" for i in range(len(synth_images))]),
                source="synthetic",
            )

        # Stats
        stats = {
            "n_real": len(real_images),
            "n_synthetic": len(synth_images) if synth_images is not None else 0,
            "image_size": 160,
            "mode": "full_image",
            "zbin_distribution_real": self._zbin_dist(real_zbins),
            "zbin_distribution_synthetic": self._zbin_dist(synth_zbins) if synth_zbins is not None else {},
        }
        with open(output_dir / "extraction_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved extraction stats to {output_dir / 'extraction_stats.json'}")

        return stats

    def _load_real_images(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load full images from real slice cache (lesion-positive only)."""
        images: list[np.ndarray] = []
        zbins: list[int] = []
        subjects: list[str] = []

        for csv_file in self.csv_files:
            csv_path = self.real_cache_dir / csv_file
            if not csv_path.exists():
                logger.warning(f"CSV not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            lesion_df = df[df["has_lesion"] == True]  # noqa: E712

            for _, row in tqdm(
                lesion_df.iterrows(), total=len(lesion_df), desc=f"Loading {csv_file}"
            ):
                filepath = self.real_slices_dir / Path(row["filepath"]).name
                if not filepath.exists():
                    continue

                data = np.load(filepath)
                image = data["image"].astype(np.float32)
                mask = data["mask"].astype(np.float32)

                # Stack as 2-channel: (2, 160, 160)
                stacked = np.stack([image, mask], axis=0)
                images.append(stacked)
                zbins.append(int(row["z_bin"]))
                subjects.append(str(row["subject_id"]))

        return (
            np.array(images, dtype=np.float32),
            np.array(zbins, dtype=np.int32),
            np.array(subjects),
        )

    def _load_synthetic_images(self) -> tuple[np.ndarray, np.ndarray]:
        """Load full images from replica NPZ files (lesion-positive only)."""
        images: list[np.ndarray] = []
        zbins: list[int] = []

        if self.synthetic_cfg is None:
            return np.empty((0, 2, 160, 160), dtype=np.float32), np.empty((0,), dtype=np.int32)

        results_base = Path(self.cfg.data.synthetic.results_base_dir)
        replicas_dir = results_base / self.experiment_name / self.synthetic_cfg.replicas_subdir

        for replica_id in tqdm(
            list(self.synthetic_cfg.replica_ids), desc="Loading replicas"
        ):
            replica_path = replicas_dir / f"replica_{replica_id:03d}.npz"
            if not replica_path.exists():
                logger.warning(f"Replica not found: {replica_path}")
                continue

            data = np.load(replica_path)
            rep_images = data["images"].astype(np.float32)  # (3000, 160, 160)
            rep_masks = data["masks"].astype(np.float32)
            rep_zbins = data["zbin"]
            lesion_present = data["lesion_present"]

            lesion_idx = np.where(lesion_present == 1)[0]

            for idx in lesion_idx:
                mask = rep_masks[idx]
                if (mask > 0).sum() == 0:
                    continue

                stacked = np.stack([rep_images[idx], mask], axis=0)
                images.append(stacked)
                zbins.append(int(rep_zbins[idx]))

        return (
            np.array(images, dtype=np.float32),
            np.array(zbins, dtype=np.int32),
        )

    def _balance_per_zbin(
        self,
        real_zbins: np.ndarray,
        synth_images: np.ndarray,
        synth_zbins: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Balance synthetic samples to match real count per z-bin."""
        rng = np.random.default_rng(self.seed)

        real_counts: dict[int, int] = {}
        for zb in real_zbins:
            real_counts[int(zb)] = real_counts.get(int(zb), 0) + 1

        synth_by_zbin: dict[int, list[int]] = {}
        for i, zb in enumerate(synth_zbins):
            synth_by_zbin.setdefault(int(zb), []).append(i)

        selected_indices: list[int] = []
        for zb, count in real_counts.items():
            available = synth_by_zbin.get(zb, [])
            if not available:
                continue
            n_select = min(count, len(available))
            chosen = rng.choice(available, size=n_select, replace=False)
            selected_indices.extend(chosen.tolist())

        selected_indices.sort()
        return synth_images[selected_indices], synth_zbins[selected_indices]

    def _save_npz(
        self,
        filepath: Path,
        images: np.ndarray,
        z_bins: np.ndarray,
        subject_ids: np.ndarray,
        source: str,
    ) -> None:
        """Save in the same format as patch extraction."""
        np.savez_compressed(
            filepath,
            patches=images,
            z_bins=z_bins,
            subject_ids=subject_ids,
            sources=np.array([source] * len(images)),
            replica_ids=np.full(len(images), -1, dtype=np.int32),
            sample_indices=np.arange(len(images), dtype=np.int32),
        )
        logger.info(f"Saved {len(images)} images to {filepath}")

    @staticmethod
    def _zbin_dist(zbins: np.ndarray) -> dict[int, int]:
        dist: dict[int, int] = {}
        for zb in zbins:
            zb = int(zb)
            dist[zb] = dist.get(zb, 0) + 1
        return dist
