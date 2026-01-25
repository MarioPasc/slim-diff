"""Data loaders for ICIP 2026 ablation study experiments.

Provides unified loading of real and synthetic data across multiple experiments.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm


class ICIPExperimentLoader:
    """Load data from ICIP 2026 ablation study structure.

    Expected directory structure:
        runs_dir/
        ├── epsilon_lp_1.5/
        │   └── replicas/
        │       ├── replica_000.npz
        │       └── ...
        ├── x0_lp_2.0/
        │   └── replicas/
        │       └── ...
        └── ...

    Attributes:
        runs_dir: Path to the runs directory containing experiment folders.
        cache_dir: Path to the slice cache directory with train/val/test.csv.
        experiments: List of discovered experiment names.
        valid_zbins: List of valid z-bins (default: 0-29).
    """

    # Pattern to match experiment names: {prediction_type}_lp_{norm}
    EXPERIMENT_PATTERN = re.compile(r"^(epsilon|velocity|x0)_lp_([\d.]+)$")

    def __init__(
        self,
        runs_dir: Path,
        cache_dir: Path,
        valid_zbins: list[int] | None = None,
    ):
        """Initialize the experiment loader.

        Args:
            runs_dir: Path to the runs directory containing experiment folders.
            cache_dir: Path to the slice cache directory with train/val/test.csv.
            valid_zbins: List of valid z-bins to include (default: 0-29).
        """
        self.runs_dir = Path(runs_dir)
        self.cache_dir = Path(cache_dir)
        self.valid_zbins = valid_zbins if valid_zbins is not None else list(range(30))
        self.experiments = self._discover_experiments()

    def _discover_experiments(self) -> list[str]:
        """Discover all valid experiment directories.

        Returns:
            Sorted list of experiment names matching the expected pattern.
        """
        experiments = []
        for d in self.runs_dir.iterdir():
            if d.is_dir() and self.EXPERIMENT_PATTERN.match(d.name):
                # Verify replicas directory exists
                replicas_dir = d / "replicas"
                if replicas_dir.exists():
                    experiments.append(d.name)
        return sorted(experiments)

    @staticmethod
    def parse_experiment_name(name: str) -> tuple[str, float]:
        """Extract prediction_type and lp_norm from experiment name.

        Args:
            name: Experiment name (e.g., "x0_lp_1.5").

        Returns:
            Tuple of (prediction_type, lp_norm).

        Raises:
            ValueError: If name doesn't match expected pattern.
        """
        match = ICIPExperimentLoader.EXPERIMENT_PATTERN.match(name)
        if not match:
            raise ValueError(f"Invalid experiment name: {name}")
        return match.group(1), float(match.group(2))

    def get_replica_paths(self, experiment: str) -> list[Path]:
        """Get sorted list of replica NPZ paths for an experiment.

        Args:
            experiment: Experiment name.

        Returns:
            List of Path objects for replica files.
        """
        replicas_dir = self.runs_dir / experiment / "replicas"
        return sorted(replicas_dir.glob("replica_*.npz"))

    def load_replica(
        self,
        experiment: str,
        replica_id: int,
        channel: str = "image",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load replica images and z-bins from NPZ file.

        Args:
            experiment: Experiment name.
            replica_id: Replica index (0-4).
            channel: Which channel to load ("image", "mask", or "joint").

        Returns:
            images: (N, H, W) or (N, 2, H, W) float32 array in [-1, 1].
            zbins: (N,) int32 array.
        """
        replica_path = self.runs_dir / experiment / "replicas" / f"replica_{replica_id:03d}.npz"
        if not replica_path.exists():
            raise FileNotFoundError(f"Replica not found: {replica_path}")

        data = np.load(replica_path)
        zbins = data["zbin"].astype(np.int32)

        # Load appropriate channel(s)
        if channel == "image":
            images = data["images"].astype(np.float32)
        elif channel == "mask":
            images = data["masks"].astype(np.float32)
        elif channel == "joint":
            # Stack image and mask as 2 channels
            img = data["images"].astype(np.float32)
            mask = data["masks"].astype(np.float32)
            images = np.stack([img, mask], axis=1)  # (N, 2, H, W)
        else:
            raise ValueError(f"Invalid channel: {channel}. Use 'image', 'mask', or 'joint'.")

        # Filter by valid z-bins
        mask = np.isin(zbins, self.valid_zbins)
        images = images[mask]
        zbins = zbins[mask]

        return images, zbins

    def load_all_replicas(
        self,
        experiment: str,
        channel: str = "image",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and concatenate all replicas for an experiment.

        Args:
            experiment: Experiment name.
            channel: Which channel to load ("image", "mask", or "joint").

        Returns:
            images: Concatenated images array.
            zbins: Concatenated z-bins array.
            replica_ids: Array indicating which replica each sample came from.
        """
        replica_paths = self.get_replica_paths(experiment)
        all_images = []
        all_zbins = []
        all_replica_ids = []

        for replica_path in replica_paths:
            replica_id = int(replica_path.stem.split("_")[-1])
            images, zbins = self.load_replica(experiment, replica_id, channel)
            all_images.append(images)
            all_zbins.append(zbins)
            all_replica_ids.append(np.full(len(images), replica_id, dtype=np.int32))

        return (
            np.concatenate(all_images, axis=0),
            np.concatenate(all_zbins, axis=0),
            np.concatenate(all_replica_ids, axis=0),
        )

    def load_real_data(
        self,
        splits: list[str] | None = None,
        channel: str = "image",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load real slices from cache directory.

        Args:
            splits: List of splits to load (default: ["test"]).
            channel: Which channel to load ("image", "mask", or "joint").

        Returns:
            images: (N, H, W) or (N, 2, H, W) float32 array in [-1, 1].
            zbins: (N,) int32 array.
        """
        if splits is None:
            splits = ["test"]

        all_images = []
        all_zbins = []

        for split in splits:
            csv_path = self.cache_dir / f"{split}.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping...")
                continue

            images, zbins = self._load_slices_from_csv(csv_path, channel)
            all_images.append(images)
            all_zbins.append(zbins)
            print(f"  Loaded {len(images)} slices from {split}")

        if not all_images:
            raise FileNotFoundError(f"No valid CSV files found for splits: {splits}")

        return (
            np.concatenate(all_images, axis=0),
            np.concatenate(all_zbins, axis=0),
        )

    def _load_slices_from_csv(
        self,
        csv_path: Path,
        channel: str = "image",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load slices from a CSV index file.

        Args:
            csv_path: Path to CSV file with columns: filepath, z_bin, etc.
            channel: Which channel to load.

        Returns:
            images: (N, H, W) or (N, 2, H, W) float32 array.
            zbins: (N,) int32 array.
        """
        df = pd.read_csv(csv_path)
        base_dir = csv_path.parent

        # Filter by valid z-bins
        zbin_col = "z_bin" if "z_bin" in df.columns else "zbin"
        df_filtered = df[df[zbin_col].isin(self.valid_zbins)]

        images_list = []
        zbins_list = []

        for _, row in tqdm(
            df_filtered.iterrows(),
            total=len(df_filtered),
            desc=f"Loading {csv_path.stem}",
            leave=False,
        ):
            filepath = base_dir / row["filepath"]
            if not filepath.exists():
                continue

            data = np.load(filepath)

            if channel == "image":
                img = data["image"].astype(np.float32)
            elif channel == "mask":
                img = data["mask"].astype(np.float32)
            elif channel == "joint":
                image = data["image"].astype(np.float32)
                mask = data["mask"].astype(np.float32)
                img = np.stack([image, mask], axis=0)  # (2, H, W)
            else:
                raise ValueError(f"Invalid channel: {channel}")

            images_list.append(img)
            zbins_list.append(row[zbin_col])

        return (
            np.stack(images_list, axis=0),
            np.array(zbins_list, dtype=np.int32),
        )

    def iter_experiments(self) -> Iterator[tuple[str, str, float]]:
        """Iterate over experiments with parsed metadata.

        Yields:
            Tuples of (experiment_name, prediction_type, lp_norm).
        """
        for exp in self.experiments:
            pred_type, lp_norm = self.parse_experiment_name(exp)
            yield exp, pred_type, lp_norm

    def get_experiment_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all experiments.

        Returns:
            DataFrame with columns: experiment, prediction_type, lp_norm, n_replicas.
        """
        rows = []
        for exp, pred_type, lp_norm in self.iter_experiments():
            n_replicas = len(self.get_replica_paths(exp))
            rows.append({
                "experiment": exp,
                "prediction_type": pred_type,
                "lp_norm": lp_norm,
                "n_replicas": n_replicas,
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"ICIPExperimentLoader("
            f"runs_dir={self.runs_dir}, "
            f"n_experiments={len(self.experiments)})"
        )
