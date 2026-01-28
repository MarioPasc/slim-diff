"""Filesystem discovery of experiments in ablation studies.

Provides discovery mechanisms for finding experiments in hierarchical
folder structures based on the ablation space definition.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterator

from omegaconf import DictConfig

from src.shared.ablation.ablation_space import AblationSpace
from src.shared.ablation.experiment_coords import ExperimentCoordinate

logger = logging.getLogger(__name__)


class ExperimentDiscoverer:
    """Discover experiments from filesystem based on folder structure.

    Supports both hierarchical (self_cond_p_{X}/{pred}_lp_{Y}) and
    legacy flat ({pred}_lp_{Y}) folder structures.

    Attributes:
        base_dir: Base directory containing experiment folders.
        space: AblationSpace defining the parameter space.
        default_self_cond_p: Default self_cond_p for legacy paths.
    """

    # Pattern for self_cond_p folders (with or without underscore before value)
    SC_FOLDER_PATTERN = re.compile(r"^self_cond_p[_]?([\d.]+)$")

    # Pattern for experiment folders (prediction_type_lp_norm)
    EXP_FOLDER_PATTERN = re.compile(r"^(\w+)_lp_([\d.]+)$")

    def __init__(
        self,
        base_dir: Path | str,
        space: AblationSpace | None = None,
        default_self_cond_p: float = 0.5,
    ):
        """Initialize the experiment discoverer.

        Args:
            base_dir: Base directory containing experiment folders.
            space: AblationSpace defining the parameter space (uses default if None).
            default_self_cond_p: Default self_cond_p for legacy paths without it.
        """
        self.base_dir = Path(base_dir)
        self.space = space or AblationSpace.default()
        self.default_self_cond_p = default_self_cond_p

    @classmethod
    def from_config(cls, cfg: DictConfig | dict) -> ExperimentDiscoverer:
        """Create discoverer from configuration.

        Args:
            cfg: Configuration with 'data.synthetic.results_base_dir' and 'ablation'.

        Returns:
            Configured ExperimentDiscoverer instance.
        """
        if isinstance(cfg, DictConfig):
            cfg = dict(cfg)

        # Try different config structures
        base_dir = None
        if "data" in cfg:
            data_cfg = cfg["data"]
            if "synthetic" in data_cfg:
                base_dir = data_cfg["synthetic"].get("results_base_dir")
            elif "replicas_base_dir" in data_cfg:
                base_dir = data_cfg["replicas_base_dir"]
            elif "runs_dir" in data_cfg:
                base_dir = data_cfg["runs_dir"]

        if base_dir is None:
            raise ValueError("Cannot find base directory in config")

        space = None
        if "ablation" in cfg:
            space = AblationSpace.from_config(cfg)

        default_sc = cfg.get("ablation", {}).get("default_self_cond_p", 0.5)

        return cls(base_dir=base_dir, space=space, default_self_cond_p=default_sc)

    def discover_all(self) -> list[ExperimentCoordinate]:
        """Discover all valid experiments in the base directory.

        Searches for experiments in hierarchical structure first,
        then falls back to legacy flat structure.

        Returns:
            Sorted list of discovered ExperimentCoordinate instances.
        """
        discovered = set()

        # First try hierarchical structure
        discovered.update(self._discover_hierarchical())

        # Then try legacy flat structure
        discovered.update(self._discover_legacy())

        return sorted(discovered, key=lambda c: (c.self_cond_p, c.prediction_type, c.lp_norm))

    def _discover_hierarchical(self) -> set[ExperimentCoordinate]:
        """Discover experiments in hierarchical folder structure.

        Structure: base_dir/self_cond_p_{X}/{pred}_lp_{Y}/

        Returns:
            Set of discovered coordinates.
        """
        discovered = set()

        if not self.base_dir.exists():
            return discovered

        for sc_dir in self.base_dir.iterdir():
            if not sc_dir.is_dir():
                continue

            sc_match = self.SC_FOLDER_PATTERN.match(sc_dir.name)
            if not sc_match:
                continue

            self_cond_p = float(sc_match.group(1))

            for exp_dir in sc_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                exp_match = self.EXP_FOLDER_PATTERN.match(exp_dir.name)
                if not exp_match:
                    continue

                prediction_type = exp_match.group(1)
                lp_norm = float(exp_match.group(2))

                # Validate against space
                try:
                    coord = ExperimentCoordinate(
                        prediction_type=prediction_type,
                        lp_norm=lp_norm,
                        self_cond_p=self_cond_p,
                    )
                    discovered.add(coord)
                except ValueError as e:
                    logger.debug(f"Skipping invalid experiment {exp_dir}: {e}")

        return discovered

    def _discover_legacy(self) -> set[ExperimentCoordinate]:
        """Discover experiments in legacy flat folder structure.

        Structure: base_dir/{pred}_lp_{Y}/

        Returns:
            Set of discovered coordinates.
        """
        discovered = set()

        if not self.base_dir.exists():
            return discovered

        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            exp_match = self.EXP_FOLDER_PATTERN.match(exp_dir.name)
            if not exp_match:
                continue

            # Skip if this looks like a self_cond_p directory (hierarchical parent)
            if self.SC_FOLDER_PATTERN.match(exp_dir.name):
                continue

            prediction_type = exp_match.group(1)
            lp_norm = float(exp_match.group(2))

            try:
                coord = ExperimentCoordinate(
                    prediction_type=prediction_type,
                    lp_norm=lp_norm,
                    self_cond_p=self.default_self_cond_p,
                )
                discovered.add(coord)
            except ValueError as e:
                logger.debug(f"Skipping invalid experiment {exp_dir}: {e}")

        return discovered

    def discover_matching(self, **filters: Any) -> list[ExperimentCoordinate]:
        """Discover experiments matching filter criteria.

        Args:
            **filters: Keyword arguments mapping axis names to required values.

        Returns:
            Sorted list of matching coordinates.
        """
        all_coords = self.discover_all()
        return [c for c in all_coords if c.matches_filter(**filters)]

    def get_experiment_path(self, coord: ExperimentCoordinate) -> Path:
        """Get the filesystem path for an experiment coordinate.

        Returns the hierarchical path (base_dir/self_cond_p_{X}/{pred}_lp_{Y}).

        Args:
            coord: Experiment coordinate.

        Returns:
            Absolute path to the experiment folder.
        """
        return self.base_dir / coord.to_folder_path()

    def get_replicas_path(
        self, coord: ExperimentCoordinate, replicas_subdir: str = "replicas"
    ) -> Path:
        """Get the path to the replicas folder for an experiment.

        Args:
            coord: Experiment coordinate.
            replicas_subdir: Name of the replicas subdirectory.

        Returns:
            Absolute path to the replicas folder.
        """
        return self.get_experiment_path(coord) / replicas_subdir

    def has_replicas(
        self, coord: ExperimentCoordinate, replicas_subdir: str = "replicas"
    ) -> bool:
        """Check if an experiment has generated replicas.

        Args:
            coord: Experiment coordinate.
            replicas_subdir: Name of the replicas subdirectory.

        Returns:
            True if replicas directory exists and contains .npz files.
        """
        replicas_dir = self.get_replicas_path(coord, replicas_subdir)
        if not replicas_dir.exists():
            return False
        return any(replicas_dir.glob("replica_*.npz"))

    def has_patches(
        self,
        coord: ExperimentCoordinate,
        patches_base_dir: Path | str | None = None,
    ) -> bool:
        """Check if an experiment has extracted patches.

        Args:
            coord: Experiment coordinate.
            patches_base_dir: Base directory for patches. If None, uses
                base_dir/../classification/patches.

        Returns:
            True if patches directory exists with real_patches.npz.
        """
        if patches_base_dir is None:
            patches_base_dir = self.base_dir.parent / "classification" / "patches"
        else:
            patches_base_dir = Path(patches_base_dir)

        patches_dir = patches_base_dir / coord.to_display_name()
        return (patches_dir / "real_patches.npz").exists()

    def iter_experiments_with_paths(
        self, **filters: Any
    ) -> Iterator[tuple[ExperimentCoordinate, Path]]:
        """Iterate over experiments yielding coordinates and paths.

        Args:
            **filters: Optional filter criteria.

        Yields:
            Tuples of (coordinate, path) for each matching experiment.
        """
        coords = self.discover_matching(**filters) if filters else self.discover_all()
        for coord in coords:
            yield coord, self.get_experiment_path(coord)

    def get_experiment_summary(self) -> list[dict[str, Any]]:
        """Get a summary of all discovered experiments.

        Returns:
            List of dicts with experiment info including paths and status.
        """
        summary = []
        for coord in self.discover_all():
            exp_path = self.get_experiment_path(coord)
            replicas_path = self.get_replicas_path(coord)

            n_replicas = 0
            if replicas_path.exists():
                n_replicas = len(list(replicas_path.glob("replica_*.npz")))

            summary.append({
                "display_name": coord.to_display_name(),
                **coord.to_dict(),
                "path": str(exp_path),
                "exists": exp_path.exists(),
                "n_replicas": n_replicas,
                "has_replicas": n_replicas > 0,
            })

        return summary

    def __repr__(self) -> str:
        """Return string representation."""
        n_discovered = len(self.discover_all())
        return f"ExperimentDiscoverer(base_dir={self.base_dir}, discovered={n_discovered})"
