"""Tests for ExperimentDiscoverer class."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.ablation import (
    AblationSpace,
    ExperimentCoordinate,
    ExperimentDiscoverer,
)


@pytest.fixture
def temp_runs_dir(tmp_path: Path) -> Path:
    """Create a temporary runs directory with test structure."""
    runs_dir = tmp_path / "runs"

    # Create hierarchical structure
    # self_cond_p0.5/x0_lp_1.5/replicas/
    # self_cond_p0.5/epsilon_lp_2.0/replicas/
    # self_cond_p0.8/velocity_lp_1.5/replicas/

    structures = [
        ("self_cond_p0.5", "x0_lp_1.5"),
        ("self_cond_p0.5", "epsilon_lp_2.0"),
        ("self_cond_p0.8", "velocity_lp_1.5"),
    ]

    for sc_folder, exp_folder in structures:
        exp_path = runs_dir / sc_folder / exp_folder / "replicas"
        exp_path.mkdir(parents=True, exist_ok=True)
        # Create dummy replica files
        (exp_path / "replica_000.npz").touch()
        (exp_path / "replica_001.npz").touch()

    return runs_dir


@pytest.fixture
def temp_legacy_runs_dir(tmp_path: Path) -> Path:
    """Create a temporary runs directory with legacy flat structure."""
    runs_dir = tmp_path / "legacy_runs"

    # Create flat structure (no self_cond_p prefix)
    # x0_lp_1.5/replicas/
    # epsilon_lp_2.0/replicas/

    structures = ["x0_lp_1.5", "epsilon_lp_2.0", "velocity_lp_2.5"]

    for exp_folder in structures:
        exp_path = runs_dir / exp_folder / "replicas"
        exp_path.mkdir(parents=True, exist_ok=True)
        (exp_path / "replica_000.npz").touch()

    return runs_dir


class TestExperimentDiscovererCreation:
    """Test ExperimentDiscoverer creation."""

    def test_create_discoverer(self, temp_runs_dir: Path):
        """Test creating a discoverer."""
        space = AblationSpace.default()
        discoverer = ExperimentDiscoverer(
            base_dir=temp_runs_dir,
            space=space,
        )
        assert discoverer.base_dir == temp_runs_dir

    def test_create_with_default_space(self, temp_runs_dir: Path):
        """Test creating discoverer with default space."""
        discoverer = ExperimentDiscoverer(base_dir=temp_runs_dir)
        assert discoverer.space is not None


class TestExperimentDiscovererDiscoverAll:
    """Test discovering all experiments."""

    def test_discover_hierarchical(self, temp_runs_dir: Path):
        """Test discovering experiments in hierarchical structure."""
        discoverer = ExperimentDiscoverer(base_dir=temp_runs_dir)
        experiments = discoverer.discover_all()

        assert len(experiments) == 3

        # Check all are ExperimentCoordinate
        for exp in experiments:
            assert isinstance(exp, ExperimentCoordinate)

        # Check specific experiments exist
        names = {exp.to_display_name() for exp in experiments}
        assert "sc_0.5__x0_lp_1.5" in names
        assert "sc_0.5__epsilon_lp_2.0" in names
        assert "sc_0.8__velocity_lp_1.5" in names

    def test_discover_legacy(self, temp_legacy_runs_dir: Path):
        """Test discovering experiments in legacy flat structure."""
        discoverer = ExperimentDiscoverer(
            base_dir=temp_legacy_runs_dir,
            default_self_cond_p=0.5,
        )
        experiments = discoverer.discover_all()

        assert len(experiments) == 3

        # All should have default self_cond_p
        for exp in experiments:
            assert exp.self_cond_p == 0.5


class TestExperimentDiscovererFiltering:
    """Test discovering with filters."""

    def test_discover_matching_prediction_type(self, temp_runs_dir: Path):
        """Test filtering by prediction type."""
        discoverer = ExperimentDiscoverer(base_dir=temp_runs_dir)
        experiments = discoverer.discover_matching(prediction_type="x0")

        assert len(experiments) == 1
        assert experiments[0].prediction_type == "x0"

    def test_discover_matching_self_cond_p(self, temp_runs_dir: Path):
        """Test filtering by self_cond_p."""
        discoverer = ExperimentDiscoverer(base_dir=temp_runs_dir)
        experiments = discoverer.discover_matching(self_cond_p=0.5)

        assert len(experiments) == 2
        for exp in experiments:
            assert exp.self_cond_p == 0.5

    def test_discover_matching_multiple_filters(self, temp_runs_dir: Path):
        """Test filtering by multiple parameters."""
        discoverer = ExperimentDiscoverer(base_dir=temp_runs_dir)
        experiments = discoverer.discover_matching(
            prediction_type="x0",
            self_cond_p=0.5,
        )

        assert len(experiments) == 1
        exp = experiments[0]
        assert exp.prediction_type == "x0"
        assert exp.self_cond_p == 0.5


class TestExperimentDiscovererPaths:
    """Test path resolution."""

    def test_get_experiment_path(self, temp_runs_dir: Path):
        """Test getting experiment path from coordinate."""
        discoverer = ExperimentDiscoverer(base_dir=temp_runs_dir)
        coord = ExperimentCoordinate("x0", 1.5, 0.5)

        path = discoverer.get_experiment_path(coord)
        expected = temp_runs_dir / "self_cond_p0.5" / "x0_lp_1.5"
        assert path == expected

    def test_get_replicas_path(self, temp_runs_dir: Path):
        """Test getting replicas path from coordinate."""
        discoverer = ExperimentDiscoverer(base_dir=temp_runs_dir)
        coord = ExperimentCoordinate("x0", 1.5, 0.5)

        path = discoverer.get_replicas_path(coord)
        expected = temp_runs_dir / "self_cond_p0.5" / "x0_lp_1.5" / "replicas"
        assert path == expected

    def test_get_replicas_path_custom_subdir(self, temp_runs_dir: Path):
        """Test getting replicas path with custom subdirectory."""
        discoverer = ExperimentDiscoverer(base_dir=temp_runs_dir)
        coord = ExperimentCoordinate("x0", 1.5, 0.5)

        path = discoverer.get_replicas_path(coord, replicas_subdir="outputs")
        expected = temp_runs_dir / "self_cond_p0.5" / "x0_lp_1.5" / "outputs"
        assert path == expected


class TestExperimentDiscovererEdgeCases:
    """Test edge cases."""

    def test_empty_directory(self, tmp_path: Path):
        """Test discovering in empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        discoverer = ExperimentDiscoverer(base_dir=empty_dir)
        experiments = discoverer.discover_all()

        assert len(experiments) == 0

    def test_nonexistent_directory(self, tmp_path: Path):
        """Test with nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"

        discoverer = ExperimentDiscoverer(base_dir=nonexistent)
        experiments = discoverer.discover_all()

        assert len(experiments) == 0

    def test_mixed_structure(self, tmp_path: Path):
        """Test with mixed hierarchical and legacy structure."""
        runs_dir = tmp_path / "mixed"

        # Create hierarchical
        (runs_dir / "self_cond_p0.8" / "x0_lp_1.5" / "replicas").mkdir(parents=True)
        (runs_dir / "self_cond_p0.8" / "x0_lp_1.5" / "replicas" / "replica_000.npz").touch()

        # Create legacy (flat)
        (runs_dir / "epsilon_lp_2.0" / "replicas").mkdir(parents=True)
        (runs_dir / "epsilon_lp_2.0" / "replicas" / "replica_000.npz").touch()

        discoverer = ExperimentDiscoverer(base_dir=runs_dir, default_self_cond_p=0.5)
        experiments = discoverer.discover_all()

        assert len(experiments) == 2

        # Check we found both
        names = {exp.to_display_name() for exp in experiments}
        assert "sc_0.8__x0_lp_1.5" in names
        assert "sc_0.5__epsilon_lp_2.0" in names
