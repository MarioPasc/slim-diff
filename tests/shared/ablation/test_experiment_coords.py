"""Tests for ExperimentCoordinate dataclass."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.ablation import ExperimentCoordinate


class TestExperimentCoordinateCreation:
    """Test ExperimentCoordinate creation and basic attributes."""

    def test_create_coordinate(self):
        """Test basic coordinate creation."""
        coord = ExperimentCoordinate(
            prediction_type="x0",
            lp_norm=1.5,
            self_cond_p=0.5,
        )
        assert coord.prediction_type == "x0"
        assert coord.lp_norm == 1.5
        assert coord.self_cond_p == 0.5

    def test_coordinate_is_frozen(self):
        """Test that coordinate is immutable."""
        coord = ExperimentCoordinate(
            prediction_type="x0",
            lp_norm=1.5,
            self_cond_p=0.5,
        )
        with pytest.raises(AttributeError):
            coord.prediction_type = "epsilon"

    def test_coordinate_equality(self):
        """Test coordinate equality comparison."""
        coord1 = ExperimentCoordinate("x0", 1.5, 0.5)
        coord2 = ExperimentCoordinate("x0", 1.5, 0.5)
        coord3 = ExperimentCoordinate("x0", 2.0, 0.5)

        assert coord1 == coord2
        assert coord1 != coord3

    def test_coordinate_hash(self):
        """Test coordinate can be used in sets/dicts."""
        coord1 = ExperimentCoordinate("x0", 1.5, 0.5)
        coord2 = ExperimentCoordinate("x0", 1.5, 0.5)
        coord3 = ExperimentCoordinate("epsilon", 1.5, 0.5)

        coord_set = {coord1, coord2, coord3}
        assert len(coord_set) == 2  # coord1 and coord2 are equal


class TestExperimentCoordinateDisplayName:
    """Test display name generation and parsing."""

    def test_to_display_name(self):
        """Test display name generation."""
        coord = ExperimentCoordinate("x0", 1.5, 0.5)
        assert coord.to_display_name() == "sc_0.5__x0_lp_1.5"

    def test_to_display_name_different_values(self):
        """Test display name with different values."""
        coord = ExperimentCoordinate("epsilon", 2.0, 0.8)
        assert coord.to_display_name() == "sc_0.8__epsilon_lp_2.0"

    def test_to_display_name_zero_self_cond(self):
        """Test display name with zero self_cond_p."""
        coord = ExperimentCoordinate("velocity", 2.5, 0.0)
        assert coord.to_display_name() == "sc_0.0__velocity_lp_2.5"

    def test_from_display_name(self):
        """Test parsing display name."""
        coord = ExperimentCoordinate.from_display_name("sc_0.5__x0_lp_1.5")
        assert coord.prediction_type == "x0"
        assert coord.lp_norm == 1.5
        assert coord.self_cond_p == 0.5

    def test_from_display_name_roundtrip(self):
        """Test display name roundtrip."""
        original = ExperimentCoordinate("epsilon", 2.0, 0.8)
        name = original.to_display_name()
        parsed = ExperimentCoordinate.from_display_name(name)
        assert parsed == original

    def test_from_display_name_invalid_format(self):
        """Test parsing invalid display name raises error."""
        with pytest.raises(ValueError):
            ExperimentCoordinate.from_display_name("invalid_format")

    def test_from_display_name_missing_parts(self):
        """Test parsing name with missing parts raises error."""
        with pytest.raises(ValueError):
            ExperimentCoordinate.from_display_name("x0_lp_1.5")  # Missing sc prefix


class TestExperimentCoordinateLegacyName:
    """Test legacy name parsing."""

    def test_from_legacy_name(self):
        """Test parsing legacy name (without self_cond_p prefix)."""
        coord = ExperimentCoordinate.from_legacy_name("x0_lp_1.5", default_self_cond_p=0.5)
        assert coord.prediction_type == "x0"
        assert coord.lp_norm == 1.5
        assert coord.self_cond_p == 0.5

    def test_from_legacy_name_different_default(self):
        """Test legacy name with different default self_cond_p."""
        coord = ExperimentCoordinate.from_legacy_name("epsilon_lp_2.0", default_self_cond_p=0.8)
        assert coord.prediction_type == "epsilon"
        assert coord.lp_norm == 2.0
        assert coord.self_cond_p == 0.8

    def test_from_legacy_name_invalid_format(self):
        """Test legacy name with invalid format raises error."""
        with pytest.raises(ValueError):
            ExperimentCoordinate.from_legacy_name("invalid")

    def test_from_legacy_name_velocity(self):
        """Test parsing legacy name with velocity prediction type."""
        coord = ExperimentCoordinate.from_legacy_name("velocity_lp_2.5")
        assert coord.prediction_type == "velocity"
        assert coord.lp_norm == 2.5
        assert coord.self_cond_p == 0.5  # Default


class TestExperimentCoordinateFolderPath:
    """Test folder path generation and parsing."""

    def test_to_folder_path(self):
        """Test folder path generation."""
        coord = ExperimentCoordinate("x0", 1.5, 0.5)
        path = coord.to_folder_path()
        assert path == Path("self_cond_p0.5") / "x0_lp_1.5"

    def test_to_folder_path_different_values(self):
        """Test folder path with different values."""
        coord = ExperimentCoordinate("epsilon", 2.0, 0.8)
        path = coord.to_folder_path()
        assert path == Path("self_cond_p0.8") / "epsilon_lp_2.0"

    def test_from_folder_path_hierarchical(self):
        """Test parsing hierarchical folder path."""
        path = Path("self_cond_p0.5/x0_lp_1.5")
        coord = ExperimentCoordinate.from_folder_path(path)
        assert coord.prediction_type == "x0"
        assert coord.lp_norm == 1.5
        assert coord.self_cond_p == 0.5

    def test_from_folder_path_legacy(self):
        """Test parsing legacy (flat) folder path."""
        path = Path("x0_lp_1.5")
        coord = ExperimentCoordinate.from_folder_path(path, default_self_cond_p=0.5)
        assert coord.prediction_type == "x0"
        assert coord.lp_norm == 1.5
        assert coord.self_cond_p == 0.5

    def test_from_folder_path_roundtrip(self):
        """Test folder path roundtrip."""
        original = ExperimentCoordinate("velocity", 2.5, 0.8)
        path = original.to_folder_path()
        parsed = ExperimentCoordinate.from_folder_path(path)
        assert parsed == original


class TestExperimentCoordinateDict:
    """Test dictionary conversion."""

    def test_to_dict(self):
        """Test converting coordinate to dictionary."""
        coord = ExperimentCoordinate("x0", 1.5, 0.5)
        d = coord.to_dict()
        assert d == {
            "prediction_type": "x0",
            "lp_norm": 1.5,
            "self_cond_p": 0.5,
        }

    def test_matches(self):
        """Test filter matching."""
        coord = ExperimentCoordinate("x0", 1.5, 0.5)

        assert coord.matches(prediction_type="x0")
        assert coord.matches(lp_norm=1.5)
        assert coord.matches(self_cond_p=0.5)
        assert coord.matches(prediction_type="x0", lp_norm=1.5)
        assert coord.matches()  # No filters = always matches

        assert not coord.matches(prediction_type="epsilon")
        assert not coord.matches(lp_norm=2.0)
        assert not coord.matches(prediction_type="x0", lp_norm=2.0)

    def test_matches_with_lists(self):
        """Test filter matching with lists of values."""
        coord = ExperimentCoordinate("x0", 1.5, 0.5)

        assert coord.matches(prediction_type=["x0", "epsilon"])
        assert coord.matches(lp_norm=[1.5, 2.0])
        assert not coord.matches(prediction_type=["epsilon", "velocity"])
