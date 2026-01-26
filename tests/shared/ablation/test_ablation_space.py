"""Tests for AblationSpace and AblationAxis classes."""

from __future__ import annotations

import pytest

from src.shared.ablation import AblationSpace, AblationAxis, ExperimentCoordinate


class TestAblationAxis:
    """Test AblationAxis dataclass."""

    def test_create_axis(self):
        """Test basic axis creation."""
        axis = AblationAxis(
            name="prediction_type",
            values=["epsilon", "velocity", "x0"],
        )
        assert axis.name == "prediction_type"
        assert axis.values == ["epsilon", "velocity", "x0"]

    def test_axis_with_folder_format(self):
        """Test axis with custom folder format."""
        axis = AblationAxis(
            name="lp_norm",
            values=[1.5, 2.0, 2.5],
            folder_format="lp_{value}",
        )
        assert axis.folder_format == "lp_{value}"


class TestAblationSpaceCreation:
    """Test AblationSpace creation."""

    def test_default_space(self):
        """Test creating default ablation space."""
        space = AblationSpace.default()

        assert "prediction_type" in space.axes
        assert "lp_norm" in space.axes
        assert "self_cond_p" in space.axes

        assert space.axes["prediction_type"].values == ["epsilon", "velocity", "x0"]
        assert space.axes["lp_norm"].values == [1.5, 2.0, 2.5]
        assert space.axes["self_cond_p"].values == [0.0, 0.5, 0.8]

    def test_create_custom_space(self):
        """Test creating custom ablation space."""
        axes = {
            "prediction_type": AblationAxis("prediction_type", ["x0", "epsilon"]),
            "lp_norm": AblationAxis("lp_norm", [1.5, 2.0]),
            "self_cond_p": AblationAxis("self_cond_p", [0.5]),
        }
        space = AblationSpace(
            axes=axes,
            hierarchy=["self_cond_p", "prediction_type_lp_norm"],
        )

        assert len(space.axes["prediction_type"].values) == 2
        assert len(space.axes["lp_norm"].values) == 2

    def test_from_config_dict(self):
        """Test creating space from config dictionary."""
        config = {
            "ablation": {
                "axes": {
                    "self_cond_p": {"values": [0.0, 0.5]},
                    "prediction_type": {"values": ["x0", "velocity"]},
                    "lp_norm": {"values": [1.5, 2.0]},
                },
                "hierarchy": ["self_cond_p", "prediction_type_lp_norm"],
            }
        }

        space = AblationSpace.from_config(config)
        assert space.axes["self_cond_p"].values == [0.0, 0.5]
        assert space.axes["prediction_type"].values == ["x0", "velocity"]
        assert space.axes["lp_norm"].values == [1.5, 2.0]


class TestAblationSpaceEnumeration:
    """Test AblationSpace coordinate enumeration."""

    def test_enumerate_coordinates(self):
        """Test enumerating all coordinates."""
        space = AblationSpace.default()
        coords = list(space.enumerate_coordinates())

        # 3 prediction_types x 3 lp_norms x 3 self_cond_p = 27 total
        assert len(coords) == 27

        # Check all are ExperimentCoordinate instances
        for coord in coords:
            assert isinstance(coord, ExperimentCoordinate)

    def test_enumerate_coordinates_subset(self):
        """Test enumeration with subset of values."""
        axes = {
            "prediction_type": AblationAxis("prediction_type", ["x0"]),
            "lp_norm": AblationAxis("lp_norm", [1.5, 2.0]),
            "self_cond_p": AblationAxis("self_cond_p", [0.5]),
        }
        space = AblationSpace(axes=axes, hierarchy=[])

        coords = list(space.enumerate_coordinates())
        assert len(coords) == 2  # 1 x 2 x 1 = 2

        # Both should have x0 and self_cond_p=0.5
        for coord in coords:
            assert coord.prediction_type == "x0"
            assert coord.self_cond_p == 0.5

        # Different lp_norms
        lp_norms = {coord.lp_norm for coord in coords}
        assert lp_norms == {1.5, 2.0}


class TestAblationSpaceFiltering:
    """Test AblationSpace filtering."""

    def test_filter_by_prediction_type(self):
        """Test filtering by single prediction type."""
        space = AblationSpace.default()
        coords = list(space.filter_coordinates(prediction_type="x0"))

        # 3 lp_norms x 3 self_cond_p = 9
        assert len(coords) == 9
        for coord in coords:
            assert coord.prediction_type == "x0"

    def test_filter_by_lp_norm(self):
        """Test filtering by lp_norm."""
        space = AblationSpace.default()
        coords = list(space.filter_coordinates(lp_norm=1.5))

        # 3 prediction_types x 3 self_cond_p = 9
        assert len(coords) == 9
        for coord in coords:
            assert coord.lp_norm == 1.5

    def test_filter_by_self_cond_p(self):
        """Test filtering by self_cond_p."""
        space = AblationSpace.default()
        coords = list(space.filter_coordinates(self_cond_p=0.5))

        # 3 prediction_types x 3 lp_norms = 9
        assert len(coords) == 9
        for coord in coords:
            assert coord.self_cond_p == 0.5

    def test_filter_by_multiple_axes(self):
        """Test filtering by multiple axes."""
        space = AblationSpace.default()
        coords = list(space.filter_coordinates(prediction_type="x0", self_cond_p=0.5))

        # 3 lp_norms
        assert len(coords) == 3
        for coord in coords:
            assert coord.prediction_type == "x0"
            assert coord.self_cond_p == 0.5

    def test_filter_with_list(self):
        """Test filtering with list of values."""
        space = AblationSpace.default()
        coords = list(space.filter_coordinates(prediction_type=["x0", "velocity"]))

        # 2 prediction_types x 3 lp_norms x 3 self_cond_p = 18
        assert len(coords) == 18
        pred_types = {coord.prediction_type for coord in coords}
        assert pred_types == {"x0", "velocity"}

    def test_filter_no_matches(self):
        """Test filtering with no matches returns empty."""
        space = AblationSpace.default()
        coords = list(space.filter_coordinates(prediction_type="nonexistent"))
        assert len(coords) == 0
