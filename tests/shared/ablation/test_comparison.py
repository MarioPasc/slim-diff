"""Tests for ComparisonSpec and related functions."""

from __future__ import annotations

import pytest

from src.shared.ablation import (
    AblationSpace,
    ExperimentCoordinate,
    ComparisonSpec,
    differ_by_one_axis,
)


class TestDifferByOneAxis:
    """Test differ_by_one_axis function."""

    def test_differ_by_prediction_type(self):
        """Test detection when only prediction_type differs."""
        coord_a = ExperimentCoordinate("x0", 1.5, 0.5)
        coord_b = ExperimentCoordinate("epsilon", 1.5, 0.5)

        result = differ_by_one_axis(coord_a, coord_b)
        assert result == "prediction_type"

    def test_differ_by_lp_norm(self):
        """Test detection when only lp_norm differs."""
        coord_a = ExperimentCoordinate("x0", 1.5, 0.5)
        coord_b = ExperimentCoordinate("x0", 2.0, 0.5)

        result = differ_by_one_axis(coord_a, coord_b)
        assert result == "lp_norm"

    def test_differ_by_self_cond_p(self):
        """Test detection when only self_cond_p differs."""
        coord_a = ExperimentCoordinate("x0", 1.5, 0.5)
        coord_b = ExperimentCoordinate("x0", 1.5, 0.8)

        result = differ_by_one_axis(coord_a, coord_b)
        assert result == "self_cond_p"

    def test_same_coordinates(self):
        """Test with identical coordinates."""
        coord_a = ExperimentCoordinate("x0", 1.5, 0.5)
        coord_b = ExperimentCoordinate("x0", 1.5, 0.5)

        result = differ_by_one_axis(coord_a, coord_b)
        assert result is None

    def test_differ_by_two_axes(self):
        """Test when two axes differ."""
        coord_a = ExperimentCoordinate("x0", 1.5, 0.5)
        coord_b = ExperimentCoordinate("epsilon", 2.0, 0.5)

        result = differ_by_one_axis(coord_a, coord_b)
        assert result is None

    def test_differ_by_all_axes(self):
        """Test when all axes differ."""
        coord_a = ExperimentCoordinate("x0", 1.5, 0.5)
        coord_b = ExperimentCoordinate("epsilon", 2.0, 0.8)

        result = differ_by_one_axis(coord_a, coord_b)
        assert result is None


class TestComparisonSpecCreation:
    """Test ComparisonSpec creation."""

    def test_create_comparison(self):
        """Test basic comparison spec creation."""
        spec = ComparisonSpec(
            name="by_prediction_type",
            varying_axis="prediction_type",
            fixed_axes={"self_cond_p": 0.5},
        )
        assert spec.name == "by_prediction_type"
        assert spec.varying_axis == "prediction_type"
        assert spec.fixed_axes == {"self_cond_p": 0.5}

    def test_create_comparison_empty_fixed(self):
        """Test comparison with no fixed axes."""
        spec = ComparisonSpec(
            name="by_lp_norm",
            varying_axis="lp_norm",
            fixed_axes={},
        )
        assert spec.fixed_axes == {}


class TestComparisonSpecCoordinates:
    """Test ComparisonSpec coordinate retrieval."""

    def test_get_coordinates_varying_prediction_type(self):
        """Test getting coordinates when varying prediction_type."""
        space = AblationSpace.default()
        spec = ComparisonSpec(
            name="pred_type_comparison",
            varying_axis="prediction_type",
            fixed_axes={"self_cond_p": 0.5, "lp_norm": 1.5},
        )

        coords = spec.get_coordinates(space)
        assert len(coords) == 3  # epsilon, velocity, x0

        # All should have fixed axes
        for coord in coords:
            assert coord.self_cond_p == 0.5
            assert coord.lp_norm == 1.5

        # Should have all prediction types
        pred_types = {coord.prediction_type for coord in coords}
        assert pred_types == {"epsilon", "velocity", "x0"}

    def test_get_coordinates_varying_lp_norm(self):
        """Test getting coordinates when varying lp_norm."""
        space = AblationSpace.default()
        spec = ComparisonSpec(
            name="lp_norm_comparison",
            varying_axis="lp_norm",
            fixed_axes={"prediction_type": "x0", "self_cond_p": 0.5},
        )

        coords = spec.get_coordinates(space)
        assert len(coords) == 3  # 1.5, 2.0, 2.5

        lp_norms = {coord.lp_norm for coord in coords}
        assert lp_norms == {1.5, 2.0, 2.5}

    def test_get_coordinates_no_fixed_axes(self):
        """Test getting coordinates with no fixed axes."""
        space = AblationSpace.default()
        spec = ComparisonSpec(
            name="all_self_cond",
            varying_axis="self_cond_p",
            fixed_axes={},
        )

        coords = spec.get_coordinates(space)
        # All 27 coordinates should be returned (3^3)
        assert len(coords) == 27


class TestComparisonSpecPairs:
    """Test ComparisonSpec pair generation."""

    def test_get_pairs(self):
        """Test getting comparison pairs."""
        space = AblationSpace.default()
        spec = ComparisonSpec(
            name="pred_type_pairs",
            varying_axis="prediction_type",
            fixed_axes={"self_cond_p": 0.5, "lp_norm": 1.5},
        )

        pairs = spec.get_pairs(space)
        # 3 prediction types -> 3 pairs: (eps, vel), (eps, x0), (vel, x0)
        assert len(pairs) == 3

        for coord_a, coord_b in pairs:
            # Fixed axes should be same
            assert coord_a.self_cond_p == coord_b.self_cond_p == 0.5
            assert coord_a.lp_norm == coord_b.lp_norm == 1.5
            # Varying axis should differ
            assert coord_a.prediction_type != coord_b.prediction_type

    def test_get_pairs_lp_norm(self):
        """Test getting pairs when varying lp_norm."""
        space = AblationSpace.default()
        spec = ComparisonSpec(
            name="lp_norm_pairs",
            varying_axis="lp_norm",
            fixed_axes={"prediction_type": "x0", "self_cond_p": 0.5},
        )

        pairs = spec.get_pairs(space)
        # 3 lp_norms -> 3 pairs
        assert len(pairs) == 3

        for coord_a, coord_b in pairs:
            assert coord_a.lp_norm != coord_b.lp_norm


class TestComparisonSpecFromConfig:
    """Test creating ComparisonSpec from config."""

    def test_from_config_dict(self):
        """Test creating from config dictionary."""
        config = {
            "name": "by_prediction_type",
            "varying_axis": "prediction_type",
            "fixed_axes": {"self_cond_p": 0.5},
        }

        spec = ComparisonSpec.from_config(config)
        assert spec.name == "by_prediction_type"
        assert spec.varying_axis == "prediction_type"
        assert spec.fixed_axes == {"self_cond_p": 0.5}

    def test_from_config_list(self):
        """Test creating multiple specs from config list."""
        configs = [
            {
                "name": "by_pred",
                "varying_axis": "prediction_type",
                "fixed_axes": {},
            },
            {
                "name": "by_lp",
                "varying_axis": "lp_norm",
                "fixed_axes": {},
            },
        ]

        specs = [ComparisonSpec.from_config(c) for c in configs]
        assert len(specs) == 2
        assert specs[0].name == "by_pred"
        assert specs[1].name == "by_lp"
