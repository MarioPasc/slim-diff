"""Experiment coordinate system for multi-axis ablation studies.

Provides an immutable coordinate type representing a point in the
experiment parameter space. Supports conversion to/from display names
and folder paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentCoordinate:
    """Immutable coordinate in the experiment parameter space.

    Represents a specific experiment configuration defined by its position
    along each ablation axis.

    Attributes:
        prediction_type: Diffusion prediction type ("epsilon", "velocity", "x0").
        lp_norm: Lp norm exponent (1.5, 2.0, 2.5).
        self_cond_p: Self-conditioning probability (0.0, 0.5, 0.8).

    Examples:
        >>> coord = ExperimentCoordinate("x0", 1.5, 0.5)
        >>> coord.to_display_name()
        'sc_0.5__x0_lp_1.5'
        >>> coord.to_folder_path()
        PosixPath('self_cond_p_0.5/x0_lp_1.5')
    """

    prediction_type: str
    lp_norm: float
    self_cond_p: float

    # Valid values for each axis
    VALID_PREDICTION_TYPES = frozenset({"epsilon", "velocity", "x0"})

    def __post_init__(self) -> None:
        """Validate coordinate values."""
        if self.prediction_type not in self.VALID_PREDICTION_TYPES:
            raise ValueError(
                f"Invalid prediction_type: {self.prediction_type}. "
                f"Must be one of {self.VALID_PREDICTION_TYPES}"
            )
        if not isinstance(self.lp_norm, (int, float)) or self.lp_norm <= 0:
            raise ValueError(f"Invalid lp_norm: {self.lp_norm}. Must be positive number.")
        if not isinstance(self.self_cond_p, (int, float)) or not (0.0 <= self.self_cond_p <= 1.0):
            raise ValueError(
                f"Invalid self_cond_p: {self.self_cond_p}. Must be in [0.0, 1.0]."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert coordinate to dictionary.

        Returns:
            Dictionary with axis names as keys.
        """
        return {
            "prediction_type": self.prediction_type,
            "lp_norm": self.lp_norm,
            "self_cond_p": self.self_cond_p,
        }

    def to_display_name(self) -> str:
        """Convert to human-readable display name.

        Format: sc_{self_cond_p}__{prediction_type}_lp_{lp_norm}

        Returns:
            Display name string (e.g., "sc_0.5__x0_lp_1.5").
        """
        return f"sc_{self.self_cond_p:.1f}__{self.prediction_type}_lp_{self.lp_norm:.1f}"

    def to_folder_path(self) -> Path:
        """Convert to hierarchical folder path.

        Format: self_cond_p_{self_cond_p}/{prediction_type}_lp_{lp_norm}

        Returns:
            Relative Path object for the experiment folder.
        """
        sc_folder = f"self_cond_p_{self.self_cond_p:.1f}"
        exp_folder = f"{self.prediction_type}_lp_{self.lp_norm:.1f}"
        return Path(sc_folder) / exp_folder

    def to_legacy_name(self) -> str:
        """Convert to legacy 2-axis experiment name.

        Format: {prediction_type}_lp_{lp_norm}

        Returns:
            Legacy name string (e.g., "x0_lp_1.5").
        """
        return f"{self.prediction_type}_lp_{self.lp_norm:.1f}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentCoordinate:
        """Create coordinate from dictionary.

        Args:
            d: Dictionary with keys 'prediction_type', 'lp_norm', 'self_cond_p'.

        Returns:
            ExperimentCoordinate instance.
        """
        return cls(
            prediction_type=str(d["prediction_type"]),
            lp_norm=float(d["lp_norm"]),
            self_cond_p=float(d["self_cond_p"]),
        )

    @classmethod
    def from_display_name(cls, name: str) -> ExperimentCoordinate:
        """Parse coordinate from display name.

        Args:
            name: Display name (e.g., "sc_0.5__x0_lp_1.5").

        Returns:
            ExperimentCoordinate instance.

        Raises:
            ValueError: If name doesn't match expected format.
        """
        # Pattern: sc_{self_cond_p}__{prediction_type}_lp_{lp_norm}
        pattern = r"^sc_([\d.]+)__(\w+)_lp_([\d.]+)$"
        match = re.match(pattern, name)
        if not match:
            raise ValueError(
                f"Invalid display name format: {name}. "
                f"Expected format: sc_{{self_cond_p}}__{{prediction_type}}_lp_{{lp_norm}}"
            )
        return cls(
            prediction_type=match.group(2),
            lp_norm=float(match.group(3)),
            self_cond_p=float(match.group(1)),
        )

    @classmethod
    def from_folder_path(cls, path: Path | str, default_self_cond_p: float = 0.5) -> ExperimentCoordinate:
        """Parse coordinate from folder path.

        Supports both hierarchical and flat (legacy) folder structures.

        Hierarchical format: self_cond_p{X}/{prediction_type}_lp_{Y}
        Legacy flat format: {prediction_type}_lp_{Y}

        Args:
            path: Folder path (relative or absolute).
            default_self_cond_p: Default self_cond_p for legacy paths.

        Returns:
            ExperimentCoordinate instance.

        Raises:
            ValueError: If path doesn't match expected format.
        """
        path = Path(path)
        parts = path.parts

        # Try hierarchical format: self_cond_p{X}/{prediction_type}_lp_{Y}
        if len(parts) >= 2:
            sc_pattern = r"^self_cond_p[_]?([\d.]+)$"
            exp_pattern = r"^(\w+)_lp_([\d.]+)$"

            sc_match = re.match(sc_pattern, parts[-2])
            exp_match = re.match(exp_pattern, parts[-1])

            if sc_match and exp_match:
                return cls(
                    prediction_type=exp_match.group(1),
                    lp_norm=float(exp_match.group(2)),
                    self_cond_p=float(sc_match.group(1)),
                )

        # Try legacy flat format: {prediction_type}_lp_{Y}
        exp_pattern = r"^(\w+)_lp_([\d.]+)$"
        last_part = parts[-1] if parts else str(path)
        exp_match = re.match(exp_pattern, last_part)

        if exp_match:
            return cls(
                prediction_type=exp_match.group(1),
                lp_norm=float(exp_match.group(2)),
                self_cond_p=default_self_cond_p,
            )

        raise ValueError(
            f"Cannot parse folder path: {path}. "
            f"Expected hierarchical: self_cond_p_{{X}}/{{pred}}_lp_{{Y}} "
            f"or legacy: {{pred}}_lp_{{Y}}"
        )

    @classmethod
    def from_legacy_name(cls, name: str, default_self_cond_p: float = 0.5) -> ExperimentCoordinate:
        """Parse coordinate from legacy 2-axis experiment name.

        Args:
            name: Legacy name (e.g., "x0_lp_1.5").
            default_self_cond_p: Default self_cond_p value.

        Returns:
            ExperimentCoordinate instance.

        Raises:
            ValueError: If name doesn't match expected format.
        """
        pattern = r"^(\w+)_lp_([\d.]+)$"
        match = re.match(pattern, name)
        if not match:
            raise ValueError(
                f"Invalid legacy name format: {name}. "
                f"Expected format: {{prediction_type}}_lp_{{lp_norm}}"
            )
        return cls(
            prediction_type=match.group(1),
            lp_norm=float(match.group(2)),
            self_cond_p=default_self_cond_p,
        )

    def matches_filter(self, **filters: Any) -> bool:
        """Check if coordinate matches filter criteria.

        Args:
            **filters: Keyword arguments mapping axis names to required values.
                Values can be single values or lists of allowed values.

        Returns:
            True if coordinate matches all filter criteria.

        Examples:
            >>> coord = ExperimentCoordinate("x0", 1.5, 0.5)
            >>> coord.matches_filter(prediction_type="x0")
            True
            >>> coord.matches_filter(lp_norm=[1.5, 2.0])
            True
            >>> coord.matches_filter(self_cond_p=0.0)
            False
        """
        coord_dict = self.to_dict()
        for key, value in filters.items():
            if key not in coord_dict:
                raise ValueError(f"Unknown filter key: {key}")
            coord_value = coord_dict[key]
            if isinstance(value, (list, tuple, set, frozenset)):
                if coord_value not in value:
                    return False
            elif coord_value != value:
                return False
        return True

    def __str__(self) -> str:
        """Return display name as string representation."""
        return self.to_display_name()

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"ExperimentCoordinate("
            f"prediction_type={self.prediction_type!r}, "
            f"lp_norm={self.lp_norm}, "
            f"self_cond_p={self.self_cond_p})"
        )
