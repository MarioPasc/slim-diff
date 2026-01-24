"""Report generation and visualization utilities."""

from src.classification.diagnostics.reporting.compact_report import (
    generate_compact_report,
)
from src.classification.diagnostics.reporting.experiment_report import (
    generate_experiment_report,
)
from src.classification.diagnostics.reporting.paired_comparison import (
    run_paired_comparison,
)

__all__ = [
    "generate_compact_report",
    "generate_experiment_report",
    "run_paired_comparison",
]
