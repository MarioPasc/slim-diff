"""Callbacks for logging segmentation metrics."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class CSVLoggingCallback(Callback):
    """Log metrics to CSV per fold."""

    def __init__(self, cfg: DictConfig, fold_idx: int):
        """Initialize callback.

        Args:
            cfg: Configuration
            fold_idx: Current fold index
        """
        super().__init__()
        self.cfg = cfg
        self.fold_idx = fold_idx

        # Create CSV path
        csv_dir = Path(cfg.experiment.output_dir) / "csv_logs"
        csv_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = csv_dir / f"fold_{fold_idx}_metrics.csv"

        self._header_written = False
        self._all_metric_names = set()

        logger.info(f"CSV logging to: {self.csv_path}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Write metrics to CSV at epoch end.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        # Skip sanity check validation
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        global_step = trainer.global_step

        # Collect metrics
        metrics = trainer.callback_metrics
        metrics_dict = {
            "epoch": epoch,
            "step": global_step,
            "fold": self.fold_idx,
        }

        for key, value in metrics.items():
            if hasattr(value, "item"):
                metrics_dict[key] = value.item()
            else:
                metrics_dict[key] = value

        # Update columns
        self._all_metric_names.update(metrics_dict.keys())

        # Write row
        self._write_row(metrics_dict)

    def _write_row(self, row_data: dict):
        """Write row to CSV.

        Args:
            row_data: Dictionary of metric values
        """
        columns = ["epoch", "step", "fold"] + sorted(
            k
            for k in self._all_metric_names
            if k not in ["epoch", "step", "fold"]
        )

        file_exists = self.csv_path.exists()

        # Check if columns have changed (new metrics appeared)
        current_columns = None
        if file_exists:
            try:
                with open(self.csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    current_columns = reader.fieldnames
            except Exception as e:
                logger.warning(f"Could not read CSV header: {e}")
                current_columns = None

        # Need to rewrite if columns changed or file doesn't exist
        need_rewrite = (not file_exists) or (current_columns is not None and set(current_columns) != set(columns))

        if need_rewrite and file_exists:
            # Read existing data
            existing_rows = []
            try:
                with open(self.csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)
            except Exception as e:
                logger.warning(f"Could not read existing CSV: {e}")
                existing_rows = []

            # Rewrite with new columns
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow(row)
                writer.writerow(row_data)
        elif not file_exists:
            # Create new file
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerow(row_data)
        else:
            # Just append
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                writer.writerow(row_data)


class FoldMetricsAggregator(Callback):
    """Aggregate metrics across all folds."""

    def __init__(self, output_dir: Path):
        """Initialize aggregator.

        Args:
            output_dir: Output directory
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.fold_metrics = []

    def on_fit_end(self, trainer, pl_module):
        """Aggregate at end of training.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        # Collect final metrics
        metrics = trainer.callback_metrics
        self.fold_metrics.append({
            k: v.item() if hasattr(v, "item") else v
            for k, v in metrics.items()
        })
