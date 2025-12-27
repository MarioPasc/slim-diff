"""CSV logging callback for JS-DDPM training.

Provides a callback that logs all metrics to a CSV file,
with one row per epoch. This complements wandb logging
by providing a simple local CSV file for analysis.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class CSVLoggingCallback(Callback):
    """Callback for logging metrics to CSV file.

    Logs all metrics that are tracked by the trainer to a CSV file,
    with one row per epoch. This includes all metrics logged via
    self.log() in the LightningModule (train/*, val/*, etc.).

    The CSV file will have columns:
    - epoch: Epoch number
    - step: Global step number
    - All logged metrics (train/loss, val/loss, train/psnr, etc.)

    Metrics are written after each validation epoch ends.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the CSV logging callback.

        Args:
            cfg: Configuration object.
        """
        super().__init__()
        self.cfg = cfg

        # Create output directory
        self.output_dir = Path(cfg.experiment.output_dir) / "csv_logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CSV file path
        self.csv_path = self.output_dir / "training_metrics.csv"

        # Track if we've written the header
        self._header_written = False

        # Store all metric names we've seen
        self._all_metric_names: set[str] = set()

        logger.info(f"CSVLoggingCallback initialized, will write to {self.csv_path}")

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called at the end of each training epoch.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        # We'll write metrics after validation epoch ends
        # to ensure we have both train and val metrics
        pass

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called at the end of each validation epoch.

        Collects all logged metrics and writes them to CSV.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        # Get current epoch and step
        current_epoch = trainer.current_epoch
        global_step = trainer.global_step

        # Collect all logged metrics from trainer's callback metrics
        # These are the metrics logged via self.log() in the LightningModule
        metrics = trainer.callback_metrics

        # Convert metrics to a regular dict and extract values
        metrics_dict = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                # Convert tensor to Python scalar
                metrics_dict[key] = value.item()
            else:
                metrics_dict[key] = value

        # Update our set of all metric names
        self._all_metric_names.update(metrics_dict.keys())

        # Add epoch and step to metrics
        row_data = {
            "epoch": current_epoch,
            "step": global_step,
            **metrics_dict,
        }

        # Write to CSV
        self._write_row(row_data)

        logger.debug(f"Wrote metrics for epoch {current_epoch} to {self.csv_path}")

    def _write_row(self, row_data: dict[str, Any]) -> None:
        """Write a single row to the CSV file.

        Args:
            row_data: Dictionary of metric names to values.
        """
        # Determine all columns (epoch, step, + all metrics we've seen)
        columns = ["epoch", "step"] + sorted(self._all_metric_names)

        # Check if file exists and we need to write header
        file_exists = self.csv_path.exists()

        # If file exists but we have new columns, we need to rewrite the file
        # with the new columns. This handles cases where new metrics appear
        # in later epochs (e.g., validation metrics that only appear after
        # the first epoch).
        if file_exists and not self._header_written:
            # Read existing data
            existing_rows = []
            try:
                with open(self.csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)
            except Exception as e:
                logger.warning(f"Could not read existing CSV: {e}, will overwrite")
                existing_rows = []

            # Rewrite file with new columns
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

                # Write existing rows (with potentially new columns as empty)
                for row in existing_rows:
                    writer.writerow(row)

                # Write new row
                writer.writerow(row_data)

            self._header_written = True

        elif not file_exists:
            # Create new file with header
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerow(row_data)

            self._header_written = True

        else:
            # Append to existing file
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writerow(row_data)
