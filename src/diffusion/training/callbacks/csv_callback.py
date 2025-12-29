"""CSV logging callback for JS-DDPM training.

Provides a callback that logs all metrics to a CSV file,
with one row per epoch. This complements wandb logging
by providing a simple local CSV file for analysis.

Histograms are saved to separate npz files for offline analysis.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class CSVLoggingCallback(Callback):
    """Callback for logging metrics to CSV and histograms to npz files.

    Logs all metrics that are tracked by the trainer to a CSV file,
    with one row per epoch. This includes all metrics logged via
    self.log() in the LightningModule (train/*, val/*, diagnostics/*).

    The CSV file will have columns:
    - epoch: Epoch number
    - step: Global step number
    - All logged scalar metrics (train/loss, val/loss, diagnostics/*, etc.)

    Histograms (timestep_distribution, token_distribution) are saved
    to separate npz files in the histograms/ subdirectory.

    Metrics are written after each validation epoch ends to ensure
    both train and val metrics are captured.
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

        # Histogram directory
        self.histogram_dir = self.output_dir / "histograms"
        self.histogram_dir.mkdir(parents=True, exist_ok=True)

        # CSV file path (renamed to performance.csv)
        self.csv_path = self.output_dir / "performance.csv"

        # Track if we've written the header
        self._header_written = False

        # Store all metric names we've seen
        self._all_metric_names: set[str] = set()

        # Accumulators for histograms
        self._histogram_data: dict[str, list] = {}

        # Cache for training metrics (to be written with validation metrics)
        self._cached_train_metrics: dict[str, float] = {}

        logger.info(f"CSVLoggingCallback initialized, will write to {self.csv_path}")
        logger.info(f"Histograms will be saved to {self.histogram_dir}")

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called at the end of each training epoch.

        Caches training metrics to be written with validation metrics.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        # Get current epoch
        current_epoch = trainer.current_epoch

        # Collect training metrics from callback_metrics
        # These are the epoch-aggregated values from self.log(..., on_epoch=True)
        metrics = trainer.callback_metrics

        # Cache training metrics
        self._cached_train_metrics = {}
        for key, value in metrics.items():
            # Only cache training metrics (skip val/*, diagnostics/*, etc.)
            if key.startswith("train/") or key == "lr":
                try:
                    if hasattr(value, "item"):
                        self._cached_train_metrics[key] = value.item()
                    elif isinstance(value, (int, float)):
                        self._cached_train_metrics[key] = value
                    else:
                        logger.debug(f"Skipping non-scalar metric: {key}")
                except (ValueError, RuntimeError):
                    logger.debug(f"Could not convert metric {key} to scalar")
                    continue

        logger.debug(f"Cached {len(self._cached_train_metrics)} training metrics for epoch {current_epoch}")

    def save_histogram(
        self,
        name: str,
        data: np.ndarray,
        epoch: int,
    ) -> None:
        """Save histogram data to npz file.

        Args:
            name: Histogram name (e.g., 'timestep_distribution').
            data: Numpy array of values.
            epoch: Current epoch number.
        """
        # Create filename with epoch number
        filename = f"{name}_epoch{epoch:04d}.npz"
        filepath = self.histogram_dir / filename

        # Save to npz
        np.savez_compressed(
            filepath,
            data=data,
            epoch=epoch,
            histogram_name=name,
        )

        logger.debug(f"Saved histogram {name} to {filepath}")

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called at the end of each validation epoch.

        Collects all logged metrics (train + val + diagnostics) and writes them to CSV.
        Also checks for histogram data logged to wandb and saves to npz.

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
        # Only include scalar metrics (not tensors with > 1 element)
        metrics_dict = {}
        for key, value in metrics.items():
            try:
                if hasattr(value, "item"):
                    # Convert tensor to Python scalar
                    metrics_dict[key] = value.item()
                elif isinstance(value, (int, float)):
                    metrics_dict[key] = value
                else:
                    # Skip non-scalar values
                    logger.debug(f"Skipping non-scalar metric: {key}")
            except (ValueError, RuntimeError):
                # Skip if conversion fails
                logger.debug(f"Could not convert metric {key} to scalar")
                continue

        # Merge cached training metrics with current metrics
        # Training metrics were cached in on_train_epoch_end
        combined_metrics = {**self._cached_train_metrics, **metrics_dict}

        # Update our set of all metric names
        self._all_metric_names.update(combined_metrics.keys())

        # Add epoch and step to metrics
        row_data = {
            "epoch": current_epoch,
            "step": global_step,
            **combined_metrics,
        }

        # Write to CSV
        self._write_row(row_data)

        logger.debug(f"Wrote {len(combined_metrics)} metrics for epoch {current_epoch} to {self.csv_path}")

        # Clear cached training metrics for next epoch
        self._cached_train_metrics = {}

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
