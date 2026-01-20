"""Data loading utilities for model comparison."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig

from .config import get_model_results_paths, validate_model_paths

logger = logging.getLogger(__name__)


class ModelDataLoader:
    """Unified data loader for a single model's results."""

    def __init__(self, model_name: str, model_dir: Path) -> None:
        """Initialize loader for a model.

        Args:
            model_name: Display name for the model.
            model_dir: Path to model results directory.
        """
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.paths = get_model_results_paths(self.model_dir)

        # Cache for loaded data
        self._performance_df: pd.DataFrame | None = None
        self._histogram_cache: dict[str, NDArray] = {}

    def load_performance_csv(self, force_reload: bool = False) -> pd.DataFrame:
        """Load performance.csv from csv_logs/.

        Args:
            force_reload: If True, reload even if cached.

        Returns:
            DataFrame with epoch-level metrics.

        Raises:
            FileNotFoundError: If CSV file doesn't exist.
        """
        if self._performance_df is not None and not force_reload:
            return self._performance_df

        csv_path = self.paths["csv"]
        if not csv_path.exists():
            raise FileNotFoundError(f"Performance CSV not found: {csv_path}")

        self._performance_df = pd.read_csv(csv_path)
        logger.debug(
            f"Loaded performance CSV for {self.model_name}: "
            f"{len(self._performance_df)} rows, {len(self._performance_df.columns)} columns"
        )

        return self._performance_df

    def load_histogram(
        self,
        histogram_name: str,
        epoch: int,
    ) -> NDArray[np.float32]:
        """Load histogram data for a specific epoch.

        Args:
            histogram_name: 'timestep_distribution' or 'token_distribution'.
            epoch: Epoch number.

        Returns:
            Numpy array of histogram values.

        Raises:
            FileNotFoundError: If histogram file doesn't exist.
        """
        cache_key = f"{histogram_name}_{epoch}"
        if cache_key in self._histogram_cache:
            return self._histogram_cache[cache_key]

        histograms_dir = self.paths["histograms_dir"]
        if not histograms_dir.exists():
            raise FileNotFoundError(f"Histograms directory not found: {histograms_dir}")

        # Format: timestep_distribution_epoch0000.npz
        filename = f"{histogram_name}_epoch{epoch:04d}.npz"
        filepath = histograms_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Histogram file not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        histogram_data = data["data"].astype(np.float32)

        self._histogram_cache[cache_key] = histogram_data
        return histogram_data

    def get_available_epochs(
        self,
        histogram_name: str = "timestep_distribution",
    ) -> list[int]:
        """Get list of epochs with histogram data.

        Args:
            histogram_name: Type of histogram.

        Returns:
            Sorted list of epoch numbers.
        """
        histograms_dir = self.paths["histograms_dir"]
        if not histograms_dir.exists():
            return []

        pattern = f"{histogram_name}_epoch*.npz"
        files = list(histograms_dir.glob(pattern))

        epochs = []
        for f in files:
            match = re.search(r"epoch(\d+)", f.name)
            if match:
                epochs.append(int(match.group(1)))

        return sorted(epochs)

    def get_metric_columns(self, pattern: str | None = None) -> list[str]:
        """Get list of metric column names matching a pattern.

        Args:
            pattern: Regex pattern to match (optional).

        Returns:
            List of matching column names.
        """
        df = self.load_performance_csv()
        columns = list(df.columns)

        if pattern is None:
            return columns

        regex = re.compile(pattern)
        return [col for col in columns if regex.search(col)]

    def get_max_epoch(self) -> int:
        """Get maximum epoch number in the data.

        Returns:
            Maximum epoch number.
        """
        df = self.load_performance_csv()
        return int(df["epoch"].max())

    def get_final_metrics(
        self,
        metrics: list[str],
        epoch: int | None = None,
    ) -> dict[str, float]:
        """Get metric values for a specific epoch.

        Args:
            metrics: List of metric column names.
            epoch: Epoch to extract (None = final epoch).

        Returns:
            Dictionary mapping metric name to value.
        """
        df = self.load_performance_csv()

        if epoch is None or epoch == -1:
            epoch = self.get_max_epoch()

        row = df[df["epoch"] == epoch]
        if row.empty:
            raise ValueError(f"No data found for epoch {epoch}")

        result = {}
        for metric in metrics:
            if metric in row.columns:
                value = row[metric].values[0]
                result[metric] = float(value) if pd.notna(value) else float("nan")
            else:
                result[metric] = float("nan")

        return result


class ComparisonDataLoader:
    """Aggregated data loader for all models."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize with configuration.

        Args:
            cfg: Configuration with model paths.
        """
        self.cfg = cfg
        self.model_paths = validate_model_paths(cfg)

        # Create individual loaders
        self.loaders: dict[str, ModelDataLoader] = {}
        for model_name, model_dir in self.model_paths.items():
            self.loaders[model_name] = ModelDataLoader(model_name, model_dir)

        logger.info(f"Initialized comparison loader with {len(self.loaders)} models")

    @property
    def model_names(self) -> list[str]:
        """Get list of model names in order."""
        return list(self.loaders.keys())

    def load_all_performance_csvs(self) -> dict[str, pd.DataFrame]:
        """Load performance CSVs for all models.

        Returns:
            Dictionary mapping model_name -> DataFrame.
        """
        result = {}
        for model_name, loader in self.loaders.items():
            try:
                result[model_name] = loader.load_performance_csv()
            except FileNotFoundError as e:
                logger.error(f"Failed to load CSV for {model_name}: {e}")

        return result

    def get_common_metrics(self) -> list[str]:
        """Get metrics present in all models.

        Returns:
            List of metric column names common to all models.
        """
        all_columns: list[set[str]] = []

        for loader in self.loaders.values():
            try:
                df = loader.load_performance_csv()
                all_columns.append(set(df.columns))
            except FileNotFoundError:
                continue

        if not all_columns:
            return []

        common = all_columns[0]
        for cols in all_columns[1:]:
            common = common.intersection(cols)

        return sorted(list(common))

    def get_max_epoch(self) -> int:
        """Get maximum epoch across all models.

        Returns:
            Maximum epoch number.
        """
        max_epochs = []
        for loader in self.loaders.values():
            try:
                max_epochs.append(loader.get_max_epoch())
            except Exception:
                continue

        return max(max_epochs) if max_epochs else 0

    def get_min_max_epoch(self) -> int:
        """Get minimum of max epochs across all models.

        Useful for comparing models trained for different durations.

        Returns:
            Minimum of max epoch numbers.
        """
        max_epochs = []
        for loader in self.loaders.values():
            try:
                max_epochs.append(loader.get_max_epoch())
            except Exception:
                continue

        return min(max_epochs) if max_epochs else 0

    def get_summary_dataframe(
        self,
        metrics: list[str],
        epoch: int | None = None,
    ) -> pd.DataFrame:
        """Get summary DataFrame with metrics for all models.

        Args:
            metrics: List of metric column names.
            epoch: Epoch to extract (None = final epoch).

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        data = []
        for model_name, loader in self.loaders.items():
            try:
                model_metrics = loader.get_final_metrics(metrics, epoch)
                model_metrics["model"] = model_name
                data.append(model_metrics)
            except Exception as e:
                logger.error(f"Failed to get metrics for {model_name}: {e}")

        df = pd.DataFrame(data)
        if "model" in df.columns:
            df = df.set_index("model")

        return df

    def get_metric_over_epochs(
        self,
        metric: str,
        max_epoch: int | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Get metric values over all epochs for all models.

        Args:
            metric: Metric column name.
            max_epoch: Maximum epoch to include (optional).

        Returns:
            Dictionary mapping model_name -> array of values.
        """
        result = {}
        for model_name, loader in self.loaders.items():
            try:
                df = loader.load_performance_csv()
                if metric not in df.columns:
                    logger.warning(f"Metric {metric} not found in {model_name}")
                    continue

                values = df[metric].values.copy()
                if max_epoch is not None:
                    epochs = df["epoch"].values
                    mask = epochs <= max_epoch
                    values = values[mask]

                result[model_name] = values.astype(np.float64)
            except Exception as e:
                logger.error(f"Failed to get metric {metric} for {model_name}: {e}")

        return result

    def get_epochs(self, max_epoch: int | None = None) -> NDArray[np.int64]:
        """Get epoch array from first model.

        Args:
            max_epoch: Maximum epoch to include (optional).

        Returns:
            Array of epoch numbers.
        """
        for loader in self.loaders.values():
            try:
                df = loader.load_performance_csv()
                epochs = df["epoch"].values.astype(np.int64)
                if max_epoch is not None:
                    epochs = epochs[epochs <= max_epoch]
                return epochs
            except Exception:
                continue

        return np.array([], dtype=np.int64)

    def get_histogram_data(
        self,
        histogram_name: str,
        epoch: int,
    ) -> dict[str, NDArray[np.float32]]:
        """Get histogram data for all models at a specific epoch.

        Args:
            histogram_name: Type of histogram.
            epoch: Epoch number.

        Returns:
            Dictionary mapping model_name -> histogram array.
        """
        result = {}
        for model_name, loader in self.loaders.items():
            try:
                result[model_name] = loader.load_histogram(histogram_name, epoch)
            except FileNotFoundError as e:
                logger.warning(f"Histogram not found for {model_name}: {e}")

        return result
