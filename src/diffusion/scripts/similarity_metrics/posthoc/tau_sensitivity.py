"""MMD-MF sensitivity to the mask binarisation threshold tau (R1.5).

Synthetic masks emitted by TASK-03 are continuous in ``[-1, +1]``; the default
binarisation in :class:`MaskMorphologyDistanceComputer` thresholds at
``mask > 0`` (i.e. ``tau = 0``). This module sweeps ``tau`` over
``{-0.3, ..., +0.3}`` and recomputes MMD-MF at each threshold to quantify how
sensitive the metric is to the threshold choice.

Design:

* Binarisation rule: ``binary = np.where(continuous > tau, 1.0, -1.0)``
  (float32 in ``{-1, +1}``), matching the convention expected by
  :meth:`MorphologicalFeatureExtractor._preprocess_mask` which then binarises
  at ``mask > 0``.
* Real features are extracted once per fold and cached across tau values.
* ``np.random.seed`` is set before each MMD compute call so the sweep is
  reproducible, and the tau=0 cell matches the TASK-04 ``fold_metrics.csv``
  value within the Monte-Carlo subsampling tolerance (``atol=0.01``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..data.fold_loaders import FoldEvalData, load_fold_eval_data
from ..metrics.mask_morphology import MaskMorphologyDistanceComputer

logger = logging.getLogger(__name__)

# Default tau sweep values (inclusive of tau=0.0, the TASK-04 reference).
DEFAULT_TAU_VALUES: tuple[float, ...] = (
    -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3,
)


@dataclass
class TauSensitivityResult:
    """Results for one ``(fold, architecture)`` cell across tau values.

    Attributes
    ----------
    fold : int
    architecture : str
    tau_values : list[float]
    mmd_mf_values, mmd_mf_stds : list[float]
        Per-tau MMD-MF mean and subsampling std (NaN if MMD-MF is undefined,
        e.g., fewer than 10 detected lesions).
    n_lesions_detected : list[int]
        Total number of connected components detected across the synthetic
        batch at each tau (reveals threshold-induced mask collapse / noise).
    optimal_tau : float
        Tau minimising the (finite) MMD-MF; NaN if no tau yielded a finite
        MMD-MF.
    """

    fold: int
    architecture: str
    tau_values: list[float]
    mmd_mf_values: list[float]
    mmd_mf_stds: list[float]
    n_lesions_detected: list[int]
    optimal_tau: float


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _binarize_to_signed(
    continuous: NDArray, tau: float,
) -> NDArray[np.float32]:
    """Binarise continuous masks at tau, returning values in ``{-1, +1}``.

    The ``{-1, +1}`` output convention is what
    :class:`MorphologicalFeatureExtractor` expects: it internally re-thresholds
    at ``mask > 0``, which is exact for this output regardless of input dtype.
    """
    cont_f32 = np.asarray(continuous, dtype=np.float32)
    return np.where(cont_f32 > float(tau), 1.0, -1.0).astype(np.float32)


def compute_tau_sensitivity(
    data: FoldEvalData,
    tau_values: Sequence[float] = DEFAULT_TAU_VALUES,
    min_lesion_size_px: int = 5,
    subset_size: int = 500,
    num_subsets: int = 100,
    degree: int = 3,
    normalize_features: bool = True,
    seed: int = 42,
) -> TauSensitivityResult:
    """Run the tau sweep for one ``(fold, architecture)`` cell.

    Parameters
    ----------
    data : FoldEvalData
        Output of :func:`load_fold_eval_data`. ``data.real_masks`` must be
        binary in ``{-1, +1}`` (as produced by the dataset); ``data.synth_masks``
        is continuous in ``[-1, +1]`` (float16).
    tau_values : Sequence[float]
    min_lesion_size_px, subset_size, num_subsets, degree, normalize_features
        Forwarded to :class:`MaskMorphologyDistanceComputer`. Defaults match
        the TASK-04 ``fold_metrics.csv`` configuration so that the tau=0 cell
        is directly comparable.
    seed : int
        Seed applied via :func:`numpy.random.seed` before each MMD-MF
        computation to make the sweep reproducible.
    """
    computer = MaskMorphologyDistanceComputer(
        min_lesion_size_px=min_lesion_size_px,
        subset_size=subset_size,
        num_subsets=num_subsets,
        degree=degree,
        normalize_features=normalize_features,
    )

    # Extract real features once (cached via the computer's internal id-keyed
    # cache, but we extract explicitly here so the cache miss happens outside
    # the tau loop).
    real_features = computer.extract_features(
        np.asarray(data.real_masks, dtype=np.float32),
        show_progress=False,
        desc=f"real_fold{data.fold}_{data.architecture}",
    )
    logger.info(
        "[tau-sweep fold=%d %s] real lesions=%d (from %d masks)",
        data.fold, data.architecture, int(real_features.shape[0]), data.n_real,
    )

    continuous = np.asarray(data.synth_masks)  # keep float16 to save RAM
    tau_list: list[float] = []
    mmd_list: list[float] = []
    mmd_std_list: list[float] = []
    n_lesions_list: list[int] = []

    for tau in tau_values:
        synth_binarised = _binarize_to_signed(continuous, tau)
        synth_features = computer.extract_features(
            synth_binarised,
            show_progress=False,
            desc=f"synth_tau_{tau:+.2f}",
        )
        n_lesions = int(synth_features.shape[0])

        # Seed for reproducible subsampling.
        np.random.seed(seed)
        # `compute()` requires the raw masks to fill in n_real / n_synth in
        # the returned MetricResult; pass them but override the features so
        # re-extraction is skipped.
        result, _, _ = computer.compute(
            real_masks=np.asarray(data.real_masks, dtype=np.float32),
            synth_masks=synth_binarised,
            show_progress=False,
            real_features=real_features,
            synth_features=synth_features,
        )

        tau_list.append(float(tau))
        mmd_list.append(float(result.value))
        mmd_std_list.append(
            float(result.std) if result.std is not None else float("nan")
        )
        n_lesions_list.append(n_lesions)
        logger.debug(
            "[tau-sweep fold=%d %s] tau=%+.2f mmd=%.4f n_lesions=%d",
            data.fold, data.architecture, tau, result.value, n_lesions,
        )

    # Optimal tau among finite MMD values; NaN propagates if all NaN.
    finite_vals = [v if np.isfinite(v) else np.inf for v in mmd_list]
    if all(np.isinf(v) for v in finite_vals):
        optimal_tau = float("nan")
    else:
        optimal_tau = float(tau_list[int(np.argmin(finite_vals))])

    return TauSensitivityResult(
        fold=int(data.fold),
        architecture=str(data.architecture),
        tau_values=tau_list,
        mmd_mf_values=mmd_list,
        mmd_mf_stds=mmd_std_list,
        n_lesions_detected=n_lesions_list,
        optimal_tau=optimal_tau,
    )


# ---------------------------------------------------------------------------
# Grid driver + output serialisation
# ---------------------------------------------------------------------------


def run_tau_sensitivity_grid(
    folds: Iterable[int],
    architectures: Iterable[str],
    cache_dir: Path | str,
    results_root: Path | str,
    output_dir: Path | str,
    tau_values: Sequence[float] = DEFAULT_TAU_VALUES,
    max_replicas: int | None = None,
    cell_dir_template: str | None = None,
    min_lesion_size_px: int = 5,
    subset_size: int = 500,
    num_subsets: int = 100,
    degree: int = 3,
    normalize_features: bool = True,
    seed: int = 42,
) -> list[TauSensitivityResult]:
    """Run the tau sweep over every ``(fold, architecture)`` cell.

    Writes ``tau_sensitivity.csv`` (long form) and
    ``tau_sensitivity_summary.csv`` (aggregated across folds) into
    ``output_dir``.
    """
    cache_dir = Path(cache_dir)
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[TauSensitivityResult] = []
    for fold in folds:
        for arch in architectures:
            data = load_fold_eval_data(
                fold=fold,
                architecture=arch,
                cache_dir=cache_dir,
                results_root=results_root,
                max_replicas=max_replicas,
                cell_dir_template=cell_dir_template,
            )
            result = compute_tau_sensitivity(
                data,
                tau_values=tau_values,
                min_lesion_size_px=min_lesion_size_px,
                subset_size=subset_size,
                num_subsets=num_subsets,
                degree=degree,
                normalize_features=normalize_features,
                seed=seed,
            )
            all_results.append(result)
            del data

    save_tau_sensitivity_outputs(all_results, output_dir)
    return all_results


def _result_rows(result: TauSensitivityResult) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for tau, mmd, mmd_std, n in zip(
        result.tau_values,
        result.mmd_mf_values,
        result.mmd_mf_stds,
        result.n_lesions_detected,
    ):
        rows.append(
            {
                "fold": int(result.fold),
                "architecture": str(result.architecture),
                "tau": float(tau),
                "mmd_mf_mean": float(mmd),
                "mmd_mf_std": float(mmd_std),
                "n_lesions": int(n),
            }
        )
    return rows


def save_tau_sensitivity_outputs(
    results: list[TauSensitivityResult],
    output_dir: Path | str,
) -> None:
    """Write ``tau_sensitivity.csv`` + ``tau_sensitivity_summary.csv``.

    * Long form: one row per (fold, architecture, tau).
    * Summary:  one row per (architecture, tau), aggregated across folds.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, float]] = []
    for r in results:
        long_rows.extend(_result_rows(r))
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(output_dir / "tau_sensitivity.csv", index=False)

    if long_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "architecture", "tau",
                "mmd_mf_mean_across_folds",
                "mmd_mf_std_across_folds",
                "n_lesions_mean",
            ]
        )
    else:
        summary = (
            long_df.groupby(["architecture", "tau"], as_index=False)
            .agg(
                mmd_mf_mean_across_folds=("mmd_mf_mean", lambda s: float(np.nanmean(s))),
                mmd_mf_std_across_folds=("mmd_mf_mean", lambda s: float(np.nanstd(s, ddof=0))),
                n_lesions_mean=("n_lesions", lambda s: float(np.mean(s))),
            )
        )
        summary_df = summary
    summary_df.to_csv(output_dir / "tau_sensitivity_summary.csv", index=False)
