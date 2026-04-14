"""Paired cross-fold comparison of shared vs decoupled architectures (R1.1/R1.3/R2.2).

With ``n=3`` folds the sample size forbids classical hypothesis testing
(Wilcoxon's minimum achievable two-sided p-value is 0.25), so the hierarchy
of evidence produced here is, in decreasing weight:

1. **Descriptives** — mean, std, and per-fold values per architecture.
2. **Paired differences** ``delta_k = decoupled_k - shared_k`` with mean/std.
3. **Effect size** — Cliff's delta (imported from
   :mod:`similarity_metrics.statistics.comparison`) and Cohen's d on the
   paired differences.
4. **Directional consistency** — whether all three folds share the same
   sign (``shared < decoupled`` for all three, or the opposite).
5. **Wilcoxon signed-rank** — computed for completeness; we do NOT claim
   significance when ``n=3``.

The convention in this module is that *lower is better* for every metric
(KID, LPIPS, MMD-MF), so a positive ``delta`` means the shared architecture
wins on that fold.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from ..statistics.comparison import compute_cliffs_delta

logger = logging.getLogger(__name__)


# (report_key, fold_metrics_csv_column)
DEFAULT_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("kid", "kid_mean"),
    ("lpips", "lpips_mean"),
    ("mmd_mf", "mmd_mf_mean"),
)


# ---------------------------------------------------------------------------
# Per-metric statistics
# ---------------------------------------------------------------------------


def _safe_std(values: np.ndarray, ddof: int) -> float:
    if values.size <= ddof:
        return float("nan")
    return float(np.std(values, ddof=ddof))


def _arch_payload(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "std": _safe_std(values, ddof=1),
        "per_fold": [float(v) for v in values.tolist()],
    }


def _wilcoxon_p(shared: np.ndarray, decoupled: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; NaN on failure.

    ``scipy.stats.wilcoxon`` raises ``ValueError`` if every difference is
    zero; returns NaN in that case.
    """
    diffs = decoupled - shared
    if np.all(diffs == 0):
        return float("nan")
    try:
        # ``mode="exact"`` is appropriate for small n; ``zero_method="wilcox"``
        # is the default and drops zeros.
        _, p = stats.wilcoxon(shared, decoupled, zero_method="wilcox")
    except ValueError:
        return float("nan")
    return float(p)


def _per_metric_stats(
    df: pd.DataFrame,
    metric_col: str,
    fold_col: str = "fold",
    arch_col: str = "architecture",
) -> dict[str, Any]:
    """Compute the full payload for one metric across folds.

    The input ``df`` must contain one row per ``(fold, architecture)`` cell
    with both architectures present for every fold. Missing pairs raise.
    """
    pivot = df.pivot(index=fold_col, columns=arch_col, values=metric_col).sort_index()
    missing_archs = [a for a in ("shared", "decoupled") if a not in pivot.columns]
    if missing_archs:
        raise ValueError(
            f"fold_metrics is missing columns for architectures: {missing_archs}"
        )
    if pivot.isna().any().any():
        raise ValueError(
            "fold_metrics contains NaN in the pivoted "
            f"{metric_col} column (per-fold pairs must be complete)."
        )

    shared = pivot["shared"].to_numpy(dtype=np.float64)
    decoupled = pivot["decoupled"].to_numpy(dtype=np.float64)

    # delta > 0 means shared is BETTER (lower is better for every supported metric).
    delta = decoupled - shared

    delta_mean = float(np.mean(delta)) if delta.size else float("nan")
    delta_std = _safe_std(delta, ddof=1)
    if delta.size >= 2 and delta_std > 0 and np.isfinite(delta_std):
        cohens_d = float(delta_mean / delta_std)
    else:
        cohens_d = float("nan")

    # Cliff's delta on the two univariate distributions.
    cliffs_delta = float(compute_cliffs_delta(decoupled, shared))

    signs = np.sign(delta).astype(int)
    all_consistent = bool(delta.size > 0 and np.all(signs == signs[0]) and signs[0] != 0)

    wilcoxon_p = _wilcoxon_p(shared, decoupled)

    return {
        "shared": _arch_payload(shared),
        "decoupled": _arch_payload(decoupled),
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "cliffs_delta": cliffs_delta,
        "cohens_d": cohens_d,
        "all_folds_consistent": all_consistent,
        "wilcoxon_p": wilcoxon_p if not np.isnan(wilcoxon_p) else None,
        "delta_per_fold": [float(v) for v in delta.tolist()],
        "fold_ids": [int(v) for v in pivot.index.tolist()],
    }


# ---------------------------------------------------------------------------
# Early-stopping epochs (optional input)
# ---------------------------------------------------------------------------


def _early_stopping_payload(
    es_path: Path | None,
) -> dict[str, Any] | None:
    """Load optional early-stopping epochs.

    Accepted schemas:

    * CSV with columns ``fold, architecture, epoch`` (one row per cell).
    * JSON mapping ``{"shared": {"0": 150, "1": 155, ...}, "decoupled": {...}}``.
    """
    if es_path is None:
        return None
    es_path = Path(es_path)
    if not es_path.exists():
        logger.warning("Early-stopping file not found, skipping: %s", es_path)
        return None

    if es_path.suffix.lower() == ".json":
        with open(es_path) as fh:
            raw = json.load(fh)
        by_arch: dict[str, dict[str, int]] = {
            str(arch): {str(k): int(v) for k, v in epochs.items()}
            for arch, epochs in raw.items()
        }
    else:
        df = pd.read_csv(es_path)
        required = {"fold", "architecture", "epoch"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Early-stopping CSV is missing columns: {missing}"
            )
        by_arch = {}
        for arch, sub in df.groupby("architecture"):
            by_arch[str(arch)] = {
                str(int(row["fold"])): int(row["epoch"]) for _, row in sub.iterrows()
            }

    out: dict[str, Any] = {}
    for arch, epoch_dict in by_arch.items():
        vals = np.array(list(epoch_dict.values()), dtype=np.float64)
        out[arch] = {
            "mean": float(np.mean(vals)) if vals.size else float("nan"),
            "std": _safe_std(vals, ddof=1),
            "per_fold": [
                int(epoch_dict[k]) for k in sorted(epoch_dict.keys(), key=int)
            ],
        }
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cross_fold_comparison(
    fold_metrics_csv: Path | str,
    output_json: Path | str | None = None,
    early_stopping_csv: Path | str | None = None,
    metric_columns: Iterable[tuple[str, str]] = DEFAULT_METRIC_COLUMNS,
) -> dict[str, Any]:
    """Produce the cross-fold shared-vs-decoupled comparison report.

    Parameters
    ----------
    fold_metrics_csv : path
        ``fold_metrics.csv`` produced by TASK-04 (one row per cell).
    output_json : path | None
        Where to write ``cross_fold_comparison.json``; if ``None``, the report
        is returned only.
    early_stopping_csv : path | None
        Optional CSV (``fold, architecture, epoch``) or JSON with per-cell
        early-stopping epochs. If absent, the corresponding report field is
        omitted (``None``).
    metric_columns : iterable of (report_key, csv_column)
        Override the default metric set.

    Returns
    -------
    dict
        The full report payload. Also written to ``output_json`` if provided.
    """
    fold_metrics_csv = Path(fold_metrics_csv)
    df = pd.read_csv(fold_metrics_csv)

    metrics_payload: dict[str, dict[str, Any]] = {}
    for key, col in metric_columns:
        if col not in df.columns:
            logger.warning(
                "cross-fold comparison: metric column %r not found, skipping",
                col,
            )
            continue
        metrics_payload[key] = _per_metric_stats(df, col)

    es_payload = _early_stopping_payload(
        Path(early_stopping_csv) if early_stopping_csv else None
    )
    report: dict[str, Any] = {"metrics": metrics_payload}
    if es_payload is not None:
        report["early_stopping_epochs"] = es_payload

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Wrote %s", output_json)

    return report
