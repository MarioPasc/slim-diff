"""Compute per-zbin KID for every ``(fold, architecture)`` cell and cache to CSV.

Downstream figures (notably :mod:`ablation_comparison_figure`) annotate each
z-bin panel with the fold-aggregated KID. Per-zbin KID is not part of the
standard ``slimdiff-metrics fold-eval`` output, so this script computes it
once and writes two artefacts:

* ``kid_per_zbin.csv``          — one row per ``(fold, architecture, zbin)``.
* ``kid_per_zbin_summary.csv``  — cross-fold aggregation with mean, std,
  and t-based 95% CI half-width.

Only the requested zbins are evaluated (default: 7 evenly-spaced bins) so
the compute cost stays at ``7 × 3 × 2 ≈ 42`` KID evaluations rather than
the full ``30 × 3 × 2``.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.diffusion.scripts.similarity_metrics.data.fold_loaders import (
    load_fold_eval_data,
)
from src.diffusion.scripts.similarity_metrics.metrics.kid import compute_per_zbin_kid

logger = logging.getLogger(__name__)


# t-distribution 97.5% quantiles for df=1,2,3,...,10 (one-sided), used to
# build 95% CIs without depending on scipy.
_T_CRIT_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def _t_crit(df: int) -> float:
    if df in _T_CRIT_975:
        return _T_CRIT_975[df]
    return 1.96  # large-n Gaussian fallback


# =============================================================================
# Core
# =============================================================================


def compute_cell_kid(
    cache_dir: Path,
    results_root: Path,
    fold: int,
    architecture: str,
    zbins: list[int],
    subset_size: int,
    num_subsets: int,
    device: str,
    batch_size: int,
    max_replicas: int | None,
) -> list[dict]:
    data = load_fold_eval_data(
        fold=fold,
        architecture=architecture,
        cache_dir=cache_dir,
        results_root=results_root,
        max_replicas=max_replicas,
    )
    rows = compute_per_zbin_kid(
        real_images=data.real_images,
        real_zbins=data.real_zbins,
        synth_images=data.synth_images,  # float16 — KIDComputer preprocesses per batch.
        synth_zbins=data.synth_zbins,
        valid_zbins=zbins,
        subset_size=subset_size,
        num_subsets=num_subsets,
        device=device,
        batch_size=batch_size,
    )
    out = []
    for r in rows:
        out.append(
            {
                "fold": int(fold),
                "architecture": architecture,
                "zbin": int(r["zbin"]),
                "kid": float(r["kid"]),
                "kid_std": float(r["kid_std"]) if r["kid_std"] is not None else float("nan"),
                "n_real": int(r["n_real"]),
                "n_synth": int(r["n_synth"]),
            }
        )
    return out


def aggregate_across_folds(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Aggregate KID across folds with 95% CI half-width (t-based)."""
    groups = per_fold.groupby(["architecture", "zbin"])
    rows = []
    for (arch, zbin), sub in groups:
        vals = sub["kid"].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        if n == 0:
            mean = float("nan")
            std = float("nan")
            ci_half = float("nan")
        else:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            sem = std / math.sqrt(n) if n > 0 else float("nan")
            ci_half = _t_crit(max(n - 1, 1)) * sem if n > 1 else 0.0
        rows.append(
            {
                "architecture": arch,
                "zbin": int(zbin),
                "kid_mean": mean,
                "kid_std_across_folds": std,
                "kid_ci95_half": ci_half,
                "n_folds": n,
            }
        )
    return pd.DataFrame(rows).sort_values(["architecture", "zbin"]).reset_index(drop=True)


# =============================================================================
# CLI
# =============================================================================


def _default_zbins(n_total: int, n_pick: int) -> list[int]:
    return [int(z) for z in np.linspace(0, n_total - 1, n_pick, dtype=int)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--architectures", nargs="+", default=["shared", "decoupled"]
    )
    p.add_argument(
        "--zbins",
        type=int,
        nargs="+",
        default=None,
        help="Z-bins to compute (default: 7 evenly spaced over 0..29).",
    )
    p.add_argument("--subset-size", type=int, default=30)
    p.add_argument("--num-subsets", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-replicas", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args() if argv is None else _parse_args_from_list(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    zbins = args.zbins or _default_zbins(30, 7)
    logger.info("Computing per-zbin KID for zbins=%s", zbins)

    per_fold_rows: list[dict] = []
    for arch in args.architectures:
        for fold in args.folds:
            logger.info("→ fold=%d arch=%s", fold, arch)
            per_fold_rows.extend(
                compute_cell_kid(
                    cache_dir=args.cache_dir,
                    results_root=args.results_root,
                    fold=fold,
                    architecture=arch,
                    zbins=zbins,
                    subset_size=args.subset_size,
                    num_subsets=args.num_subsets,
                    device=args.device,
                    batch_size=args.batch_size,
                    max_replicas=args.max_replicas,
                )
            )
    per_fold_df = pd.DataFrame(per_fold_rows)
    summary_df = aggregate_across_folds(per_fold_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_fold_csv = args.output_dir / "kid_per_zbin.csv"
    summary_csv = args.output_dir / "kid_per_zbin_summary.csv"
    per_fold_df.to_csv(per_fold_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Wrote %s (%d rows)", per_fold_csv, len(per_fold_df))
    logger.info("Wrote %s (%d rows)", summary_csv, len(summary_df))
    return summary_csv


def _parse_args_from_list(argv: list[str]) -> argparse.Namespace:
    import sys as _sys

    old = _sys.argv
    try:
        _sys.argv = ["compute_zbin_kid", *argv]
        return _parse_args()
    finally:
        _sys.argv = old


if __name__ == "__main__":
    main()
