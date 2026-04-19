"""Fold-aware similarity-metrics evaluation for the ICIP 2026 camera-ready.

Orchestrates KID, LPIPS, and MMD-MF computation across the
``3 folds x 2 architectures = 6 cells`` grid produced by TASK-03.

Outputs (written to ``output_dir``):

* ``fold_metrics.csv``              -- one row per cell (6 rows)
* ``summary_metrics.csv``           -- one row per architecture (2 rows)
* ``wasserstein_per_feature.csv``   -- per-feature Wasserstein for MMD-MF
* ``eval_sample_counts.json``       -- per-cell sample counts

In single-cell mode (``--fold N --architecture X``), only that cell is
computed and partial files are written; subsequent ``--aggregate-only``
passes merge them into the 4 final artifacts.

Run either as::

    python -m src.diffusion.scripts.similarity_metrics.fold_evaluation ...
    slimdiff-metrics fold-eval ...
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .data.fold_loaders import (
    DEFAULT_CELL_DIR_TEMPLATE,
    FoldEvalData,
    FoldLoaderError,
    load_fold_eval_data,
)
from .metrics.kid import KIDComputer
from .metrics.lpips import LPIPSComputer
from .metrics.mask_morphology import MaskMorphologyDistanceComputer

logger = logging.getLogger(__name__)


_WASSERSTEIN_FEATURE_ORDER: tuple[str, ...] = (
    "area",
    "perimeter",
    "circularity",
    "solidity",
    "extent",
    "eccentricity",
    "major_axis_length",
    "minor_axis_length",
    "equivalent_diameter",
    "geometric_mean",
)


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class CellMetrics:
    """Per-cell scalar metrics for one ``(fold, architecture)`` pair."""

    fold: int
    architecture: str
    kid_mean: float
    kid_std: float
    lpips_mean: float
    lpips_std: float
    mmd_mf_mean: float
    mmd_mf_std: float
    n_real: int
    n_synth: int


# =============================================================================
# Config I/O
# =============================================================================


def load_config(config_path: Path | str | None) -> dict[str, Any]:
    """Load the YAML config; empty dict if unset."""
    if config_path is None:
        return {}
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config not found: %s (using defaults + CLI overrides)", config_path)
        return {}
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


def merge_config_with_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge YAML config with CLI arguments; CLI takes precedence."""
    paths = config.get("paths", {}) or {}
    metrics_cfg = config.get("metrics", {}) or {}
    compute_cfg = config.get("compute", {}) or {}

    merged: dict[str, Any] = {
        "results_root": args.results_root or paths.get("results_root"),
        "cache_dir": args.cache_dir or paths.get("cache_dir"),
        "output_dir": args.output_dir or paths.get("output_dir"),
        "cell_dir_template": (
            args.cell_dir_template
            or paths.get("cell_dir_template")
            or DEFAULT_CELL_DIR_TEMPLATE
        ),
        "architectures": config.get("architectures", ["shared", "decoupled"]),
        "n_splits": int(config.get("folds", {}).get("n_splits", 3)),
        "kid": _with_defaults(
            metrics_cfg.get("kid"),
            {"enabled": True, "subset_size": 1000, "num_subsets": 100, "degree": 3},
        ),
        "lpips": _with_defaults(
            metrics_cfg.get("lpips"),
            {"enabled": True, "net": "vgg", "n_pairs": 1000},
        ),
        "mmd_mf": _with_defaults(
            metrics_cfg.get("mmd_mf"),
            {
                "enabled": True,
                "min_lesion_size_px": 5,
                "subset_size": 500,
                "num_subsets": 100,
                "degree": 3,
                "normalize_features": True,
            },
        ),
        "device": args.device or compute_cfg.get("device", "cuda:0"),
        "batch_size": int(args.batch_size or compute_cfg.get("batch_size", 32)),
        "seed": int(compute_cfg.get("seed", 42) if args.seed is None else args.seed),
        "max_replicas": (
            args.max_replicas
            if args.max_replicas is not None
            else compute_cfg.get("max_replicas")
        ),
    }
    return merged


def _with_defaults(user: dict[str, Any] | None, defaults: dict[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    if user:
        out.update(user)
    return out


# =============================================================================
# Per-cell compute
# =============================================================================


def compute_cell_metrics(
    data: FoldEvalData,
    cfg: dict[str, Any],
    device: str,
) -> tuple[CellMetrics, dict[str, float]]:
    """Compute KID, LPIPS, and MMD-MF for one cell.

    Memory strategy: KID and LPIPS accept raw float16 arrays and preprocess
    per-batch internally, so the full ``(N, 3, 299, 299)`` tensor is never
    materialised. Only MMD-MF needs a transient float32 copy of masks.

    Returns
    -------
    CellMetrics
        Scalar metrics (with NaN placeholders for disabled or undefined metrics).
    wasserstein : dict[str, float]
        Per-feature Wasserstein distances, always keyed by
        ``_WASSERSTEIN_FEATURE_ORDER`` (NaN if MMD-MF is disabled or failed).
    """
    kid_cfg = cfg["kid"]
    lpips_cfg = cfg["lpips"]
    mmd_cfg = cfg["mmd_mf"]
    batch_size = int(cfg["batch_size"])
    seed = int(cfg["seed"])

    logger.info(
        "[fold=%d %s] n_real=%d n_synth=%d device=%s",
        data.fold, data.architecture, data.n_real, data.n_synth, device,
    )

    # -------- KID ------------------------------------------------------------
    kid_mean = float("nan")
    kid_std = float("nan")
    if kid_cfg.get("enabled", True):
        kid_computer = KIDComputer(
            subset_size=int(kid_cfg.get("subset_size", 1000)),
            num_subsets=int(kid_cfg.get("num_subsets", 100)),
            degree=int(kid_cfg.get("degree", 3)),
            device=device,
            batch_size=batch_size,
        )
        kid_result = kid_computer.compute(
            data.real_images, data.synth_images, show_progress=False,
        )
        kid_mean = float(kid_result.value)
        kid_std = float(kid_result.std) if kid_result.std is not None else float("nan")
        del kid_computer, kid_result
        _release_gpu_memory(device)

    # -------- LPIPS ----------------------------------------------------------
    lpips_mean = float("nan")
    lpips_std = float("nan")
    if lpips_cfg.get("enabled", True):
        lpips_computer = LPIPSComputer(
            net=str(lpips_cfg.get("net", "vgg")),
            device=device,
            batch_size=batch_size,
        )
        lpips_result = lpips_computer.compute_pairwise(
            data.real_images,
            data.synth_images,
            n_pairs=int(lpips_cfg.get("n_pairs", 1000)),
            seed=seed,
            show_progress=False,
        )
        lpips_mean = float(lpips_result.value)
        lpips_std = float(lpips_result.std) if lpips_result.std is not None else float("nan")
        del lpips_computer, lpips_result
        _release_gpu_memory(device)

    # -------- MMD-MF + per-feature Wasserstein -------------------------------
    mmd_mean = float("nan")
    mmd_std = float("nan")
    wasserstein = {name: float("nan") for name in _WASSERSTEIN_FEATURE_ORDER}
    if mmd_cfg.get("enabled", True):
        mmd_computer = MaskMorphologyDistanceComputer(
            min_lesion_size_px=int(mmd_cfg.get("min_lesion_size_px", 5)),
            subset_size=int(mmd_cfg.get("subset_size", 500)),
            num_subsets=int(mmd_cfg.get("num_subsets", 100)),
            degree=int(mmd_cfg.get("degree", 3)),
            normalize_features=bool(mmd_cfg.get("normalize_features", True)),
        )
        synth_masks_f32 = data.synth_masks_f32()
        mmd_result, real_feat, synth_feat = mmd_computer.compute(
            real_masks=data.real_masks,
            synth_masks=synth_masks_f32,
            show_progress=False,
        )
        mmd_mean = float(mmd_result.value)
        mmd_std = float(mmd_result.std) if mmd_result.std is not None else float("nan")

        wass = mmd_computer.compute_per_feature_wasserstein(
            real_features=real_feat,
            synth_features=synth_feat,
            show_progress=False,
        )
        for key in _WASSERSTEIN_FEATURE_ORDER:
            wasserstein[key] = float(wass.get(key, float("nan")))
        del synth_masks_f32, mmd_computer, mmd_result, real_feat, synth_feat, wass
        _release_gpu_memory(device)

    return (
        CellMetrics(
            fold=int(data.fold),
            architecture=str(data.architecture),
            kid_mean=kid_mean,
            kid_std=kid_std,
            lpips_mean=lpips_mean,
            lpips_std=lpips_std,
            mmd_mf_mean=mmd_mean,
            mmd_mf_std=mmd_std,
            n_real=data.n_real,
            n_synth=data.n_synth,
        ),
        wasserstein,
    )


def _release_gpu_memory(device: str) -> None:
    """Drop cached allocations if running on CUDA."""
    gc.collect()
    if str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


# =============================================================================
# Drivers
# =============================================================================


def run_single_cell(
    fold: int,
    architecture: str,
    cache_dir: Path,
    results_root: Path,
    output_dir: Path,
    cfg: dict[str, Any],
    device: str,
    cell_dir_template: str | None = None,
) -> tuple[CellMetrics, dict[str, float], int]:
    """Evaluate one cell and write partial output files.

    Returns
    -------
    metrics, wasserstein, n_replicas
    """
    data = load_fold_eval_data(
        fold=fold,
        architecture=architecture,
        cache_dir=cache_dir,
        results_root=results_root,
        max_replicas=cfg.get("max_replicas"),
        cell_dir_template=cell_dir_template,
    )
    metrics, wasserstein = compute_cell_metrics(data, cfg, device)
    save_outputs(
        all_metrics=[metrics],
        all_wasserstein=[wasserstein],
        sample_counts=[{"n_replicas": data.n_replicas}],
        output_dir=output_dir,
        mode="partial",
    )
    return metrics, wasserstein, data.n_replicas


def run_grid(
    folds: Iterable[int],
    architectures: Iterable[str],
    cache_dir: Path,
    results_root: Path,
    output_dir: Path,
    cfg: dict[str, Any],
    device: str,
    cell_dir_template: str | None = None,
) -> tuple[list[CellMetrics], list[dict[str, float]]]:
    """Evaluate the full grid sequentially (one cell at a time)."""
    all_metrics: list[CellMetrics] = []
    all_wasserstein: list[dict[str, float]] = []
    sample_counts: list[dict[str, Any]] = []

    for fold in folds:
        for arch in architectures:
            logger.info("=== cell fold=%d architecture=%s ===", fold, arch)
            data = load_fold_eval_data(
                fold=fold,
                architecture=arch,
                cache_dir=cache_dir,
                results_root=results_root,
                max_replicas=cfg.get("max_replicas"),
                cell_dir_template=cell_dir_template,
            )
            metrics, wasserstein = compute_cell_metrics(data, cfg, device)
            all_metrics.append(metrics)
            all_wasserstein.append(wasserstein)
            sample_counts.append({"n_replicas": data.n_replicas})
            # Free the cell's arrays before moving on.
            del data
            _release_gpu_memory(device)

    save_outputs(
        all_metrics=all_metrics,
        all_wasserstein=all_wasserstein,
        sample_counts=sample_counts,
        output_dir=output_dir,
        mode="grid",
    )
    return all_metrics, all_wasserstein


# =============================================================================
# Output serialization
# =============================================================================


def _fold_metrics_row(m: CellMetrics) -> dict[str, Any]:
    return {
        "fold": m.fold,
        "architecture": m.architecture,
        "kid_mean": m.kid_mean,
        "kid_std": m.kid_std,
        "lpips_mean": m.lpips_mean,
        "lpips_std": m.lpips_std,
        "mmd_mf_mean": m.mmd_mf_mean,
        "mmd_mf_std": m.mmd_mf_std,
        "n_real": m.n_real,
        "n_synth": m.n_synth,
    }


def _wasserstein_row(m: CellMetrics, w: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {"fold": m.fold, "architecture": m.architecture}
    for key in _WASSERSTEIN_FEATURE_ORDER:
        row[key] = float(w.get(key, float("nan")))
    return row


def _summary_row(arch: str, cells: list[CellMetrics]) -> dict[str, Any]:
    """Aggregate across folds for a single architecture.

    ``*_mean`` is ``np.nanmean`` of per-fold means; ``*_std_across_folds`` is
    ``np.nanstd(..., ddof=0)`` (population std, n=3). If all folds produced
    NaN for a metric, the aggregate is NaN.
    """
    def agg(field: str) -> tuple[float, float]:
        vals = np.asarray([getattr(c, field) for c in cells], dtype=np.float64)
        valid = vals[~np.isnan(vals)]
        if valid.size == 0:
            return float("nan"), float("nan")
        if valid.size < vals.size:
            logger.warning(
                "architecture=%s %s: %d/%d folds NaN, aggregating over %d",
                arch, field, vals.size - valid.size, vals.size, valid.size,
            )
        return float(np.mean(valid)), float(np.std(valid, ddof=0))

    kid_mean, kid_std = agg("kid_mean")
    lpips_mean, lpips_std = agg("lpips_mean")
    mmd_mean, mmd_std = agg("mmd_mf_mean")
    return {
        "architecture": arch,
        "kid_mean": kid_mean,
        "kid_std_across_folds": kid_std,
        "lpips_mean": lpips_mean,
        "lpips_std_across_folds": lpips_std,
        "mmd_mf_mean": mmd_mean,
        "mmd_mf_std_across_folds": mmd_std,
    }


def save_outputs(
    all_metrics: list[CellMetrics],
    all_wasserstein: list[dict[str, float]],
    sample_counts: list[dict[str, Any]],
    output_dir: Path | str,
    mode: str = "grid",
) -> None:
    """Write the 4 evaluation artifacts.

    Parameters
    ----------
    mode : {"grid", "partial"}
        ``grid``: write the final ``fold_metrics.csv``, ``summary_metrics.csv``,
        ``wasserstein_per_feature.csv``, ``eval_sample_counts.json``.
        ``partial``: write per-cell ``*_f{fold}_{arch}.{csv|json}`` files for
        later merging via :func:`aggregate_partials`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(all_metrics) != len(all_wasserstein) or len(all_metrics) != len(sample_counts):
        raise ValueError(
            "save_outputs expects metrics / wasserstein / sample_counts to be the same length"
        )

    if mode == "grid":
        fold_rows = [_fold_metrics_row(m) for m in all_metrics]
        pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)

        wass_rows = [_wasserstein_row(m, w) for m, w in zip(all_metrics, all_wasserstein)]
        pd.DataFrame(wass_rows).to_csv(output_dir / "wasserstein_per_feature.csv", index=False)

        archs_ordered: list[str] = []
        for m in all_metrics:
            if m.architecture not in archs_ordered:
                archs_ordered.append(m.architecture)
        summary_rows = [
            _summary_row(arch, [m for m in all_metrics if m.architecture == arch])
            for arch in archs_ordered
        ]
        pd.DataFrame(summary_rows).to_csv(output_dir / "summary_metrics.csv", index=False)

        counts_payload = {
            "cells": [
                {
                    "fold": m.fold,
                    "architecture": m.architecture,
                    "n_real": m.n_real,
                    "n_synth": m.n_synth,
                    "n_replicas": int(sc["n_replicas"]),
                }
                for m, sc in zip(all_metrics, sample_counts)
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(output_dir / "eval_sample_counts.json", "w") as fh:
            json.dump(counts_payload, fh, indent=2)

    elif mode == "partial":
        for m, w, sc in zip(all_metrics, all_wasserstein, sample_counts):
            suffix = f"f{m.fold}_{m.architecture}"
            pd.DataFrame([_fold_metrics_row(m)]).to_csv(
                output_dir / f"fold_metrics_{suffix}.csv", index=False
            )
            pd.DataFrame([_wasserstein_row(m, w)]).to_csv(
                output_dir / f"wasserstein_per_feature_{suffix}.csv", index=False
            )
            payload = {
                "cells": [
                    {
                        "fold": m.fold,
                        "architecture": m.architecture,
                        "n_real": m.n_real,
                        "n_synth": m.n_synth,
                        "n_replicas": int(sc["n_replicas"]),
                    }
                ],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(output_dir / f"eval_sample_counts_{suffix}.json", "w") as fh:
                json.dump(payload, fh, indent=2)

    else:
        raise ValueError(f"mode must be 'grid' or 'partial', got {mode!r}")


def aggregate_partials(partial_dir: Path | str, output_dir: Path | str) -> None:
    """Merge all ``*_f{fold}_{arch}`` partial files into the 4 final outputs.

    Looks in ``partial_dir`` for:
    ``fold_metrics_f*.csv``, ``wasserstein_per_feature_f*.csv``,
    ``eval_sample_counts_f*.json``. Writes the merged artifacts into
    ``output_dir`` (may equal ``partial_dir``).
    """
    partial_dir = Path(partial_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_files = sorted(partial_dir.glob("fold_metrics_f*.csv"))
    wass_files = sorted(partial_dir.glob("wasserstein_per_feature_f*.csv"))
    count_files = sorted(partial_dir.glob("eval_sample_counts_f*.json"))
    if not fold_files:
        raise FileNotFoundError(
            f"No partial fold_metrics_f*.csv files found in {partial_dir}"
        )

    cells_metrics: list[CellMetrics] = []
    for fp in fold_files:
        row = pd.read_csv(fp).iloc[0].to_dict()
        cells_metrics.append(
            CellMetrics(
                fold=int(row["fold"]),
                architecture=str(row["architecture"]),
                kid_mean=float(row["kid_mean"]),
                kid_std=float(row["kid_std"]),
                lpips_mean=float(row["lpips_mean"]),
                lpips_std=float(row["lpips_std"]),
                mmd_mf_mean=float(row["mmd_mf_mean"]),
                mmd_mf_std=float(row["mmd_mf_std"]),
                n_real=int(row["n_real"]),
                n_synth=int(row["n_synth"]),
            )
        )

    wass_dicts: list[dict[str, float]] = []
    for fp in wass_files:
        row = pd.read_csv(fp).iloc[0].to_dict()
        wass_dicts.append({k: float(row[k]) for k in _WASSERSTEIN_FEATURE_ORDER})

    sample_counts: list[dict[str, Any]] = []
    for fp in count_files:
        with open(fp) as fh:
            payload = json.load(fh)
        cell = payload["cells"][0]
        sample_counts.append({"n_replicas": int(cell["n_replicas"])})

    save_outputs(
        all_metrics=cells_metrics,
        all_wasserstein=wass_dicts,
        sample_counts=sample_counts,
        output_dir=output_dir,
        mode="grid",
    )


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fold_evaluation",
        description=(
            "Fold-aware KID / LPIPS / MMD-MF evaluation for the ICIP 2026 "
            "camera-ready (shared vs decoupled x 3 folds)."
        ),
    )
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config.")
    parser.add_argument("--results-root", type=str, help="Parent of slimdiff_cr_* dirs.")
    parser.add_argument("--cache-dir", type=str, help="Slice cache root (has folds/).")
    parser.add_argument("--output-dir", type=str, help="Where to write the 4 artifacts.")
    parser.add_argument(
        "--cell-dir-template",
        type=str,
        default=None,
        help="Override default '{results_root}/slimdiff_cr_{architecture}_fold_{fold}'.",
    )
    parser.add_argument("--fold", type=int, default=None, help="Single-cell fold id.")
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        choices=["shared", "decoupled"],
        help="Single-cell architecture.",
    )
    parser.add_argument("--max-replicas", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0 | cpu.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Merge partial files from --partial-dir; skip all computation.",
    )
    parser.add_argument(
        "--partial-dir",
        type=str,
        default=None,
        help="Directory holding *_f{fold}_{arch}.* partials (for --aggregate-only).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _validate_paths(cfg: dict[str, Any], need_results: bool = True) -> list[str]:
    """Return a list of missing-required-path error messages (empty if OK)."""
    errors: list[str] = []
    if not cfg.get("cache_dir"):
        errors.append("--cache-dir is required (or set paths.cache_dir in the config).")
    if need_results and not cfg.get("results_root"):
        errors.append("--results-root is required (or set paths.results_root).")
    if not cfg.get("output_dir"):
        errors.append("--output-dir is required (or set paths.output_dir).")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = load_config(args.config)
    cfg = merge_config_with_args(config, args)

    # Aggregate-only path needs only --partial-dir and --output-dir.
    if args.aggregate_only:
        if not args.partial_dir:
            print("Error: --partial-dir is required with --aggregate-only", file=sys.stderr)
            return 2
        output_dir = cfg.get("output_dir") or args.partial_dir
        aggregate_partials(Path(args.partial_dir), Path(output_dir))
        logger.info("Aggregated partials from %s into %s", args.partial_dir, output_dir)
        return 0

    errors = _validate_paths(cfg, need_results=True)
    if errors:
        for err in errors:
            print(f"Error: {err}", file=sys.stderr)
        return 2

    # Single-cell vs grid mode
    single_fold_set = args.fold is not None
    single_arch_set = args.architecture is not None
    if single_fold_set ^ single_arch_set:
        print(
            "Error: --fold and --architecture must be supplied together (single-cell mode).",
            file=sys.stderr,
        )
        return 2

    cache_dir = Path(cfg["cache_dir"])
    results_root = Path(cfg["results_root"])
    output_dir = Path(cfg["output_dir"])

    try:
        if single_fold_set:
            run_single_cell(
                fold=int(args.fold),
                architecture=str(args.architecture),
                cache_dir=cache_dir,
                results_root=results_root,
                output_dir=output_dir,
                cfg=cfg,
                device=str(cfg["device"]),
                cell_dir_template=cfg["cell_dir_template"],
            )
        else:
            folds = list(range(int(cfg["n_splits"])))
            run_grid(
                folds=folds,
                architectures=list(cfg["architectures"]),
                cache_dir=cache_dir,
                results_root=results_root,
                output_dir=output_dir,
                cfg=cfg,
                device=str(cfg["device"]),
                cell_dir_template=cfg["cell_dir_template"],
            )
    except FoldLoaderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
