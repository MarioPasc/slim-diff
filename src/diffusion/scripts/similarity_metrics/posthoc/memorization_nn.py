"""Nearest-neighbor memorization diagnostic (R2 follow-up).

For each ``(fold, architecture)`` cell we extract InceptionV3 pool3 features
(2048-D) and compute nearest-neighbor L2 distances against the *training*
slices of that fold:

* ``D_test  = NN(H_f -> T_f)``: each held-out test slice's distance to its
  nearest training slice. Baseline -- how close untrained reals are to train.
* ``D_synth = NN(S_{f,arch} -> T_f)``: each synthetic slice's distance to its
  nearest training slice. Control -- how close synth samples are to train.

A model that simply generalises well will produce ``D_synth`` distributions
indistinguishable from ``D_test``. A model that memorises will have
``D_synth`` shifted toward zero (synth samples sit closer to train slices
than the test distribution does).

We report three indicators per cell:

* ``mem_ratio = median(D_synth) / median(D_test)`` -- values << 1 are
  suspicious; ~1 is consistent with generalisation.
* ``suspect_count`` and ``suspect_frac`` -- the count/fraction of synth
  samples with ``D_synth < min(D_test)`` (Carlini-style "closer to *any*
  train sample than the closest held-out test sample is").
* ``wasserstein_1(D_synth, D_test)`` -- a full-distribution gap that is
  insensitive to the choice of summary statistic.

Outputs (written to ``--output-dir``, default ``posthoc_output``):

* ``memorization_nn.csv``         -- one row per (fold, architecture)
* ``memorization_nn_summary.csv`` -- aggregated across folds per architecture
* ``memorization_nn.json``        -- full payload with percentiles and config
* ``tables/table_memorization_nn.tex`` -- compact summary table for the supp.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.diffusion.data.kfold import KFoldManager
from src.diffusion.scripts.similarity_metrics.data.fold_loaders import (
    _load_real_from_fold_csv,
    _load_replicas,
    resolve_cell_dir,
)
from src.diffusion.scripts.similarity_metrics.metrics.feature_nn import (
    FeatureNNComputer,
)
from src.diffusion.scripts.similarity_metrics.metrics.kid import (
    InceptionFeatureExtractor,
)

logger = logging.getLogger(__name__)

DEFAULT_ARCHS: tuple[str, ...] = ("shared", "decoupled", "zero_coupling")
DEFAULT_FOLDS: tuple[int, ...] = (0, 1, 2)
ARCH_LABELS: dict[str, str] = {
    "shared": "Shared (ours)",
    "decoupled": "Decoupled",
    "zero_coupling": "Zero-coupling",
}


# ----------------------------------------------------------------------------
# Feature extraction wrapper (chunked to keep RAM bounded for ~100k samples)
# ----------------------------------------------------------------------------
def extract_features_chunked(
    extractor: InceptionFeatureExtractor,
    images: np.ndarray,
    chunk: int = 2000,
    label: str = "features",
) -> np.ndarray:
    """Run ``extract_features`` on chunks of ``chunk`` samples at a time.

    The underlying ``preprocess_for_inception`` materialises the whole input
    as a ``(N, 3, 299, 299)`` uint8 tensor (~28 GB for 100k samples), so we
    feed it in slices and stack the resulting feature tensors on the CPU.
    """
    parts: list[np.ndarray] = []
    n = int(images.shape[0])
    iterator = tqdm(range(0, n, chunk), desc=f"InceptionV3 {label}", leave=False)
    for start in iterator:
        chunk_imgs = np.asarray(images[start : start + chunk], dtype=np.float32)
        feat = extractor.extract_features(chunk_imgs, show_progress=False)
        parts.append(feat.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ----------------------------------------------------------------------------
# Per-cell pipeline
# ----------------------------------------------------------------------------
@dataclass
class _DistanceSummary:
    """Per-cell aggregation of one NN-distance distribution."""

    mean: float
    std: float
    median: float
    p1: float
    p5: float
    p25: float
    p75: float
    p99: float
    min_: float


def _summarise(dists: np.ndarray) -> _DistanceSummary:
    return _DistanceSummary(
        mean=float(np.mean(dists)),
        std=float(np.std(dists, ddof=0)),
        median=float(np.median(dists)),
        p1=float(np.percentile(dists, 1)),
        p5=float(np.percentile(dists, 5)),
        p25=float(np.percentile(dists, 25)),
        p75=float(np.percentile(dists, 75)),
        p99=float(np.percentile(dists, 99)),
        min_=float(np.min(dists)),
    )


def _wasserstein_1(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 distance between two 1-D empirical distributions."""
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(a, b))
    except Exception:  # pragma: no cover -- scipy is a hard dependency upstream
        # Fallback: integral of |CDF_a - CDF_b| via sorted-quantile diff.
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        # Resample to a common quantile grid.
        q = np.linspace(0.0, 1.0, num=min(len(a_sorted), len(b_sorted)))
        qa = np.quantile(a_sorted, q)
        qb = np.quantile(b_sorted, q)
        return float(np.mean(np.abs(qa - qb)))


def _row(
    fold: int,
    arch: str,
    n_train: int,
    n_test: int,
    n_synth: int,
    n_replicas: int,
    synth_summary: _DistanceSummary,
    test_summary: _DistanceSummary,
    suspect_count: int,
    w1: float,
) -> dict:
    ratio = synth_summary.median / test_summary.median if test_summary.median > 0 else float("nan")
    return {
        "fold": int(fold),
        "architecture": str(arch),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_synth": int(n_synth),
        "n_replicas": int(n_replicas),
        "d_synth_mean": synth_summary.mean,
        "d_synth_std": synth_summary.std,
        "d_synth_median": synth_summary.median,
        "d_synth_p1": synth_summary.p1,
        "d_synth_p5": synth_summary.p5,
        "d_synth_p25": synth_summary.p25,
        "d_synth_p75": synth_summary.p75,
        "d_synth_p99": synth_summary.p99,
        "d_synth_min": synth_summary.min_,
        "d_test_mean": test_summary.mean,
        "d_test_std": test_summary.std,
        "d_test_median": test_summary.median,
        "d_test_p1": test_summary.p1,
        "d_test_p5": test_summary.p5,
        "d_test_min": test_summary.min_,
        "mem_ratio": ratio,
        "suspect_count": int(suspect_count),
        "suspect_frac": float(suspect_count) / float(n_synth) if n_synth else float("nan"),
        "wasserstein_1": float(w1),
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_memorization_nn(
    folds: Iterable[int],
    architectures: Iterable[str],
    cache_dir: Path,
    results_root: Path,
    output_dir: Path,
    device: str = "cuda:0",
    batch_size: int = 16,
    chunk_size_features: int = 2000,
    chunk_size_nn: int = 2000,
    max_replicas: int | None = None,
    cell_dir_template: str | None = None,
) -> pd.DataFrame:
    """Compute the NN memorization diagnostic and write all outputs."""
    cache_dir = Path(cache_dir)
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    extractor = InceptionFeatureExtractor(device=device, batch_size=batch_size)
    nn_computer = FeatureNNComputer(
        device=device, batch_size=batch_size, chunk_size=chunk_size_nn
    )

    manager = KFoldManager.from_meta_json(cache_dir / "folds" / "folds_meta.json")

    rows: list[dict] = []
    for fold in folds:
        fold_dir = manager.get_cache_dir_for_fold(fold)
        train_csv = fold_dir / "train.csv"
        test_csv = fold_dir / "test.csv"
        logger.info("--- fold=%d ---", fold)
        logger.info("Loading train slices from %s", train_csv)
        train_imgs, _, _ = _load_real_from_fold_csv(train_csv, cache_dir)
        logger.info("Loading test slices from %s", test_csv)
        test_imgs, _, _ = _load_real_from_fold_csv(test_csv, cache_dir)
        n_train = int(train_imgs.shape[0])
        n_test = int(test_imgs.shape[0])

        logger.info("Extracting features: train n=%d, test n=%d", n_train, n_test)
        train_feat = extract_features_chunked(
            extractor, train_imgs, chunk=chunk_size_features, label=f"fold{fold} train"
        )
        del train_imgs
        test_feat = extract_features_chunked(
            extractor, test_imgs, chunk=chunk_size_features, label=f"fold{fold} test"
        )
        del test_imgs

        # Baseline: NN(test -> train)
        logger.info("Computing baseline NN(test -> train)")
        test_to_train, _ = nn_computer._compute_nn_distances_chunked(
            query_features=test_feat,
            reference_features=train_feat,
            description=None,
        )
        test_summary = _summarise(test_to_train)
        del test_feat

        for arch in architectures:
            cell_dir = resolve_cell_dir(
                results_root=results_root,
                architecture=arch,
                fold=fold,
                cell_dir_template=cell_dir_template,
            )
            logger.info("[fold=%d arch=%s] loading synth from %s", fold, arch, cell_dir)
            synth_imgs, _, _, _, n_reps = _load_replicas(
                cell_dir / "replicas", max_replicas=max_replicas
            )
            n_synth = int(synth_imgs.shape[0])
            logger.info(
                "[fold=%d arch=%s] extracting features for %d samples (%d replicas)",
                fold, arch, n_synth, n_reps,
            )
            synth_feat = extract_features_chunked(
                extractor, synth_imgs, chunk=chunk_size_features,
                label=f"fold{fold} {arch}",
            )
            del synth_imgs

            logger.info("[fold=%d arch=%s] computing NN(synth -> train)", fold, arch)
            synth_to_train, _ = nn_computer._compute_nn_distances_chunked(
                query_features=synth_feat,
                reference_features=train_feat,
                description=None,
            )
            del synth_feat

            synth_summary = _summarise(synth_to_train)
            suspect_count = int(np.sum(synth_to_train < test_summary.min_))
            w1 = _wasserstein_1(synth_to_train, test_to_train)

            rows.append(
                _row(
                    fold=fold,
                    arch=arch,
                    n_train=n_train,
                    n_test=n_test,
                    n_synth=n_synth,
                    n_replicas=n_reps,
                    synth_summary=synth_summary,
                    test_summary=test_summary,
                    suspect_count=suspect_count,
                    w1=w1,
                )
            )
            logger.info(
                "[fold=%d arch=%s] median(D_synth)=%.4f median(D_test)=%.4f ratio=%.3f "
                "suspects=%d/%d (%.2f%%) W1=%.4f",
                fold, arch,
                synth_summary.median, test_summary.median,
                rows[-1]["mem_ratio"],
                suspect_count, n_synth, 100.0 * suspect_count / max(n_synth, 1),
                w1,
            )

            del synth_to_train
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        del train_feat
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    csv_path = output_dir / "memorization_nn.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    summary_df = aggregate_across_folds(df)
    summary_path = output_dir / "memorization_nn_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Wrote %s", summary_path)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "batch_size": batch_size,
        "chunk_size_features": chunk_size_features,
        "chunk_size_nn": chunk_size_nn,
        "feature_extractor": "InceptionV3 pool3 (2048-D)",
        "distance": "L2",
        "indicators": {
            "mem_ratio": "median(D_synth) / median(D_test); ~1 = generalisation, <<1 = memorization.",
            "suspect_count": "Synth samples with D_synth < min(D_test).",
            "wasserstein_1": "W1(D_synth, D_test).",
        },
        "per_cell": df.to_dict(orient="records"),
        "across_folds": summary_df.to_dict(orient="records"),
    }
    json_path = output_dir / "memorization_nn.json"
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", json_path)

    table_path = output_dir / "tables" / "table_memorization_nn.tex"
    table_path.write_text(generate_latex_table(summary_df))
    logger.info("Wrote %s", table_path)

    return df


def aggregate_across_folds(per_cell: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-cell rows into one row per architecture."""
    rows: list[dict] = []
    for arch, sub in per_cell.groupby("architecture"):
        rows.append(
            {
                "architecture": str(arch),
                "n_folds": int(len(sub)),
                "d_synth_median_mean": float(np.mean(sub["d_synth_median"])),
                "d_synth_median_std": float(np.std(sub["d_synth_median"], ddof=0)),
                "d_test_median_mean": float(np.mean(sub["d_test_median"])),
                "d_test_median_std": float(np.std(sub["d_test_median"], ddof=0)),
                "mem_ratio_mean": float(np.mean(sub["mem_ratio"])),
                "mem_ratio_std": float(np.std(sub["mem_ratio"], ddof=0)),
                "suspect_frac_mean": float(np.mean(sub["suspect_frac"])),
                "suspect_frac_std": float(np.std(sub["suspect_frac"], ddof=0)),
                "wasserstein_1_mean": float(np.mean(sub["wasserstein_1"])),
                "wasserstein_1_std": float(np.std(sub["wasserstein_1"], ddof=0)),
                "n_synth_total": int(sub["n_synth"].sum()),
                "suspect_count_total": int(sub["suspect_count"].sum()),
            }
        )
    df = pd.DataFrame(rows)
    # Stable architecture order
    order = {"decoupled": 0, "zero_coupling": 1, "shared": 2}
    df["__order"] = df["architecture"].map(order).fillna(99)
    df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# LaTeX emission
# ----------------------------------------------------------------------------
def generate_latex_table(summary_df: pd.DataFrame) -> str:
    body_lines: list[str] = []
    for _, row in summary_df.iterrows():
        label = ARCH_LABELS.get(row["architecture"], row["architecture"].capitalize())
        ratio = f"${row['mem_ratio_mean']:.3f} \\pm {row['mem_ratio_std']:.3f}$"
        suspect = f"{100.0 * row['suspect_frac_mean']:.2f}\\%"
        w1 = f"${row['wasserstein_1_mean']:.3f}$"
        body_lines.append(f"{label} & {ratio} & {suspect} & {w1} \\\\")

    body = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Nearest-neighbour memorisation diagnostic in InceptionV3 "
        r"pool3 feature space ($L_2$). Per-cell distances were aggregated to a "
        r"median and then averaged across the 3 stratified folds. "
        r"$R = \mathrm{median}(D_{\text{synth}\to\text{train}}) / "
        r"\mathrm{median}(D_{\text{test}\to\text{train}})$: values close to 1 "
        r"are consistent with generalisation; values "
        r"$\ll 1$ indicate that synthetic samples sit anomalously close to the "
        r"training slices. Suspect~\% is the share of synthetic samples whose "
        r"distance to some train slice is lower than the minimum distance from "
        r"any test slice to the train set. $W_1$ is the Wasserstein-1 gap "
        r"between $D_{\text{synth}}$ and $D_{\text{test}}$.}" "\n"
        r"\label{tab:memorization_nn}" "\n"
        r"\begin{tabular}{lccc}" "\n"
        r"\toprule" "\n"
        r"Architecture & $R$ & Suspect~\% & $W_1$ \\" "\n"
        r"\midrule" "\n"
        + "\n".join(body_lines) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    return body


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memorization_nn",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS),
        help="Fold ids (default: 0 1 2).",
    )
    p.add_argument(
        "--architectures", nargs="+", default=list(DEFAULT_ARCHS),
        help="Architecture cells to evaluate (default: shared decoupled zero_coupling).",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16,
                   help="InceptionV3 forward batch size (default: 16, fits 8 GB VRAM).")
    p.add_argument("--chunk-size-features", type=int, default=2000,
                   help="Samples per CPU->GPU chunk during feature extraction.")
    p.add_argument("--chunk-size-nn", type=int, default=2000,
                   help="Samples per chunk during NN distance computation.")
    p.add_argument("--max-replicas", type=int, default=None,
                   help="Cap replicas per cell (default: all available).")
    p.add_argument("--cell-dir-template", type=str, default=None,
                   help="Override {results_root}/slimdiff_cr_{architecture}_fold_{fold}.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    run_memorization_nn(
        folds=args.folds,
        architectures=args.architectures,
        cache_dir=args.cache_dir,
        results_root=args.results_root,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        chunk_size_features=args.chunk_size_features,
        chunk_size_nn=args.chunk_size_nn,
        max_replicas=args.max_replicas,
        cell_dir_template=args.cell_dir_template,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
