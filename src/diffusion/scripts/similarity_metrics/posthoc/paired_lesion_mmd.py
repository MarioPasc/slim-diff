"""Paired-lesion joint MMD (MMD-PL) — image-given-mask coupling diagnostic.

Whereas

* ``KID`` and ``LPIPS`` test the **image** marginal (does the FLAIR look real?),
* ``MMD-MF`` tests the **mask** marginal (do the masks have realistic
  morphology?),

MMD-PL tests the **image-given-mask joint** (do the pixels under the mask
actually correspond to lesions, or is the mask a plausible shape dropped on
unrelated tissue?). Every feature in the MMD-PL vector is a function of both
the FLAIR slice and the mask polygon — none of them duplicates the pure-
morphology features used by MMD-MF.

Per-lesion features (5-D)
-------------------------
For each connected mask component (>= ``min_lesion_size_px`` pixels):

1. ``delta``      = mean(FLAIR | mask) − mean(FLAIR | perilesion ring).
   The ring is a 5-pixel dilation of the mask minus the mask itself,
   restricted to brain tissue.
2. ``z_in``       = (mean(FLAIR | mask) − mean(FLAIR | brain)) / std(FLAIR | brain).
   z-scored hyperintensity of the patch under the mask, relative to brain
   tissue on the same slice.
3. ``sigma_norm`` = std(FLAIR | mask) / std(FLAIR | brain).
   Normalised intra-lesion intensity variance (texture proxy).
4. ``edge_align`` = mean(|grad(FLAIR)|) along the mask boundary, normalised by
   mean(|grad(FLAIR)|) on brain tissue. Does the FLAIR have an edge where the
   mask draws one?
5. ``frac_bright`` = #{x ∈ mask : FLAIR(x) > mean_brain + std_brain} / #mask.
   Fraction of "lesion" pixels that are above the brain hyperintensity
   threshold. A randomly placed plausible mask will have this near the
   baseline rate; a correctly placed lesion mask will be near 1.

All features are computed per connected component, not per slice. The
features for real and synth populations form distributions; we report:

* per-feature mean/median/std and Wasserstein-1 (real vs synth), so we see
  *which axis* a model breaks on;
* joint polynomial-kernel MMD on the 5-D vector (kernel: ``(x·y/D + 1)^3``,
  the same family used by KID and MMD-MF for consistency);
* falsification scalar ``mean(frac_bright)`` — a memorisation-free,
  domain-grounded sanity check.

Outputs (under ``--output-dir``):

* ``paired_lesion_mmd.csv``         — per-cell MMD + per-feature stats
* ``paired_lesion_features.csv``    — per-cell summary of the 5-D feature
  distributions (mean, std, p5, p50, p95, n_lesions)
* ``paired_lesion_mmd.json``        — full payload
* ``tables/table_paired_lesion_mmd.tex`` — supplementary LaTeX table
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    sobel,
)
from scipy.ndimage import label as scipy_label

from src.diffusion.data.kfold import KFoldManager
from src.diffusion.scripts.similarity_metrics.data.fold_loaders import (
    _load_real_from_fold_csv,
    _load_replicas,
    resolve_cell_dir,
)

logger = logging.getLogger(__name__)

DEFAULT_ARCHS: tuple[str, ...] = ("shared", "decoupled", "zero_coupling")
DEFAULT_FOLDS: tuple[int, ...] = (0, 1, 2)
ARCH_LABELS: dict[str, str] = {
    "shared": "Shared (ours)",
    "decoupled": "Decoupled",
    "zero_coupling": "Zero-coupling",
}

FEATURE_NAMES: tuple[str, ...] = (
    "delta",
    "z_in",
    "sigma_norm",
    "edge_align",
    "frac_bright",
)


# ----------------------------------------------------------------------------
# Per-slice feature extraction
# ----------------------------------------------------------------------------
def _brain_mask(image: NDArray[np.float32], background: float = -0.95) -> NDArray[np.bool_]:
    """Return a coarse brain-tissue mask from a slice.

    FLAIR background outside the brain sits at -1 by convention (see
    ``transforms.intensity_norm``). We treat any pixel above ``background``
    as brain. A 1-pixel erosion removes the bright edge ring.
    """
    return binary_erosion(image > background, iterations=1)


def _binarise_mask(mask: NDArray[np.float32], tau: float) -> NDArray[np.bool_]:
    return mask > tau


def _per_lesion_features(
    image: NDArray[np.float32],
    binary_mask: NDArray[np.bool_],
    brain_mask: NDArray[np.bool_],
    min_lesion_size_px: int = 5,
    ring_radius_px: int = 5,
) -> list[dict[str, float]]:
    """Compute 5-D per-component image-given-mask features for one slice."""
    if not binary_mask.any() or not brain_mask.any():
        return []

    brain_pixels = image[brain_mask]
    mu_brain = float(np.mean(brain_pixels))
    sigma_brain = float(np.std(brain_pixels))
    if sigma_brain <= 1e-6:
        return []

    grad_x = sobel(image, axis=0, mode="reflect")
    grad_y = sobel(image, axis=1, mode="reflect")
    grad_mag = np.hypot(grad_x, grad_y).astype(np.float32)
    mean_grad_brain = float(np.mean(grad_mag[brain_mask]))
    if mean_grad_brain <= 1e-6:
        mean_grad_brain = 1e-6

    bright_threshold = mu_brain + sigma_brain

    labelled, n_components = scipy_label(binary_mask)
    out: list[dict[str, float]] = []
    for comp_id in range(1, int(n_components) + 1):
        comp = labelled == comp_id
        size = int(comp.sum())
        if size < min_lesion_size_px:
            continue

        ring = binary_dilation(comp, iterations=ring_radius_px) & ~comp & brain_mask
        # If the lesion is at the brain boundary the ring may be empty -- skip.
        if not ring.any():
            continue

        comp_brain = comp & brain_mask
        if not comp_brain.any():
            continue

        mu_in = float(np.mean(image[comp_brain]))
        sigma_in = float(np.std(image[comp_brain]))
        mu_ring = float(np.mean(image[ring]))

        # Boundary = 1-pixel outer rim of the component.
        boundary = binary_dilation(comp, iterations=1) & ~comp
        boundary = boundary & brain_mask
        if boundary.any():
            edge_align = float(np.mean(grad_mag[boundary])) / mean_grad_brain
        else:
            edge_align = float("nan")

        frac_bright = float(np.mean(image[comp_brain] > bright_threshold))

        out.append(
            {
                "delta": mu_in - mu_ring,
                "z_in": (mu_in - mu_brain) / sigma_brain,
                "sigma_norm": sigma_in / sigma_brain if sigma_brain > 0 else float("nan"),
                "edge_align": edge_align,
                "frac_bright": frac_bright,
            }
        )

    return out


def _stack_features(rows: list[dict[str, float]]) -> NDArray[np.float64]:
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.asarray(
        [[r[name] for name in FEATURE_NAMES] for r in rows], dtype=np.float64
    )


def _extract_population(
    images: NDArray[np.float32],
    masks: NDArray[np.float32],
    tau: float,
    min_lesion_size_px: int,
    ring_radius_px: int,
) -> NDArray[np.float64]:
    """Iterate slices and concatenate per-lesion feature rows."""
    rows: list[dict[str, float]] = []
    for i in range(images.shape[0]):
        img = np.asarray(images[i], dtype=np.float32)
        mask = np.asarray(masks[i], dtype=np.float32)
        if not np.any(mask > tau):
            continue
        binary = _binarise_mask(mask, tau)
        brain = _brain_mask(img)
        rows.extend(
            _per_lesion_features(
                image=img,
                binary_mask=binary,
                brain_mask=brain,
                min_lesion_size_px=min_lesion_size_px,
                ring_radius_px=ring_radius_px,
            )
        )
    feats = _stack_features(rows)
    if feats.size == 0:
        return feats
    # Drop rows with any NaN/Inf (edge cases at brain boundary).
    finite = np.all(np.isfinite(feats), axis=1)
    return feats[finite]


# ----------------------------------------------------------------------------
# Polynomial-kernel MMD (matches KID / MMD-MF kernel choice)
# ----------------------------------------------------------------------------
def _polynomial_kernel(
    x: NDArray[np.float64], y: NDArray[np.float64], degree: int = 3
) -> NDArray[np.float64]:
    """``(x·y / D + 1)^d`` — the kernel family used by KID and MMD-MF."""
    d = x.shape[1]
    return (x @ y.T / float(d) + 1.0) ** degree


def _mmd2_unbiased(
    x: NDArray[np.float64], y: NDArray[np.float64], degree: int = 3
) -> float:
    """Unbiased estimator of squared MMD between two empirical distributions."""
    m = x.shape[0]
    n = y.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    kxx = _polynomial_kernel(x, x, degree=degree)
    kyy = _polynomial_kernel(y, y, degree=degree)
    kxy = _polynomial_kernel(x, y, degree=degree)
    # Drop diagonal terms for unbiased estimate.
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    term_xx = kxx.sum() / (m * (m - 1))
    term_yy = kyy.sum() / (n * (n - 1))
    term_xy = kxy.sum() * 2.0 / (m * n)
    return float(term_xx + term_yy - term_xy)


def _subset_mmd(
    real_feats: NDArray[np.float64],
    synth_feats: NDArray[np.float64],
    subset_size: int = 500,
    num_subsets: int = 100,
    degree: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, int]:
    """Average squared-MMD over random subsets (mirrors the MMD-MF estimator)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_real = real_feats.shape[0]
    n_synth = synth_feats.shape[0]
    if n_real == 0 or n_synth == 0:
        return float("nan"), float("nan"), 0
    effective_size = max(2, min(subset_size, n_real, n_synth))
    values: list[float] = []
    for _ in range(num_subsets):
        ri = rng.choice(n_real, size=effective_size, replace=False)
        si = rng.choice(n_synth, size=effective_size, replace=False)
        v = _mmd2_unbiased(real_feats[ri], synth_feats[si], degree=degree)
        if np.isfinite(v):
            values.append(v)
    if not values:
        return float("nan"), float("nan"), 0
    return float(np.mean(values)), float(np.std(values)), effective_size


def _wasserstein_1(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(a, b))


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def _summarise_feature(values: NDArray[np.float64]) -> dict[str, float]:
    if values.size == 0:
        return {k: float("nan") for k in ("mean", "std", "median", "p5", "p95")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def _standardise(
    real: NDArray[np.float64], synth: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Z-score both feature matrices using the real distribution's stats."""
    if real.size == 0 or synth.size == 0:
        return real, synth
    mu = real.mean(axis=0, keepdims=True)
    sigma = real.std(axis=0, keepdims=True)
    sigma = np.where(sigma > 1e-8, sigma, 1.0)
    return (real - mu) / sigma, (synth - mu) / sigma


def run_paired_lesion_mmd(
    folds: Iterable[int],
    architectures: Iterable[str],
    cache_dir: Path,
    results_root: Path,
    output_dir: Path,
    tau: float = 0.0,
    min_lesion_size_px: int = 5,
    ring_radius_px: int = 5,
    subset_size: int = 500,
    num_subsets: int = 100,
    degree: int = 3,
    seed: int = 42,
    max_replicas: int | None = None,
    cell_dir_template: str | None = None,
) -> pd.DataFrame:
    """Run MMD-PL across the (fold, architecture) grid and write outputs."""
    cache_dir = Path(cache_dir)
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    manager = KFoldManager.from_meta_json(cache_dir / "folds" / "folds_meta.json")
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    feature_rows: list[dict] = []

    for fold in folds:
        fold_dir = manager.get_cache_dir_for_fold(fold)
        test_csv = fold_dir / "test.csv"
        logger.info("--- fold=%d ---", fold)
        logger.info("Loading real (test) slices from %s", test_csv)
        real_imgs, real_masks, _ = _load_real_from_fold_csv(test_csv, cache_dir)
        logger.info(
            "fold=%d real n_slices=%d -- extracting features (tau=%.2f, min=%dpx)",
            fold, real_imgs.shape[0], tau, min_lesion_size_px,
        )
        real_feats = _extract_population(
            real_imgs, real_masks, tau=tau,
            min_lesion_size_px=min_lesion_size_px,
            ring_radius_px=ring_radius_px,
        )
        logger.info("fold=%d real n_lesions=%d", fold, real_feats.shape[0])
        del real_imgs, real_masks

        # Save the per-cell summary for the real reference.
        for j, name in enumerate(FEATURE_NAMES):
            stats = _summarise_feature(real_feats[:, j])
            feature_rows.append(
                {
                    "fold": int(fold),
                    "architecture": "real",
                    "feature": name,
                    "n": int(real_feats.shape[0]),
                    **stats,
                }
            )

        for arch in architectures:
            cell_dir = resolve_cell_dir(
                results_root=results_root,
                architecture=arch,
                fold=fold,
                cell_dir_template=cell_dir_template,
            )
            logger.info("[fold=%d arch=%s] loading synth from %s", fold, arch, cell_dir)
            synth_imgs, synth_masks, _, _, n_reps = _load_replicas(
                cell_dir / "replicas", max_replicas=max_replicas
            )
            n_synth_slices = int(synth_imgs.shape[0])
            logger.info(
                "[fold=%d arch=%s] extracting features over %d slices",
                fold, arch, n_synth_slices,
            )
            # NOTE: do NOT cast the whole array to float32 here -- with 108k
            # 160x160 slices the upfront cast materialises ~11 GB on top of
            # the fp16 buffer and OOMs the 31 GB workstation. The per-slice
            # cast inside _extract_population is enough.
            synth_feats = _extract_population(
                synth_imgs,
                synth_masks,
                tau=tau,
                min_lesion_size_px=min_lesion_size_px,
                ring_radius_px=ring_radius_px,
            )
            logger.info("[fold=%d arch=%s] synth n_lesions=%d", fold, arch, synth_feats.shape[0])
            del synth_imgs, synth_masks

            # Per-feature stats and Wasserstein-1 vs real.
            per_feature_w1: dict[str, float] = {}
            for j, name in enumerate(FEATURE_NAMES):
                stats = _summarise_feature(synth_feats[:, j])
                feature_rows.append(
                    {
                        "fold": int(fold),
                        "architecture": arch,
                        "feature": name,
                        "n": int(synth_feats.shape[0]),
                        **stats,
                    }
                )
                per_feature_w1[name] = _wasserstein_1(
                    real_feats[:, j], synth_feats[:, j]
                )

            # Joint MMD on z-scored features (so all five dims contribute on
            # comparable scales -- otherwise sigma_norm and frac_bright would
            # be drowned out by edge_align magnitudes).
            real_z, synth_z = _standardise(real_feats, synth_feats)
            mmd_mean, mmd_std, eff_subset = _subset_mmd(
                real_z, synth_z,
                subset_size=subset_size,
                num_subsets=num_subsets,
                degree=degree,
                rng=rng,
            )

            falsification = float(
                np.mean(synth_feats[:, FEATURE_NAMES.index("frac_bright")])
                if synth_feats.size
                else float("nan")
            )
            real_falsification = float(
                np.mean(real_feats[:, FEATURE_NAMES.index("frac_bright")])
                if real_feats.size
                else float("nan")
            )

            row = {
                "fold": int(fold),
                "architecture": arch,
                "n_real_lesions": int(real_feats.shape[0]),
                "n_synth_lesions": int(synth_feats.shape[0]),
                "n_synth_slices": n_synth_slices,
                "n_replicas": int(n_reps),
                "mmd_pl_mean": mmd_mean,
                "mmd_pl_std": mmd_std,
                "subset_size": int(eff_subset),
                "num_subsets": int(num_subsets),
                "frac_bright_synth_mean": falsification,
                "frac_bright_real_mean": real_falsification,
            }
            for name in FEATURE_NAMES:
                row[f"w1_{name}"] = per_feature_w1[name]
            rows.append(row)

            logger.info(
                "[fold=%d arch=%s] n_synth_lesions=%d MMD-PL=%.4f +/- %.4f "
                "frac_bright synth=%.3f real=%.3f W1[delta]=%.4f",
                fold, arch,
                synth_feats.shape[0],
                row["mmd_pl_mean"], row["mmd_pl_std"],
                row["frac_bright_synth_mean"], row["frac_bright_real_mean"],
                row["w1_delta"],
            )

            del synth_feats

        del real_feats

    df = pd.DataFrame(rows)
    csv_path = output_dir / "paired_lesion_mmd.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(df))

    features_df = pd.DataFrame(feature_rows)
    features_path = output_dir / "paired_lesion_features.csv"
    features_df.to_csv(features_path, index=False)
    logger.info("Wrote %s (%d rows)", features_path, len(features_df))

    summary_df = aggregate_across_folds(df)
    summary_path = output_dir / "paired_lesion_mmd_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Wrote %s (%d rows)", summary_path, len(summary_df))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tau": tau,
        "min_lesion_size_px": min_lesion_size_px,
        "ring_radius_px": ring_radius_px,
        "subset_size": subset_size,
        "num_subsets": num_subsets,
        "degree": degree,
        "seed": seed,
        "feature_names": list(FEATURE_NAMES),
        "kernel": f"polynomial (x.y/D + 1)^{degree}",
        "standardisation": "z-score on real reference distribution per fold",
        "per_cell": df.to_dict(orient="records"),
        "per_cell_per_feature": features_df.to_dict(orient="records"),
        "across_folds": summary_df.to_dict(orient="records"),
    }
    json_path = output_dir / "paired_lesion_mmd.json"
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", json_path)

    table_path = output_dir / "tables" / "table_paired_lesion_mmd.tex"
    table_path.write_text(generate_latex_table(summary_df))
    logger.info("Wrote %s", table_path)

    return df


def aggregate_across_folds(per_cell: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-cell rows into one row per architecture (mean across folds)."""
    out: list[dict] = []
    for arch, sub in per_cell.groupby("architecture"):
        record = {
            "architecture": str(arch),
            "n_folds": int(len(sub)),
            "n_synth_lesions_total": int(sub["n_synth_lesions"].sum()),
            "mmd_pl_mean_across_folds": float(np.mean(sub["mmd_pl_mean"])),
            "mmd_pl_std_across_folds": float(np.std(sub["mmd_pl_mean"], ddof=0)),
            "frac_bright_synth_mean": float(np.mean(sub["frac_bright_synth_mean"])),
            "frac_bright_synth_std": float(np.std(sub["frac_bright_synth_mean"], ddof=0)),
            "frac_bright_real_mean": float(np.mean(sub["frac_bright_real_mean"])),
        }
        for name in FEATURE_NAMES:
            record[f"w1_{name}_mean"] = float(np.mean(sub[f"w1_{name}"]))
            record[f"w1_{name}_std"] = float(np.std(sub[f"w1_{name}"], ddof=0))
        out.append(record)
    df = pd.DataFrame(out)
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
        mmd = f"${row['mmd_pl_mean_across_folds']:.3f} \\pm {row['mmd_pl_std_across_folds']:.3f}$"
        fb_synth = f"{100.0 * row['frac_bright_synth_mean']:.1f}\\%"
        fb_real = f"{100.0 * row['frac_bright_real_mean']:.1f}\\%"
        w1_delta = f"${row['w1_delta_mean']:.3f}$"
        w1_z = f"${row['w1_z_in_mean']:.3f}$"
        body_lines.append(
            f"{label} & {mmd} & {fb_synth} ({fb_real}) & {w1_delta} & {w1_z} \\\\"
        )

    body = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{MMD-PL: paired-lesion image-given-mask coupling. Every "
        r"feature requires both the FLAIR slice and the mask polygon; none "
        r"overlaps the pure-morphology axes covered by MMD-MF. ``Bright frac.\ "
        r"(real)'' is the share of synthetic lesion pixels above the brain "
        r"hyperintensity threshold, with the real baseline in parentheses. "
        r"$W_1[\Delta]$ and $W_1[z_{\text{in}}]$ are the per-feature "
        r"Wasserstein-1 gaps to the real distribution. Mean $\pm$ std across "
        r"3 stratified folds. Lower is better for MMD-PL and the $W_1$ "
        r"columns; closer to the real baseline is better for the bright "
        r"fraction.}" "\n"
        r"\label{tab:mmd_pl}" "\n"
        r"\begin{tabular}{lcccc}" "\n"
        r"\toprule" "\n"
        r"Architecture & MMD-PL $\downarrow$ & "
        r"Bright frac. (real) & $W_1[\Delta]$ & $W_1[z_{\text{in}}]$ \\" "\n"
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
        prog="paired_lesion_mmd",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    p.add_argument(
        "--architectures", nargs="+", default=list(DEFAULT_ARCHS),
        help="Architectures to evaluate (default: shared decoupled zero_coupling).",
    )
    p.add_argument("--tau", type=float, default=0.0,
                   help="Mask binarisation threshold (default: 0.0).")
    p.add_argument("--min-lesion-size-px", type=int, default=5)
    p.add_argument("--ring-radius-px", type=int, default=5)
    p.add_argument("--subset-size", type=int, default=500)
    p.add_argument("--num-subsets", type=int, default=100)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-replicas", type=int, default=None)
    p.add_argument("--cell-dir-template", type=str, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_paired_lesion_mmd(
        folds=args.folds,
        architectures=args.architectures,
        cache_dir=args.cache_dir,
        results_root=args.results_root,
        output_dir=args.output_dir,
        tau=args.tau,
        min_lesion_size_px=args.min_lesion_size_px,
        ring_radius_px=args.ring_radius_px,
        subset_size=args.subset_size,
        num_subsets=args.num_subsets,
        degree=args.degree,
        seed=args.seed,
        max_replicas=args.max_replicas,
        cell_dir_template=args.cell_dir_template,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
