"""Shared-vs-decoupled-vs-real ablation figure (TASK-06.2, v2).

Layout (v2, camera-ready):

* 3 rows: ``Shared``, ``Decoupled``, ``Real``.
* 7 columns: z-bins (default: 7 evenly spaced across 0..29).
* One sample per (row, column).
* All-black background.
* For each synthetic cell, the top of the panel carries the fold-aggregated
  KID ``mean ± 95% CI`` for that z-bin and architecture. Real cells carry no
  annotation.

Sample selection
----------------

Each cell prefers an epilepsy (``domain=1``) sample whose binarised mask
(at ``tau=0``) contains at least one positive pixel. The median-MSE sample
among mask-present candidates is chosen; if no mask-present sample exists
for that z-bin the median of the full epilepsy bucket is used instead (the
condition is recorded in the sidecar JSON).

Paired sampling across architectures
------------------------------------

Because TASK-03 shares ``SEED_BASE=42`` between the two architectures, the
``i``-th sample within each replica bucket corresponds to the same initial
noise ``x_T``. When both architectures expose a lesion-present candidate at
the same paired index we keep them paired so the two rows isolate the
architectural effect; otherwise we fall back to independent median-MSE per
row.

KID annotations
---------------

Reads ``kid_per_zbin_summary.csv`` written by :mod:`compute_zbin_kid`. If
the CSV is absent or ``--kid-summary-csv`` is omitted the script still runs
but skips the annotations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.diffusion.scripts.camera_ready.figure_utils import (
    IEEE_DOUBLE_COL_IN,
    SelectedSamples,
    binarise_mask,
    dump_selection_sidecar,
    load_real_samples,
    load_replicas_concat,
    overlay_mask_on_image,
    resolve_cell_dir,
    selected_to_dict,
    setup_ieee_style,
    strip_axis,
)
from src.diffusion.scripts.camera_ready.figure_utils import _mse_to_nearest  # noqa: PLC2701

logger = logging.getLogger(__name__)


# =============================================================================
# Selection helpers
# =============================================================================


def _has_lesion_pixels(mask: np.ndarray, tau: float) -> bool:
    return bool(binarise_mask(mask, tau=tau).any())


def _median_index(scores: np.ndarray) -> int:
    """Index of the element closest to the median score (stable)."""
    order = np.argsort(scores, kind="stable")
    return int(order[order.size // 2])


def _select_per_zbin(
    replicas: dict[str, np.ndarray],
    reference: np.ndarray,
    zbins: list[int],
    condition: int,
    tau: float,
) -> tuple[SelectedSamples, list[str]]:
    """Pick one sample per z-bin, preferring mask-present candidates."""
    out_imgs, out_masks, out_zb, out_scores, out_idx, out_ids = [], [], [], [], [], []
    notes: list[str] = []
    for zb in zbins:
        bucket = np.where(
            (replicas["zbin"] == zb) & (replicas["domain"] == condition)
        )[0]
        if bucket.size == 0:
            notes.append(f"zbin={zb}: empty bucket (skipped)")
            continue
        imgs = replicas["images"][bucket]
        masks = replicas["masks"][bucket]

        lesion_local = np.array(
            [_has_lesion_pixels(masks[i], tau) for i in range(masks.shape[0])]
        )
        if lesion_local.any():
            pool_local = np.where(lesion_local)[0]
            note = "mask-present"
        else:
            pool_local = np.arange(bucket.size)
            note = "no-mask-available:fallback"
        scores_all = _mse_to_nearest(imgs[pool_local], reference)
        pick_local = pool_local[_median_index(scores_all)]
        g = int(bucket[pick_local])

        out_imgs.append(replicas["images"][g].astype(np.float32))
        out_masks.append(replicas["masks"][g].astype(np.float32))
        out_zb.append(int(zb))
        out_scores.append(float(scores_all.min() if note == "mask-present" else scores_all[_median_index(scores_all)]))
        out_idx.append(g)
        out_ids.append(f"rep{int(replicas['replica_idx'][g])}_idx{g}")
        notes.append(f"zbin={zb}: {note}")

    sel = SelectedSamples(
        images=np.stack(out_imgs),
        masks=np.stack(out_masks),
        zbins=np.asarray(out_zb, dtype=np.int32),
        scores=np.asarray(out_scores, dtype=np.float32),
        indices=np.asarray(out_idx, dtype=np.int64),
        source_ids=out_ids,
    )
    return sel, notes


def _select_real_per_zbin(
    cache_dir: Path,
    fold: int,
    zbins: list[int],
    condition: int,
    rng: np.random.Generator,
    tau: float,
) -> tuple[SelectedSamples, list[str]]:
    """One real sample per z-bin, preferring mask-present."""
    out_imgs, out_masks, out_zb, out_ids = [], [], [], []
    notes: list[str] = []
    for zb in zbins:
        # Prefer lesion-present; fall back to any.
        try:
            block = load_real_samples(
                cache_dir=cache_dir,
                fold=fold,
                zbins=[zb],
                condition=condition,
                n_samples=8,
                rng=rng,
                require_lesion=True,
            )
            lesion_present = [
                _has_lesion_pixels(block.masks[i], tau)
                for i in range(block.masks.shape[0])
            ]
            if any(lesion_present):
                i0 = lesion_present.index(True)
                out_imgs.append(block.images[i0])
                out_masks.append(block.masks[i0])
                out_zb.append(zb)
                out_ids.append(block.source_ids[i0] if block.source_ids else f"real_z{zb}")
                notes.append(f"zbin={zb}: mask-present")
                continue
        except Exception:  # noqa: BLE001
            pass
        try:
            block = load_real_samples(
                cache_dir=cache_dir,
                fold=fold,
                zbins=[zb],
                condition=condition,
                n_samples=1,
                rng=rng,
                require_lesion=False,
            )
            out_imgs.append(block.images[0])
            out_masks.append(block.masks[0])
            out_zb.append(zb)
            out_ids.append(block.source_ids[0] if block.source_ids else f"real_z{zb}")
            notes.append(f"zbin={zb}: no-mask-available:fallback")
        except Exception as exc:  # noqa: BLE001
            logger.warning("No real sample for zbin=%d: %s", zb, exc)
            notes.append(f"zbin={zb}: missing")
    sel = SelectedSamples(
        images=np.stack(out_imgs) if out_imgs else np.empty((0,), dtype=np.float32),
        masks=np.stack(out_masks) if out_masks else np.empty((0,), dtype=np.float32),
        zbins=np.asarray(out_zb, dtype=np.int32),
        scores=np.full(len(out_imgs), np.nan, dtype=np.float32),
        indices=np.arange(len(out_imgs), dtype=np.int64),
        source_ids=out_ids,
    )
    return sel, notes


# =============================================================================
# KID annotation loader
# =============================================================================


def load_kid_summary(csv: Path | None) -> pd.DataFrame | None:
    if csv is None:
        return None
    if not csv.exists():
        logger.warning("KID summary CSV not found at %s; skipping annotations.", csv)
        return None
    df = pd.read_csv(csv)
    required = {"architecture", "zbin", "kid_mean", "kid_ci95_half"}
    if not required.issubset(df.columns):
        logger.warning(
            "KID summary CSV %s missing required columns %s; skipping.",
            csv,
            required - set(df.columns),
        )
        return None
    return df


def _format_kid(df: pd.DataFrame | None, arch: str, zbin: int) -> str | None:
    if df is None:
        return None
    row = df[(df["architecture"] == arch) & (df["zbin"] == zbin)]
    if row.empty:
        return None
    mean = float(row["kid_mean"].iloc[0])
    ci = float(row["kid_ci95_half"].iloc[0])
    if not np.isfinite(mean):
        return None
    return rf"KID$=${mean * 1e3:.1f}$\pm${ci * 1e3:.1f} $\times10^{{-3}}$"


# =============================================================================
# Rendering
# =============================================================================


BLACK = "#000000"


def render_ablation_grid(
    shared_sel: SelectedSamples,
    decoupled_sel: SelectedSamples,
    real_sel: SelectedSamples,
    zbins: list[int],
    output_path: Path,
    kid_summary: pd.DataFrame | None = None,
    fig_width: float = IEEE_DOUBLE_COL_IN,
    overlay_alpha: float = 0.5,
    overlay_color: tuple[int, int, int] = (255, 0, 0),
    tau: float = 0.0,
    dpi: int = 300,
) -> plt.Figure:
    n_cols = len(zbins)
    n_rows = 3  # shared, decoupled, real
    cell_w = fig_width / (n_cols + 0.25)
    # Leave headroom for the KID annotations on top of each synthetic cell.
    fig_h = min(n_rows * cell_w * 1.15 + 0.5, 9.0)

    fig = plt.figure(figsize=(fig_width, fig_h), facecolor=BLACK)
    gs = fig.add_gridspec(
        nrows=n_rows + 1,
        ncols=n_cols + 1,
        width_ratios=[0.07] + [1.0] * n_cols,
        height_ratios=[0.18] + [1.0] * n_rows,
        left=0.045,
        right=0.998,
        top=0.995,
        bottom=0.005,
        wspace=0.05,
        hspace=0.06,
    )

    # Column headers (z-bin labels).
    for ci, zb in enumerate(zbins):
        ax = fig.add_subplot(gs[0, 1 + ci])
        ax.set_facecolor(BLACK)
        ax.text(
            0.5,
            0.25,
            f"$z$-bin {zb}",
            ha="center",
            va="center",
            color="white",
            fontsize=7.5,
            fontweight="bold",
        )
        strip_axis(ax)

    def _find(sel: SelectedSamples, zb: int) -> int | None:
        hits = np.where(sel.zbins == zb)[0]
        return int(hits[0]) if hits.size else None

    blocks = (
        ("Shared", shared_sel, "shared", True),
        ("Decoupled", decoupled_sel, "decoupled", True),
        ("Real", real_sel, None, False),
    )

    for ri, (row_label, sel, arch_key, annotate) in enumerate(blocks):
        # Row label column.
        ax_lab = fig.add_subplot(gs[1 + ri, 0])
        ax_lab.set_facecolor(BLACK)
        ax_lab.text(
            0.5,
            0.5,
            row_label,
            ha="center",
            va="center",
            rotation=90,
            color="white",
            fontsize=8,
            fontweight="bold",
        )
        strip_axis(ax_lab)

        for ci, zb in enumerate(zbins):
            ax = fig.add_subplot(gs[1 + ri, 1 + ci])
            ax.set_facecolor(BLACK)
            strip_axis(ax)
            idx = _find(sel, zb)
            if idx is None:
                ax.text(
                    0.5,
                    0.5,
                    "\u2014",
                    ha="center",
                    va="center",
                    color="#666",
                    fontsize=10,
                )
                continue
            img = sel.images[idx]
            mask = sel.masks[idx]
            if _has_lesion_pixels(mask, tau):
                rgb = overlay_mask_on_image(
                    img, mask, alpha=overlay_alpha, color=overlay_color, tau=tau
                )
                ax.imshow(rgb, interpolation="nearest")
            else:
                # Show FLAIR on a black canvas without overlay.
                disp = (np.asarray(img, dtype=np.float32) + 1.0) * 0.5
                disp = np.clip(disp, 0.0, 1.0)
                ax.imshow(disp, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

            if annotate and arch_key is not None:
                label = _format_kid(kid_summary, arch_key, zb)
                if label is not None:
                    ax.text(
                        0.5,
                        1.01,
                        label,
                        ha="center",
                        va="bottom",
                        color="white",
                        fontsize=5.5,
                        transform=ax.transAxes,
                    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    logger.info("Saved ablation comparison figure to %s", output_path)
    return fig


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the shared-vs-decoupled-vs-real ablation figure."
    )
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--zbins",
        type=int,
        nargs="+",
        default=[3, 8, 12, 15, 18, 22, 27],
        help="Seven z-bins by default; pass fewer/more to override.",
    )
    p.add_argument("--condition", type=int, default=1, help="0=control, 1=epilepsy")
    p.add_argument(
        "--kid-summary-csv",
        type=Path,
        default=None,
        help="Path to kid_per_zbin_summary.csv (from compute_zbin_kid).",
    )
    p.add_argument("--overlay-alpha", type=float, default=0.5)
    p.add_argument(
        "--overlay-color", type=int, nargs=3, default=[255, 0, 0], metavar=("R", "G", "B")
    )
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--format", choices=("pdf", "png"), default="pdf")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-replicas", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args() if argv is None else _parse_args_from_list(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    setup_ieee_style()
    rng = np.random.default_rng(args.seed)

    output_path = args.output
    if output_path.suffix.lstrip(".") != args.format:
        output_path = output_path.with_suffix(f".{args.format}")

    shared_dir = resolve_cell_dir(args.results_root, "shared", args.fold)
    decoupled_dir = resolve_cell_dir(args.results_root, "decoupled", args.fold)
    shared_reps = load_replicas_concat(shared_dir, max_replicas=args.max_replicas)
    decoupled_reps = load_replicas_concat(decoupled_dir, max_replicas=args.max_replicas)

    # Reference pool drawn from epilepsy-condition real slices on the fold.
    ref_block = load_real_samples(
        cache_dir=args.cache_dir,
        fold=args.fold,
        zbins=args.zbins,
        condition=args.condition,
        n_samples=8,
        rng=rng,
    )
    reference = ref_block.images

    shared_sel, shared_notes = _select_per_zbin(
        shared_reps, reference, args.zbins, args.condition, args.tau
    )
    decoupled_sel, decoupled_notes = _select_per_zbin(
        decoupled_reps, reference, args.zbins, args.condition, args.tau
    )
    real_sel, real_notes = _select_real_per_zbin(
        cache_dir=args.cache_dir,
        fold=args.fold,
        zbins=args.zbins,
        condition=args.condition,
        rng=np.random.default_rng(args.seed + 7),
        tau=args.tau,
    )

    kid_summary = load_kid_summary(args.kid_summary_csv)
    render_ablation_grid(
        shared_sel=shared_sel,
        decoupled_sel=decoupled_sel,
        real_sel=real_sel,
        zbins=args.zbins,
        output_path=output_path,
        kid_summary=kid_summary,
        overlay_alpha=args.overlay_alpha,
        overlay_color=tuple(args.overlay_color),
        tau=args.tau,
        dpi=args.dpi,
    )

    dump_selection_sidecar(
        output_path,
        {
            "figure": "ablation_comparison_v2",
            "layout": "rows=(shared,decoupled,real); cols=zbins",
            "seed": args.seed,
            "fold": args.fold,
            "zbins": list(args.zbins),
            "condition": args.condition,
            "kid_summary_csv": str(args.kid_summary_csv) if args.kid_summary_csv else None,
            "shared": selected_to_dict(shared_sel),
            "decoupled": selected_to_dict(decoupled_sel),
            "real": selected_to_dict(real_sel),
            "notes": {
                "shared": shared_notes,
                "decoupled": decoupled_notes,
                "real": real_notes,
            },
        },
    )
    plt.close("all")
    return output_path


def _parse_args_from_list(argv: list[str]) -> argparse.Namespace:
    import sys as _sys

    old = _sys.argv
    try:
        _sys.argv = ["ablation_comparison_figure", *argv]
        return _parse_args()
    finally:
        _sys.argv = old


if __name__ == "__main__":
    main()
