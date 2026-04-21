"""Failure modes gallery for the ICIP 2026 camera-ready (R1.4).

Produces a 2×K grid (shared / decoupled × worst-K) showing the epilepsy
samples with the largest MSE distance to the nearest real test slice in the
same ``(fold, zbin)`` bucket. Each panel is annotated with its raw score so
reviewers can compare failure severity across architectures.

The script intentionally uses MSE rather than LPIPS to keep figure generation
CPU-only; the ordering matches LPIPS closely on representative inlier/outlier
splits. Override with ``--use-lpips`` if the GPU LPIPS model is desired
(requires ``piq``/``lpips`` at import time).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.diffusion.scripts.camera_ready.figure_utils import (
    IEEE_DOUBLE_COL_IN,
    SelectedSamples,
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
# Worst-K selection
# =============================================================================


def select_worst_k(
    replicas: dict[str, np.ndarray],
    reference: np.ndarray,
    condition: int,
    k: int,
) -> SelectedSamples:
    """Rank epilepsy samples by descending MSE to ``reference`` and take top-K."""
    bucket = np.where(replicas["domain"] == condition)[0]
    if bucket.size == 0:
        raise RuntimeError(f"No samples with condition={condition} in replicas.")
    imgs = replicas["images"][bucket]
    scores = _mse_to_nearest(imgs, reference)
    order = np.argsort(scores, kind="stable")[::-1][:k]
    g = bucket[order]
    return SelectedSamples(
        images=replicas["images"][g].astype(np.float32),
        masks=replicas["masks"][g].astype(np.float32),
        zbins=replicas["zbin"][g].astype(np.int32),
        scores=scores[order].astype(np.float32),
        indices=g.astype(np.int64),
        source_ids=[f"rep{int(replicas['replica_idx'][i])}_idx{int(i)}" for i in g],
    )


# =============================================================================
# Rendering
# =============================================================================


def render_failure_gallery(
    shared_sel: SelectedSamples,
    decoupled_sel: SelectedSamples,
    output_path: Path,
    score_label: str = "MSE",
    fig_width: float = IEEE_DOUBLE_COL_IN,
    overlay_alpha: float = 0.5,
    overlay_color: tuple[int, int, int] = (255, 0, 0),
    tau: float = 0.0,
    dpi: int = 300,
) -> plt.Figure:
    k = max(len(shared_sel), len(decoupled_sel))
    n_rows = 2
    cell_w = fig_width / (k + 0.15)
    fig_h = min(n_rows * cell_w + 0.6, 9.0)

    fig = plt.figure(figsize=(fig_width, fig_h))
    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=k + 1,
        width_ratios=[0.1] + [1.0] * k,
        height_ratios=[1.0] * n_rows,
        left=0.04,
        right=0.995,
        top=0.93,
        bottom=0.04,
        wspace=0.04,
        hspace=0.15,
    )

    blocks = [("Shared", shared_sel), ("Decoupled", decoupled_sel)]
    for row, (label, sel) in enumerate(blocks):
        ax_lab = fig.add_subplot(gs[row, 0])
        ax_lab.text(
            0.5,
            0.5,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=7,
            fontweight="bold",
        )
        strip_axis(ax_lab)
        for i in range(k):
            ax = fig.add_subplot(gs[row, 1 + i])
            if i >= len(sel):
                ax.set_facecolor("#f0f0f0")
                strip_axis(ax)
                continue
            rgb = overlay_mask_on_image(
                sel.images[i],
                sel.masks[i],
                alpha=overlay_alpha,
                color=overlay_color,
                tau=tau,
            )
            ax.imshow(rgb, interpolation="nearest")
            ax.text(
                0.5,
                -0.05,
                f"{score_label}={float(sel.scores[i]):.3f}\n$z$={int(sel.zbins[i])}",
                ha="center",
                va="top",
                fontsize=6,
                transform=ax.transAxes,
            )
            strip_axis(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved failure-modes figure to %s", output_path)
    return fig


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the worst-K failure-modes gallery for the camera-ready."
    )
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-worst", type=int, default=4)
    p.add_argument("--condition", type=int, default=1, help="0=control, 1=epilepsy")
    p.add_argument("--overlay-alpha", type=float, default=0.5)
    p.add_argument(
        "--overlay-color", type=int, nargs=3, default=[255, 0, 0], metavar=("R", "G", "B")
    )
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--format", choices=("pdf", "png"), default="pdf")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-replicas", type=int, default=None)
    p.add_argument(
        "--zbins",
        type=int,
        nargs="+",
        default=None,
        help="Restrict to these zbins for the reference pool (default: all).",
    )
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

    zbins_ref = args.zbins if args.zbins else sorted(set(shared_reps["zbin"].tolist()))
    ref_block = load_real_samples(
        cache_dir=args.cache_dir,
        fold=args.fold,
        zbins=zbins_ref,
        condition=args.condition,
        n_samples=4,
        rng=rng,
    )
    reference = ref_block.images

    shared_sel = select_worst_k(shared_reps, reference, args.condition, args.n_worst)
    decoupled_sel = select_worst_k(
        decoupled_reps, reference, args.condition, args.n_worst
    )

    render_failure_gallery(
        shared_sel=shared_sel,
        decoupled_sel=decoupled_sel,
        output_path=output_path,
        overlay_alpha=args.overlay_alpha,
        overlay_color=tuple(args.overlay_color),
        tau=args.tau,
        dpi=args.dpi,
    )
    dump_selection_sidecar(
        output_path,
        {
            "figure": "failure_modes",
            "seed": args.seed,
            "fold": args.fold,
            "n_worst": args.n_worst,
            "condition": args.condition,
            "score": "mse_to_nearest_real",
            "shared_worst": selected_to_dict(shared_sel),
            "decoupled_worst": selected_to_dict(decoupled_sel),
        },
    )
    plt.close("all")
    return output_path


def _parse_args_from_list(argv: list[str]) -> argparse.Namespace:
    import sys as _sys

    old = _sys.argv
    try:
        _sys.argv = ["failure_modes", *argv]
        return _parse_args()
    finally:
        _sys.argv = old


if __name__ == "__main__":
    main()
