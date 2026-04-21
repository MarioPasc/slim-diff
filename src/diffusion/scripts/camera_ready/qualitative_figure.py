"""Primary qualitative figure for the ICIP 2026 camera-ready (R1.4).

Renders a 3-z-bin × 3-row × ``n_samples`` × 2-modality grid:

* Rows:  Generated (control), Generated (epilepsy), Real.
* Cols:  z-low / z-mid / z-high (each split into FLAIR | Overlay).
* Samples: ``n_samples`` per (row, z-bin) bucket, stacked vertically.

Selection is median-ranked by MSE to the nearest real reference slice in the
same ``(fold, zbin, condition)`` bucket, which gives a representative (not
cherry-picked) view of the generator's output. The selected sample indices
are logged to a JSON sidecar next to the figure for reproducibility.

Usage
-----

::

    python -m src.diffusion.scripts.camera_ready.qualitative_figure \
        --results-root /path/to/results \
        --cache-dir /path/to/slice_cache \
        --fold 0 --architecture shared \
        --output /path/to/figures/qualitative_grid.pdf \
        --zbins 5 15 25 --n-samples 2 --selection-mode median --format pdf
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
    SelectionMode,
    add_row_label,
    dump_selection_sidecar,
    load_real_samples,
    load_replicas_concat,
    load_synthetic_samples,
    overlay_mask_on_image,
    rescale_to_display,
    resolve_cell_dir,
    selected_to_dict,
    setup_ieee_style,
    strip_axis,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Rendering
# =============================================================================


def _plot_cell(
    ax_flair: plt.Axes,
    ax_over: plt.Axes,
    image: np.ndarray,
    mask: np.ndarray,
    show_overlay: bool,
    overlay_alpha: float,
    overlay_color: tuple[int, int, int],
    tau: float,
) -> None:
    disp = rescale_to_display(image)
    ax_flair.imshow(disp, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    strip_axis(ax_flair)
    if show_overlay:
        rgb = overlay_mask_on_image(
            image, mask, alpha=overlay_alpha, color=overlay_color, tau=tau
        )
        ax_over.imshow(rgb, interpolation="nearest")
    else:
        ax_over.imshow(disp, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    strip_axis(ax_over)


def render_qualitative_grid(
    synth_ctrl: SelectedSamples,
    synth_epi: SelectedSamples,
    real_ref: SelectedSamples,
    zbins: list[int],
    n_samples: int,
    output_path: Path,
    fig_width: float = IEEE_DOUBLE_COL_IN,
    overlay_alpha: float = 0.5,
    overlay_color: tuple[int, int, int] = (255, 0, 0),
    tau: float = 0.0,
    dpi: int = 300,
) -> plt.Figure:
    """Build the grid figure and save it to ``output_path``.

    Layout is 3 row-blocks (control synth, epilepsy synth, real); each block
    has ``n_samples`` rows × ``2 * len(zbins)`` columns (FLAIR | Overlay per
    z-bin). Column headers span each z-bin's two modality sub-columns.
    """
    n_zbins = len(zbins)
    n_cols = 2 * n_zbins
    n_rows = 3 * n_samples
    # Two header rows (z-bin labels and sub-labels), one extra row at the top.
    header_rows = 1
    # Keep the figure height bounded to IEEE's 9 in max.
    cell_h = fig_width / (n_cols * 1.02)
    fig_h = min(n_rows * cell_h + 0.8, 9.0)

    fig = plt.figure(figsize=(fig_width, fig_h))
    gs = fig.add_gridspec(
        nrows=n_rows + header_rows,
        ncols=n_cols + 1,  # extra leftmost narrow column for row labels
        width_ratios=[0.07] + [1.0] * n_cols,
        height_ratios=[0.18] + [1.0] * n_rows,
        left=0.04,
        right=0.995,
        top=0.96,
        bottom=0.01,
        wspace=0.04,
        hspace=0.05,
    )

    # --- column headers --------------------------------------------------------
    for zi, zbin in enumerate(zbins):
        base = 1 + 2 * zi
        ax_hdr = fig.add_subplot(gs[0, base : base + 2])
        ax_hdr.text(
            0.5,
            0.35,
            f"$z$-bin {zbin}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        ax_hdr.text(0.27, -0.3, "FLAIR", ha="center", va="center", fontsize=6.5)
        ax_hdr.text(0.77, -0.3, "Overlay", ha="center", va="center", fontsize=6.5)
        ax_hdr.set_xticks([])
        ax_hdr.set_yticks([])
        for s in ax_hdr.spines.values():
            s.set_visible(False)

    # --- plot row blocks -------------------------------------------------------
    blocks = [
        ("Gen.\n(control)", synth_ctrl, False),
        ("Gen.\n(epilepsy)", synth_epi, True),
        ("Real", real_ref, True),
    ]
    for block_idx, (label, samples, overlay_lesion) in enumerate(blocks):
        # Slice samples by z-bin for deterministic column placement.
        for s in range(n_samples):
            r = header_rows + block_idx * n_samples + s
            for zi, zbin in enumerate(zbins):
                bucket = np.where(samples.zbins == zbin)[0]
                if bucket.size < s + 1:
                    # Missing data: draw empty panels rather than crashing.
                    ax_f = fig.add_subplot(gs[r, 1 + 2 * zi])
                    ax_o = fig.add_subplot(gs[r, 2 + 2 * zi])
                    for ax in (ax_f, ax_o):
                        ax.set_facecolor("#f0f0f0")
                        ax.text(
                            0.5,
                            0.5,
                            "\u2014",
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="#888",
                        )
                        strip_axis(ax)
                    continue
                idx = bucket[s]
                ax_f = fig.add_subplot(gs[r, 1 + 2 * zi])
                ax_o = fig.add_subplot(gs[r, 2 + 2 * zi])
                # For the control block, never draw the overlay (mask should
                # be blank). Still call the helper so a stray lesion pixel in
                # the predicted mask is visibly exposed.
                _plot_cell(
                    ax_f,
                    ax_o,
                    samples.images[idx],
                    samples.masks[idx],
                    show_overlay=overlay_lesion,
                    overlay_alpha=overlay_alpha,
                    overlay_color=overlay_color,
                    tau=tau,
                )

            # Left row label, once per block (on the middle sample of the block).
            if s == n_samples // 2:
                ax_lab = fig.add_subplot(gs[r, 0])
                ax_lab.text(
                    0.5,
                    0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    rotation=90,
                )
                strip_axis(ax_lab)
            else:
                ax_lab = fig.add_subplot(gs[r, 0])
                strip_axis(ax_lab)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved qualitative figure to %s", output_path)
    return fig


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the camera-ready qualitative grid figure."
    )
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--architecture", type=str, default="shared")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--zbins", type=int, nargs="+", default=[5, 15, 25])
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument(
        "--selection-mode",
        choices=("median", "best", "random", "worst"),
        default="median",
    )
    p.add_argument("--overlay-alpha", type=float, default=0.5)
    p.add_argument(
        "--overlay-color", type=int, nargs=3, default=[255, 0, 0], metavar=("R", "G", "B")
    )
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--format", choices=("pdf", "png"), default="pdf")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-replicas",
        type=int,
        default=None,
        help="Cap replicas loaded per cell (default: all).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main(argv: list[str] | None = None) -> Path:
    """CLI entry point. Returns the path of the saved figure."""
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

    # Load replicas once and reuse for both conditions.
    cell_dir = resolve_cell_dir(args.results_root, args.architecture, args.fold)
    replicas = load_replicas_concat(cell_dir, max_replicas=args.max_replicas)

    selection_mode: SelectionMode = args.selection_mode  # type: ignore[assignment]

    synth_ctrl = load_synthetic_samples(
        results_root=args.results_root,
        fold=args.fold,
        architecture=args.architecture,
        zbins=args.zbins,
        condition=0,
        n_samples=args.n_samples,
        selection_mode=selection_mode,
        cache_dir=args.cache_dir,
        rng=np.random.default_rng(args.seed),
        replicas=replicas,
    )
    synth_epi = load_synthetic_samples(
        results_root=args.results_root,
        fold=args.fold,
        architecture=args.architecture,
        zbins=args.zbins,
        condition=1,
        n_samples=args.n_samples,
        selection_mode=selection_mode,
        cache_dir=args.cache_dir,
        rng=np.random.default_rng(args.seed + 1),
        replicas=replicas,
    )
    real_ref = load_real_samples(
        cache_dir=args.cache_dir,
        fold=args.fold,
        zbins=args.zbins,
        condition=1,  # lesion-present real row for visual comparison
        n_samples=args.n_samples,
        rng=rng,
    )

    render_qualitative_grid(
        synth_ctrl=synth_ctrl,
        synth_epi=synth_epi,
        real_ref=real_ref,
        zbins=args.zbins,
        n_samples=args.n_samples,
        output_path=output_path,
        overlay_alpha=args.overlay_alpha,
        overlay_color=tuple(args.overlay_color),
        tau=args.tau,
        dpi=args.dpi,
    )

    sidecar = dump_selection_sidecar(
        output_path,
        {
            "figure": "qualitative_grid",
            "seed": args.seed,
            "fold": args.fold,
            "architecture": args.architecture,
            "zbins": list(args.zbins),
            "n_samples": args.n_samples,
            "selection_mode": selection_mode,
            "synth_control": selected_to_dict(synth_ctrl),
            "synth_epilepsy": selected_to_dict(synth_epi),
            "real_epilepsy": selected_to_dict(real_ref),
        },
    )
    logger.info("Selection sidecar written to %s", sidecar)
    plt.close("all")
    return output_path


def _parse_args_from_list(argv: list[str]) -> argparse.Namespace:
    import sys as _sys

    old = _sys.argv
    try:
        _sys.argv = ["qualitative_figure", *argv]
        return _parse_args()
    finally:
        _sys.argv = old


if __name__ == "__main__":
    main()
