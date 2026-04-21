"""τ-sensitivity scatter: Shared vs Decoupled MMD-MF.

Consumes ``tau_sensitivity_summary.csv`` from TASK-05 (see
``evaluations/posthoc_output/``) and produces a diagnostic scatter plot:

* x-axis: fold-aggregated MMD-MF mean for the **Shared** architecture.
* y-axis: fold-aggregated MMD-MF mean for the **Decoupled** architecture.
* Error bars: fold-wise ``mmd_mf_std_across_folds`` on both axes.
* Points: one per binarisation threshold τ, sorted by τ and colour-encoded
  with a perceptually-uniform colormap. Adjacent τ values are connected by
  a thin line to make the sweep trajectory visible.
* The ``y = x`` diagonal is drawn as a visual reference: points above the
  diagonal indicate that Decoupled has larger MMD-MF than Shared at the same
  τ (i.e. the Shared model is closer to the real lesion-morphology manifold
  at that τ).

Input CSV columns (from :mod:`src.diffusion.scripts.similarity_metrics.posthoc`):

``architecture, tau, mmd_mf_mean_across_folds, mmd_mf_std_across_folds, n_lesions_mean``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from src.diffusion.scripts.camera_ready.figure_utils import (
    IEEE_DOUBLE_COL_IN,
    IEEE_SINGLE_COL_IN,
    dump_selection_sidecar,
    setup_ieee_style,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data
# =============================================================================


def load_tau_summary(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    required = {
        "architecture",
        "tau",
        "mmd_mf_mean_across_folds",
        "mmd_mf_std_across_folds",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tau summary CSV missing columns: {missing}")
    return df


def pivot_by_tau(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per τ with ``shared_*``/``decoupled_*`` columns."""
    pivot = df.pivot_table(
        index="tau",
        columns="architecture",
        values=["mmd_mf_mean_across_folds", "mmd_mf_std_across_folds"],
    )
    pivot.columns = [f"{metric}_{arch}" for metric, arch in pivot.columns]
    pivot = pivot.reset_index().sort_values("tau").reset_index(drop=True)
    return pivot


# =============================================================================
# Rendering
# =============================================================================


def _build_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.stack([x[:-1], y[:-1]], axis=1), np.stack([x[1:], y[1:]], axis=1)],
        axis=1,
    )


def render_tau_scatter(
    pivot: pd.DataFrame,
    output_path: Path,
    fig_width: float = IEEE_SINGLE_COL_IN,
    fig_height: float | None = None,
    cmap_name: str = "viridis",
    dpi: int = 300,
    annotate: bool = True,
    zoom_xlim: tuple[float, float] = (4.5, 5.5),
    zoom_ylim: tuple[float, float] = (13.0, 14.0),
    inset_bounds: tuple[float, float, float, float] = (12.0, 0.0, 8.0, 9.0),
) -> plt.Figure:
    """Render the τ-sensitivity scatter with an inset zoom.

    The main axes show the full Shared/Decoupled MMD-MF plane including the
    ``y=x`` reference. All per-point ``τ`` annotations live inside an inset
    axes that zooms into ``(zoom_xlim, zoom_ylim)`` and is placed in data
    coordinates at ``inset_bounds`` (x0, y0, width, height). ``indicate_inset_zoom``
    draws the rectangle + connecting lines between the two axes.
    """
    if fig_height is None:
        fig_height = fig_width * 0.95

    x = pivot["mmd_mf_mean_across_folds_shared"].to_numpy(dtype=float)
    y = pivot["mmd_mf_mean_across_folds_decoupled"].to_numpy(dtype=float)
    xerr = pivot["mmd_mf_std_across_folds_shared"].to_numpy(dtype=float)
    yerr = pivot["mmd_mf_std_across_folds_decoupled"].to_numpy(dtype=float)
    taus = pivot["tau"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=taus.min(), vmax=taus.max())
    segments = _build_segments(x, y)

    def _plot_sweep(axis: plt.Axes, lw: float, scatter_s: float, cap: float) -> object:
        lc = LineCollection(
            segments, cmap=cmap, norm=norm, linewidth=lw, alpha=0.6, zorder=1
        )
        lc.set_array(0.5 * (taus[:-1] + taus[1:]))
        axis.add_collection(lc)
        axis.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt="none", ecolor="0.65", elinewidth=0.6, capsize=cap, capthick=0.5,
            zorder=2,
        )
        return axis.scatter(
            x, y,
            c=taus, cmap=cmap, norm=norm,
            s=scatter_s, edgecolor="white", linewidth=0.5, zorder=3,
        )

    # Main axes: full plane, no per-point τ annotations.
    sc = _plot_sweep(ax, lw=0.8, scatter_s=24, cap=1.5)

    # Axes limits chosen so (a) all data + error bars are visible and
    # (b) the data-coord inset_bounds fit inside the plot. Equal aspect.
    data_lo = float(min(x.min() - xerr.max(), y.min() - yerr.max()))
    data_hi = float(max(x.max() + xerr.max(), y.max() + yerr.max()))
    ix0, iy0, iw, ih = inset_bounds
    lo = min(0.0, data_lo, ix0, iy0) - 0.5
    hi = max(data_hi, ix0 + iw, iy0 + ih) + 0.5
    ax.plot([lo, hi], [lo, hi], color="0.3", linestyle="--", linewidth=0.7, zorder=0)
    ax.text(hi, hi, r" $y=x$", ha="left", va="center", fontsize=6, color="0.3")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"MMD-MF, Shared")
    ax.set_ylabel(r"MMD-MF, Decoupled")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.tick_params(axis="both", labelsize=6)

    # Inset zoom — positioned in data coordinates per the caller's spec.
    axins = ax.inset_axes(list(inset_bounds), transform=ax.transData)
    _plot_sweep(axins, lw=1.0, scatter_s=30, cap=2.0)
    if annotate:
        for xi, yi, t in zip(x, y, taus, strict=True):
            """
            if zoom_xlim[0] <= xi <= zoom_xlim[1] and zoom_ylim[0] <= yi <= zoom_ylim[1]:
                axins.annotate(
                    rf"$\tau={t:g}$",
                    xy=(xi, yi),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=5.5,
                    color="0.2",
                    zorder=4,
                )
            """
    axins.set_xlim(*zoom_xlim)
    axins.set_ylim(*zoom_ylim)
    axins.tick_params(axis="both", labelsize=5)
    axins.grid(True, linestyle=":", linewidth=0.3, alpha=0.5)
    for spine in axins.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("0.35")

    ax.indicate_inset_zoom(axins, edgecolor="0.35", linewidth=0.6, alpha=0.8)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"binarisation threshold $\tau$", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)

    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.12, top=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved tau-sensitivity scatter to %s", output_path)
    return fig


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--width",
        type=float,
        default=IEEE_SINGLE_COL_IN,
        help=f"Figure width in inches (default: single-col {IEEE_SINGLE_COL_IN}).",
    )
    p.add_argument("--height", type=float, default=None)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--format", choices=("pdf", "png"), default="pdf")
    p.add_argument(
        "--no-annotate",
        action="store_true",
        help="Suppress per-point τ annotations in the inset.",
    )
    p.add_argument(
        "--zoom-xlim", type=float, nargs=2, default=[5.1, 5.4], metavar=("XLO", "XHI")
    )
    p.add_argument(
        "--zoom-ylim", type=float, nargs=2, default=[12.2, 13.6], metavar=("YLO", "YHI")
    )
    p.add_argument(
        "--inset-bounds",
        type=float,
        nargs=4,
        default=[11.6, 0.6, 12.0, 9.0],
        metavar=("X0", "Y0", "W", "H"),
        help="Inset bounds in data coords: (x0, y0, width, height).",
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

    out = args.output
    if out.suffix.lstrip(".") != args.format:
        out = out.with_suffix(f".{args.format}")

    df = load_tau_summary(args.summary_csv)
    pivot = pivot_by_tau(df)
    if pivot.empty:
        raise RuntimeError(f"τ summary pivot is empty for {args.summary_csv}")

    fig_width = min(float(args.width), IEEE_DOUBLE_COL_IN)
    render_tau_scatter(
        pivot=pivot,
        output_path=out,
        fig_width=fig_width,
        fig_height=args.height,
        cmap_name=args.cmap,
        dpi=args.dpi,
        annotate=not args.no_annotate,
        zoom_xlim=tuple(args.zoom_xlim),
        zoom_ylim=tuple(args.zoom_ylim),
        inset_bounds=tuple(args.inset_bounds),
    )
    dump_selection_sidecar(
        out,
        {
            "figure": "tau_scatter",
            "summary_csv": str(args.summary_csv),
            "n_points": int(pivot.shape[0]),
            "tau_values": pivot["tau"].tolist(),
        },
    )
    plt.close("all")
    return out


def _parse_args_from_list(argv: list[str]) -> argparse.Namespace:
    import sys as _sys

    old = _sys.argv
    try:
        _sys.argv = ["tau_scatter_figure", *argv]
        return _parse_args()
    finally:
        _sys.argv = old


if __name__ == "__main__":
    main()
