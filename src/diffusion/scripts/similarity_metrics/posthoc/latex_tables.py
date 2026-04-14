"""LaTeX table emitters for the ICIP 2026 camera-ready (TASK-05).

Generates three self-contained LaTeX files under ``{output_dir}/tables/``:

* ``table_ablation.tex``       — shared vs decoupled ablation (NEW for CR).
* ``table_main_updated.tex``   — updated main Table 1 with cross-fold std on
  the camera-ready cell and a footnote for the single-split cells of the
  original ablation.
* ``table_tau_sensitivity.tex``— compact tau-sweep summary.

The emitters pull all numeric content from TASK-04 outputs + TASK-05
``tau_sensitivity.csv``; no values are hard-coded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Decimal places: match the paper's existing Table 1 (KID/LPIPS: 3 dp,
# MMD-MF: 2 dp). See TASK-05 anti-patterns.
METRIC_DECIMALS: dict[str, int] = {
    "kid": 3,
    "lpips": 3,
    "mmd_mf": 2,
}

ARCH_LABELS: dict[str, str] = {
    "shared": "Shared (ours)",
    "decoupled": "Decoupled",
}


def _fmt(value: float, decimals: int) -> str:
    """Format a float with the specified decimal places; NaN → ``---``."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "---"
    return f"{value:.{decimals}f}"


def _fmt_mean_std(mean: float, std: float, decimals: int, bold: bool = False) -> str:
    cell = f"${_fmt(mean, decimals)} \\pm {_fmt(std, decimals)}$"
    if bold:
        cell = f"$\\mathbf{{{_fmt(mean, decimals)} \\pm {_fmt(std, decimals)}}}$"
    return cell


def _summary_from_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a ``fold_metrics.csv`` into per-architecture mean/std.

    Aggregation is ``np.nanmean`` and ``np.nanstd(ddof=0)`` over folds — same
    convention as TASK-04's ``summary_metrics.csv``.
    """
    if "architecture" not in fold_df.columns:
        raise ValueError("fold_metrics.csv is missing the 'architecture' column")

    rows = []
    for arch, sub in fold_df.groupby("architecture"):
        row = {"architecture": str(arch)}
        for metric_key, col in (
            ("kid", "kid_mean"),
            ("lpips", "lpips_mean"),
            ("mmd_mf", "mmd_mf_mean"),
        ):
            if col not in sub.columns:
                row[f"{metric_key}_mean"] = float("nan")
                row[f"{metric_key}_std"] = float("nan")
                continue
            vals = sub[col].to_numpy(dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            row[f"{metric_key}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{metric_key}_std"] = (
                float(np.std(finite, ddof=0)) if finite.size else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table A — Ablation (shared vs decoupled)
# ---------------------------------------------------------------------------


def generate_ablation_table(
    fold_metrics_csv: Path | str,
    output_path: Path | str,
) -> str:
    """Emit the shared-vs-decoupled ablation table.

    Bolds the winning cell per metric column (lower is better). Decoupled is
    listed first, then shared — matching the caption's narrative (decoupled =
    alternative, shared = ours).
    """
    fold_metrics_csv = Path(fold_metrics_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fold_metrics_csv)
    summary = _summary_from_fold_metrics(df).set_index("architecture")

    # Winners per metric column (ignoring NaNs).
    winners: dict[str, str] = {}
    for key in ("kid", "lpips", "mmd_mf"):
        means = summary[f"{key}_mean"]
        finite = means.dropna()
        if finite.empty:
            continue
        winners[key] = str(finite.idxmin())

    def row(arch: str) -> str:
        label = ARCH_LABELS.get(arch, arch.capitalize())
        cells = [label]
        for key in ("kid", "lpips", "mmd_mf"):
            m = summary.loc[arch, f"{key}_mean"]
            s = summary.loc[arch, f"{key}_std"]
            cells.append(
                _fmt_mean_std(m, s, METRIC_DECIMALS[key], bold=(winners.get(key) == arch))
            )
        return " & ".join(cells) + r" \\"

    present = [a for a in ("decoupled", "shared") if a in summary.index]
    body_rows = "\n".join(row(a) for a in present)

    body = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Shared vs.\ decoupled bottleneck ($x_0$-prediction, $L_{\gamma=1.5}$). "
        r"Mean $\pm$ std across 3 stratified folds. Lower is better; best value per "
        r"column in \textbf{bold}.}" "\n"
        r"\label{tab:ablation}" "\n"
        r"\begin{tabular}{lccc}" "\n"
        r"\toprule" "\n"
        r"Architecture & KID $\downarrow$ & LPIPS $\downarrow$ & MMD-MF $\downarrow$ \\" "\n"
        r"\midrule" "\n"
        f"{body_rows}\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    output_path.write_text(body)
    logger.info("Wrote %s", output_path)
    return body


# ---------------------------------------------------------------------------
# Table B — Updated main table (cross-fold std for the camera-ready cell)
# ---------------------------------------------------------------------------


def generate_main_updated_table(
    fold_metrics_csv: Path | str,
    output_path: Path | str,
    architecture: str = "shared",
) -> str:
    """Emit the updated main Table 1 with cross-fold ± for the camera-ready cell.

    Only the ``(x_0, L_{1.5})`` cell was retrained across folds; the remaining
    eight cells of the original ablation keep their single-split numbers — a
    footnote flags the mismatch honestly. The body renders a single row for
    the retrained cell; downstream the paper TeX includes the remaining rows.
    """
    fold_metrics_csv = Path(fold_metrics_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fold_metrics_csv)
    summary = _summary_from_fold_metrics(df).set_index("architecture")

    if architecture not in summary.index:
        raise ValueError(
            f"architecture {architecture!r} not present in {fold_metrics_csv}"
        )

    kid = _fmt_mean_std(
        summary.loc[architecture, "kid_mean"],
        summary.loc[architecture, "kid_std"],
        METRIC_DECIMALS["kid"],
    )
    lpips = _fmt_mean_std(
        summary.loc[architecture, "lpips_mean"],
        summary.loc[architecture, "lpips_std"],
        METRIC_DECIMALS["lpips"],
    )
    mmd = _fmt_mean_std(
        summary.loc[architecture, "mmd_mf_mean"],
        summary.loc[architecture, "mmd_mf_std"],
        METRIC_DECIMALS["mmd_mf"],
    )

    body = (
        r"% Updated Table 1 — only the (x_0, L_{1.5}) row carries cross-fold" "\n"
        r"% variance (3 stratified folds). Other cells keep the original" "\n"
        r"% single-split numbers; see footnote." "\n"
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Updated Table~1. The $(x_0, L_{1.5})$ row (boxed) reports "
        r"mean $\pm$ std across 3 stratified folds (camera-ready); other rows "
        r"retain the original single-split values.\textsuperscript{$\dagger$}}" "\n"
        r"\label{tab:main_updated}" "\n"
        r"\begin{tabular}{llccc}" "\n"
        r"\toprule" "\n"
        r"Prediction & $L_\gamma$ & KID $\downarrow$ & LPIPS $\downarrow$ & MMD-MF $\downarrow$ \\" "\n"
        r"\midrule" "\n"
        r"\multicolumn{5}{l}{\emph{Remaining ablation cells: import the legacy single-split values}} \\" "\n"
        r"\midrule" "\n"
        f"$x_0$ & $1.5$ & {kid} & {lpips} & {mmd} \\\\" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\begin{flushleft}\footnotesize{$\dagger$ Only the $(x_0, L_{1.5})$ " "\n"
        r"cell was retrained for the camera-ready; cross-fold variance " "\n"
        r"therefore applies to that cell only.}\end{flushleft}" "\n"
        r"\end{table}" "\n"
    )
    output_path.write_text(body)
    logger.info("Wrote %s", output_path)
    return body


# ---------------------------------------------------------------------------
# Table C — tau sensitivity
# ---------------------------------------------------------------------------


def generate_tau_sensitivity_table(
    tau_csv: Path | str,
    output_path: Path | str,
    architecture: str = "shared",
) -> str:
    """Emit a compact tau-sensitivity table for one architecture.

    Rows: tau values from the sweep.
    Cols: mean MMD-MF across folds (from ``tau_sensitivity_summary.csv``
    semantics, but we re-aggregate here to avoid depending on file order).
    """
    tau_csv = Path(tau_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tau_csv)
    subset = df[df["architecture"] == architecture].copy()
    if subset.empty:
        raise ValueError(
            f"No rows for architecture={architecture!r} in {tau_csv}"
        )

    agg = (
        subset.groupby("tau", as_index=False)
        .agg(
            mmd_mean=("mmd_mf_mean", lambda s: float(np.nanmean(s))),
            mmd_std=("mmd_mf_mean", lambda s: float(np.nanstd(s, ddof=0))),
            n_lesions_mean=("n_lesions", lambda s: float(np.mean(s))),
        )
        .sort_values("tau")
    )

    body_rows = []
    for _, row in agg.iterrows():
        tau_str = f"${row['tau']:+.2f}$"
        cell = _fmt_mean_std(row["mmd_mean"], row["mmd_std"], METRIC_DECIMALS["mmd_mf"])
        body_rows.append(f"{tau_str} & {cell} & {int(round(row['n_lesions_mean']))} \\\\")

    body = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{MMD-MF sensitivity to the mask binarisation threshold "
        rf"$\tau$ for the {ARCH_LABELS.get(architecture, architecture)} "
        r"architecture; mean $\pm$ std across 3 stratified folds.}" "\n"
        rf"\label{{tab:tau_sensitivity_{architecture}}}" "\n"
        r"\begin{tabular}{ccc}" "\n"
        r"\toprule" "\n"
        r"$\tau$ & MMD-MF $\downarrow$ & \# lesions (avg.) \\" "\n"
        r"\midrule" "\n"
        + "\n".join(body_rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    output_path.write_text(body)
    logger.info("Wrote %s", output_path)
    return body


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


def generate_all_tables(
    fold_metrics_csv: Path | str,
    output_dir: Path | str,
    tau_csv: Path | str | None = None,
) -> list[Path]:
    """Generate every available table. Skips any whose inputs are missing."""
    fold_metrics_csv = Path(fold_metrics_csv)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    ablation_path = tables_dir / "table_ablation.tex"
    generate_ablation_table(fold_metrics_csv, ablation_path)
    written.append(ablation_path)

    main_path = tables_dir / "table_main_updated.tex"
    generate_main_updated_table(fold_metrics_csv, main_path)
    written.append(main_path)

    if tau_csv is not None and Path(tau_csv).exists():
        for arch in ("shared", "decoupled"):
            df = pd.read_csv(tau_csv)
            if arch in df["architecture"].unique():
                tau_path = tables_dir / f"table_tau_sensitivity_{arch}.tex"
                generate_tau_sensitivity_table(tau_csv, tau_path, architecture=arch)
                written.append(tau_path)
    else:
        logger.info("tau_sensitivity.csv not provided; skipping tau table.")

    return written
