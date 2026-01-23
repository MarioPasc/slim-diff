"""Report generation for classification experiments.

Generates LaTeX tables, comparison plots, and per-z-bin heatmaps
suitable for ICIP 2026 paper submission.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.classification.evaluation.metrics import ExperimentResult
from src.classification.evaluation.statistical_tests import PermutationTestResult

logger = logging.getLogger(__name__)


def generate_comparison_table(
    results: list[ExperimentResult],
    permutation_results: dict[str, PermutationTestResult] | None = None,
    output_path: Path | None = None,
    fmt: str = "latex",
) -> pd.DataFrame:
    """Generate cross-experiment comparison table.

    Args:
        results: List of ExperimentResult (one per experiment+mode).
        permutation_results: Optional dict mapping experiment_name to test results.
        output_path: Path to save the table.
        fmt: Output format ("latex", "markdown", "csv").

    Returns:
        DataFrame with comparison metrics.
    """
    rows = []
    for res in results:
        # Parse experiment name into prediction type and Lp value
        parts = res.experiment_name.split("_lp_")
        pred_type = parts[0] if len(parts) == 2 else res.experiment_name
        lp_value = parts[1] if len(parts) == 2 else ""

        row = {
            "Experiment": res.experiment_name,
            "Prediction": pred_type,
            "Lp": lp_value,
            "Mode": res.input_mode,
            "AUC (mean)": f"{res.mean_auc:.3f}",
            "AUC (std)": f"{res.std_auc:.3f}",
            "95% CI": f"[{res.pooled_ci_lower:.3f}, {res.pooled_ci_upper:.3f}]",
        }

        if permutation_results and res.experiment_name in permutation_results:
            perm = permutation_results[res.experiment_name]
            row["p-value"] = f"{perm.p_value:.4f}"
            row["Significant"] = "Yes" if perm.significant else "No"
        else:
            row["p-value"] = "-"
            row["Significant"] = "-"

        rows.append(row)

    df = pd.DataFrame(rows)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "latex":
            latex_str = df.to_latex(index=False, escape=False)
            output_path.with_suffix(".tex").write_text(latex_str)
        elif fmt == "markdown":
            md_str = df.to_markdown(index=False)
            output_path.with_suffix(".md").write_text(md_str)
        elif fmt == "csv":
            df.to_csv(output_path.with_suffix(".csv"), index=False)

        logger.info(f"Saved comparison table to {output_path}")

    return df


def generate_paper_figures(
    results: list[ExperimentResult],
    permutation_results: dict[str, PermutationTestResult] | None = None,
    control_result: ExperimentResult | None = None,
    figures_dir: Path = Path("figures"),
) -> None:
    """Generate publication-quality figures.

    Produces:
    - AUC bar chart with 95% CI error bars
    - Per-z-bin AUC heatmap across experiments

    Args:
        results: Experiment results to plot.
        permutation_results: Optional permutation test results.
        control_result: Optional real-vs-real control result.
        figures_dir: Output directory for figures.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figure generation.")
        return

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: AUC bar chart ---
    _plot_auc_bars(results, control_result, figures_dir / "auc_comparison.pdf")

    # --- Figure 2: Per-z-bin heatmap ---
    _plot_zbin_heatmap(results, figures_dir / "zbin_heatmap.pdf")

    logger.info(f"Figures saved to {figures_dir}")


def _plot_auc_bars(
    results: list[ExperimentResult],
    control_result: ExperimentResult | None,
    output_path: Path,
) -> None:
    """Plot AUC bar chart with CI error bars."""
    import matplotlib.pyplot as plt

    names = [r.experiment_name for r in results]
    aucs = [r.mean_auc for r in results]
    ci_lower = [r.pooled_ci_lower for r in results]
    ci_upper = [r.pooled_ci_upper for r in results]

    # Error bars (asymmetric)
    errors = [
        [auc - lo for auc, lo in zip(aucs, ci_lower)],
        [hi - auc for auc, hi in zip(aucs, ci_upper)],
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, aucs, yerr=errors, capsize=4, color="steelblue", alpha=0.8)

    # Chance level
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (AUC=0.5)")

    # Control
    if control_result is not None:
        ax.axhline(
            y=control_result.mean_auc, color="green", linestyle=":",
            linewidth=1.5, label=f"Control (AUC={control_result.mean_auc:.3f})"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Lesion Discrimination: Real vs. Synthetic")
    ax.set_ylim(0.3, 0.8)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_zbin_heatmap(results: list[ExperimentResult], output_path: Path) -> None:
    """Plot per-z-bin AUC heatmap across experiments."""
    import matplotlib.pyplot as plt

    # Collect all z-bins
    all_zbins: set[int] = set()
    for r in results:
        all_zbins.update(r.per_zbin_mean_auc.keys())
    zbins_sorted = sorted(all_zbins)

    if not zbins_sorted:
        logger.warning("No per-z-bin data available for heatmap.")
        return

    # Build matrix (experiments x z-bins)
    matrix = np.full((len(results), len(zbins_sorted)), np.nan)
    for i, r in enumerate(results):
        for j, zb in enumerate(zbins_sorted):
            if zb in r.per_zbin_mean_auc:
                matrix[i, j] = r.per_zbin_mean_auc[zb]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.4, vmax=0.7)
    ax.set_xticks(range(len(zbins_sorted)))
    ax.set_xticklabels(zbins_sorted, fontsize=7)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r.experiment_name for r in results], fontsize=8)
    ax.set_xlabel("Z-bin")
    ax.set_ylabel("Experiment")
    ax.set_title("Per-Z-bin AUC (lower = better synthetic quality)")
    fig.colorbar(im, ax=ax, label="AUC-ROC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_experiment_result(result: ExperimentResult, output_path: Path) -> None:
    """Save experiment result to JSON.

    Args:
        result: ExperimentResult to save.
        output_path: Output JSON file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "experiment_name": result.experiment_name,
        "input_mode": result.input_mode,
        "mean_auc": result.mean_auc,
        "std_auc": result.std_auc,
        "pooled_auc": result.pooled_auc,
        "pooled_ci_lower": result.pooled_ci_lower,
        "pooled_ci_upper": result.pooled_ci_upper,
        "per_zbin_mean_auc": {str(k): v for k, v in result.per_zbin_mean_auc.items()},
        "n_folds": len(result.fold_results),
        "fold_aucs": [fr.global_metrics.auc_roc for fr in result.fold_results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved result to {output_path}")


def load_experiment_result_summary(json_path: Path) -> dict:
    """Load a saved experiment result summary from JSON."""
    with open(json_path) as f:
        return json.load(f)
