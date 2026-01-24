"""CLI script for running all experiments and generating reports.

Usage:
    python -m src.classification run-all --config <path> [--include-control]
    python -m src.classification report --config <path> [--format latex]
"""

from __future__ import annotations

import argparse
import gc
import logging
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.classification.data.data_module import ControlDataModule
from src.classification.evaluation.metrics import (
    ExperimentResult,
    aggregate_fold_metrics,
    compute_fold_metrics,
)
from src.classification.evaluation.reporting import (
    generate_comparison_table,
    generate_paper_figures,
    load_experiment_result_summary,
    save_experiment_result,
)
from src.classification.evaluation.statistical_tests import (
    PermutationTestResult,
    permutation_test_auc,
)
from src.classification.scripts.run_experiment import run_experiment
from src.classification.training.lit_module import ClassificationLightningModule

logger = logging.getLogger(__name__)


def run_all(args: argparse.Namespace) -> None:
    """Run classification for all experiments and input modes.

    Each experiment/mode is run as a separate subprocess to prevent memory
    accumulation from numpy arrays that Python's allocator doesn't return to
    the OS. Results are saved to disk by each subprocess and loaded at the end.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = OmegaConf.load(args.config)
    input_modes: list[str] = list(cfg.data.input_modes)
    experiments = [exp.name for exp in cfg.data.synthetic.experiments]

    use_dithering = getattr(args, "dithering", False)
    use_full_image = getattr(args, "full_image", False)

    failed: list[str] = []
    total = len(experiments) * len(input_modes)
    completed = 0

    for exp_name in experiments:
        for mode in input_modes:
            completed += 1
            logger.info(f"{'='*60}")
            logger.info(f"[{completed}/{total}] Running: {exp_name} / {mode}")
            logger.info(f"{'='*60}")

            # Run as subprocess to guarantee memory isolation
            cmd = [
                sys.executable, "-m", "src.classification", "run",
                "--config", str(args.config),
                "--experiment", exp_name,
                "--input-mode", mode,
            ]
            if use_dithering:
                cmd.append("--dithering")
            if use_full_image:
                cmd.append("--full-image")

            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                logger.error(f"FAILED: {exp_name} / {mode} (exit code {result.returncode})")
                failed.append(f"{exp_name}/{mode}")
            else:
                logger.info(f"Completed: {exp_name} / {mode}")

    # Control experiment (small patches, run in-process)
    if args.include_control and cfg.evaluation.control.enabled:
        logger.info("Running real-vs-real control experiment...")
        _run_control(cfg, input_modes[0])

    if failed:
        logger.warning(f"{len(failed)} experiment(s) failed: {failed}")
    else:
        logger.info(f"All {total} experiments completed successfully.")

    logger.info("Run 'python -m src.classification report --config ...' to generate tables.")


def generate_report(args: argparse.Namespace) -> None:
    """Generate report from saved results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = OmegaConf.load(args.config)
    results_dir = Path(cfg.output.base_dir) / cfg.output.results_subdir
    figures_dir = Path(cfg.output.base_dir) / cfg.output.figures_subdir
    tables_dir = Path(cfg.output.base_dir) / cfg.output.tables_subdir

    # Load saved results
    results: list[ExperimentResult] = []
    result_files = sorted(results_dir.rglob("experiment_result.json"))

    if not result_files:
        logger.error(f"No results found in {results_dir}. Run experiments first.")
        return

    # We can only generate tables from saved summaries
    summaries = [load_experiment_result_summary(f) for f in result_files]
    logger.info(f"Loaded {len(summaries)} experiment results")

    # Generate table from summaries
    rows = []
    for s in summaries:
        parts = s["experiment_name"].split("_lp_")
        pred_type = parts[0] if len(parts) == 2 else s["experiment_name"]
        lp_value = parts[1] if len(parts) == 2 else ""
        rows.append({
            "Experiment": s["experiment_name"],
            "Prediction": pred_type,
            "Lp": lp_value,
            "Mode": s["input_mode"],
            "AUC (mean)": f"{s['mean_auc']:.3f}",
            "AUC (std)": f"{s['std_auc']:.3f}",
            "95% CI": f"[{s['pooled_ci_lower']:.3f}, {s['pooled_ci_upper']:.3f}]",
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    tables_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    if fmt == "latex":
        table_path = tables_dir / "comparison_table.tex"
        table_path.write_text(df.to_latex(index=False, escape=False))
    elif fmt == "markdown":
        table_path = tables_dir / "comparison_table.md"
        table_path.write_text(df.to_markdown(index=False))
    else:
        table_path = tables_dir / "comparison_table.csv"
        df.to_csv(table_path, index=False)

    logger.info(f"Table saved to {table_path}")


def _run_control(
    cfg,
    input_mode: Literal["joint", "image_only", "mask_only"],
) -> ExperimentResult:
    """Run the real-vs-real control experiment."""
    patches_dir = Path(cfg.output.base_dir) / cfg.output.patches_subdir / "control"
    n_folds = cfg.data.kfold.n_folds
    n_repeats = cfg.evaluation.control.n_repeats

    all_fold_results = []

    for repeat in range(n_repeats):
        pl.seed_everything(cfg.experiment.seed + repeat, workers=True)

        dm = ControlDataModule(
            cfg=cfg, patches_dir=patches_dir, input_mode=input_mode, repeat_idx=repeat
        )

        for fold_idx in range(n_folds):
            dm.set_fold(fold_idx)
            dm.prepare_data()
            dm.setup()

            model = ClassificationLightningModule(
                cfg=cfg, in_channels=dm.in_channels, fold_idx=fold_idx
            )

            trainer = pl.Trainer(
                max_epochs=cfg.training.max_epochs,
                callbacks=[
                    pl.callbacks.EarlyStopping(
                        monitor=cfg.training.early_stopping.monitor,
                        mode=cfg.training.early_stopping.mode,
                        patience=cfg.training.early_stopping.patience,
                    ),
                ],
                precision=cfg.training.precision,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                deterministic=True,
            )
            trainer.fit(model, datamodule=dm)

            model.clear_test_outputs()
            trainer.test(model, dataloaders=dm.val_dataloader())
            outputs = model.get_test_outputs()

            fold_result = compute_fold_metrics(
                probs=outputs["probs"],
                labels=outputs["labels"],
                z_bins=outputs["z_bins"],
                fold_idx=fold_idx + repeat * n_folds,
                bootstrap_n=cfg.evaluation.bootstrap.n_iterations,
                confidence_level=cfg.evaluation.bootstrap.confidence_level,
                bootstrap_seed=cfg.evaluation.bootstrap.seed + fold_idx + repeat * 100,
            )
            all_fold_results.append(fold_result)

    result = aggregate_fold_metrics(
        fold_results=all_fold_results,
        experiment_name="control",
        input_mode=input_mode,
        bootstrap_n=cfg.evaluation.bootstrap.n_iterations,
        confidence_level=cfg.evaluation.bootstrap.confidence_level,
        bootstrap_seed=cfg.evaluation.bootstrap.seed,
    )

    logger.info(
        f"Control experiment: AUC={result.mean_auc:.4f} +/- {result.std_auc:.4f} "
        f"(expected ~0.5)"
    )

    results_dir = Path(cfg.output.base_dir) / cfg.output.results_subdir / "control" / input_mode
    save_experiment_result(result, results_dir / "experiment_result.json")

    return result


def _generate_full_report(
    cfg,
    results: list[ExperimentResult],
    perm_results: dict[str, PermutationTestResult],
    control_result: ExperimentResult | None,
) -> None:
    """Generate full comparison report with tables and figures."""
    tables_dir = Path(cfg.output.base_dir) / cfg.output.tables_subdir
    figures_dir = Path(cfg.output.base_dir) / cfg.output.figures_subdir

    # Group results by input mode for separate tables
    modes = set(r.input_mode for r in results)
    for mode in sorted(modes):
        mode_results = [r for r in results if r.input_mode == mode]
        mode_perm = {
            r.experiment_name: perm_results.get(f"{r.experiment_name}_{mode}")
            for r in mode_results
            if f"{r.experiment_name}_{mode}" in perm_results
        }

        generate_comparison_table(
            results=mode_results,
            permutation_results=mode_perm,
            output_path=tables_dir / f"comparison_{mode}",
            fmt="latex",
        )

    # Figures (use joint mode by default)
    joint_results = [r for r in results if r.input_mode == "joint"]
    if joint_results:
        generate_paper_figures(
            results=joint_results,
            permutation_results={
                r.experiment_name: perm_results.get(f"{r.experiment_name}_joint")
                for r in joint_results
                if f"{r.experiment_name}_joint" in perm_results
            },
            control_result=control_result,
            figures_dir=figures_dir,
        )

    logger.info("Report generation complete.")
