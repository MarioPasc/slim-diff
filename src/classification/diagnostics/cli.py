"""CLI entry point for diagnostic analyses.

Usage:
    python -m src.classification.diagnostics run-all \
        --config src/classification/diagnostics/config/diagnostics.yaml \
        --experiment velocity_lp_1.5

    python -m src.classification.diagnostics run-all \
        --config src/classification/diagnostics/config/diagnostics.yaml \
        --experiment all

    python -m src.classification.diagnostics dither \
        --config <path> --experiment <name>

    python -m src.classification.diagnostics gradcam \
        --config <path> --experiment <name>

    python -m src.classification.diagnostics spectral \
        --config <path> --experiment <name>

    python -m src.classification.diagnostics texture \
        --config <path> --experiment <name>

    python -m src.classification.diagnostics stats \
        --config <path> --experiment <name>

    python -m src.classification.diagnostics full-image \
        --config <path> --experiment <name>

    python -m src.classification.diagnostics report \
        --config <path> --experiment <name>

    python -m src.classification.diagnostics aggregate \
        --config <path>
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the diagnostics module."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_config(config_path: str):
    """Load and return the diagnostics configuration."""
    cfg = OmegaConf.load(config_path)
    return cfg


def _discover_experiments(cfg) -> list[str]:
    """Discover available experiments from the patches directory."""
    patches_dir = Path(cfg.data.patches_base_dir)
    if not patches_dir.exists():
        logger.error(f"Patches directory not found: {patches_dir}")
        return []

    experiments = sorted([
        d.name for d in patches_dir.iterdir()
        if d.is_dir() and (d / "real_patches.npz").exists() and d.name != "control"
    ])
    return experiments


def _run_dither(cfg, experiment_name: str) -> dict:
    """Run dithering and re-classification analysis."""
    from src.classification.diagnostics.preprocessing.dithering import dither_and_reclassify
    return dither_and_reclassify(cfg, experiment_name)


def _run_gradcam(cfg, experiment_name: str, gpu: int = 0) -> dict:
    """Run GradCAM analysis."""
    from src.classification.diagnostics.xai.gradcam import run_gradcam_analysis
    device = f"cuda:{gpu}" if cfg.experiment.device == "cuda" else "cpu"
    return run_gradcam_analysis(cfg, experiment_name, device=device)


def _run_spectral(cfg, experiment_name: str) -> dict:
    """Run spectral analysis."""
    from src.classification.diagnostics.feature_probes.spectral import run_spectral_analysis
    return run_spectral_analysis(cfg, experiment_name)


def _run_texture(cfg, experiment_name: str) -> dict:
    """Run texture analysis."""
    from src.classification.diagnostics.feature_probes.texture import run_texture_analysis
    return run_texture_analysis(cfg, experiment_name)


def _run_bands(cfg, experiment_name: str) -> dict:
    """Run frequency band analysis."""
    from src.classification.diagnostics.feature_probes.frequency_bands import run_band_analysis
    return run_band_analysis(cfg, experiment_name)


def _run_pixel_stats(cfg, experiment_name: str) -> dict:
    """Run per-pixel statistical tests."""
    from src.classification.diagnostics.statistical.pixel_stats import run_pixel_stats
    return run_pixel_stats(cfg, experiment_name)


def _run_distributions(cfg, experiment_name: str) -> dict:
    """Run distribution comparison tests."""
    from src.classification.diagnostics.statistical.distribution_tests import run_distribution_tests
    return run_distribution_tests(cfg, experiment_name)


def _run_boundary(cfg, experiment_name: str) -> dict:
    """Run boundary analysis."""
    from src.classification.diagnostics.statistical.boundary_analysis import run_boundary_analysis
    return run_boundary_analysis(cfg, experiment_name)


def _run_wavelet(cfg, experiment_name: str) -> dict:
    """Run wavelet analysis."""
    from src.classification.diagnostics.statistical.wavelet_analysis import run_wavelet_analysis
    return run_wavelet_analysis(cfg, experiment_name)


def _run_background(cfg, experiment_name: str) -> dict:
    """Run background consistency analysis."""
    from src.classification.diagnostics.full_image.background_analysis import run_background_analysis
    return run_background_analysis(cfg, experiment_name)


def _run_spatial_correlation(cfg, experiment_name: str) -> dict:
    """Run spatial correlation analysis."""
    from src.classification.diagnostics.full_image.spatial_correlation import run_spatial_correlation
    return run_spatial_correlation(cfg, experiment_name)


def _run_global_frequency(cfg, experiment_name: str) -> dict:
    """Run global frequency analysis."""
    from src.classification.diagnostics.full_image.global_frequency import run_global_frequency
    return run_global_frequency(cfg, experiment_name)


def _run_channel_decomposition(cfg, experiment_name: str, gpu: int = 0) -> dict:
    """Run channel decomposition XAI analysis."""
    from src.classification.diagnostics.xai.channel_decomposition import run_channel_decomposition
    device = f"cuda:{gpu}" if cfg.experiment.device == "cuda" else "cpu"
    return run_channel_decomposition(cfg, experiment_name, device=device)


def _run_spectral_attribution(cfg, experiment_name: str, gpu: int = 0) -> dict:
    """Run spectral attribution XAI analysis."""
    from src.classification.diagnostics.xai.spectral_attribution import run_spectral_attribution
    device = f"cuda:{gpu}" if cfg.experiment.device == "cuda" else "cpu"
    return run_spectral_attribution(cfg, experiment_name, device=device)


def _run_feature_space(cfg, experiment_name: str, gpu: int = 0) -> dict:
    """Run feature space analysis."""
    from src.classification.diagnostics.xai.feature_space import run_feature_space_analysis
    device = f"cuda:{gpu}" if cfg.experiment.device == "cuda" else "cpu"
    return run_feature_space_analysis(cfg, experiment_name, device=device)


def _run_integrated_gradients(cfg, experiment_name: str, gpu: int = 0) -> dict:
    """Run Integrated Gradients analysis."""
    from src.classification.diagnostics.xai.integrated_gradients import run_integrated_gradients
    device = f"cuda:{gpu}" if cfg.experiment.device == "cuda" else "cpu"
    return run_integrated_gradients(cfg, experiment_name, device=device)


def _run_report(cfg, experiment_name: str) -> None:
    """Generate summary report (legacy)."""
    from src.classification.diagnostics.reporting.summary_report import generate_report
    generate_report(cfg, experiment_name)


def _run_full_report(cfg, experiment_name: str) -> dict:
    """Generate comprehensive structured experiment report."""
    from src.classification.diagnostics.reporting.experiment_report import generate_experiment_report
    return generate_experiment_report(cfg, experiment_name)


def _run_compact_report(cfg, experiment_name: str) -> dict:
    """Generate compact LLM-readable YAML report."""
    from src.classification.diagnostics.reporting.compact_report import generate_compact_report
    return generate_compact_report(cfg, experiment_name)


def _run_aggregate(cfg) -> dict:
    """Aggregate all experiment CSVs into cross-experiment summaries."""
    from src.classification.diagnostics.reporting.cross_experiment import aggregate_experiments
    return aggregate_experiments(cfg)


def _run_paired_comparison(cfg) -> dict:
    """Run paired comparison across experiments."""
    from src.classification.diagnostics.reporting.paired_comparison import run_paired_comparison
    return run_paired_comparison(cfg)


def run_all(cfg, experiment_name: str, gpu: int = 0, skip: list[str] | None = None) -> dict:
    """Run all diagnostic analyses for one experiment.

    Args:
        cfg: Diagnostics configuration.
        experiment_name: Experiment to analyze.
        gpu: GPU device index.
        skip: List of analysis names to skip.

    Returns:
        Dict of all results keyed by analysis name.
    """
    skip = skip or []
    results = {}

    # Phase 1: Independent analyses (patches-based)
    analyses = [
        #("dither", _run_dither),
        ("spectral", _run_spectral),
        ("texture", _run_texture),
        ("bands", _run_bands),
        ("pixel_stats", _run_pixel_stats),
        ("distributions", _run_distributions),
        ("boundary", _run_boundary),
        ("wavelet", _run_wavelet),
    ]

    for name, fn in analyses:
        if name in skip:
            logger.info(f"Skipping {name}")
            continue
        logger.info(f"Running {name} analysis...")
        try:
            results[name] = fn(cfg, experiment_name)
        except Exception as e:
            logger.error(f"Failed {name}: {e}", exc_info=True)
            results[name] = {"error": str(e)}

    # Phase 1b: Full image analyses
    full_image_analyses = [
        ("background", _run_background),
        ("spatial_correlation", _run_spatial_correlation),
        ("global_frequency", _run_global_frequency),
    ]

    for name, fn in full_image_analyses:
        if name in skip:
            logger.info(f"Skipping {name}")
            continue
        logger.info(f"Running {name} analysis...")
        try:
            results[name] = fn(cfg, experiment_name)
        except Exception as e:
            logger.error(f"Failed {name}: {e}", exc_info=True)
            results[name] = {"error": str(e)}

    # Phase 2: GradCAM (depends on trained checkpoints)
    if "gradcam" not in skip:
        logger.info("Running GradCAM analysis...")
        try:
            results["gradcam"] = _run_gradcam(cfg, experiment_name, gpu)
        except Exception as e:
            logger.error(f"Failed gradcam: {e}", exc_info=True)
            results["gradcam"] = {"error": str(e)}

    # Phase 2b: Enhanced XAI (depends on trained checkpoints)
    xai_analyses = [
        ("channel_decomposition", _run_channel_decomposition),
        ("spectral_attribution", _run_spectral_attribution),
        ("feature_space", _run_feature_space),
        ("integrated_gradients", _run_integrated_gradients),
    ]

    for name, fn in xai_analyses:
        if name in skip:
            logger.info(f"Skipping {name}")
            continue
        logger.info(f"Running {name} analysis...")
        try:
            results[name] = fn(cfg, experiment_name, gpu)
        except Exception as e:
            logger.error(f"Failed {name}: {e}", exc_info=True)
            results[name] = {"error": str(e)}

    # Phase 3: Reports
    if "report" not in skip:
        logger.info("Generating experiment report...")
        try:
            _run_full_report(cfg, experiment_name)
        except Exception as e:
            logger.error(f"Failed experiment report: {e}", exc_info=True)

    if "compact_report" not in skip:
        logger.info("Generating compact report...")
        try:
            _run_compact_report(cfg, experiment_name)
        except Exception as e:
            logger.error(f"Failed compact report: {e}", exc_info=True)

    logger.info(f"All analyses complete for '{experiment_name}'")
    return results


def _run_all_experiments_subprocess(
    config_path: str,
    experiments: list[str],
    component: str,
    gpu: int = 0,
    skip: list[str] | None = None,
) -> None:
    """Run diagnostics for all experiments using subprocess isolation.

    Each experiment runs in a separate subprocess to prevent memory
    accumulation from numpy/torch allocations.

    Args:
        config_path: Path to diagnostics.yaml.
        experiments: List of experiment names.
        component: Which component to run (e.g., "all", "dither", etc.).
        gpu: GPU device index.
        skip: Analyses to skip (only for run-all).
    """
    failed: list[str] = []
    total = len(experiments)

    for i, exp_name in enumerate(experiments, 1):
        logger.info(f"{'='*60}")
        logger.info(f"[{i}/{total}] Diagnostics for: {exp_name} ({component})")
        logger.info(f"{'='*60}")

        # Build subprocess command
        cmd = [
            sys.executable, "-m", "src.classification.diagnostics",
            component if component != "all" else "run-all",
            "--config", config_path,
            "--experiment", exp_name,
        ]
        if component in ("run-all", "gradcam"):
            cmd.extend(["--gpu", str(gpu)])
        if skip and component == "run-all":
            cmd.extend(["--skip"] + skip)

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            logger.error(f"FAILED: {exp_name} (exit code {result.returncode})")
            failed.append(exp_name)
        else:
            logger.info(f"Completed: {exp_name}")

    if failed:
        logger.warning(f"{len(failed)}/{total} experiment(s) failed: {failed}")
    else:
        logger.info(f"All {total} experiments completed successfully.")


def run_from_args(args: argparse.Namespace) -> None:
    """Entry point from the classification __main__.py."""
    _setup_logging()
    cfg = _load_config(args.config)

    component = args.component

    # Handle no-experiment commands
    if component == "aggregate":
        _run_aggregate(cfg)
        return
    if component == "paired-comparison":
        _run_paired_comparison(cfg)
        return

    experiment = args.experiment
    gpu = args.gpu

    # Handle --experiment all
    if experiment == "all":
        experiments = _discover_experiments(cfg)
        if not experiments:
            logger.error("No experiments found. Check data.patches_base_dir in config.")
            sys.exit(1)
        logger.info(f"Discovered {len(experiments)} experiments: {experiments}")
        _run_all_experiments_subprocess(
            config_path=args.config,
            experiments=experiments,
            component=component,
            gpu=gpu,
        )
        return

    if component == "all":
        run_all(cfg, experiment, gpu=gpu)
    elif component == "dither":
        _run_dither(cfg, experiment)
    elif component == "gradcam":
        _run_gradcam(cfg, experiment, gpu)
    elif component == "channel-decomp":
        _run_channel_decomposition(cfg, experiment, gpu)
    elif component == "spectral-attr":
        _run_spectral_attribution(cfg, experiment, gpu)
    elif component == "feature-space":
        _run_feature_space(cfg, experiment, gpu)
    elif component == "integrated-gradients":
        _run_integrated_gradients(cfg, experiment, gpu)
    elif component == "spectral":
        _run_spectral(cfg, experiment)
    elif component == "texture":
        _run_texture(cfg, experiment)
    elif component == "stats":
        _run_pixel_stats(cfg, experiment)
        _run_distributions(cfg, experiment)
        _run_boundary(cfg, experiment)
        _run_wavelet(cfg, experiment)
    elif component == "full-image":
        _run_background(cfg, experiment)
        _run_spatial_correlation(cfg, experiment)
        _run_global_frequency(cfg, experiment)
    elif component == "report":
        _run_report(cfg, experiment)
    elif component == "full-report":
        _run_full_report(cfg, experiment)
    elif component == "compact-report":
        _run_compact_report(cfg, experiment)
    else:
        logger.error(f"Unknown component: {component}")
        sys.exit(1)


def main() -> None:
    """Standalone CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="jsddpm-diagnostics",
        description="Diagnostic tools for real vs. synthetic MRI analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run-all ---
    p_all = subparsers.add_parser("run-all", help="Run all diagnostic analyses.")
    p_all.add_argument("--config", required=True, help="Path to diagnostics.yaml")
    p_all.add_argument("--experiment", required=True,
                       help="Experiment name or 'all' to run all detected experiments")
    p_all.add_argument("--gpu", type=int, default=0, help="GPU device index")
    p_all.add_argument("--skip", nargs="*", default=[], help="Analyses to skip")
    p_all.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # --- Aggregate (no experiment required) ---
    p_agg = subparsers.add_parser(
        "aggregate", help="Aggregate all experiment CSVs into cross-experiment summaries."
    )
    p_agg.add_argument("--config", required=True, help="Path to diagnostics.yaml")
    p_agg.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # --- Individual subcommands ---
    gpu_commands = {"gradcam", "channel-decomp", "spectral-attr", "feature-space", "integrated-gradients"}
    for cmd_name, cmd_help in [
        #("dither", "Run dithering and re-classification"),
        ("gradcam", "Run GradCAM analysis"),
        ("channel-decomp", "Run channel decomposition XAI analysis"),
        ("spectral-attr", "Run spectral attribution XAI analysis"),
        ("feature-space", "Run feature space analysis"),
        ("integrated-gradients", "Run Integrated Gradients analysis"),
        ("spectral", "Run spectral (PSD) analysis"),
        ("texture", "Run texture (GLCM, LBP) analysis"),
        ("bands", "Run frequency band analysis"),
        ("pixel-stats", "Run per-pixel statistical tests"),
        ("distributions", "Run distribution comparison tests"),
        ("boundary", "Run lesion boundary analysis"),
        ("wavelet", "Run wavelet decomposition analysis"),
        ("background", "Run background consistency analysis"),
        ("spatial-correlation", "Run spatial autocorrelation analysis"),
        ("global-frequency", "Run global frequency analysis"),
        ("report", "Generate summary report"),
        ("full-report", "Generate structured experiment report with recommendations"),
        ("compact-report", "Generate compact LLM-readable YAML report"),
    ]:
        p = subparsers.add_parser(cmd_name, help=cmd_help)
        p.add_argument("--config", required=True, help="Path to diagnostics.yaml")
        p.add_argument("--experiment", required=True,
                       help="Experiment name or 'all' to run all detected experiments")
        if cmd_name in gpu_commands:
            p.add_argument("--gpu", type=int, default=0, help="GPU device index")
        p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # --- No-experiment commands ---
    p_paired = subparsers.add_parser("paired-comparison",
                                     help="Run paired comparison across experiments")
    p_paired.add_argument("--config", required=True, help="Path to diagnostics.yaml")
    p_paired.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))
    cfg = _load_config(args.config)

    cmd = args.command

    # Handle no-experiment commands
    if cmd == "aggregate":
        _run_aggregate(cfg)
        return
    if cmd == "paired-comparison":
        _run_paired_comparison(cfg)
        return

    experiment = args.experiment

    # Handle --experiment all for any command
    if experiment == "all":
        experiments = _discover_experiments(cfg)
        if not experiments:
            logger.error("No experiments found. Check data.patches_base_dir in config.")
            sys.exit(1)
        logger.info(f"Discovered {len(experiments)} experiments: {experiments}")
        _run_all_experiments_subprocess(
            config_path=args.config,
            experiments=experiments,
            component=cmd,
            gpu=getattr(args, "gpu", 0),
            skip=getattr(args, "skip", None),
        )
        return

    dispatch = {
        "run-all": lambda: run_all(cfg, experiment, gpu=args.gpu, skip=args.skip),
        "dither": lambda: _run_dither(cfg, experiment),
        "gradcam": lambda: _run_gradcam(cfg, experiment, args.gpu),
        "channel-decomp": lambda: _run_channel_decomposition(cfg, experiment, args.gpu),
        "spectral-attr": lambda: _run_spectral_attribution(cfg, experiment, args.gpu),
        "feature-space": lambda: _run_feature_space(cfg, experiment, args.gpu),
        "integrated-gradients": lambda: _run_integrated_gradients(cfg, experiment, args.gpu),
        "spectral": lambda: _run_spectral(cfg, experiment),
        "texture": lambda: _run_texture(cfg, experiment),
        "bands": lambda: _run_bands(cfg, experiment),
        "pixel-stats": lambda: _run_pixel_stats(cfg, experiment),
        "distributions": lambda: _run_distributions(cfg, experiment),
        "boundary": lambda: _run_boundary(cfg, experiment),
        "wavelet": lambda: _run_wavelet(cfg, experiment),
        "background": lambda: _run_background(cfg, experiment),
        "spatial-correlation": lambda: _run_spatial_correlation(cfg, experiment),
        "global-frequency": lambda: _run_global_frequency(cfg, experiment),
        "report": lambda: _run_report(cfg, experiment),
        "full-report": lambda: _run_full_report(cfg, experiment),
        "compact-report": lambda: _run_compact_report(cfg, experiment),
    }

    if cmd in dispatch:
        dispatch[cmd]()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
