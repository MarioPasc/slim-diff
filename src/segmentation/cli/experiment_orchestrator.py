#!/usr/bin/env python3
"""Experiment Orchestrator for K-fold Segmentation Training.

This script orchestrates multiple segmentation experiments across different models
and data configurations. It creates a planification CSV, assigns experiments to GPUs,
and executes training runs in a structured manner.

Usage:
    # --dry-run: create planification and folder structure only
    python -m src.segmentation.cli.experiment_orchestrator \
        --experiments real_only,real_synthetic_balance \
        --models unet,dynunet,swinunetr \
        --output-dir /media/hddb/mario/results/epilepsy/segmentation \
        --device 0 

    python -m src.segmentation.cli.experiment_orchestrator \
        --experiments real_synthetic_concat,synthetic_only \
        --models unet,dynunet,swinunetr \
        --output-dir /media/hddb/mario/results/epilepsy/segmentation \
        --device 1 

    # Full execution
    python -m src.segmentation.cli.experiment_orchestrator \\
        --experiments real_only,real_synthetic_balance,real_synthetic_concat,synthetic_only \\
        --models dynunet,swinunetr,unet,unetplusplus \\
        --folds 5 \\
        --output-dir ./outputs/experiments \\
        --device 0,1,2,3
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from omegaconf import OmegaConf

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")


# Default values
DEFAULT_MODELS = ["dynunet", "swinunetr", "unet", "unetplusplus"]
DEFAULT_EXPERIMENTS = [
    "real_only",
    "real_synthetic_balance",
    "real_synthetic_concat",
    "synthetic_only",
]
DEFAULT_DEVICE = "0"
DEFAULT_FOLDS = 5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Orchestrate multiple k-fold segmentation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - create planification only
python -m src.segmentation.cli.experiment_orchestrator \
--experiments real_only,real_synthetic_balance \
--models unet,dynunet \
--output-dir ./outputs/experiments \
--device 0,1 \
--dry-run

  # Full execution across 4 GPUs
  python -m src.segmentation.cli.experiment_orchestrator \\
      --experiments real_only,real_synthetic_balance,real_synthetic_concat,synthetic_only \\
      --models dynunet,swinunetr,unet,unetplusplus \\
      --folds 5 \\
      --output-dir ./outputs/experiments \\
      --device 0,1,2,3

  # Single experiment with specific folds
  python -m src.segmentation.cli.experiment_orchestrator \\
      --experiments real_only \\
      --models unet \\
      --folds 0,1,2 \\
      --output-dir ./outputs/test \\
      --device 0
        """,
    )

    # Experiment selection
    parser.add_argument(
        "--experiments",
        type=str,
        default=",".join(DEFAULT_EXPERIMENTS),
        help=f"Comma-separated list of experiments to run (default: %(default)s). "
        f"Must match YAML files in src/segmentation/config/experiments/",
    )

    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated list of models to train (default: %(default)s). "
        f"Must match YAML files in src/segmentation/config/models/",
    )

    # K-fold configuration
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Folds to run. Can be: "
        "(1) Single integer for n_folds (e.g., '5' = 5 folds, run all), "
        "(2) Comma-separated fold indices (e.g., '0,1,2' = run folds 0,1,2 only). "
        "Overrides experiment config. Default: use experiment config.",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory for all experiments",
    )

    # GPU configuration
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Comma-separated list of GPU indices (default: %(default)s). "
        f"Experiments will be distributed across devices in round-robin fashion.",
    )

    # Execution mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create planification and folder structure only, without training",
    )

    # Advanced options
    parser.add_argument(
        "--config-dir",
        type=str,
        default="src/segmentation/config",
        help="Base configuration directory (default: %(default)s)",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run experiments sequentially instead of parallel (useful for debugging)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def validate_configs(
    experiments: List[str],
    models: List[str],
    config_dir: Path,
) -> Tuple[bool, List[str]]:
    """Validate that all experiment and model configs exist.

    Args:
        experiments: List of experiment names
        models: List of model names
        config_dir: Base configuration directory

    Returns:
        (is_valid, error_messages) tuple
    """
    errors = []

    # Check experiments
    exp_dir = config_dir / "experiments"
    for exp in experiments:
        exp_file = exp_dir / f"{exp}.yaml"
        if not exp_file.exists():
            errors.append(f"Experiment config not found: {exp_file}")

    # Check models
    model_dir = config_dir / "models"
    for model in models:
        model_file = model_dir / f"{model}.yaml"
        if not model_file.exists():
            errors.append(f"Model config not found: {model_file}")

    # Check master config
    master_file = config_dir / "master.yaml"
    if not master_file.exists():
        errors.append(f"Master config not found: {master_file}")

    return len(errors) == 0, errors


def create_planification(
    experiments: List[str],
    models: List[str],
    output_dir: Path,
    devices: List[str],
) -> List[dict]:
    """Create experiment planification.

    Args:
        experiments: List of experiment names
        models: List of model names
        output_dir: Base output directory
        devices: List of GPU device indices

    Returns:
        List of planification entries (dicts)
    """
    planification = []
    device_idx = 0

    for exp in experiments:
        for model in models:
            # Create output folder for this experiment-model combination
            exp_folder = output_dir / f"{exp}_{model}"

            # Assign device in round-robin fashion
            assigned_device = devices[device_idx % len(devices)]
            device_idx += 1

            entry = {
                "experiment": exp,
                "model": model,
                "output_folder": str(exp_folder),
                "device": assigned_device,
                "status": "pending",
            }

            planification.append(entry)

    return planification


def save_planification(
    planification: List[dict],
    output_dir: Path,
) -> Path:
    """Save planification to CSV.

    Args:
        planification: List of planification entries
        output_dir: Output directory

    Returns:
        Path to saved CSV
    """
    csv_path = output_dir / "experiment_planification.csv"

    fieldnames = ["experiment", "model", "output_folder", "device", "status"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(planification)

    logger.info(f"Saved experiment planification: {csv_path}")
    return csv_path


def parse_folds_arg(folds_str: str | None) -> Tuple[int | None, List[int] | None]:
    """Parse the --folds argument.

    Args:
        folds_str: Folds string from CLI

    Returns:
        (n_folds, folds_to_run) tuple.
        - If single int: (n, None) = create n folds, run all
        - If comma-separated: (None, [0,1,2]) = run specific folds
        - If None: (None, None) = use experiment config
    """
    if folds_str is None:
        return None, None

    # Try to parse as single integer (n_folds)
    try:
        n_folds = int(folds_str)
        return n_folds, None
    except ValueError:
        pass

    # Parse as comma-separated fold indices
    try:
        folds_to_run = [int(f.strip()) for f in folds_str.split(",")]
        return None, folds_to_run
    except ValueError:
        logger.error(f"Invalid --folds argument: {folds_str}")
        sys.exit(1)


def execute_gpu_queue(
    gpu_id: str,
    experiments: List[dict],
    config_dir: Path,
    master_config: Path,
    folds_arg: str | None,
    dry_run: bool,
) -> List[Tuple[dict, bool]]:
    """Execute all experiments assigned to a single GPU sequentially.

    Args:
        gpu_id: GPU device ID
        experiments: List of experiment entries for this GPU
        config_dir: Base configuration directory
        master_config: Path to master config
        folds_arg: Folds argument to pass to kfold_segmentation
        dry_run: Whether to run in dry-run mode

    Returns:
        List of (entry, success) tuples
    """
    results = []
    for i, entry in enumerate(experiments, 1):
        logger.info(f"[GPU {gpu_id}] Running experiment {i}/{len(experiments)}: "
                    f"{entry['experiment']} + {entry['model']}")
        success = execute_experiment(entry, config_dir, master_config, folds_arg, dry_run)
        results.append((entry, success))
    return results


def execute_experiment(
    entry: dict,
    config_dir: Path,
    master_config: Path,
    folds_arg: str | None,
    dry_run: bool,
) -> bool:
    """Execute a single experiment-model combination.

    Args:
        entry: Planification entry dict
        config_dir: Base configuration directory
        master_config: Path to master config
        folds_arg: Folds argument to pass to kfold_segmentation
        dry_run: Whether to run in dry-run mode

    Returns:
        True if successful, False otherwise
    """
    exp = entry["experiment"]
    model = entry["model"]
    output_folder = entry["output_folder"]
    device = entry["device"]

    logger.info(f"{'='*80}")
    logger.info(f"Executing: {exp} + {model} on GPU {device}")
    logger.info(f"Output: {output_folder}")
    logger.info(f"{'='*80}")

    # Build command
    cmd = [
        "python",
        "-m",
        "src.segmentation.cli.kfold_segmentation",
        "--model",
        model,
        "--config",
        str(master_config),
        "--output-dir",
        output_folder,
        "--experiment-config",
        str(config_dir / "experiments" / f"{exp}.yaml"),
    ]

    # Add folds argument if provided
    if folds_arg is not None:
        n_folds, folds_to_run = parse_folds_arg(folds_arg)
        if n_folds is not None:
            cmd.extend(["--n-folds", str(n_folds)])
        if folds_to_run is not None:
            cmd.extend(["--folds"] + [str(f) for f in folds_to_run])

    # Add dry-run flag if needed
    if dry_run:
        cmd.append("--dry-run")

    # Set CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device

    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Environment: CUDA_VISIBLE_DEVICES={device}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,
        )
        logger.info(f"✓ Completed: {exp} + {model}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed: {exp} + {model}")
        logger.error(f"Error: {e}")
        return False


def main():
    """Main entry point for experiment orchestrator."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Banner
    logger.info("=" * 80)
    logger.info("EXPERIMENT ORCHESTRATOR - K-FOLD SEGMENTATION")
    logger.info("=" * 80)

    # Parse lists
    experiments = [e.strip() for e in args.experiments.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    devices = [d.strip() for d in args.device.split(",")]

    logger.info(f"Experiments: {experiments}")
    logger.info(f"Models: {models}")
    logger.info(f"Devices: {devices}")
    logger.info(f"Folds: {args.folds or 'use experiment config'}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTION'}")
    logger.info(f"Execution: {'SEQUENTIAL' if args.sequential else 'PARALLEL (per device)'}")

    # Validate configurations
    config_dir = Path(args.config_dir)
    is_valid, errors = validate_configs(experiments, models, config_dir)

    if not is_valid:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    logger.info("✓ All configurations validated")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create planification
    logger.info("\nCreating experiment planification...")
    planification = create_planification(experiments, models, output_dir, devices)

    # Save planification
    csv_path = save_planification(planification, output_dir)

    # Print planification table
    logger.info("\nExperiment Planification:")
    logger.info("-" * 80)
    logger.info(f"{'Experiment':<30} {'Model':<15} {'Device':<10} {'Output Folder'}")
    logger.info("-" * 80)
    for entry in planification:
        logger.info(
            f"{entry['experiment']:<30} {entry['model']:<15} "
            f"GPU {entry['device']:<7} {entry['output_folder']}"
        )
    logger.info("-" * 80)
    logger.info(f"Total experiments: {len(planification)}")

    # Execute experiments
    master_config = config_dir / "master.yaml"
    successful = 0
    failed = 0

    if args.sequential or len(devices) == 1:
        # Sequential execution (original behavior)
        logger.info("\nRunning experiments SEQUENTIALLY")
        for i, entry in enumerate(planification, 1):
            logger.info(f"\n[{i}/{len(planification)}] Processing: {entry['experiment']} + {entry['model']}")

            success = execute_experiment(
                entry,
                config_dir,
                master_config,
                args.folds,
                args.dry_run,
            )

            if success:
                entry["status"] = "completed" if not args.dry_run else "dry_run_ok"
                successful += 1
            else:
                entry["status"] = "failed"
                failed += 1

            # Update planification CSV with status
            save_planification(planification, output_dir)
    else:
        # Parallel execution: group experiments by GPU and run GPU queues in parallel
        logger.info(f"\nRunning experiments in PARALLEL across {len(devices)} GPUs")

        # Group experiments by GPU
        gpu_queues: Dict[str, List[dict]] = defaultdict(list)
        for entry in planification:
            gpu_queues[entry["device"]].append(entry)

        # Log queue distribution
        for gpu_id, queue in gpu_queues.items():
            logger.info(f"  GPU {gpu_id}: {len(queue)} experiments")

        # Run GPU queues in parallel using ProcessPoolExecutor
        # Each GPU gets its own process that runs experiments sequentially
        with ProcessPoolExecutor(max_workers=len(devices)) as executor:
            futures = {}
            for gpu_id, queue in gpu_queues.items():
                future = executor.submit(
                    execute_gpu_queue,
                    gpu_id,
                    queue,
                    config_dir,
                    master_config,
                    args.folds,
                    args.dry_run,
                )
                futures[future] = gpu_id

            # Collect results as they complete
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    results = future.result()
                    for entry, success in results:
                        if success:
                            entry["status"] = "completed" if not args.dry_run else "dry_run_ok"
                            successful += 1
                        else:
                            entry["status"] = "failed"
                            failed += 1
                    logger.info(f"GPU {gpu_id} completed all experiments")
                except Exception as e:
                    logger.error(f"GPU {gpu_id} queue failed with error: {e}")
                    # Mark all experiments for this GPU as failed
                    for entry in gpu_queues[gpu_id]:
                        entry["status"] = "failed"
                        failed += 1

        # Save final planification
        save_planification(planification, output_dir)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ORCHESTRATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total experiments: {len(planification)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Planification saved to: {csv_path}")

    if args.dry_run:
        logger.info("\nDry run complete. Review the planification and folder structure.")
        logger.info("Run without --dry-run to execute training.")
    else:
        logger.info(f"\nResults saved to: {output_dir}")

    if failed > 0:
        logger.warning(f"\n{failed} experiment(s) failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
