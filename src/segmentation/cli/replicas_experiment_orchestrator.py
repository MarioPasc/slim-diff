#!/usr/bin/env python3
"""Dataset Expansion Experiment Orchestrator.

This script orchestrates experiments to measure the effect of synthetic data expansion
on segmentation performance (DICE score). It trains models with synthetic-only data
at different expansion levels (x1 to x10 replicas) and evaluates on real test data.

Each replica contains ~9000 images (balanced classes), approximately matching the
real training dataset size (~9119 images). This allows us to study how increasing
synthetic data quantity affects generalization to real data.

Experiment structure:
    - 10 expansion levels (x1, x2, ..., x10)
    - 3 models (dynunet, unet, swinunetr)
    - Total: 30 experiments

Output organization:
    output_dir/
    └── dataset_expansion_experiment/
        ├── experiment_planification.csv  (with expansion, model, status columns)
        ├── x1/
        │   ├── synth_x1_dynunet/
        │   ├── synth_x1_unet/
        │   └── synth_x1_swinunetr/
        ├── x2/
        │   └── ...
        └── x10/
            └── ...

Usage:
    # Dry run - create planification only
    python -m src.segmentation.cli.replicas_experiment_orchestrator \
        --expansions x1,x2,x3,x4,x5 \
        --models dynunet,unet,swinunetr \
        --output-dir /media/hddb/mario/results/epilepsy/segmentation \
        --device 0 \
        --dry-run

    # Full execution across 2 GPUs
    python -m src.segmentation.cli.replicas_experiment_orchestrator \\
        --expansions x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 \\
        --models dynunet,unet,swinunetr \\
        --folds 5 \\
        --output-dir /media/hddb/mario/results/epilepsy/segmentation \\
        --device 0,1
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import tempfile
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
DEFAULT_MODELS = ["dynunet", "unet", "swinunetr"]
DEFAULT_EXPANSIONS = [f"x{i}" for i in range(1, 11)]  # x1, x2, ..., x10
DEFAULT_DEVICE = "0"
DEFAULT_FOLDS = 5

# Experiment subfolder name
EXPERIMENT_SUBFOLDER = "dataset_expansion_experiment"



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Orchestrate dataset expansion experiments (synth train -> real test)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - create planification only
  python -m src.segmentation.cli.replicas_experiment_orchestrator \\
      --expansions x1,x2,x3 \\
      --models dynunet,unet,swinunetr \\
      --output-dir ./outputs/experiments \\
      --device 0 \\
      --dry-run

  # Full execution across 2 GPUs
  python -m src.segmentation.cli.replicas_experiment_orchestrator \\
      --expansions x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 \\
      --models dynunet,unet,swinunetr \\
      --folds 5 \\
      --output-dir ./outputs/experiments \\
      --device 0,1

  # Single expansion with specific folds
  python -m src.segmentation.cli.replicas_experiment_orchestrator \\
      --expansions x5 \\
      --models unet \\
      --folds 0,1,2 \\
      --output-dir ./outputs/test \\
      --device 0
        """,
    )

    # Expansion selection (replaces --experiments)
    parser.add_argument(
        "--expansions",
        type=str,
        default=",".join(DEFAULT_EXPANSIONS),
        help=f"Comma-separated list of expansion levels to run (default: %(default)s). "
        f"Must match YAML files in src/segmentation/config/replicas/ (e.g., x1.yaml, x2.yaml)",
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
    expansions: List[str],
    models: List[str],
    config_dir: Path,
) -> Tuple[bool, List[str]]:
    """Validate that all expansion and model configs exist.

    Args:
        expansions: List of expansion levels (x1, x2, ..., x10)
        models: List of model names
        config_dir: Base configuration directory

    Returns:
        (is_valid, error_messages) tuple
    """
    errors = []

    # Check expansion configs (in config/replicas/)
    replicas_dir = config_dir / "replicas"
    for exp in expansions:
        exp_file = replicas_dir / f"{exp}.yaml"
        if not exp_file.exists():
            errors.append(f"Expansion config not found: {exp_file}")

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
    expansions: List[str],
    models: List[str],
    output_dir: Path,
    devices: List[str],
) -> List[dict]:
    """Create experiment planification for dataset expansion experiment.

    Args:
        expansions: List of expansion levels (x1, x2, ..., x10)
        models: List of model names
        output_dir: Base output directory (will contain dataset_expansion_experiment/)
        devices: List of GPU device indices

    Returns:
        List of planification entries (dicts)
    """
    planification = []
    device_idx = 0

    # Base experiment directory
    experiment_base = output_dir / EXPERIMENT_SUBFOLDER

    for expansion in expansions:
        for model in models:
            # Output structure: output_dir/dataset_expansion_experiment/x{N}/synth_x{N}_{model}
            exp_folder = experiment_base / expansion / f"synth_{expansion}_{model}"

            # Assign device in round-robin fashion
            assigned_device = devices[device_idx % len(devices)]
            device_idx += 1

            # Extract numeric expansion level for easier analysis
            expansion_num = int(expansion.replace("x", ""))

            entry = {
                "expansion": expansion,
                "expansion_num": expansion_num,
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
        output_dir: Output directory (should be dataset_expansion_experiment/)

    Returns:
        Path to saved CSV
    """
    csv_path = output_dir / "experiment_planification.csv"

    fieldnames = ["expansion", "expansion_num", "model", "output_folder", "device", "status"]

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
                    f"{entry['expansion']} + {entry['model']}")
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
    """Execute a single expansion-model combination.

    All expansion configs (x1-x10) use synthetic training only mode:
    - Training: synthetic data only
    - Validation/Test: real data only

    Args:
        entry: Planification entry dict
        config_dir: Base configuration directory
        master_config: Path to master config
        folds_arg: Folds argument to pass to kfold_segmentation
        dry_run: Whether to run in dry-run mode

    Returns:
        True if successful, False otherwise
    """
    expansion = entry["expansion"]
    model = entry["model"]
    output_folder = entry["output_folder"]
    device = entry["device"]

    logger.info(f"{'='*80}")
    logger.info(f"Executing: {expansion} + {model} on GPU {device}")
    logger.info(f"Output: {output_folder}")
    logger.info(f"{'='*80}")

    # All expansion configs are synthetic training only - modify config to enable
    # real data for validation/test while training with synthetic only
    experiment_config_path = config_dir / "replicas" / f"{expansion}.yaml"
    temp_config_file = None

    logger.info(f"Synthetic training only mode - enabling real data for validation and test")

    # Load the expansion config
    exp_cfg = OmegaConf.load(experiment_config_path)

    # Override to enable real data and set synthetic training_only mode
    exp_cfg.data.real.enabled = True
    exp_cfg.data.synthetic.training_only = True

    # Create a temporary config file with the overrides
    temp_config_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        delete=False,
        prefix=f'{expansion}_modified_{model}_'
    )
    OmegaConf.save(exp_cfg, temp_config_file.name)
    temp_config_file.close()
    experiment_config_path = Path(temp_config_file.name)

    logger.info(f"Using modified config: {experiment_config_path}")

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
        str(experiment_config_path),
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

    success = False
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,
        )
        logger.info(f"✓ Completed: {expansion} + {model}")
        success = True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed: {expansion} + {model}")
        logger.error(f"Error: {e}")
        success = False
    finally:
        # Clean up temporary config file
        if temp_config_file is not None:
            try:
                os.unlink(temp_config_file.name)
                logger.debug(f"Cleaned up temporary config: {temp_config_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary config: {e}")

    return success


def main():
    """Main entry point for dataset expansion experiment orchestrator."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Banner
    logger.info("=" * 80)
    logger.info("DATASET EXPANSION EXPERIMENT ORCHESTRATOR")
    logger.info("Synthetic Train -> Real Val/Test")
    logger.info("=" * 80)

    # Parse lists
    expansions = [e.strip() for e in args.expansions.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    devices = [d.strip() for d in args.device.split(",")]

    logger.info(f"Expansions: {expansions}")
    logger.info(f"Models: {models}")
    logger.info(f"Devices: {devices}")
    logger.info(f"Folds: {args.folds or 'use experiment config'}")
    logger.info(f"Output dir: {args.output_dir}/{EXPERIMENT_SUBFOLDER}/")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTION'}")
    logger.info(f"Execution: {'SEQUENTIAL' if args.sequential else 'PARALLEL (per device)'}")

    # Validate configurations
    config_dir = Path(args.config_dir)
    is_valid, errors = validate_configs(expansions, models, config_dir)

    if not is_valid:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    logger.info("✓ All configurations validated")

    # Create output directory (includes dataset_expansion_experiment subfolder)
    output_dir = Path(args.output_dir)
    experiment_dir = output_dir / EXPERIMENT_SUBFOLDER
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create planification
    logger.info("\nCreating experiment planification...")
    planification = create_planification(expansions, models, output_dir, devices)

    # Save planification to experiment_dir
    csv_path = save_planification(planification, experiment_dir)

    # Print planification table
    logger.info("\nExperiment Planification:")
    logger.info("-" * 100)
    logger.info(f"{'Expansion':<12} {'Model':<15} {'Device':<10} {'Output Folder'}")
    logger.info("-" * 100)
    for entry in planification:
        logger.info(
            f"{entry['expansion']:<12} {entry['model']:<15} "
            f"GPU {entry['device']:<7} {entry['output_folder']}"
        )
    logger.info("-" * 100)
    logger.info(f"Total experiments: {len(planification)}")

    # Execute experiments
    master_config = config_dir / "master.yaml"
    successful = 0
    failed = 0

    if args.sequential or len(devices) == 1:
        # Sequential execution (original behavior)
        logger.info("\nRunning experiments SEQUENTIALLY")
        for i, entry in enumerate(planification, 1):
            logger.info(f"\n[{i}/{len(planification)}] Processing: {entry['expansion']} + {entry['model']}")

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
            save_planification(planification, experiment_dir)
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
        save_planification(planification, experiment_dir)

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
        logger.info(f"\nResults saved to: {experiment_dir}")

    if failed > 0:
        logger.warning(f"\n{failed} experiment(s) failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
