#!/usr/bin/env python3
"""Evaluate trained audition classifier and generate comprehensive report.

This script loads a trained classifier, runs inference on the test set,
and generates a detailed evaluation report including:
- Global AUC-ROC and PR-AUC with bootstrap confidence intervals
- Per-zbin AUC analysis
- ROC and PR curve visualizations
- Interpretation of synthetic data quality

Usage:
    python -m src.diffusion.audition.scripts.evaluate_audition --config path/to/audition.yaml

Example:
    python -m src.diffusion.audition.scripts.evaluate_audition \
        --config src/diffusion/audition/config/audition.yaml \
        --checkpoint outputs/audition/checkpoints/best.ckpt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from ..data.data_module import AuditionDataModule
from ..evaluation.metrics import generate_evaluation_report
from ..training.lit_module import AuditionLightningModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_best_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Find the best checkpoint in the checkpoints directory.

    Args:
        checkpoints_dir: Directory containing checkpoints.

    Returns:
        Path to best checkpoint or None if not found.
    """
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        return None

    # Look for checkpoints with 'val_auc' in name
    ckpts = list(checkpoints_dir.glob("*.ckpt"))
    if not ckpts:
        return None

    # Filter out 'last.ckpt' and sort by name (assumes val_auc is in filename)
    best_ckpts = [c for c in ckpts if "last" not in c.name]
    if best_ckpts:
        # Sort by modification time (most recent)
        best_ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return best_ckpts[0]

    # Fall back to last.ckpt
    last_ckpt = checkpoints_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    return ckpts[0] if ckpts else None


def main() -> None:
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate audition classifier and generate report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.diffusion.audition.scripts.evaluate_audition \\
        --config src/diffusion/audition/config/audition.yaml

    python -m src.diffusion.audition.scripts.evaluate_audition \\
        --config src/diffusion/audition/config/audition.yaml \\
        --checkpoint outputs/audition/checkpoints/best.ckpt
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to audition configuration YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-detects best if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_best_checkpoint(Path(cfg.output.checkpoints_dir))

    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Set seed
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Create data module
    logger.info("Initializing data module...")
    data_module = AuditionDataModule(cfg)
    data_module.setup(stage="test")

    # Load model
    logger.info("Loading model from checkpoint...")
    model = AuditionLightningModule.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,
        strict=True,
        weights_only=False
    )
    model.eval()

    # Get test dataloader
    test_loader = data_module.test_dataloader()

    # Run inference
    logger.info("Running inference on test set...")
    all_probs = []
    all_labels = []
    all_zbins = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for batch in test_loader:
            patches = batch["patch"].to(device)
            labels = batch["label"]
            z_bins = batch["z_bin"]

            logits = model(patches)
            probs = torch.sigmoid(logits).squeeze(1).cpu()

            all_probs.append(probs)
            all_labels.append(labels)
            all_zbins.append(z_bins)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    z_bins = torch.cat(all_zbins).numpy()

    logger.info(f"Collected {len(probs)} test samples")

    # Generate report
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.output.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating evaluation report...")
    report = generate_evaluation_report(
        probs=probs,
        labels=labels,
        z_bins=z_bins,
        output_dir=output_dir,
        cfg=cfg,
    )

    logger.info(f"Results saved to: {output_dir}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
