  # Diffusion Audition Module

Real vs. Synthetic lesion patch classifier for quality assessment of generated samples.

## Overview

This module implements a binary classifier to distinguish between real and synthetic epilepsy lesion patches. It provides tools for patch extraction, classifier training with z-bin stratification, and comprehensive AUC-based evaluation with bootstrap confidence intervals.

## Structure

```
src/diffusion/audition/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── audition.yaml           # Full configuration file
├── data/
│   ├── __init__.py
│   ├── patch_extractor.py      # Extract lesion-centered patches (280 lines)
│   ├── dataset.py              # PyTorch Dataset for patches (140 lines)
│   └── data_module.py          # Lightning DataModule with z-bin stratification (180 lines)
├── models/
│   ├── __init__.py
│   └── classifier.py           # Simple CNN + ResNet classifier options (200 lines)
├── training/
│   ├── __init__.py
│   ├── lit_module.py           # Lightning module with AUC metrics (200 lines)
│   └── callbacks.py            # CSV logging callbacks (110 lines)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py              # AUC analysis with bootstrap CI, per-zbin (360 lines)
└── scripts/
    ├── __init__.py
    ├── extract_patches.py      # Step 1: Extract patches CLI
    ├── train_classifier.py     # Step 2: Train classifier CLI
    └── evaluate_audition.py    # Step 3: Generate evaluation report CLI
```

## Usage

### Step 1: Extract Patches

Extract lesion-centered patches from real and synthetic data:

```bash
python -m src.diffusion.audition.scripts.extract_patches \
    --config src/diffusion/audition/config/audition.yaml
```

### Step 2: Train Classifier

Train the binary classifier with z-bin stratification:

```bash
python -m src.diffusion.audition.scripts.train_classifier \
    --config src/diffusion/audition/config/audition.yaml
```

### Step 3: Evaluate

Generate comprehensive evaluation report with AUC metrics:

```bash
python -m src.diffusion.audition.scripts.evaluate_audition \
    --config src/diffusion/audition/config/audition.yaml
```

## Configuration

All settings are controlled via [audition.yaml](config/audition.yaml). Key parameters include:
- Patch extraction settings (size, stride, lesion threshold)
- Train/val/test splits with z-bin stratification
- Classifier architecture (SimpleCNN, ResNet18, ResNet34)
- Training hyperparameters (optimizer, scheduler, epochs)
- Evaluation metrics (AUC, bootstrap confidence intervals)     