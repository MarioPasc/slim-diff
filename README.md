# SLIM-Diff: Shared Latent Image-Mask Diffusion

> A compact joint diffusion model for synthesizing paired FLAIR MRI slices and lesion masks in low-data epilepsy imaging scenarios.

![Graphical Abstract](docs/icip2026/graphical-abstract-readme.png)

## Abstract

Focal Cortical Dysplasia (FCD) is a leading cause of drug-resistant epilepsy, yet automatic detection remains challenging due to subtle MRI features and extremely limited labeled data. We present **SLIM-Diff**, a compact joint diffusion model that simultaneously synthesizes paired FLAIR images and lesion masks through a shared-bottleneck U-Net architecture.

Key findings:
- **x0-prediction parameterization** yields **37x better mask quality** than epsilon-prediction (measured by MMD on morphological features)
- **Tunable Lp loss geometry** (p=2.25) provides optimal trade-off between image fidelity and mask accuracy
- **Self-conditioning** significantly improves sample quality across all metrics
- The model successfully generates anatomically plausible lesion masks with realistic morphology

## Key Features

- **Single shared-bottleneck U-Net** for joint image+mask synthesis
- **x0-prediction parameterization** (37x better mask quality than epsilon-prediction)
- **Tunable Lp loss geometry** for image/mask fidelity trade-offs
- **Z-position and pathology class conditioning** for controllable generation
- **Self-conditioning** for improved sample quality
- **Anatomical prior conditioning** via cross-attention for spatial guidance
- **Focal Frequency Loss (FFL)** for improved edge sharpness

## Installation

```bash
pip install slim-diff
```

For development:

```bash
git clone https://github.com/mpascualgonzalez/slim-diff.git
cd slim-diff
pip install -e ".[dev]"
```

## Quick Start

SLIM-Diff provides four main CLI commands for the complete synthesis pipeline:

### 1. Prepare Slice Cache

Convert 3D MRI volumes into 2D slice cache for efficient training:

```bash
slimdiff-cache --config configs/examples/cache_example.yaml
```

### 2. Train Model

Train the joint diffusion model:

```bash
slimdiff-train --config configs/examples/training_example.yaml
```

### 3. Generate Samples

Generate synthetic image-mask pairs using one of two methods:

**Command-line arguments:**
```bash
slimdiff-generate --config config.yaml --ckpt model.ckpt --out_dir output/
```

**JSON specification (recommended for reproducible experiments):**
```bash
slimdiff-generate-spec --spec generation_spec.json
```

The JSON specification supports two formats:

<details>
<summary>Detailed format - explicit per-zbin specification</summary>

```json
{
    "checkpoint_path": "/path/to/model.ckpt",
    "config_path": "src/diffusion/config/jsddpm.yaml",
    "output_dir": "./outputs/generation_run_001",
    "seed": 42,
    "device": "cuda",
    "format": "detailed",
    "samples": [
        {"zbin": 0, "control": 10, "lesion": 5},
        {"zbin": 5, "control": 20, "lesion": 10},
        {"zbin": 10, "control": 15, "lesion": 15}
    ]
}
```
</details>

<details>
<summary>Compact format - defaults with optional overrides</summary>

```json
{
    "checkpoint_path": "/path/to/model.ckpt",
    "config_path": "src/diffusion/config/jsddpm.yaml",
    "output_dir": "./outputs/generation_run_002",
    "seed": 123,
    "device": "cuda",
    "format": "compact",
    "defaults": {
        "control": 10,
        "lesion": 10,
        "zbins": "all"
    },
    "overrides": [
        {"zbin": 0, "control": 5, "lesion": 2},
        {"zbin": 29, "control": 5, "lesion": 2}
    ]
}
```
</details>

Additional generation options:
```bash
# Validate specification without generating
slimdiff-generate-spec --spec spec.json --validate-only

# Dry run (show plan)
slimdiff-generate-spec --spec spec.json --dry-run

# Verbose output
slimdiff-generate-spec --spec spec.json --verbose
```

### 4. Compute Similarity Metrics

Evaluate generated samples against real data:

```bash
# Image quality metrics (KID, FID, LPIPS)
slimdiff-metrics image-metrics --config metrics_config.yaml

# Mask morphology metrics (MMD-MF)
slimdiff-metrics mask-metrics --config metrics_config.yaml

# Generate publication-ready plots
slimdiff-metrics plot --config metrics_config.yaml
```

## Results

### Similarity Metrics Comparison

![Similarity Metrics](docs/icip2026/similarity-metrics-readme.png)

### Quantitative Results

![Results Table](docs/icip2026/table-readme.png)

## Configuration

Example configuration files are provided in `configs/examples/`:

- `cache_example.yaml` - Slice cache setup
- `training_example.yaml` - Model training
- `generation_spec_example.json` - JSON generation specification
- `metrics_example.yaml` - Similarity metrics computation

See the [configuration documentation](configs/examples/) for detailed parameter descriptions.

## Model Architecture

SLIM-Diff uses a shared-bottleneck U-Net architecture that processes both the FLAIR image and lesion mask jointly:

```
Input: [image, mask] (2 channels)
  |
  v
Encoder (shared) --> Bottleneck (shared) --> Decoder (shared)
  |                                              |
  +-- Z-position embedding                       |
  +-- Pathology class embedding                  |
  +-- Anatomical prior (cross-attention)         |
                                                 v
                                    Output: [image, mask] (2 channels)
```

Key architectural choices:
- **Prediction type**: x0-prediction (directly predicts clean sample)
- **Loss function**: Lp norm (p=2.25) + Focal Frequency Loss
- **Conditioning**: Class embeddings + optional anatomical cross-attention
- **Self-conditioning**: 50% probability during training

## Citation

If you use SLIM-Diff in your research, please cite:

```bibtex
@inproceedings{pascual2026slimdiff,
    title={{SLIM-Diff}: A Compact Joint Diffusion Model for Low-Data Epilepsy Lesion Synthesis},
    author={Pascual Gonz{\'a}lez, Mario and [co-authors]},
    booktitle={IEEE International Conference on Image Processing (ICIP)},
    year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by [funding sources]. We thank [collaborators] for their valuable feedback and contributions.
