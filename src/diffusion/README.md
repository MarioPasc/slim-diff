# JS-DDPM: Joint-Synthesis Denoising Diffusion Probabilistic Model

A 2D Joint-Synthesis DDPM for generating paired (FLAIR slice, lesion mask slice) samples, conditioned on pathology presence and slice depth (z-position).

## Quick Start

### 1. Installation

```bash
conda activate jsddpm
pip install -r requirements.txt
```

**Important**: Ensure `einops` is installed:
```bash
pip install einops
```

### 2. Build Slice Cache

Convert 3D volumes into 2D slices and cache them:

```bash
python -m src.diffusion.data.caching --config src/diffusion/config/jsddpm.yaml
```

This will:
- Load 3D volumes from epilepsy and control datasets
- Resample to 128×128×128
- Extract axial slices respecting `z_range` configuration
- Save as .npz files with metadata
- Create train/val/test CSV indices

**Output**: `{cache_dir}/slices/` and `{cache_dir}/{train,val,test}.csv`

### 3. Train Model

```bash
python -m src.diffusion.training.runners.train --config src/diffusion/config/jsddpm.yaml
```

Optional arguments:
- `--seed SEED`: Override random seed
- `--output_dir DIR`: Override output directory
- `--resume PATH`: Resume from checkpoint

**Output**: Model checkpoints, logs, and visualizations in `outputs/jsddpm/`

### 4. Generate Synthetic Samples

```bash
python -m src.diffusion.training.runners.generate \
    --config src/diffusion/config/jsddpm.yaml \
    --ckpt outputs/jsddpm/checkpoints/best.ckpt \
    --out_dir ./generated_samples \
    --n_per_condition 100
```

Optional arguments:
- `--z_bins "0,12,25,37,49"`: Specific z-bins to generate (comma-separated)
- `--classes "0,1"`: Classes to generate (0=control, 1=lesion)
- `--seed SEED`: Random seed for reproducibility

**Output**: Generated .npz files and index CSV in `generated_samples/`

## Configuration

All settings are controlled via `src/diffusion/config/jsddpm.yaml`.

### Key Parameters

#### Z-Range Filtering (NEW)

Control which axial slices to use:

```yaml
slice_sampling:
  z_range: [40, 100]  # Only use slices 40-100 (middle of volume)
```

**Important**: This affects:
- ✅ Slice caching (only specified slices are saved)
- ✅ Training (model only sees these slices)
- ✅ Generation (only z_bins from this range are available)

**Use cases**:
- Train on middle slices only: `[40, 100]` (excludes top/bottom of brain)
- Full volume: `[0, 127]` (all 128 slices)
- Top half: `[0, 63]`
- Bottom half: `[64, 127]`

#### Data Paths

```yaml
data:
  root_dir: "/path/to/epilepsy/data"
  cache_dir: "${data.root_dir}/slice_cache"

  epilepsy:
    name: "Dataset210_MRIe_none"
    modality_index: 0  # 0=FLAIR, 1=T1N

  control:
    name: "Dataset310_MRIcontrol_none"
    modality_index: 0
```

#### Model Architecture

```yaml
model:
  channels: [64, 128, 256, 256]
  attention_levels: [false, false, true, true]
  num_res_blocks: 2
  num_head_channels: 32
  norm_num_groups: 32
```

#### Training

```yaml
training:
  batch_size: 32
  max_epochs: 200
  precision: "16-mixed"

  optimizer:
    lr: 1.0e-4
    weight_decay: 1.0e-4
```

#### Conditioning

```yaml
conditioning:
  z_bins: 50  # Quantize 128 slices into 50 bins
  # Total tokens: 100 (50 z_bins × 2 classes)
```

Token mapping:
- Tokens 0-49: Control/no-lesion slices at different z-positions
- Tokens 50-99: Lesion-present slices at different z-positions

## Module Structure

```
src/diffusion/
├── config/
│   └── jsddpm.yaml           # Main configuration file
├── model/
│   ├── factory.py            # Model, scheduler, sampler builders
│   ├── components/
│   │   └── conditioning.py   # Token computation
│   └── embeddings/
│       └── zpos.py           # Z-position encoding
├── data/
│   ├── caching.py           # Slice cache builder ⭐ z_range filtering
│   ├── dataset.py           # PyTorch Dataset
│   ├── transforms.py        # MONAI transforms
│   └── splits.py            # Train/val/test splitting
├── losses/
│   ├── uncertainty.py       # Kendall uncertainty weighting
│   └── diffusion_losses.py # Epsilon MSE loss
├── training/
│   ├── lit_modules.py       # Lightning training module
│   ├── metrics.py           # PSNR, SSIM, Dice
│   ├── callbacks/
│   │   └── epoch_callbacks.py  # Visualization callback
│   └── runners/
│       ├── train.py         # Training entrypoint
│       └── generate.py      # Generation entrypoint ⭐ z_range filtering
└── utils/
    ├── io.py                # File I/O
    ├── seeding.py          # Reproducibility
    └── logging.py          # Logger setup
```

⭐ = Files implementing z_range functionality

## Workflow

### Complete Pipeline

```bash
# 1. Build cache (respects z_range)
python -m src.diffusion.data.caching --config src/diffusion/config/jsddpm.yaml

# 2. Train model
python -m src.diffusion.training.runners.train --config src/diffusion/config/jsddpm.yaml

# 3. Generate samples (respects z_range)
python -m src.diffusion.training.runners.generate \
    --config src/diffusion/config/jsddpm.yaml \
    --ckpt outputs/jsddpm/checkpoints/jsddpm-epoch_0100-val_loss_0.0123.ckpt \
    --out_dir ./synthetic_data \
    --n_per_condition 200
```

### Using SLURM

```bash
sbatch slurm/train_jsddpm.sh
```

The script:
- Auto-assigns available GPU
- Activates jsddpm conda environment
- Builds cache
- Trains model
- Saves outputs to `{RESULTS_DST}`

## Data Format

### Input (3D Volumes)

- **Epilepsy**: NIfTI files in `Dataset210_MRIe_none/imagesTr/` and `labelsTr/`
- **Control**: NIfTI files in `Dataset310_MRIcontrol_none/imagesTr/` (no labels)
- **Format**: Skull-stripped, KDE-normalized, ~188×232×196, ~1mm isotropic

### Output (2D Slices)

Each cached slice is a .npz file with:
- `image`: (128, 128) float32 in [-1, 1]
- `mask`: (128, 128) float32 in {-1, +1}
- Metadata: subject_id, z_index, z_bin, pathology_class, token, etc.

### Generated Samples

Same format as cached slices:
- `generated_samples/samples/{sample_id}.npz`
- `generated_samples/generated_samples.csv` (index)

## Conditioning

### Z-Position Encoding

1. **Z-index**: 0-127 (128 slices after resampling)
2. **Z-normalization**: `z_norm = z_index / 127`
3. **Z-bin**: `z_bin = floor(z_norm * 50)` clamped to [0, 49]

### Pathology Class

Per-slice classification:
- **Class 0**: No lesion (control OR epilepsy slice with no lesion pixels)
- **Class 1**: Lesion present (epilepsy slice with lesion pixels)

### Token Computation

```python
token = z_bin + pathology_class * 50

# Examples:
# z_bin=0, class=0 → token=0   (control, bottom slice)
# z_bin=25, class=0 → token=25 (control, middle slice)
# z_bin=0, class=1 → token=50  (lesion, bottom slice)
# z_bin=25, class=1 → token=75 (lesion, middle slice)
```

## Visualization

During training, a 2×5 grid is generated each validation epoch:

```
Row 1: [Control z=0] [Control z=12] [Control z=25] [Control z=37] [Control z=49]
Row 2: [Lesion z=0]  [Lesion z=12]  [Lesion z=25]  [Lesion z=37]  [Lesion z=49]
       (with mask)   (with mask)    (with mask)    (with mask)    (with mask)
```

Saved to: `outputs/jsddpm/viz/epoch_{epoch:04d}.png`

## Metrics

### Training Metrics

- **Loss**: Uncertainty-weighted MSE (Kendall et al.)
  - `loss_image`: MSE on FLAIR channel
  - `loss_mask`: MSE on mask channel
  - `log_var_image`, `log_var_mask`: Learned uncertainty weights

### Validation Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **Dice** (optional): Overlap for mask channel

## Testing

Run all smoke tests:

```bash
python -m pytest src/diffusion/tests/test_smoke.py -v
```

Expected output:
```
======================== 20 passed, 1 warning in 2.26s =========================
```

## Troubleshooting

### ImportError: No module named 'einops'

```bash
pip install einops
```

### CUDA out of memory

Reduce batch size in `jsddpm.yaml`:
```yaml
training:
  batch_size: 16  # or 8
```

### Cache directory not found

Check paths in YAML:
```yaml
data:
  root_dir: "/correct/path/to/data"
```

### Z-range filtering not working

Ensure you rebuild the cache after changing `z_range`:
```bash
# Delete old cache
rm -rf /path/to/slice_cache

# Rebuild with new z_range
python -m src.diffusion.data.caching --config src/diffusion/config/jsddpm.yaml
```

## References

- DDPM: [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- Uncertainty Weighting: [Multi-Task Learning Using Uncertainty (Kendall et al., 2018)](https://arxiv.org/abs/1705.07115)
- MONAI: [Medical Open Network for AI](https://monai.io/)

## Citation

If you use this code, please cite:

```bibtex
@software{jsddpm2024,
  title={JS-DDPM: Joint-Synthesis Denoising Diffusion Probabilistic Model},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/js-ddpm-epilepsy}
}
```
