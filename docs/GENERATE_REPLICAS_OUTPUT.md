# Generate Replicas Script Output

This document describes the output produced by `slurm/generate_replicas_array.sh` and the underlying `generate_replicas.py` runner.

## Overview

The script generates synthetic MRI FLAIR images and lesion masks that match the statistical distribution of the test set. Each "replica" is an independent synthetic dataset with the same number of samples per condition (z-bin, lesion presence, domain) as the real test set.

## Output Directory Structure

```
{OUT_DIR}/
└── replicas/
    ├── replica_000.npz          # Compressed NumPy archive for replica 0
    ├── replica_000_meta.json    # JSON metadata for replica 0
    ├── replica_001.npz
    ├── replica_001_meta.json
    ├── ...
    ├── replica_009.npz
    └── replica_009_meta.json
```

## NPZ File Contents

Each `replica_XXX.npz` file contains the following arrays:

| Array Name | Shape | Dtype | Description |
|------------|-------|-------|-------------|
| `images` | `(N, 128, 128)` | `float16` | Generated FLAIR images, normalized to [-1, 1] |
| `masks` | `(N, 128, 128)` | `float16` | Generated lesion masks, normalized to [-1, 1] |
| `zbin` | `(N,)` | `int32` | Z-bin index for each sample (0 to z_bins-1) |
| `lesion_present` | `(N,)` | `uint8` | 0 = no lesion, 1 = lesion present |
| `domain` | `(N,)` | `uint8` | 0 = control, 1 = epilepsy |
| `condition_token` | `(N,)` | `int32` | Model conditioning token = zbin + lesion_present * z_bins |
| `seed` | `(N,)` | `int64` | Per-sample SHA256-derived seed (for reproducibility) |
| `k_index` | `(N,)` | `int32` | Sample index within its condition (0 to n_slices-1) |
| `replica_id` | `(N,)` | `int32` | Replica identifier (constant per file) |

Where `N` = total samples in test distribution (sum of all `n_slices` in test CSV).

## JSON Metadata Contents

Each `replica_XXX_meta.json` file contains:

```json
{
  "replica_id": 0,
  "seed_base": 42,
  "n_samples": 4523,
  "n_conditions": 150,
  "generation_timestamp": "2025-01-10T14:32:15.123456",
  "config": {
    "checkpoint_path": "/path/to/best.ckpt",
    "config_path": "/path/to/config.yaml",
    "z_bins": 30,
    "batch_size": 16,
    "use_ema": true,
    "ema_loaded": true,
    "anatomical_conditioning": true,
    "output_dtype": "float16"
  },
  "sampler": {
    "type": "DDIM",
    "num_inference_steps": 200,
    "eta": 0.0,
    "guidance_scale": 1.0
  },
  "domain_mapping": {"control": 0, "epilepsy": 1},
  "domain_mapping_inverse": {"0": "control", "1": "epilepsy"}
}
```

## Loading the Data

```python
import numpy as np

# Load a single replica
data = np.load("replicas/replica_000.npz")

# Access arrays
images = data["images"]          # (N, 128, 128) float16
masks = data["masks"]            # (N, 128, 128) float16
zbins = data["zbin"]             # (N,) int32
lesion_present = data["lesion_present"]  # (N,) uint8
domains = data["domain"]         # (N,) uint8
seeds = data["seed"]             # (N,) int64

# Filter by condition
epilepsy_lesion_mask = (domains == 1) & (lesion_present == 1)
epilepsy_lesion_images = images[epilepsy_lesion_mask]
```

## Reproducibility

Each sample is generated with a deterministic seed computed via SHA256:

```python
seed = SHA256(seed_base, replica_id, zbin, lesion_present, domain_int, sample_index)
```

This guarantees:
- Identical outputs when re-running with same parameters
- Different but deterministic samples across replicas (different `replica_id`)
- Independence from hardware, Python version, or process state

## SLURM Job Array Mapping

The script uses SLURM job arrays where:
- `SLURM_ARRAY_TASK_ID` = replica ID (0-9 by default)
- Each array task generates one complete replica
- All tasks are independent and can run in parallel

## Typical Output Sizes

For a test distribution with ~4500 samples:
- NPZ file: ~70-100 MB (compressed, float16)
- JSON metadata: ~1 KB
- Total per replica: ~100 MB
- 10 replicas: ~1 GB
