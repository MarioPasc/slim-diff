# TASK 03 — Training & Generation Orchestration (Fold × Architecture Grid)

## Context

The ICIP 2026 camera-ready requires running **6 training experiments**: 3 folds × 2 architectures (shared bottleneck, decoupled bottleneck). After training, each model generates 90,000 synthetic samples (replicas) for evaluation. This task creates the configuration files, SLURM job scripts, and a small orchestration layer so that the 6-run grid can be launched reliably on the Picasso cluster.

The existing codebase already has a working train+generate pipeline (see `slurm/icip2026/lp_ablation/x0/lp_1.5/train_generate.sh` for reference). Your job is to adapt this for the fold × architecture matrix while reusing the existing `slimdiff-train` and `generate_replicas.py` entrypoints.

## Your Deliverable

1. **6 YAML config files** (one per fold × architecture combination) derived from the best configuration: x₀-prediction, L₁.₅ for image, L₂.₀ for mask (or the closest existing config).
2. **6 SLURM job scripts** that each run train → generate for one (fold, architecture) combination.
3. **1 launcher script** (`launch_camera_ready.sh`) that submits all 6 jobs.
4. A **README** documenting the experiment matrix and expected outputs.

## Files You Own (create)

```
slurm/camera_ready/
├── README.md
├── launch_camera_ready.sh          # Submits all 6 jobs
├── shared_fold_0/
│   ├── config.yaml
│   └── train_generate.sh
├── shared_fold_1/
│   ├── config.yaml
│   └── train_generate.sh
├── shared_fold_2/
│   ├── config.yaml
│   └── train_generate.sh
├── decoupled_fold_0/
│   ├── config.yaml
│   └── train_generate.sh
├── decoupled_fold_1/
│   ├── config.yaml
│   └── train_generate.sh
└── decoupled_fold_2/
    ├── config.yaml
    └── train_generate.sh
```

Also create:
- `configs/camera_ready/base_shared.yaml` — base config for shared bottleneck
- `configs/camera_ready/base_decoupled.yaml` — base config for decoupled bottleneck

## Files You Must NOT Modify

- `src/diffusion/model/*`
- `src/diffusion/data/*`
- `src/diffusion/training/lit_modules.py`
- `src/diffusion/training/runners/train.py`
- `src/diffusion/training/runners/generate_replicas.py`

## Configuration Specification

### Base Configuration (both architectures share these)

Derive from the existing `slurm/icip2026/lp_ablation/x0/lp_1.5/x0_lp_1.5.yaml` (or the closest match). The critical training parameters that **must be identical** across all 6 runs:

```yaml
scheduler:
  prediction_type: "sample"   # x0-prediction (MONAI uses "sample" for x0)
  schedule: "cosine"
  num_train_timesteps: 1000

training:
  optimizer:
    type: "AdamW"
    lr: 1.0e-4
    weight_decay: 1.0e-4
  lr_scheduler:
    type: "CosineAnnealingLR"
  max_epochs: 1000
  early_stopping:
    enabled: true
    patience: 25                # Paper says 25, verify against existing configs
    monitor: "val/loss"
  ema:
    enabled: true
    decay: 0.999
  gradient_clip_val: 1.0
  precision: "16-mixed"

loss:
  lp_norm:
    p: 1.5                     # Best Lp for image quality

sampler:
  type: "DDIM"
  num_inference_steps: 300      # Paper says 300
  eta: 0.2
```

### Architecture-Specific Config

For **shared**:
```yaml
model:
  bottleneck_mode: "shared"     # Default / current behavior
```

For **decoupled**:
```yaml
model:
  bottleneck_mode: "decoupled"  # Uses the new DecoupledMiddleBlock
```

### Fold-Specific Config

Each config must point to the correct fold's data CSVs:

```yaml
data:
  cache_dir: "${DATA_SRC}/slice_cache/folds/fold_${FOLD_ID}"
```

Or, if the fold system uses a different mechanism (e.g., a `fold_id` key that the dataloader interprets), adapt accordingly. The key constraint: **the training data for fold k must exclude all patients in fold k's test set**.

### Generation Configuration

After training, each job generates replicas:

```yaml
# In the SLURM script, not the YAML
NUM_REPLICAS=20
N_SAMPLES_PER_MODE=150    # 150 samples per (zbin, domain)
GEN_BATCH_SIZE=32
SEED_BASE=42              # Same base seed for all runs
```

This produces 20 × 150 × 30 zbins × 2 domains = 180,000 samples per replica... Actually, re-check. The paper says "20 replicas × 150 images × 30 z-bins" = 90,000. The `--n_samples_per_mode` flag in `generate_replicas.py` controls samples per (zbin, domain) combination. With 30 zbins × 2 domains = 60 modes, and 150 samples per mode, each replica has 9,000 samples. With 20 replicas, total = 180,000. Verify against the existing SLURM scripts — the original used `NUM_REPLICAS=5` and `N_SAMPLES_PER_MODE=50`. Scale to match the paper's stated 90,000.

**Seed determinism**: The `SEED_BASE` must be identical across architectures within the same fold, so that the noise realizations are comparable. Different folds may use different seed bases (e.g., `SEED_BASE = 42 + fold_id * 1000`), or the same base if the generate_replicas script already incorporates the replica_id.

### Output Directory Structure

```
results/camera_ready/
├── shared_fold_0/
│   ├── checkpoints/
│   ├── logs/
│   ├── replicas/
│   │   ├── replica_000.npz
│   │   ├── replica_001.npz
│   │   └── ...
│   └── config.yaml        # Copy of the config used
├── shared_fold_1/
│   └── ...
├── shared_fold_2/
│   └── ...
├── decoupled_fold_0/
│   └── ...
├── decoupled_fold_1/
│   └── ...
└── decoupled_fold_2/
    └── ...
```

## SLURM Job Script Template

Adapt from `slurm/icip2026/lp_ablation/x0/lp_1.5/train_generate.sh`. Key changes:

1. **Job name**: `slimdiff_cr_{arch}_{fold}` (e.g., `slimdiff_cr_shared_f0`)
2. **Config path**: points to the fold-specific config
3. **Results directory**: `results/camera_ready/{arch}_fold_{fold_id}`
4. **Data directory**: points to the correct fold's CSVs
5. **GPU request**: same as existing (`--gres=gpu:2`, `--constraint=dgx`)
6. **Time limit**: same as existing (3 days)
7. **Generation phase**: identical to existing, just uses fold-specific checkpoint

Important SLURM script features to preserve from the template:
- Conda environment activation (`jsddpm`)
- DDP verification for multi-GPU
- Cache directory check
- Config file backup to results directory
- Sequential replica generation loop
- Timing instrumentation

### Launcher Script

```bash
#!/usr/bin/env bash
# launch_camera_ready.sh — Submit all 6 camera-ready experiments
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting ICIP 2026 camera-ready experiments..."
echo "Matrix: 3 folds × 2 architectures = 6 jobs"
echo ""

for arch in shared decoupled; do
    for fold in 0 1 2; do
        JOB_DIR="${SCRIPT_DIR}/${arch}_fold_${fold}"
        JOB_SCRIPT="${JOB_DIR}/train_generate.sh"
        
        if [ ! -f "${JOB_SCRIPT}" ]; then
            echo "ERROR: Missing ${JOB_SCRIPT}"
            exit 1
        fi
        
        JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $NF}')
        echo "  Submitted ${arch}_fold_${fold}: job ${JOB_ID}"
    done
done

echo ""
echo "All 6 jobs submitted. Monitor with: squeue -u \$USER"
```

## Acceptance Criteria (Testable)

These are all file-level checks, no GPU required.

### Test 1: All 6 Configs Exist and Parse
```bash
for arch in shared decoupled; do
    for fold in 0 1 2; do
        CONFIG="slurm/camera_ready/${arch}_fold_${fold}/config.yaml"
        [ -f "$CONFIG" ] || { echo "MISSING: $CONFIG"; exit 1; }
        python -c "from omegaconf import OmegaConf; OmegaConf.load('$CONFIG')" || exit 1
    done
done
echo "All configs valid."
```

### Test 2: Critical Hyperparameters Identical Across All 6 Configs
```python
def test_hyperparameter_consistency():
    """All 6 configs share identical training hyperparameters."""
    configs = load_all_6_configs()
    
    MUST_MATCH = [
        "scheduler.prediction_type",
        "scheduler.schedule",
        "scheduler.num_train_timesteps",
        "training.optimizer.lr",
        "training.optimizer.weight_decay",
        "training.early_stopping.patience",
        "training.ema.decay",
        "training.gradient_clip_val",
        "loss.lp_norm.p",
        "sampler.num_inference_steps",
        "sampler.eta",
    ]
    
    for key in MUST_MATCH:
        values = [get_nested(cfg, key) for cfg in configs]
        assert len(set(str(v) for v in values)) == 1, \
            f"{key} differs across configs: {values}"
```

### Test 3: Architecture Configs Differ Only in bottleneck_mode
```python
def test_arch_configs_minimal_diff():
    """Shared vs decoupled configs for the same fold differ only in bottleneck_mode."""
    for fold in [0, 1, 2]:
        shared = load_config(f"shared_fold_{fold}")
        decoupled = load_config(f"decoupled_fold_{fold}")
        
        assert shared.model.bottleneck_mode == "shared"
        assert decoupled.model.bottleneck_mode == "decoupled"
        
        # Remove bottleneck_mode and compare
        shared.model.pop("bottleneck_mode")
        decoupled.model.pop("bottleneck_mode")
        assert OmegaConf.to_yaml(shared) == OmegaConf.to_yaml(decoupled)
```

### Test 4: SLURM Scripts Are Executable
```bash
for script in slurm/camera_ready/*/train_generate.sh; do
    [ -x "$script" ] || { echo "Not executable: $script"; exit 1; }
    bash -n "$script" || { echo "Syntax error in: $script"; exit 1; }
done
echo "All SLURM scripts pass syntax check."
```

### Test 5: Output Directories Are Distinct
```python
def test_output_dirs_unique():
    """Each config points to a unique output directory."""
    dirs = set()
    for cfg in load_all_6_configs():
        out = cfg.experiment.output_dir
        assert out not in dirs, f"Duplicate output_dir: {out}"
        dirs.add(out)
```

## Anti-Patterns — Do NOT:

- Do NOT change ANY training hyperparameter between shared and decoupled. The only difference is `bottleneck_mode`.
- Do NOT use a different random seed for generation between shared and decoupled within the same fold. Noise realizations must be comparable.
- Do NOT hardcode absolute paths in the YAML configs. Use `${VARIABLE}` placeholders that the SLURM script fills via `sed`.
- Do NOT create a single monolithic SLURM job that runs all 6 sequentially. They must be independent jobs that can run in parallel.
- Do NOT modify existing SLURM scripts in `slurm/icip2026/`.

## Notes

- Check whether the `early_stopping.patience` in the existing ICIP configs is 25 (paper) or 50 (jsddpm.yaml). The camera-ready should use 25 to match the paper.
- The existing SLURM template uses `jsddpm-train` and `jsddpm-cache` CLI names. The `pyproject.toml` shows the current names are `slimdiff-train`, `slimdiff-cache`, etc. Use whatever is correct in the installed environment; check `pyproject.toml` `[project.scripts]`.
- Verify that the loss config handles image and mask channels with potentially different Lp norms. The best config from Table 1 is x₀ + L₁.₅ for KID/LPIPS (image) and x₀ + L₂.₀ for MMD-MF (mask). If the current loss config applies a single `p` to both channels, this is a point to verify — the paper may have used p=1.5 globally and the mask performance at L₂.₀ was from a different run.
