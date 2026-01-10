# Quick Start Guide - SLURM Training

## TL;DR - Best Setup for SLURM

```bash
# 1. One-time W&B setup
conda activate jsddpm
pip install wandb
wandb login  # Enter API key from https://wandb.ai/authorize

# 2. Submit job with W&B logging
sbatch slurm/train_jsddpm_wandb.sh

# 3. Monitor from anywhere
# Open https://wandb.ai in your browser
```

---

## Two Options for SLURM

### Option A: Weights & Biases (Recommended) ⭐

**Pros:**
- ✅ View training in real-time from anywhere
- ✅ No port forwarding or SSH tunnels
- ✅ Works on any cluster
- ✅ Free for academics

**Setup:**
```bash
# Install (if not already)
pip install wandb

# Login once
wandb login
# Paste API key from: https://wandb.ai/authorize
```

**Run:**
```bash
sbatch slurm/train_jsddpm_wandb.sh
```

**Monitor:**
- Open https://wandb.ai in browser
- See metrics, losses, visualizations in real-time
- Works from laptop, phone, etc.

---

### Option B: TensorBoard (Default)

**Pros:**
- ✅ No external dependencies
- ✅ All data stays on cluster

**Cons:**
- ❌ Can't view during training
- ❌ Requires copying files or complex port forwarding

**Run:**
```bash
sbatch slurm/train_jsddpm.sh
```

**View after training:**
```bash
# On login node, copy logs
cp -r /path/to/outputs/jsddpm/logs ./local_logs

# Start TensorBoard
tensorboard --logdir ./local_logs --port 6006

# On your laptop
ssh -L 6006:localhost:6006 user@cluster-login-node
# Open http://localhost:6006
```

---

## Configuration Files

| File | Logger | Use Case |
|------|--------|----------|
| `jsddpm.yaml` | TensorBoard | Default, local training |
| `jsddpm_wandb.yaml` | W&B | SLURM, remote monitoring |

To use W&B with default config, the script automatically modifies it.

---

## Logging Comparison

| Feature | W&B | TensorBoard |
|---------|-----|-------------|
| Real-time on SLURM | ✅ Yes | ❌ No |
| Setup complexity | Easy | Complex |
| View from anywhere | ✅ Yes | ❌ No |
| Port forwarding | ❌ Not needed | ✅ Required |
| **Recommended** | ✅ **SLURM** | ⚠️ Local only |

---

## Troubleshooting

### "wandb: ERROR Error uploading"

Compute nodes may not have internet. Use offline mode:

Edit `train_jsddpm_wandb.sh`:
```bash
# Uncomment this line:
export WANDB_MODE="offline"
```

Then sync later from login node:
```bash
wandb sync /path/to/outputs/jsddpm/logs/wandb/offline-run-*
```

### TensorBoard "Address already in use"

Use different port:
```bash
tensorboard --logdir ./logs --port 6007
```

---

## Complete Example

### Using W&B (Recommended)

```bash
# 1. Setup W&B (once)
conda activate jsddpm
pip install wandb
wandb login

# 2. Submit job
sbatch slurm/train_jsddpm_wandb.sh

# 3. Monitor
# Open https://wandb.ai
# You'll see:
# - Training/validation losses
# - PSNR, SSIM metrics
# - Generated image grids
# - GPU usage, system metrics
```

### Using TensorBoard

```bash
# 1. Submit job
sbatch slurm/train_jsddpm.sh

# 2. Wait for training to finish

# 3. View logs (from login node)
tensorboard --logdir /path/to/outputs/jsddpm/logs --port 6006

# 4. Forward port to laptop
ssh -L 6006:localhost:6006 user@cluster
# Open http://localhost:6006
```

---

## Which Should You Use?

**Use W&B if:**
- ✅ You want to monitor training in real-time
- ✅ You're on a SLURM cluster
- ✅ You want to compare multiple runs easily
- ✅ You want system metrics (GPU usage, etc.)

**Use TensorBoard if:**
- ✅ You're training locally
- ✅ You don't want external dependencies
- ✅ You can wait until training finishes to view logs
- ✅ All data must stay on-premises

**For your SLURM setup: Use W&B** ⭐

---

## Generating Synthetic Samples

After training, use the generation script to create synthetic FLAIR/lesion samples:

### Quick Start

```bash
# Generate samples from a trained checkpoint
sbatch slurm/generate_jsddpm.sh /path/to/checkpoint.ckpt

# Example:
sbatch slurm/generate_jsddpm.sh \
  /mnt/home/users/tic_163_uma/mpascual/fscratch/results/jsddpm/checkpoints/jsddpm-epoch=0050-val_loss=0.1234.ckpt
```

### Customizing Generation Parameters

Edit `slurm/generate_jsddpm.sh` before submitting:

```bash
# In generate_jsddpm.sh, modify these variables:
Z_BINS="0,12,25,37,49"       # Specific z-positions (or empty for all)
CLASSES="0,1"                # 0=control, 1=lesion (or empty for both)
N_PER_CONDITION="100"        # Samples per (z_bin, class) pair
SEED="42"                    # Random seed for reproducibility
```

### Output Structure

```
/results/jsddpm/
├── checkpoints/             # Training checkpoints
│   └── jsddpm-epoch=0050-val_loss=0.1234.ckpt
└── generated_samples/       # Generated output (created by generate script)
    ├── samples/             # Individual .npz files
    │   ├── z00_c0_s0000.npz
    │   ├── z00_c0_s0001.npz
    │   └── ...
    ├── generated_samples.csv      # Index of all samples
    └── generation_config.yaml     # Generation parameters used
```

### Generation Performance

- **Time estimate**: ~0.5-1 second per sample on GPU
- **Memory**: ~4-8 GB GPU memory
- **Example**: 1000 samples = ~10-15 minutes

---

## Training Features

### Automatic Cache Management

The training script now intelligently handles dataset caching:

- **First run**: Automatically creates cache from raw data
- **Subsequent runs**: Detects existing cache and skips caching step
- **Manual rebuild**: Delete `${DATA_SRC}/slice_cache` to force rebuild

```bash
# To rebuild cache:
rm -rf /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy/slice_cache
sbatch slurm/train_jsddpm.sh
```

### WandB Offline Mode for Clusters

The config is now set to use offline mode by default for cluster nodes without internet:

- Logs saved locally: `/results/jsddpm/logs/wandb/offline-run-*`
- Sync later from login node with internet:
  ```bash
  wandb sync /path/to/offline-run-*
  ```

### Disabled Progress Bars

TQDM progress bars are disabled for SLURM (they're misleading in log files). Instead, all metrics are logged:

**Training metrics logged:**
- `train/loss`, `train/loss_image`, `train/loss_mask`
- `train/psnr`, `train/ssim`
- `train/log_var_image`, `train/log_var_mask` (uncertainty weights)
- `train/weight_image`, `train/weight_mask` (actual weights = exp(-log_var))
- `train/lr` (learning rate)

**Validation metrics logged:**
- `val/loss`, `val/loss_image`, `val/loss_mask`
- `val/psnr`, `val/ssim`, `val/dice`
- `val/log_var_image`, `val/log_var_mask`
- `val/weight_image`, `val/weight_mask`

---

## See Also

- `LOGGING_GUIDE.md` - Detailed logging setup guide
- `CHANGES.md` - Recent fixes and changes
- `src/diffusion/README.md` - Complete module documentation
