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

## See Also

- `LOGGING_GUIDE.md` - Detailed logging setup guide
- `CHANGES.md` - Recent fixes and changes
- `src/diffusion/README.md` - Complete module documentation
