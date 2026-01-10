# Logging on SLURM Clusters - Complete Guide

## The Problem with TensorBoard on SLURM

When running on SLURM compute nodes:
- ❌ No direct browser access to compute nodes
- ❌ TensorBoard web interface requires port forwarding
- ❌ Compute node names are assigned dynamically
- ❌ Can't easily view training progress in real-time

## Solutions

### ✅ Option 1: Weights & Biases (RECOMMENDED)

**Best for**: Real-time monitoring from anywhere, cluster computing

#### Setup

1. **Install W&B** (if not already):
   ```bash
   conda activate jsddpm
   pip install wandb
   ```

2. **Login to W&B** (one-time setup):
   ```bash
   wandb login
   ```

   Get your API key from: https://wandb.ai/authorize

3. **Modify config** to use W&B:
   ```yaml
   # In jsddpm.yaml, change:
   logging:
     logger:
       type: "wandb"  # Change from "tensorboard"
       wandb:
         project: "jsddpm-epilepsy"
         entity: "your-username"  # Your W&B username
   ```

4. **Run training normally**:
   ```bash
   sbatch slurm/train_jsddpm.sh
   ```

5. **View from anywhere**:
   - Go to https://wandb.ai
   - See metrics, visualizations, system stats in real-time
   - Access from laptop, phone, etc.

#### Benefits
- ✅ Real-time monitoring from anywhere
- ✅ Automatic cloud sync
- ✅ No port forwarding needed
- ✅ Free for academics
- ✅ Stores model artifacts
- ✅ Experiment comparison built-in

#### W&B Login on Cluster

If compute nodes don't have internet, use **offline mode**:

```yaml
logging:
  logger:
    type: "wandb"
    wandb:
      project: "jsddpm-epilepsy"
      offline: true  # Sync later from login node
```

Then sync from login node:
```bash
wandb sync outputs/jsddpm/logs/wandb/offline-run-*
```

---

### ✅ Option 2: TensorBoard with Post-Training Viewing

**Best for**: If you don't want to use W&B and can wait to view logs

#### During Training (SLURM)

Use default TensorBoard configuration (already in `jsddpm.yaml`):
```yaml
logging:
  logger:
    type: "tensorboard"
```

Logs are saved to disk at: `{output_dir}/logs/`

#### After Training (View on Login Node)

**Method A: Copy to login node and view there**

```bash
# On login node
rsync -avz compute_node:/path/to/outputs/jsddpm/logs ./local_logs

# Start TensorBoard on login node
tensorboard --logdir ./local_logs --port 6006

# Forward to your laptop
ssh -L 6006:localhost:6006 user@login-node
# Then open http://localhost:6006 in browser
```

**Method B: While job is running (advanced)**

```bash
# 1. Find your compute node
squeue -u $USER

# 2. SSH to compute node
ssh compute-node-name

# 3. Start TensorBoard
tensorboard --logdir /path/to/outputs/jsddpm/logs --port 6006

# 4. From login node, forward port
ssh -L 6006:compute-node-name:6006 compute-node-name

# 5. From laptop
ssh -L 6006:localhost:6006 user@login-node
```

This is **complex** and not recommended.

---

### ✅ Option 3: CSV Logger (Simplest Fallback)

**Best for**: Minimal dependencies, offline analysis

#### Implementation

Add CSV logger support to `training/runners/train.py`:

```python
from pytorch_lightning.loggers import CSVLogger

def build_logger(cfg: DictConfig) -> pl.loggers.Logger:
    log_cfg = cfg.logging.logger
    output_dir = Path(cfg.experiment.output_dir)

    if log_cfg.type == "csv":
        return CSVLogger(
            save_dir=output_dir / "logs",
            name="metrics",
        )
    # ... rest of code
```

#### Config

```yaml
logging:
  logger:
    type: "csv"
```

#### View Results

```bash
# Metrics saved to CSV files
cat outputs/jsddpm/logs/metrics/version_0/metrics.csv

# Or plot with pandas:
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/jsddpm/logs/metrics/version_0/metrics.csv')
df.plot(x='epoch', y=['train/loss', 'val/loss'])
plt.savefig('training_curves.png')
"
```

---

## Recommended Setup for SLURM

### For Your Use Case

I recommend **W&B** for the following reasons:

1. **Real-time monitoring**: See training progress from anywhere
2. **No complexity**: No SSH tunnels or port forwarding
3. **Visualization grid**: Images logged automatically
4. **Free for academics**: Sign up with .edu email
5. **Reliable**: Works on all clusters

### Quick Start with W&B

1. **One-time setup**:
   ```bash
   pip install wandb
   wandb login  # Enter API key from https://wandb.ai/authorize
   ```

2. **Modify config**:
   ```yaml
   logging:
     logger:
       type: "wandb"
       wandb:
         project: "jsddpm-epilepsy"
         entity: "your-username"
   ```

3. **Run**:
   ```bash
   sbatch slurm/train_jsddpm.sh
   ```

4. **Monitor**:
   - Open https://wandb.ai in browser
   - See real-time metrics, losses, visualizations
   - Check system metrics (GPU usage, memory, etc.)

---

## Comparison Table

| Feature | W&B | TensorBoard (Post) | CSV Logger |
|---------|-----|-------------------|------------|
| Real-time viewing | ✅ Yes | ❌ No | ❌ No |
| Remote access | ✅ Easy | ⚠️ Complex | ❌ No |
| Port forwarding needed | ❌ No | ✅ Yes | ❌ No |
| Visualization grids | ✅ Yes | ✅ Yes | ❌ No |
| Metric comparison | ✅ Built-in | ⚠️ Manual | ⚠️ Manual |
| Free | ✅ Yes (academic) | ✅ Yes | ✅ Yes |
| Setup complexity | ⭐ Easy | ⭐⭐⭐ Hard | ⭐ Easy |
| **Recommended for SLURM** | ✅ **Yes** | ❌ No | ⚠️ Fallback |

---

## Troubleshooting

### W&B: "wandb: ERROR Error uploading"

**Solution**: Use offline mode on compute nodes
```yaml
logging:
  logger:
    wandb:
      offline: true
```

Sync later from login node:
```bash
wandb sync outputs/jsddpm/logs/wandb/offline-run-*
```

### TensorBoard: "Address already in use"

**Solution**: Use different port
```bash
tensorboard --logdir ./logs --port 6007
```

### W&B: "Login failed"

**Solution**: Set API key as environment variable
```bash
# Add to SLURM script
export WANDB_API_KEY="your-api-key-here"
```

Or in SLURM script:
```bash
#SBATCH --export=WANDB_API_KEY=your-key-here
```

---

## Modified SLURM Script for W&B

```bash
#!/usr/bin/env bash
#SBATCH -J jsddpm_wandb
# ... other SBATCH directives ...

# Set W&B API key (get from https://wandb.ai/authorize)
export WANDB_API_KEY="paste-your-api-key-here"

# Or use offline mode (sync later)
export WANDB_MODE="offline"

# Rest of script as normal...
conda activate jsddpm
jsddpm-train --config "${MODIFIED_CONFIG}"
```

---

## Summary

**For SLURM clusters, use W&B:**

```bash
# Setup (once)
pip install wandb
wandb login

# Modify jsddpm.yaml
logging:
  logger:
    type: "wandb"

# Train
sbatch slurm/train_jsddpm.sh

# Monitor from laptop
# Open https://wandb.ai
```

**For local training, TensorBoard is fine:**

```bash
# Start training
python -m src.diffusion.training.runners.train --config src/diffusion/config/jsddpm.yaml

# In another terminal
tensorboard --logdir outputs/jsddpm/logs
# Open http://localhost:6006
```
