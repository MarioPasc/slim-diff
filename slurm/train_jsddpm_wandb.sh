#!/usr/bin/env bash
#SBATCH -J log_train_jsddpm_flair_soco_wandb
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

START_TIME=$(date +%s)
echo "Job started at: $(date)"

# ========================================================================
# W&B CONFIGURATION
# ========================================================================
# Option 1: Set your W&B API key here (get from https://wandb.ai/authorize)
# export WANDB_API_KEY="paste-your-api-key-here"

# Option 2: Use offline mode (sync later from login node)
# export WANDB_MODE="offline"

# Option 3: API key already set in ~/.netrc (wandb login was run)
# No need to set anything here

# Uncomment ONE of the above options

# ========================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# ========================================================================
EXPERIMENT_NAME="jsddpm"
CONDA_ENV_NAME="jsddpm"

REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/js-ddpm-epilepsy"
DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy"
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/${EXPERIMENT_NAME}"
CONFIG_FILE="${REPO_SRC}/src/diffusion/config/${EXPERIMENT_NAME}.yaml"

# Dynamic GPU assignment
export CUDA_VISIBLE_DEVICES=0
for i in {0..7}; do
  if [[ -z $(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]] && nvidia-smi -i $i &>/dev/null; then
    export CUDA_VISIBLE_DEVICES=$i
    echo "✅ Auto-assigned to available GPU: $i"
    break
  fi
done

# ---------- Load conda module and activate prebuilt env ----------
# Try common module names seen on clusters
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done
# If environment module is not needed because conda is already in PATH, continue
if [ "$module_loaded" -eq 0 ]; then
  echo "[env] no conda module loaded; assuming conda already in PATH."
fi

# Activate your existing env named ${CONDA_ENV_NAME} (precreated by you, offline)
# Support both old and new activation methods
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  # Fallback if only 'source activate' exists in module
  source activate "${CONDA_ENV_NAME}"
fi

# Verify
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch, os; print('CUDA', torch.cuda.is_available())"
echo "[torch] $(python -c 'import torch; print(torch.__version__)')"
echo "[cuda devices] $(python -c 'import torch; print(torch.cuda.device_count())')"

# Check W&B installation
echo "[wandb] $(python -c 'import wandb; print(wandb.__version__)' 2>/dev/null || echo 'not installed')"

# ========================================================================
# EXPERIMENT EXECUTION
# ========================================================================

# 1. Create the results directory (execution folder)
if [ ! -d "${RESULTS_DST}" ]; then
    echo "Creating results directory: ${RESULTS_DST}"
    mkdir -p "${RESULTS_DST}"
fi

# 2. Backup copy the configuration file to the execution folder
CONFIG_BASENAME=$(basename "${CONFIG_FILE}")
MODIFIED_CONFIG="${RESULTS_DST}/${CONFIG_BASENAME}"

echo "Copying config file to: ${MODIFIED_CONFIG}"
cp "${CONFIG_FILE}" "${MODIFIED_CONFIG}"

# 3. Modify the configuration file
# Change data: root_dir: to DATA_SRC
# Change logging: save_dir: to RESULTS_DST
# Change logger type to wandb
echo "Modifying configuration file..."
sed -i "s|  root_dir: .*|  root_dir: \"${DATA_SRC}\"|" "${MODIFIED_CONFIG}"
sed -i "s|  output_dir: .*|  output_dir: \"${RESULTS_DST}\"|" "${MODIFIED_CONFIG}"

# IMPORTANT: Change logger to wandb
sed -i "s|    type: \"tensorboard\"|    type: \"wandb\"|" "${MODIFIED_CONFIG}"

echo "✅ Configuration modified to use W&B logging"

# 4. Run the training script
# Navigate to the repository root to ensure imports work correctly
cd "${REPO_SRC}"
echo "Working directory: $(pwd)"

# Build cache first
echo "Building slice cache..."
jsddpm-cache --config "${MODIFIED_CONFIG}"

echo "Starting training with W&B logging..."
echo "View progress at: https://wandb.ai"

jsddpm-train --config "${MODIFIED_CONFIG}"

echo "Training completed."

# If using offline mode, remind to sync
if [ "${WANDB_MODE:-}" = "offline" ]; then
    echo ""
    echo "⚠️  W&B was run in OFFLINE mode"
    echo "To sync logs, run from login node:"
    echo "wandb sync ${RESULTS_DST}/logs/wandb/offline-run-*"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Job finished at: $(date)"
echo "Total execution time: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "✅ Experiment completed successfully."
