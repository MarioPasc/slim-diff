#!/usr/bin/env bash
#SBATCH -J log_jssdpm_lp_norm_x0_anatomical_cond
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# ========================================================================
# JS-DDPM Training with Lp Norm + Focal Frequency Loss (Complete)
# ========================================================================
# This experiment combines multiple techniques for improved lesion synthesis:
#   - Lp norm (p=1.5): Robust to outliers, balances sensitivity
#   - FFL: Improves high-frequency component synthesis (edge sharpness)
#   - Lesion overweighting: Upweights lesion pixels in both channels
#   - Kendall uncertainty: Learns optimal weighting between Lp+FFL
# ========================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Job started at: $(date)"

# ========================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# ========================================================================
EXPERIMENT_NAME="jssdpm_lp_norm_x0"
CONDA_ENV_NAME="jsddpm"

REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/js-ddpm-epilepsy"
DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy"
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/${EXPERIMENT_NAME}"
CONFIG_FILE="${REPO_SRC}/slurm/v_prediction/anatomical_cond/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}.yaml"

# ---------- Load conda module and activate prebuilt env ----------
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done
if [ "$module_loaded" -eq 0 ]; then
  echo "[env] no conda module loaded; assuming conda already in PATH."
fi

# Activate your existing env named ${CONDA_ENV_NAME}
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

# Verify
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch, os; print('CUDA', torch.cuda.is_available())"
echo "[torch] $(python -c 'import torch; print(torch.__version__)')"
echo "[cuda devices] $(python -c 'import torch; print(torch.cuda.device_count())')"

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
echo "Modifying configuration file..."
sed -i "s|  root_dir: .*|  root_dir: \"${DATA_SRC}\"|" "${MODIFIED_CONFIG}"
sed -i "s|  cache_dir: .*|  cache_dir: \"${DATA_SRC}/slice_cache\"|" "${MODIFIED_CONFIG}"
sed -i "s|  output_dir: .*|  output_dir: \"${RESULTS_DST}\"|" "${MODIFIED_CONFIG}"

# 4. Run the training script
cd "${REPO_SRC}"
echo "Working directory: $(pwd)"

# Determine cache directory
CACHE_DIR="${DATA_SRC}/slice_cache"
echo "Cache directory: ${CACHE_DIR}"

# Only run caching if cache directory doesn't exist
if [ ! -d "${CACHE_DIR}" ]; then
    echo "Cache directory not found. Running caching step..."
    jsddpm-cache --config "${MODIFIED_CONFIG}"
    echo "Caching completed."
else
    echo "Cache directory exists. Skipping caching step."
    echo "   Found: ${CACHE_DIR}"
    echo "   To rebuild cache, delete this directory and re-run."
fi

echo "Starting training with config: ${MODIFIED_CONFIG}"

jsddpm-train --config "${MODIFIED_CONFIG}"

echo "Training completed."

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Job finished at: $(date)"
echo "Total execution time: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Experiment completed successfully."
