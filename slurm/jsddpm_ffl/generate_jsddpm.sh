#!/usr/bin/env bash
#SBATCH -J log_generate_jsddpm_ffl
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# ========================================================================
# JS-DDPM Generation Script for FFL Experiment
# ========================================================================
# This script generates synthetic FLAIR/lesion samples from the FFL-trained model
#
# Usage:
#   sbatch slurm/jsddpm_ffl/generate_jsddpm.sh <checkpoint_path>
#
# Example:
#   sbatch slurm/jsddpm_ffl/generate_jsddpm.sh /path/to/checkpoint.ckpt
# ========================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Generation job started at: $(date)"

# ========================================================================
# CONFIGURATION
# ========================================================================
EXPERIMENT_NAME="jsddpm_ffl"
CONDA_ENV_NAME="jsddpm"

# Paths - UPDATE THESE FOR YOUR CLUSTER
REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/js-ddpm-epilepsy"
CONFIG_FILE="${REPO_SRC}/slurm/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}.yaml"

# Checkpoint path - from command line argument
if [ $# -eq 0 ]; then
    echo "ERROR: No checkpoint path provided"
    echo "Usage: sbatch slurm/jsddpm_ffl/generate_jsddpm.sh <checkpoint_path>"
    exit 1
fi
CHECKPOINT_PATH="$1"

# Verify checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

# Extract experiment directory from checkpoint path
CHECKPOINT_DIR=$(dirname "${CHECKPOINT_PATH}")
EXPERIMENT_DIR=$(dirname "${CHECKPOINT_DIR}")
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Experiment directory: ${EXPERIMENT_DIR}"

# Output directory for generated samples
OUTPUT_DIR="${EXPERIMENT_DIR}/generated_samples"
echo "Output directory: ${OUTPUT_DIR}"

# Generation parameters
Z_BINS=""                    # e.g., "0,12,25,37,49" or empty for all
CLASSES=""                   # e.g., "0,1" or empty for default
N_PER_CONDITION=""           # e.g., "100" or empty for default from config
SEED="42"

# Dynamic GPU assignment
export CUDA_VISIBLE_DEVICES=0
for i in {0..7}; do
  if [[ -z $(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]] && nvidia-smi -i $i &>/dev/null; then
    export CUDA_VISIBLE_DEVICES=$i
    echo "Auto-assigned to available GPU: $i"
    break
  fi
done

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

# Activate your existing env
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

# Verify environment
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch, os; print('CUDA', torch.cuda.is_available())"
echo "[torch] $(python -c 'import torch; print(torch.__version__)')"
echo "[cuda devices] $(python -c 'import torch; print(torch.cuda.device_count())')"

# ========================================================================
# GENERATION EXECUTION
# ========================================================================

cd "${REPO_SRC}"
echo "Working directory: $(pwd)"

# Create output directory
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "Creating output directory: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
fi

# Build generation command
GEN_CMD="jsddpm-generate --config ${CONFIG_FILE} --ckpt ${CHECKPOINT_PATH} --out_dir ${OUTPUT_DIR} --seed ${SEED}"

# Add optional parameters if specified
if [ -n "${Z_BINS}" ]; then
    GEN_CMD="${GEN_CMD} --z_bins ${Z_BINS}"
fi

if [ -n "${CLASSES}" ]; then
    GEN_CMD="${GEN_CMD} --classes ${CLASSES}"
fi

if [ -n "${N_PER_CONDITION}" ]; then
    GEN_CMD="${GEN_CMD} --n_per_condition ${N_PER_CONDITION}"
fi

echo "================================================================"
echo "Starting generation with command:"
echo "${GEN_CMD}"
echo "================================================================"

# Run generation
eval "${GEN_CMD}"

echo "Generation completed successfully!"

# Print summary
if [ -f "${OUTPUT_DIR}/generation_config.yaml" ]; then
    echo ""
    echo "Generation Summary:"
    cat "${OUTPUT_DIR}/generation_config.yaml"
fi

if [ -f "${OUTPUT_DIR}/generated_samples.csv" ]; then
    SAMPLE_COUNT=$(tail -n +2 "${OUTPUT_DIR}/generated_samples.csv" | wc -l)
    echo ""
    echo "Total samples generated: ${SAMPLE_COUNT}"
    echo "Sample index: ${OUTPUT_DIR}/generated_samples.csv"
    echo "Samples location: ${OUTPUT_DIR}/samples/"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Job finished at: $(date)"
echo "Total execution time: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Generation completed successfully."
