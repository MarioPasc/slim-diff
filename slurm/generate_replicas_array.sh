#!/usr/bin/env bash
#SBATCH -J jsddpm_gen_replicas
#SBATCH --array=0-14              # 15 replicas (adjust as needed)
#SBATCH --time=04:00:00           # 4 hours per replica (conservative)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=dgx
#SBATCH --output=logs/gen_replicas/%x.%j.out
#SBATCH --error=logs/gen_replicas/%x.%j.err

# ========================================================================
# JS-DDPM Replica Generation Script for SLURM Job Arrays
# ========================================================================
# Generates deterministic replicas of synthetic samples matching the test
# distribution. Each job array task generates one replica.
#
# Usage:
#   sbatch slurm/generate_replicas_array.slurm
#
# Prerequisites:
#   1. Model trained with export_to_checkpoint: true for EMA weights
#   2. test_zbin_distribution.csv exists
# ========================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================================================="
echo "Replica Generation Job Started"
echo "=========================================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"

# ============== CONFIGURATION ==============
# Paths - UPDATE THESE FOR YOUR CLUSTER
MODEL="jsddpm_sinus_kendall_weighted_anatomicalprior"

REPO_ROOT="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/js-ddpm-epilepsy"
DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy"
CONFIG_ORIG="${REPO_ROOT}/slurm/${MODEL}/${MODEL}.yaml"
CHECKPOINT="/mnt/home/users/tic_163_uma/mpascual/fscratch/weights_${MODEL}/best.ckpt"
TEST_CSV="${REPO_ROOT}/docs/test_analysis/test_zbin_distribution.csv"
OUT_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/replicas_${MODEL}"

# Generation parameters
NUM_REPLICAS=15
BATCH_SIZE=32
SEED_BASE=42
DTYPE="float16"
CONDA_ENV_NAME="jsddpm"
# ===========================================

# Validate configuration
if [ ! -f "${CONFIG_ORIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG_ORIG}"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

if [ ! -f "${TEST_CSV}" ]; then
    echo "ERROR: Test distribution CSV not found: ${TEST_CSV}"
    exit 1
fi

# Create output and log directories
mkdir -p "${OUT_DIR}/replicas"
mkdir -p "${REPO_ROOT}/logs/gen_replicas"

# Copy and modify config for cluster paths (only first task does this)
CONFIG="${OUT_DIR}/${MODEL}.yaml"
if [ ! -f "${CONFIG}" ]; then
    echo "Copying and modifying config for cluster paths..."
    cp "${CONFIG_ORIG}" "${CONFIG}"
    sed -i "s|  root_dir: .*|  root_dir: \"${DATA_SRC}\"|" "${CONFIG}"
    sed -i "s|  cache_dir: .*|  cache_dir: \"${DATA_SRC}/slice_cache\"|" "${CONFIG}"
    sed -i "s|  output_dir: .*|  output_dir: \"${OUT_DIR}\"|" "${CONFIG}"
    echo "Modified config saved to: ${CONFIG}"
else
    echo "Using existing modified config: ${CONFIG}"
fi

# Dynamic GPU assignment
export CUDA_VISIBLE_DEVICES=0
for i in {0..7}; do
  if [[ -z $(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]] && nvidia-smi -i $i &>/dev/null; then
    export CUDA_VISIBLE_DEVICES=$i
    echo "Auto-assigned to available GPU: $i"
    break
  fi
done

# Load conda module and activate environment
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done

if [ "$module_loaded" -eq 0 ]; then
  echo "[env] no conda module loaded; assuming conda already in PATH."
fi

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

# Verify environment
echo "=========================================================================="
echo "Environment"
echo "=========================================================================="
echo "Python: $(which python)"
echo "Python version: $(python -c 'import sys; print(sys.version.split()[0])')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

# Navigate to repository root
cd "${REPO_ROOT}"
echo "Working directory: $(pwd)"

echo "=========================================================================="
echo "Configuration"
echo "=========================================================================="
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Test CSV: ${TEST_CSV}"
echo "Output: ${OUT_DIR}"
echo "Replica ID: ${SLURM_ARRAY_TASK_ID} / ${NUM_REPLICAS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Seed base: ${SEED_BASE}"
echo "Output dtype: ${DTYPE}"

echo "=========================================================================="
echo "Starting Generation"
echo "=========================================================================="

python -m src.diffusion.training.runners.generate_replicas \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --test_dist_csv "${TEST_CSV}" \
    --out_dir "${OUT_DIR}" \
    --replica_id "${SLURM_ARRAY_TASK_ID}" \
    --num_replicas "${NUM_REPLICAS}" \
    --batch_size "${BATCH_SIZE}" \
    --seed_base "${SEED_BASE}" \
    --dtype "${DTYPE}" \
    --use_ema \
    --device cuda

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=========================================================================="
echo "Replica ${SLURM_ARRAY_TASK_ID} Complete!"
echo "=========================================================================="
echo "End time: $(date)"
echo "Elapsed: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Output: ${OUT_DIR}/replicas/replica_$(printf '%03d' ${SLURM_ARRAY_TASK_ID}).npz"
