#!/usr/bin/env bash
#SBATCH -J xai_proposal_full
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:2
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

START_TIME=$(date +%s)
echo "Job started at: $(date)"

# ========================================================================
# EXPERIMENT CONFIGURATION
# ========================================================================
EXPERIMENT_NAME="xai_proposal_full"
CONDA_ENV_NAME="jsddpm"

REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/js-ddpm-epilepsy"
DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy"
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/${EXPERIMENT_NAME}"
CONFIG_FILE="${REPO_SRC}/slurm/ablation_study_xai/proposal/proposal.yaml"

# ========================================================================
# GENERATION CONFIGURATION
# ========================================================================
NUM_REPLICAS=5
N_SAMPLES_PER_MODE=50
GEN_BATCH_SIZE=32
SEED_BASE=42
GEN_DTYPE="float32"

# ========================================================================
# CLASSIFICATION & DIAGNOSTICS CONFIGURATION
# ========================================================================
CLASSIFICATION_DIR="${RESULTS_DST}/classification"
DIAGNOSTICS_DIR="${RESULTS_DST}/diagnostics"
RESULTS_BASE_DIR=$(dirname "${RESULTS_DST}")

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

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch, os; print('CUDA', torch.cuda.is_available())"
echo "[torch] $(python -c 'import torch; print(torch.__version__)')"
echo "[cuda devices] $(python -c 'import torch; print(torch.cuda.device_count())')"

GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "[gpu] Available GPUs: ${GPU_COUNT}"

# ========================================================================
# PHASE 1: TRAINING
# ========================================================================
echo ""
echo "=========================================================================="
echo "PHASE 1: Training — ${EXPERIMENT_NAME}"
echo "=========================================================================="

mkdir -p "${RESULTS_DST}"

CONFIG_BASENAME=$(basename "${CONFIG_FILE}")
MODIFIED_CONFIG="${RESULTS_DST}/${CONFIG_BASENAME}"

echo "Copying config file to: ${MODIFIED_CONFIG}"
cp "${CONFIG_FILE}" "${MODIFIED_CONFIG}"

echo "Modifying configuration file paths for cluster..."
sed -i "s|  cache_dir: .*|  cache_dir: \"${DATA_SRC}/slice_cache\"|" "${MODIFIED_CONFIG}"
sed -i "s|  output_dir: .*|  output_dir: \"${RESULTS_DST}\"|" "${MODIFIED_CONFIG}"

cd "${REPO_SRC}"
echo "Working directory: $(pwd)"

CACHE_DIR="${DATA_SRC}/slice_cache"
echo "Cache directory: ${CACHE_DIR}"

if [ ! -d "${CACHE_DIR}" ]; then
    echo "Cache directory not found. Running caching step..."
    CACHE_CONFIG_SRC="${REPO_SRC}/src/diffusion/config/cache/epilepsy.yaml"
    CACHE_CONFIG="${RESULTS_DST}/cache_epilepsy.yaml"
    cp "${CACHE_CONFIG_SRC}" "${CACHE_CONFIG}"
    sed -i "s|cache_dir: .*|cache_dir: \"${CACHE_DIR}\"|" "${CACHE_CONFIG}"
    sed -i "s|root_dir: .*|root_dir: \"${DATA_SRC}\"|" "${CACHE_CONFIG}"
    jsddpm-cache --config "${CACHE_CONFIG}"
    echo "Caching completed."
else
    echo "Cache directory exists. Skipping caching step."
fi

echo "Starting training with config: ${MODIFIED_CONFIG}"
jsddpm-train --config "${MODIFIED_CONFIG}"

TRAIN_END_TIME=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END_TIME - START_TIME))
echo "Training completed in $(($TRAIN_ELAPSED / 3600))h $((($TRAIN_ELAPSED / 60) % 60))m $(($TRAIN_ELAPSED % 60))s"

# ========================================================================
# PHASE 2: GENERATION
# ========================================================================
echo ""
echo "=========================================================================="
echo "PHASE 2: Replica Generation"
echo "=========================================================================="

CKPT_DIR="${RESULTS_DST}/checkpoints"
echo "Looking for checkpoint in: ${CKPT_DIR}"

if [ ! -d "${CKPT_DIR}" ]; then
    echo "ERROR: Checkpoint directory not found: ${CKPT_DIR}"
    exit 1
fi

CHECKPOINT=$(find "${CKPT_DIR}" -name "*.ckpt" -type f | head -n 1)

if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: No checkpoint file found in ${CKPT_DIR}"
    exit 1
fi

echo "Found checkpoint: ${CHECKPOINT}"

REPLICAS_OUT_DIR="${RESULTS_DST}/replicas"
mkdir -p "${REPLICAS_OUT_DIR}"

GEN_START_TIME=$(date +%s)

for REPLICA_ID in $(seq 0 $((NUM_REPLICAS - 1))); do
    echo "Generating replica ${REPLICA_ID} / $((NUM_REPLICAS - 1)) ..."

    python -m src.diffusion.training.runners.generate_replicas \
        --config "${MODIFIED_CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --out_dir "${RESULTS_DST}" \
        --replica_id ${REPLICA_ID} \
        --num_replicas ${NUM_REPLICAS} \
        --batch_size ${GEN_BATCH_SIZE} \
        --seed_base ${SEED_BASE} \
        --dtype ${GEN_DTYPE} \
        --use_ema \
        --device cuda \
        --uniform_modes_generation \
        --n_samples_per_mode ${N_SAMPLES_PER_MODE}

    echo "  Replica ${REPLICA_ID} done."
done

GEN_END_TIME=$(date +%s)
GEN_ELAPSED=$((GEN_END_TIME - GEN_START_TIME))
echo "Generation time: $(($GEN_ELAPSED / 3600))h $((($GEN_ELAPSED / 60) % 60))m $(($GEN_ELAPSED % 60))s"
