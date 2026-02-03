#!/usr/bin/env bash
#SBATCH -J iwinac_sc_p0.2_x0_lp1.5
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

START_TIME=$(date +%s)
echo "Job started at: $(date)"

# ========================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# ========================================================================
EXPERIMENT_NAME="iwinac_sc_p0.2_x0_lp1.5"
CONDA_ENV_NAME="jsddpm"

REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/slim-diff"
DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy"
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/iwinac_self_cond_ablation/${EXPERIMENT_NAME}"
CONFIG_FILE="${REPO_SRC}/slurm/iwinac2026/self_cond_probability_ablation/x0_lp_1.5/self_cond_p_0.2/iwinac_sc_p0.2_x0_lp1.5.yaml"

# ========================================================================
# GENERATION CONFIGURATION (after training)
# ========================================================================
NUM_REPLICAS=5
N_SAMPLES_PER_MODE=50
GEN_BATCH_SIZE=32
SEED_BASE=42
GEN_DTYPE="float16"

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
# DDP VERIFICATION
# ========================================================================
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "[ddp] Available GPUs: ${GPU_COUNT}"
echo "[ddp] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Requested 2 GPUs but only ${GPU_COUNT} available"
    echo "Training will proceed but may not use DDP as expected"
fi

# ========================================================================
# EXPERIMENT EXECUTION
# ========================================================================

# 1. Create the results directory
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
    CACHE_CONFIG_SRC="${REPO_SRC}/src/diffusion/config/cache/epilepsy.yaml"
    CACHE_CONFIG="${RESULTS_DST}/cache_epilepsy.yaml"
    cp "${CACHE_CONFIG_SRC}" "${CACHE_CONFIG}"
    sed -i "s|cache_dir: .*|cache_dir: \"${CACHE_DIR}\"|" "${CACHE_CONFIG}"
    sed -i "s|root_dir: .*|root_dir: \"${DATA_SRC}\"|" "${CACHE_CONFIG}"
    jsddpm-cache --config "${CACHE_CONFIG}"
    echo "Caching completed."
else
    echo "Cache directory exists. Skipping caching step."
    echo "   Found: ${CACHE_DIR}"
    echo "   To rebuild cache, delete this directory and re-run."
fi

echo "Starting DDP training with config: ${MODIFIED_CONFIG}"
echo "Using ${GPU_COUNT} GPUs with DDP strategy"

jsddpm-train --config "${MODIFIED_CONFIG}"

TRAIN_END_TIME=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END_TIME - START_TIME))
echo "=========================================================================="
echo "Training completed at: $(date)"
echo "Training time: $(($TRAIN_ELAPSED / 3600))h $((($TRAIN_ELAPSED / 60) % 60))m $(($TRAIN_ELAPSED % 60))s"
echo "=========================================================================="

# ========================================================================
# GENERATION PHASE: Sequential Replica Generation
# ========================================================================
echo ""
echo "=========================================================================="
echo "Starting Generation Phase"
echo "=========================================================================="

# Find the checkpoint in RESULTS_DST/checkpoints
CKPT_DIR="${RESULTS_DST}/checkpoints"
echo "Looking for checkpoint in: ${CKPT_DIR}"

if [ ! -d "${CKPT_DIR}" ]; then
    echo "ERROR: Checkpoint directory not found: ${CKPT_DIR}"
    echo "Training may have failed or checkpointing was disabled."
    exit 1
fi

CHECKPOINT=$(find "${CKPT_DIR}" -name "*.ckpt" -type f | head -n 1)

if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: No checkpoint file found in ${CKPT_DIR}"
    exit 1
fi

echo "Found checkpoint: ${CHECKPOINT}"

# Create output directory for replicas
REPLICAS_OUT_DIR="${RESULTS_DST}/replicas"
mkdir -p "${REPLICAS_OUT_DIR}"
echo "Replicas output directory: ${REPLICAS_OUT_DIR}"

echo ""
echo "Generation Configuration:"
echo "  - Number of replicas: ${NUM_REPLICAS}"
echo "  - Samples per (zbin, domain): ${N_SAMPLES_PER_MODE}"
echo "  - Batch size: ${GEN_BATCH_SIZE}"
echo "  - Seed base: ${SEED_BASE}"
echo "  - Output dtype: ${GEN_DTYPE}"
echo ""

# Generate replicas sequentially
GEN_START_TIME=$(date +%s)

for REPLICA_ID in $(seq 0 $((NUM_REPLICAS - 1))); do
    REPLICA_START_TIME=$(date +%s)

    echo "=========================================================================="
    echo "Generating replica ${REPLICA_ID} / $((NUM_REPLICAS - 1))"
    echo "Started at: $(date)"
    echo "=========================================================================="

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

    REPLICA_END_TIME=$(date +%s)
    REPLICA_ELAPSED=$((REPLICA_END_TIME - REPLICA_START_TIME))

    echo "Replica ${REPLICA_ID} completed in $(($REPLICA_ELAPSED / 3600))h $((($REPLICA_ELAPSED / 60) % 60))m $(($REPLICA_ELAPSED % 60))s"
    echo "Output: ${REPLICAS_OUT_DIR}/replica_$(printf '%03d' ${REPLICA_ID}).npz"
    echo ""
done

GEN_END_TIME=$(date +%s)
GEN_ELAPSED=$((GEN_END_TIME - GEN_START_TIME))

echo "=========================================================================="
echo "All replicas generated!"
echo "Generation time: $(($GEN_ELAPSED / 3600))h $((($GEN_ELAPSED / 60) % 60))m $(($GEN_ELAPSED % 60))s"
echo "=========================================================================="

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================================================="
echo "Job finished at: $(date)"
echo "=========================================================================="
echo "Summary:"
echo "  - Training time:   $(($TRAIN_ELAPSED / 3600))h $((($TRAIN_ELAPSED / 60) % 60))m $(($TRAIN_ELAPSED % 60))s"
echo "  - Generation time: $(($GEN_ELAPSED / 3600))h $((($GEN_ELAPSED / 60) % 60))m $(($GEN_ELAPSED % 60))s"
echo "  - Total time:      $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo ""
echo "Outputs:"
echo "  - Checkpoint: ${CHECKPOINT}"
echo "  - Replicas:   ${REPLICAS_OUT_DIR}/"
echo ""
echo "Experiment completed successfully."
