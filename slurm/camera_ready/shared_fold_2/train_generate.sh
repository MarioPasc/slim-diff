#!/usr/bin/env bash
#SBATCH -J slimdiff_cr_shared_fold_2
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:2
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# =============================================================================
# SLIM-Diff ICIP 2026 camera-ready: shared bottleneck × fold 2
# =============================================================================
# One of 6 cells in the 3-fold × 2-architecture grid. See:
#   docs/icip2026/rebuttal_plans/TASK_03_ORCH_training_orchestration.md
#   slurm/camera_ready/README.md
#
# Reproducibility:
#   - SEED_BASE=42 is shared across ALL 6 cells so that x_T noise realisations
#     produced by `generate_replicas.py` are byte-identical for matched
#     (replica, zbin, domain, sample_index) across architectures within the
#     same fold (the SHA256 seeding is architecture-agnostic). This enables
#     paired shared-vs-decoupled qualitative comparison in TASK-06.
#   - Fold CSVs are produced by `slimdiff-kfold` with seed=42; the first
#     camera-ready job to run builds them atomically, subsequent jobs skip.
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Job started at: $(date)"

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
ARCH="shared"                               # {shared, decoupled}
FOLD_ID=2                                   # {0, 1, 2}
EXPERIMENT_NAME="slimdiff_cr_${ARCH}_fold_${FOLD_ID}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-jsddpm}"

# --- Picasso paths (override via env; launch_camera_ready.sh exports -----
# --- them from picasso_paths.yaml) ---------------------------------------
REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/slim-diff}"
DATA_SRC="${DATA_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/home/users/tic_163_uma/mpascual/fscratch/results/icip2026/camera_ready}"
RESULTS_DST="${RESULTS_ROOT}/${EXPERIMENT_NAME}"

CONFIG_SRC="${REPO_SRC}/slurm/camera_ready/${ARCH}_fold_${FOLD_ID}/config.yaml"

# --- Derived ----------------------------------------------------------------
CACHE_DIR="${DATA_SRC}/slice_cache"
FOLD_CACHE_DIR="${CACHE_DIR}/folds/fold_${FOLD_ID}"

# --- Generation phase parameters (held constant across all 6 cells) --------
NUM_REPLICAS=20
N_SAMPLES_PER_MODE=150                      # 150 × 30 zbins × 2 domains = 9 000 / replica
GEN_BATCH_SIZE=32
SEED_BASE=42                                # CRITICAL: identical across cells
GEN_DTYPE="float16"

# =============================================================================
# CONDA ENV ACTIVATION
# =============================================================================
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
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('CUDA', torch.cuda.is_available())"
echo "[torch] $(python -c 'import torch; print(torch.__version__)')"
echo "[cuda devices] $(python -c 'import torch; print(torch.cuda.device_count())')"

# =============================================================================
# DDP VERIFICATION
# =============================================================================
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "[ddp] Available GPUs: ${GPU_COUNT}"
echo "[ddp] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: camera-ready config requests 2 GPUs but only ${GPU_COUNT} available"
    echo "Training will fall back but DDP configuration may not behave as expected."
fi

# =============================================================================
# BASE SLICE CACHE (build once across all jobs if missing)
# =============================================================================
cd "${REPO_SRC}"
echo "Working directory: $(pwd)"

if [ ! -d "${CACHE_DIR}" ]; then
    echo "Base cache directory not found. Running caching step..."
    CACHE_CONFIG_SRC="${REPO_SRC}/src/diffusion/config/cache/epilepsy.yaml"
    CACHE_CONFIG="${RESULTS_DST}/cache_epilepsy.yaml"
    mkdir -p "${RESULTS_DST}"
    cp "${CACHE_CONFIG_SRC}" "${CACHE_CONFIG}"
    sed -i "s|cache_dir: .*|cache_dir: \"${CACHE_DIR}\"|" "${CACHE_CONFIG}"
    sed -i "s|root_dir: .*|root_dir: \"${DATA_SRC}\"|"   "${CACHE_CONFIG}"
    slimdiff-cache --config "${CACHE_CONFIG}"
    echo "Base caching completed."
else
    echo "Base cache directory exists. Skipping caching step."
    echo "   Found: ${CACHE_DIR}"
fi

# =============================================================================
# FOLD CSV GENERATION (idempotent; atomic swap inside slimdiff-kfold)
# =============================================================================
# Rationale: running inside the job (not from the login node) avoids mount
# issues. `slimdiff-kfold` writes `folds.tmp/` then `os.replace`-swaps to
# `folds/`, so concurrent jobs are safe — the first that wins the rename
# wins; subsequent jobs see `train.csv` and skip.
if [ ! -f "${FOLD_CACHE_DIR}/train.csv" ]; then
    echo "Fold CSV not found at ${FOLD_CACHE_DIR}/train.csv. Running slimdiff-kfold..."
    slimdiff-kfold \
        --cache-dir "${CACHE_DIR}" \
        --n-folds 3 \
        --seed 42
    echo "Fold CSV generation completed."
else
    echo "Fold CSV exists. Skipping slimdiff-kfold."
    echo "   Found: ${FOLD_CACHE_DIR}/train.csv"
fi

# =============================================================================
# CONFIG STAGING: copy + sed-patch placeholders to absolute Picasso paths
# =============================================================================
mkdir -p "${RESULTS_DST}"
CONFIG_BASENAME=$(basename "${CONFIG_SRC}")
MODIFIED_CONFIG="${RESULTS_DST}/${CONFIG_BASENAME}"

echo "Copying config file to: ${MODIFIED_CONFIG}"
cp "${CONFIG_SRC}" "${MODIFIED_CONFIG}"

echo "Modifying configuration file..."
sed -i "s|  output_dir: .*|  output_dir: \"${RESULTS_DST}\"|"     "${MODIFIED_CONFIG}"
sed -i "s|  cache_dir: .*|  cache_dir: \"${FOLD_CACHE_DIR}\"|"    "${MODIFIED_CONFIG}"

# =============================================================================
# TRAINING
# =============================================================================
echo "Starting DDP training with config: ${MODIFIED_CONFIG}"
echo "Using ${GPU_COUNT} GPUs with DDP strategy"

slimdiff-train --config "${MODIFIED_CONFIG}"

TRAIN_END_TIME=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END_TIME - START_TIME))
echo "=========================================================================="
echo "Training completed at: $(date)"
echo "Training time: $(($TRAIN_ELAPSED / 3600))h $((($TRAIN_ELAPSED / 60) % 60))m $(($TRAIN_ELAPSED % 60))s"
echo "=========================================================================="

# =============================================================================
# GENERATION PHASE: Sequential Replica Generation
# =============================================================================
echo ""
echo "=========================================================================="
echo "Starting Generation Phase"
echo "=========================================================================="

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

REPLICAS_OUT_DIR="${RESULTS_DST}/replicas"
mkdir -p "${REPLICAS_OUT_DIR}"
echo "Replicas output directory: ${REPLICAS_OUT_DIR}"

echo ""
echo "Generation Configuration:"
echo "  - Number of replicas:      ${NUM_REPLICAS}"
echo "  - Samples per (zbin,dom):  ${N_SAMPLES_PER_MODE}"
echo "  - Batch size:              ${GEN_BATCH_SIZE}"
echo "  - Seed base (shared):      ${SEED_BASE}"
echo "  - Output dtype:            ${GEN_DTYPE}"
echo ""

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
echo "  - Experiment:      ${EXPERIMENT_NAME}"
echo "  - Architecture:    ${ARCH}"
echo "  - Fold:            ${FOLD_ID}"
echo "  - Training time:   $(($TRAIN_ELAPSED / 3600))h $((($TRAIN_ELAPSED / 60) % 60))m $(($TRAIN_ELAPSED % 60))s"
echo "  - Generation time: $(($GEN_ELAPSED / 3600))h $((($GEN_ELAPSED / 60) % 60))m $(($GEN_ELAPSED % 60))s"
echo "  - Total time:      $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo ""
echo "Outputs:"
echo "  - Checkpoint: ${CHECKPOINT}"
echo "  - Replicas:   ${REPLICAS_OUT_DIR}/"
echo ""
echo "Experiment completed successfully."
