#!/usr/bin/env bash
# =============================================================================
# worker_bottleneck_only.sh — run on server3 inside a detached tmux session
# =============================================================================
# Trains and generates replicas for the BottleneckSharedTwinDDPM (PRL pivot,
# bottleneck-only coupling) sequentially across folds. One fold at a time
# (single GPU). For each fold:
#   1. Stage per-fold config to ${RESULTS_DST}/config.yaml (sed-patched).
#   2. slimdiff-train  → checkpoints + train.log.
#   3. Sequential generate_replicas loop (NUM_REPLICAS replicas).
#
# Env vars expected (exported by launch_bottleneck_only_server3.sh, or by hand):
#   REMOTE_REPO          /home/mariopascual/projects/slim-diff
#   REMOTE_DATA          /media/hddb/mario/slimdiff/slice_cache/slice_cache
#   REMOTE_RESULTS       /media/hddb/mario/slimdiff/results
#   CONDA_ENV            slimdiff
#   GPU_ID               1
#
# Args:
#   --mode {smoke|prod}  smoke: max_epochs=3, num_replicas=1, separate results subdir
#                        prod : config-defined max_epochs, num_replicas=10
#   --folds 0,1,2        Comma-separated fold ids to process sequentially.
# =============================================================================

set -euo pipefail

MODE="prod"
FOLDS="0,1,2"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)  MODE="$2";  shift 2 ;;
        --folds) FOLDS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "prod" ]]; then
    echo "ERROR: --mode must be 'smoke' or 'prod', got '${MODE}'" >&2
    exit 1
fi

# ----------------------------------------------------------------------
# Conda activation (server3 has /opt/anaconda; no module system).
# ----------------------------------------------------------------------
# shellcheck disable=SC1091
source /opt/anaconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-slimdiff}"

# Default env overrides (so this worker can run by hand without the launcher).
: "${REMOTE_REPO:=/home/mariopascual/projects/slim-diff}"
: "${REMOTE_DATA:=/media/hddb/mario/slimdiff/slice_cache/slice_cache}"
: "${REMOTE_RESULTS:=/media/hddb/mario/slimdiff/results}"
: "${GPU_ID:=1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REMOTE_REPO}/src:${REMOTE_REPO}:${PYTHONPATH:-}"
cd "${REMOTE_REPO}"

if [[ "$MODE" == "smoke" ]]; then
    NUM_REPLICAS=1
    MAX_EPOCHS_OVERRIDE=3
    RESULTS_PARENT="${REMOTE_RESULTS}/bottleneck_only_smoke"
    EXP_PREFIX="bottleneck_only_smoke"
else
    NUM_REPLICAS=10
    MAX_EPOCHS_OVERRIDE=""
    RESULTS_PARENT="${REMOTE_RESULTS}"
    EXP_PREFIX="slimdiff_pivot_bottleneck_only"
fi

# Hyperparams held constant across all cells (matches Picasso scripts).
N_SAMPLES_PER_MODE=150
GEN_BATCH_SIZE=32
SEED_BASE=42
GEN_DTYPE="float16"

echo "=========================================================================="
echo "[worker_bottleneck_only] $(date)"
echo "  host          : $(hostname)"
echo "  mode          : ${MODE}"
echo "  folds         : ${FOLDS}"
echo "  CUDA_VISIBLE  : ${CUDA_VISIBLE_DEVICES}"
echo "  conda env     : ${CONDA_DEFAULT_ENV:-unknown}"
echo "  python        : $(which python)"
echo "  torch version : $(python -c 'import torch; print(torch.__version__)')"
echo "  cuda avail    : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  REMOTE_REPO   : ${REMOTE_REPO}"
echo "  REMOTE_DATA   : ${REMOTE_DATA}"
echo "  REMOTE_RESULTS: ${RESULTS_PARENT}"
echo "  num_replicas  : ${NUM_REPLICAS}"
echo "=========================================================================="

IFS=',' read -ra FOLD_ARR <<<"${FOLDS}"
for K in "${FOLD_ARR[@]}"; do
    EXP_NAME="${EXP_PREFIX}_fold_${K}"
    if [[ "$MODE" == "smoke" ]]; then
        RESULTS_DST="${RESULTS_PARENT}/fold_${K}"
    else
        RESULTS_DST="${RESULTS_PARENT}/${EXP_NAME}"
    fi
    LOG_DIR="${RESULTS_DST}/logs"
    mkdir -p "${LOG_DIR}"

    SRC_CFG="${REMOTE_REPO}/slurm/server3/bottleneck_only_fold_${K}/config.yaml"
    MOD_CFG="${RESULTS_DST}/config.yaml"
    FOLD_CACHE="${REMOTE_DATA}/folds/fold_${K}"

    if [[ ! -f "${SRC_CFG}" ]]; then
        echo "ERROR: ${SRC_CFG} missing" >&2
        exit 1
    fi
    if [[ ! -d "${FOLD_CACHE}" ]]; then
        echo "ERROR: ${FOLD_CACHE} missing on server3" >&2
        exit 1
    fi

    cp "${SRC_CFG}" "${MOD_CFG}"
    sed -i "s|^  output_dir: .*|  output_dir: \"${RESULTS_DST}\"|" "${MOD_CFG}"
    sed -i "s|^  cache_dir: .*|  cache_dir: \"${FOLD_CACHE}\"|"    "${MOD_CFG}"
    # Patch wandb run name + experiment name for traceability.
    sed -i "s|^  name: \"slimdiff_cr_bottleneck_only_fold_.*|  name: \"${EXP_NAME}\"|" "${MOD_CFG}"
    sed -i "s|^      name: \"slimdiff_cr_bottleneck_only_fold_.*|      name: \"${EXP_NAME}\"|" "${MOD_CFG}"
    if [[ -n "${MAX_EPOCHS_OVERRIDE}" ]]; then
        sed -i "s|^  max_epochs: .*|  max_epochs: ${MAX_EPOCHS_OVERRIDE}|" "${MOD_CFG}"
        # Smoke: also disable early stopping (3 epochs is below patience).
        sed -i "s|^    enabled: true$|    enabled: false|" "${MOD_CFG}" || true
    fi

    echo ""
    echo "=========================================================================="
    echo "[fold ${K}] experiment: ${EXP_NAME}"
    echo "[fold ${K}] config    : ${MOD_CFG}"
    echo "[fold ${K}] results   : ${RESULTS_DST}"
    echo "[fold ${K}] cache     : ${FOLD_CACHE}"
    echo "[fold ${K}] start     : $(date)"
    echo "=========================================================================="

    TRAIN_START=$(date +%s)
    slimdiff-train --config "${MOD_CFG}" 2>&1 | tee -a "${LOG_DIR}/train.log"
    TRAIN_END=$(date +%s)
    echo "[fold ${K}] train elapsed: $((TRAIN_END - TRAIN_START))s"

    CKPT_DIR="${RESULTS_DST}/checkpoints"
    CHECKPOINT=$(find "${CKPT_DIR}" -name "*.ckpt" -type f 2>/dev/null | head -n 1 || true)
    if [[ -z "${CHECKPOINT}" ]]; then
        echo "ERROR: No checkpoint found in ${CKPT_DIR} after training." >&2
        exit 1
    fi
    echo "[fold ${K}] checkpoint: ${CHECKPOINT}"

    REPLICAS_OUT_DIR="${RESULTS_DST}/replicas"
    mkdir -p "${REPLICAS_OUT_DIR}"

    GEN_START=$(date +%s)
    for R in $(seq 0 $((NUM_REPLICAS - 1))); do
        echo "[fold ${K}] replica ${R}/$((NUM_REPLICAS - 1)) start: $(date)"
        python -m src.diffusion.training.runners.generate_replicas \
            --config "${MOD_CFG}" \
            --checkpoint "${CHECKPOINT}" \
            --out_dir "${RESULTS_DST}" \
            --replica_id "${R}" \
            --num_replicas "${NUM_REPLICAS}" \
            --batch_size "${GEN_BATCH_SIZE}" \
            --seed_base "${SEED_BASE}" \
            --dtype "${GEN_DTYPE}" \
            --use_ema \
            --device cuda \
            --uniform_modes_generation \
            --n_samples_per_mode "${N_SAMPLES_PER_MODE}" \
            2>&1 | tee -a "${LOG_DIR}/generate.log"
    done
    GEN_END=$(date +%s)
    echo "[fold ${K}] generate elapsed: $((GEN_END - GEN_START))s"
    echo "[fold ${K}] DONE  : $(date)"
done

echo ""
echo "=========================================================================="
echo "[worker_bottleneck_only] all folds complete: $(date)"
echo "=========================================================================="
