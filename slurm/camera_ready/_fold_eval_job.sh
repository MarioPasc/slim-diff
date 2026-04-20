#!/usr/bin/env bash
# =============================================================================
# _fold_eval_job.sh — SLURM compute job for fold-aware similarity metrics
# =============================================================================
# Submitted by run_fold_eval.sh (which pre-downloads model weights on the login
# node). Runs `slimdiff-metrics fold-eval` on the 3-fold × 2-arch grid.
#
# Outputs (written to ${EVAL_OUTPUT_DIR}):
#   fold_metrics.csv, summary_metrics.csv, wasserstein_per_feature.csv,
#   eval_sample_counts.json
#
# SLURM resources come from picasso_paths.yaml → slurm.fold_eval.
# =============================================================================
#SBATCH -J slimdiff_cr_fold_eval
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

START_TIME=$(date +%s)
echo "Job started at: $(date)"

# =============================================================================
# BOOTSTRAP — hardcoded Picasso defaults
# =============================================================================
REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/slim-diff}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-slimdiff}"

# =============================================================================
# STEP 0 — CONDA ENV ACTIVATION (must precede YAML loading)
# =============================================================================
# shellcheck disable=SC1091
source "${REPO_SRC}/slurm/camera_ready/_activate_conda.sh"
python -c "import torch; print('[torch] CUDA', torch.cuda.is_available(), '— devices', torch.cuda.device_count())"

# =============================================================================
# LOAD PATHS FROM YAML
# =============================================================================
PATHS_YAML="${PATHS_YAML:-${REPO_SRC}/slurm/camera_ready/picasso_paths.yaml}"
LOADER="${REPO_SRC}/slurm/camera_ready/_load_paths.py"

if [ ! -f "${PATHS_YAML}" ]; then
    echo "ERROR: ${PATHS_YAML} not found." >&2
    exit 1
fi
if [ ! -f "${LOADER}" ]; then
    echo "ERROR: ${LOADER} not found." >&2
    exit 1
fi

eval "$(python "${LOADER}" "${PATHS_YAML}")"

# Point torchvision / torch-fidelity at pre-downloaded weights (no internet on
# compute nodes). TORCH_HOME is emitted by _load_paths.py from picasso_paths.yaml.
export TORCH_HOME
echo "  TORCH_HOME           = ${TORCH_HOME}"

echo "Loaded paths:"
echo "  REPO_SRC             = ${REPO_SRC}"
echo "  CACHE_DIR            = ${CACHE_DIR}"
echo "  RESULTS_ROOT         = ${RESULTS_ROOT}"
echo "  EVAL_OUTPUT_DIR      = ${EVAL_OUTPUT_DIR}"
echo "  EVAL_CONFIG_TEMPLATE = ${EVAL_CONFIG_TEMPLATE}"
echo "  EVAL_DEVICE          = ${EVAL_DEVICE}"
echo "  CONDA_ENV_NAME       = ${CONDA_ENV_NAME}"

# =============================================================================
# PRE-FLIGHT
# =============================================================================
cd "${REPO_SRC}"

EVAL_CONFIG_ABS="${REPO_SRC}/${EVAL_CONFIG_TEMPLATE}"
if [ ! -f "${EVAL_CONFIG_ABS}" ]; then
    echo "ERROR: fold-eval config template not found: ${EVAL_CONFIG_ABS}" >&2
    exit 1
fi

FOLD_META="${CACHE_DIR}/folds/folds_meta.json"
if [ ! -f "${FOLD_META}" ]; then
    echo "ERROR: ${FOLD_META} not found — run build_cache_and_kfold.sh first." >&2
    exit 1
fi

# Verify pre-downloaded weights exist
WEIGHTS_DIR="${TORCH_HOME}/hub/checkpoints"
for w in vgg16-397923af.pth weights-inception-2015-12-05-6726825d.pth; do
    if [ ! -f "${WEIGHTS_DIR}/${w}" ]; then
        echo "ERROR: ${WEIGHTS_DIR}/${w} missing — run 'bash run_fold_eval.sh' from login node." >&2
        exit 1
    fi
done
echo "Pre-downloaded weights OK."

# Verify every expected replicas/ directory exists
missing=0
for arch in shared decoupled; do
    for k in 0 1 2; do
        cell="${RESULTS_ROOT}/slimdiff_cr_${arch}_fold_${k}/replicas"
        if ! compgen -G "${cell}/replica_*.npz" > /dev/null; then
            echo "WARNING: no replica_*.npz under ${cell}" >&2
            missing=$((missing + 1))
        fi
    done
done
if [ "${missing}" -gt 0 ]; then
    echo "WARNING: ${missing} cell(s) missing replicas — fold-eval will skip or error." >&2
fi

mkdir -p "${EVAL_OUTPUT_DIR}"

# =============================================================================
# RUN
# =============================================================================
echo ""
echo "Starting fold-aware evaluation ..."
slimdiff-metrics fold-eval \
    --config "${EVAL_CONFIG_ABS}" \
    --results-root "${RESULTS_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --device "${EVAL_DEVICE}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Fold-eval finished in $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s."
echo "Outputs:"
echo "  ${EVAL_OUTPUT_DIR}/fold_metrics.csv"
echo "  ${EVAL_OUTPUT_DIR}/summary_metrics.csv"
echo "  ${EVAL_OUTPUT_DIR}/wasserstein_per_feature.csv"
echo "  ${EVAL_OUTPUT_DIR}/eval_sample_counts.json"
echo ""
echo "Next: sbatch slurm/camera_ready/run_posthoc.sh"
