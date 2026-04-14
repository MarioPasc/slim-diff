#!/usr/bin/env bash
# =============================================================================
# run_fold_eval.sh — Fold-aware similarity metrics (TASK-04) on the 6-cell grid
# =============================================================================
# Submits `slimdiff-metrics fold-eval` on the 3-fold × 2-arch grid after all
# 6 generation jobs have finished. Writes:
#
#   ${EVAL_OUTPUT_DIR}/fold_metrics.csv
#   ${EVAL_OUTPUT_DIR}/summary_metrics.csv
#   ${EVAL_OUTPUT_DIR}/wasserstein_per_feature.csv
#   ${EVAL_OUTPUT_DIR}/eval_sample_counts.json
#
# Requires: generation artefacts under
#   ${RESULTS_ROOT}/slimdiff_cr_{shared,decoupled}_fold_{0,1,2}/replicas/replica_*.npz
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
# LOAD PATHS FROM YAML
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATHS_YAML="${PATHS_YAML:-${SCRIPT_DIR}/picasso_paths.yaml}"
LOADER="${SCRIPT_DIR}/_load_paths.py"

if [ ! -f "${PATHS_YAML}" ]; then
    echo "ERROR: ${PATHS_YAML} not found." >&2
    exit 1
fi

eval "$(python "${LOADER}" "${PATHS_YAML}")"

echo "Loaded paths:"
echo "  REPO_SRC             = ${REPO_SRC}"
echo "  CACHE_DIR            = ${CACHE_DIR}"
echo "  RESULTS_ROOT         = ${RESULTS_ROOT}"
echo "  EVAL_OUTPUT_DIR      = ${EVAL_OUTPUT_DIR}"
echo "  EVAL_CONFIG_TEMPLATE = ${EVAL_CONFIG_TEMPLATE}"
echo "  EVAL_DEVICE          = ${EVAL_DEVICE}"
echo "  CONDA_ENV_NAME       = ${CONDA_ENV_NAME}"

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

echo "[python] $(which python)"
python -c "import torch; print('CUDA', torch.cuda.is_available(), '— devices', torch.cuda.device_count())"

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

# Verify every expected replicas/ directory exists — otherwise fold-eval will
# fail mid-run. This catches "submitted before all 6 training jobs finished".
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
