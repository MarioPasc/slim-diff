#!/usr/bin/env bash
# =============================================================================
# launch_zero_coupling.sh — Submit 3 zero-coupling baseline experiments
# =============================================================================
# SASHIMI 2026 ablation: 3 folds × 1 architecture (IndependentTwinDDPM).
# Each job runs `train_generate.sh` in its own cell directory.
#
# Paths + knobs come from picasso_paths.yaml (same directory as the
# camera-ready launcher). SEED_BASE=42 is shared across ALL cells
# (including the existing shared/decoupled) for paired comparison.
#
# Usage (from the repo root on Picasso):
#     bash slurm/camera_ready/launch_zero_coupling.sh
# =============================================================================

set -euo pipefail

# =============================================================================
# BOOTSTRAP
# =============================================================================
REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/slim-diff}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-slimdiff}"

# shellcheck disable=SC1091
source "${REPO_SRC}/slurm/camera_ready/_activate_conda.sh"

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

# Load paths into the current shell.
eval "$(python "${LOADER}" "${PATHS_YAML}")"

echo "Loaded paths from ${PATHS_YAML}:"
echo "  REPO_SRC        = ${REPO_SRC}"
echo "  DATA_SRC        = ${DATA_SRC}"
echo "  RESULTS_ROOT    = ${RESULTS_ROOT}"
echo "  CONDA_ENV_NAME  = ${CONDA_ENV_NAME}"
echo ""
echo "Submitting SASHIMI 2026 zero-coupling baseline experiments..."
echo "Matrix: 3 folds × 1 architecture = 3 jobs"
echo "Shared SEED_BASE=42 across all cells."
echo ""

SBATCH_EXPORT="ALL,REPO_SRC=${REPO_SRC},DATA_SRC=${DATA_SRC},RESULTS_ROOT=${RESULTS_ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME}"

CELLS_DIR="${REPO_SRC}/slurm/camera_ready"

SUBMITTED=0
for fold in 0 1 2; do
    JOB_DIR="${CELLS_DIR}/zero_coupling_fold_${fold}"
    JOB_SCRIPT="${JOB_DIR}/train_generate.sh"
    CONFIG_FILE="${JOB_DIR}/config.yaml"

    if [ ! -f "${JOB_SCRIPT}" ]; then
        echo "ERROR: Missing SLURM script ${JOB_SCRIPT}" >&2
        exit 1
    fi
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Missing config ${CONFIG_FILE}" >&2
        exit 1
    fi
    if [ ! -x "${JOB_SCRIPT}" ]; then
        echo "ERROR: Not executable: ${JOB_SCRIPT}" >&2
        echo "       Run:  chmod +x ${JOB_SCRIPT}" >&2
        exit 1
    fi

    JOB_ID=$(sbatch --export="${SBATCH_EXPORT}" "${JOB_SCRIPT}" | awk '{print $NF}')
    echo "  Submitted zero_coupling_fold_${fold}: job ${JOB_ID}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "All ${SUBMITTED} jobs submitted. Monitor with: squeue -u \$USER"
