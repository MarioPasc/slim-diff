#!/usr/bin/env bash
# =============================================================================
# launch_camera_ready.sh — Submit all 6 ICIP 2026 camera-ready experiments
# =============================================================================
# Grid: 3 folds × 2 architectures = 6 independent SLURM jobs.
# Each job runs `train_generate.sh` in its own cell directory.
#
# Paths + knobs come from picasso_paths.yaml (same directory). This launcher:
#   1) Sources the YAML via _load_paths.py (emits shell-safe export lines).
#   2) Passes REPO_SRC / DATA_SRC / RESULTS_ROOT / CONDA_ENV_NAME to every
#      sbatch via --export=ALL,VAR=val so each per-cell script picks them up
#      through its ${VAR:-default} fallback pattern.
#
# All 6 jobs share SEED_BASE=42 for generation (architecture-agnostic x_T
# seeding lets TASK-06 pair shared-vs-decoupled samples). The first job to
# run builds the 3-fold CSVs if absent; subsequent jobs reuse them (atomic +
# idempotent). Better: run `build_cache_and_kfold.sh` first, then this.
#
# Usage (from the repo root on Picasso):
#     bash slurm/camera_ready/launch_camera_ready.sh
# =============================================================================

set -euo pipefail

# =============================================================================
# BOOTSTRAP — hardcoded Picasso defaults
# =============================================================================
# Only these two paths are hardcoded; everything else comes from
# picasso_paths.yaml. Activating the conda env first ensures the YAML loader
# (PyYAML) is available even when the launcher is invoked from a bare shell.
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
echo "  CACHE_DIR       = ${CACHE_DIR}"
echo "  CONDA_ENV_NAME  = ${CONDA_ENV_NAME}"
echo ""
echo "Submitting ICIP 2026 camera-ready experiments..."
echo "Matrix: 3 folds × 2 architectures = 6 jobs"
echo "Shared SEED_BASE=42 across all cells."
echo ""

SBATCH_EXPORT="ALL,REPO_SRC=${REPO_SRC},DATA_SRC=${DATA_SRC},RESULTS_ROOT=${RESULTS_ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME}"

SUBMITTED=0
for arch in shared decoupled; do
    for fold in 0 1 2; do
        JOB_DIR="${SCRIPT_DIR}/${arch}_fold_${fold}"
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
        echo "  Submitted ${arch}_fold_${fold}: job ${JOB_ID}"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "All ${SUBMITTED} jobs submitted. Monitor with: squeue -u \$USER"
