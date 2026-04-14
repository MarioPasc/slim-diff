#!/usr/bin/env bash
# =============================================================================
# _activate_conda.sh — shared conda activation for slurm/camera_ready/ scripts
# =============================================================================
# Sourced (not executed) by build_cache_and_kfold.sh, run_fold_eval.sh,
# run_posthoc.sh, and launch_camera_ready.sh. Reads ${CONDA_ENV_NAME} from the
# caller's shell (each script sets a hardcoded bootstrap default before
# sourcing). Leaves ``python`` + the ``slimdiff-*`` entrypoints on ``$PATH``.
#
# Usage:
#     CONDA_ENV_NAME="${CONDA_ENV_NAME:-slimdiff}"
#     source "${REPO_SRC}/slurm/camera_ready/_activate_conda.sh"
# =============================================================================

: "${CONDA_ENV_NAME:?_activate_conda.sh: CONDA_ENV_NAME must be set}"

module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
        module load "$m" && module_loaded=1 && break
    fi
done
if [ "$module_loaded" -eq 0 ]; then
    echo "[_activate_conda] no conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    # shellcheck disable=SC1090
    source activate "${CONDA_ENV_NAME}"
fi

echo "[_activate_conda] env=${CONDA_ENV_NAME}  python=$(which python)"
python -c "import sys; print('[_activate_conda] Python', sys.version.split()[0])"
