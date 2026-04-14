#!/usr/bin/env bash
#SBATCH -J slimdiff_cr_build
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=cpu
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# =============================================================================
# build_cache_and_kfold.sh — ICIP 2026 camera-ready prereq builder
# =============================================================================
# One-shot CPU job that produces the two artefacts the training grid depends
# on, reading all paths from picasso_paths.yaml:
#
#   1) ${CACHE_DIR}/                          — slice cache (slimdiff-cache)
#   2) ${CACHE_DIR}/folds/fold_{0,1,2}/*.csv  — 3-fold CSVs (slimdiff-kfold)
#
# Both steps are idempotent — skip if the output already exists. Running this
# explicitly before `launch_camera_ready.sh` is the recommended path; running
# it as part of the first training job also works (it serialises cache build
# behind one GPU-hour).
#
# SLURM resource overrides come from picasso_paths.yaml → slurm.build.
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Job started at: $(date)"

# =============================================================================
# LOAD PATHS FROM YAML
# =============================================================================
# Under sbatch, $0 points into /var/spool/slurmd/, not the repo. Resolution
# order:
#   1) $PATHS_YAML (explicit override)
#   2) $SLURM_SUBMIT_DIR/slurm/camera_ready/picasso_paths.yaml
#      (recommended: `sbatch slurm/camera_ready/<script>.sh` from repo root)
#   3) $(dirname "$0")/picasso_paths.yaml (direct invocation `bash …`)
if [ -z "${PATHS_YAML:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] \
        && [ -f "${SLURM_SUBMIT_DIR}/slurm/camera_ready/picasso_paths.yaml" ]; then
        PATHS_YAML="${SLURM_SUBMIT_DIR}/slurm/camera_ready/picasso_paths.yaml"
    else
        PATHS_YAML="$(cd "$(dirname "$0")" && pwd)/picasso_paths.yaml"
    fi
fi
LOADER="$(dirname "${PATHS_YAML}")/_load_paths.py"

if [ ! -f "${PATHS_YAML}" ]; then
    echo "ERROR: ${PATHS_YAML} not found." >&2
    echo "Hint: sbatch from the repo root (sbatch slurm/camera_ready/<script>.sh)" >&2
    echo "      or export PATHS_YAML=/abs/path/to/picasso_paths.yaml before sbatch." >&2
    exit 1
fi
if [ ! -f "${LOADER}" ]; then
    echo "ERROR: ${LOADER} not found." >&2
    exit 1
fi

eval "$(python "${LOADER}" "${PATHS_YAML}")"

echo "Loaded paths:"
echo "  REPO_SRC              = ${REPO_SRC}"
echo "  DATA_SRC              = ${DATA_SRC}"
echo "  CACHE_DIR             = ${CACHE_DIR}"
echo "  CACHE_CONFIG_TEMPLATE = ${CACHE_CONFIG_TEMPLATE}"
echo "  KFOLD_N_FOLDS         = ${KFOLD_N_FOLDS}"
echo "  KFOLD_SEED            = ${KFOLD_SEED}"
echo "  CONDA_ENV_NAME        = ${CONDA_ENV_NAME}"

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
python -c "import sys; print('Python', sys.version.split()[0])"

# =============================================================================
# STEP 1 — SLICE CACHE
# =============================================================================
cd "${REPO_SRC}"

if [ -d "${CACHE_DIR}" ] && [ -f "${CACHE_DIR}/train.csv" ] \
    && [ -f "${CACHE_DIR}/val.csv" ] && [ -f "${CACHE_DIR}/test.csv" ]; then
    echo "[cache] ${CACHE_DIR} already populated — skipping slice cache build."
else
    echo "[cache] Building slice cache in ${CACHE_DIR} ..."

    CACHE_CONFIG_SRC="${REPO_SRC}/${CACHE_CONFIG_TEMPLATE}"
    if [ ! -f "${CACHE_CONFIG_SRC}" ]; then
        echo "ERROR: Cache config template not found: ${CACHE_CONFIG_SRC}" >&2
        exit 1
    fi

    # Stage a patched copy alongside the cache dir so re-runs can inspect it.
    mkdir -p "${CACHE_DIR}"
    CACHE_CONFIG_STAGED="${CACHE_DIR}/_cache_config.yaml"
    cp "${CACHE_CONFIG_SRC}" "${CACHE_CONFIG_STAGED}"
    sed -i "s|cache_dir: .*|cache_dir: \"${CACHE_DIR}\"|" "${CACHE_CONFIG_STAGED}"
    sed -i "s|root_dir: .*|root_dir: \"${DATA_SRC}\"|"   "${CACHE_CONFIG_STAGED}"

    echo "[cache] using patched config ${CACHE_CONFIG_STAGED}"
    slimdiff-cache --config "${CACHE_CONFIG_STAGED}"
    echo "[cache] done."
fi

# =============================================================================
# STEP 2 — K-FOLD CSVS
# =============================================================================
FOLD_ROOT="${CACHE_DIR}/folds"
ALL_FOLDS_READY=1
for k in $(seq 0 $((KFOLD_N_FOLDS - 1))); do
    if [ ! -f "${FOLD_ROOT}/fold_${k}/train.csv" ]; then
        ALL_FOLDS_READY=0
        break
    fi
done

if [ "${ALL_FOLDS_READY}" -eq 1 ]; then
    echo "[kfold] ${FOLD_ROOT}/ already populated — skipping slimdiff-kfold."
else
    echo "[kfold] Building ${KFOLD_N_FOLDS}-fold CSVs (seed=${KFOLD_SEED}) ..."
    slimdiff-kfold \
        --cache-dir "${CACHE_DIR}" \
        --n-folds "${KFOLD_N_FOLDS}" \
        --seed "${KFOLD_SEED}"
    echo "[kfold] done."
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Prereq build finished in $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s."
echo "Next: bash slurm/camera_ready/launch_camera_ready.sh"
