#!/usr/bin/env bash
# =============================================================================
# run_posthoc.sh — TASK-05 post-hoc analyses (tau sweep + stats + LaTeX tables)
# =============================================================================
# Runs the unified post-hoc CLI with `--only all`, reading every path/knob
# from picasso_paths.yaml. Writes to ${POSTHOC_OUTPUT_DIR}:
#
#   tau_sensitivity.csv
#   tau_sensitivity_summary.csv
#   cross_fold_comparison.json
#   tables/table_ablation.tex
#   tables/table_main_updated.tex
#   tables/table_tau_sensitivity_{shared,decoupled}.tex
#
# Requires:
#   * ${EVAL_OUTPUT_DIR}/fold_metrics.csv  (produced by run_fold_eval.sh)
#   * ${CACHE_DIR}/folds/                   (produced by build_cache_and_kfold.sh)
#   * ${RESULTS_ROOT}/slimdiff_cr_{arch}_fold_{k}/replicas/replica_*.npz
#
# CPU-only: the tau sweep is feature-extraction bound (scikit-image) and MMD
# uses a polynomial kernel (NumPy). No GPU is used.
# =============================================================================
#SBATCH -J slimdiff_cr_posthoc
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=cpu
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

START_TIME=$(date +%s)
echo "Job started at: $(date)"

# =============================================================================
# BOOTSTRAP — hardcoded Picasso defaults
# =============================================================================
# Only these two paths are hardcoded; the rest come from picasso_paths.yaml
# (which cannot be read until the conda env is active, because the loader
# needs PyYAML). Override via env vars before `sbatch` if needed.
REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/slim-diff}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-slimdiff}"

# =============================================================================
# STEP 0 — CONDA ENV ACTIVATION (must precede YAML loading)
# =============================================================================
# shellcheck disable=SC1091
source "${REPO_SRC}/slurm/camera_ready/_activate_conda.sh"

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

FOLD_METRICS="${EVAL_OUTPUT_DIR}/fold_metrics.csv"

echo "Loaded paths:"
echo "  REPO_SRC                   = ${REPO_SRC}"
echo "  CACHE_DIR                  = ${CACHE_DIR}"
echo "  RESULTS_ROOT               = ${RESULTS_ROOT}"
echo "  EVAL_OUTPUT_DIR            = ${EVAL_OUTPUT_DIR}"
echo "  POSTHOC_OUTPUT_DIR         = ${POSTHOC_OUTPUT_DIR}"
echo "  POSTHOC_TAU_VALUES         = ${POSTHOC_TAU_VALUES:-<default>}"
echo "  POSTHOC_MIN_LESION_SIZE_PX = ${POSTHOC_MIN_LESION_SIZE_PX}"
echo "  POSTHOC_SUBSET_SIZE        = ${POSTHOC_SUBSET_SIZE}"
echo "  POSTHOC_NUM_SUBSETS        = ${POSTHOC_NUM_SUBSETS}"
echo "  POSTHOC_EARLY_STOPPING_CSV = ${POSTHOC_EARLY_STOPPING_CSV:-<none>}"
echo "  CONDA_ENV_NAME             = ${CONDA_ENV_NAME}"

# =============================================================================
# PRE-FLIGHT
# =============================================================================
cd "${REPO_SRC}"

if [ ! -f "${FOLD_METRICS}" ]; then
    echo "ERROR: ${FOLD_METRICS} not found — run run_fold_eval.sh first." >&2
    exit 1
fi

mkdir -p "${POSTHOC_OUTPUT_DIR}"

# =============================================================================
# BUILD CLI ARGS
# =============================================================================
POSTHOC_ARGS=(
    --only all
    --fold-metrics "${FOLD_METRICS}"
    --results-root "${RESULTS_ROOT}"
    --cache-dir    "${CACHE_DIR}"
    --output-dir   "${POSTHOC_OUTPUT_DIR}"
    --min-lesion-size-px "${POSTHOC_MIN_LESION_SIZE_PX}"
    --subset-size        "${POSTHOC_SUBSET_SIZE}"
    --num-subsets        "${POSTHOC_NUM_SUBSETS}"
)

# tau_values: optional space-separated list.
if [ -n "${POSTHOC_TAU_VALUES}" ]; then
    # shellcheck disable=SC2206
    TAU_ARR=( ${POSTHOC_TAU_VALUES} )
    POSTHOC_ARGS+=( --tau-values "${TAU_ARR[@]}" )
fi

# early_stopping_csv: optional path; resolve relative to repo_src.
if [ -n "${POSTHOC_EARLY_STOPPING_CSV}" ]; then
    case "${POSTHOC_EARLY_STOPPING_CSV}" in
        /*) ES_PATH="${POSTHOC_EARLY_STOPPING_CSV}" ;;
        *)  ES_PATH="${REPO_SRC}/${POSTHOC_EARLY_STOPPING_CSV}" ;;
    esac
    if [ -f "${ES_PATH}" ]; then
        POSTHOC_ARGS+=( --early-stopping-csv "${ES_PATH}" )
    else
        echo "WARNING: early-stopping CSV not found at ${ES_PATH} — omitting." >&2
    fi
fi

# =============================================================================
# RUN
# =============================================================================
echo ""
echo "Running: python -m src.diffusion.scripts.similarity_metrics.posthoc.cli ${POSTHOC_ARGS[*]}"
python -m src.diffusion.scripts.similarity_metrics.posthoc.cli "${POSTHOC_ARGS[@]}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Post-hoc finished in $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s."
echo "Outputs under: ${POSTHOC_OUTPUT_DIR}"
