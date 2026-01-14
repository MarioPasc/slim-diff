#!/usr/bin/env bash
#SBATCH -J seg_experiments
#SBATCH --array=0-11             # 4 experiments x 3 models = 12 jobs (indices 0-11)
#SBATCH --time=2-00:00:00        # 2 days per experiment (conservative)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=dgx
#SBATCH --output=logs/seg_experiments/%x.%A_%a.out
#SBATCH --error=logs/seg_experiments/%x.%A_%a.err

# ========================================================================
# Segmentation Experiments SLURM Array Script
# ========================================================================
# Runs segmentation experiments as an array job, one model-experiment pair
# per job. Each job requests 1 GPU.
#
# This script modifies the master YAML config on-the-fly to set cluster paths:
#   - experiment.output_dir
#   - data.real.cache_dir
#   - data.synthetic.samples_dir
#
# Usage:
#   sbatch slurm/segmentation_experiments_array.sh
#
# To run a subset of jobs:
#   sbatch --array=0-5 slurm/segmentation_experiments_array.sh
#
# Array mapping (24 combinations):
#   Index = experiment_idx * NUM_MODELS + model_idx
#   experiment_idx = Index / NUM_MODELS
#   model_idx = Index % NUM_MODELS
# ========================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "=========================================================================="
echo "Segmentation Experiment Job Started"
echo "=========================================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"

# ============== CONFIGURATION ==============
# Paths - UPDATE THESE FOR YOUR CLUSTER
REPO_ROOT="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/js-ddpm-epilepsy"
OUTPUT_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/segmentation_experiments"
CACHE_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy/slice_cache"
SAMPLES_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy/replicas"

USE_NEGATIVE_CASES=false  # Whether to include negative cases (no lesion slices)

# Conda environment
CONDA_ENV_NAME="jsddpm"

# K-fold configuration (optional, leave empty to use experiment config)
FOLDS=""  # e.g., "5" for 5-fold, or "0,1,2" for specific folds

# Define experiments and models
EXPERIMENTS=(
    "real_only"
    "real_synthetic_balance"
    "real_traditional_augmentation"
    "real_synthetic_traditional_augmentation"
)

MODELS=(
    "segresnet"
    "attentionunet"
    "unetr"
)

NUM_EXPERIMENTS=${#EXPERIMENTS[@]}
NUM_MODELS=${#MODELS[@]}
# ===========================================

# Calculate which experiment and model to run based on array task ID
EXPERIMENT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_MODELS))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID % NUM_MODELS))

# Validate indices
if [ ${EXPERIMENT_IDX} -ge ${NUM_EXPERIMENTS} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} out of range"
    echo "       Max task ID should be $((NUM_EXPERIMENTS * NUM_MODELS - 1))"
    exit 1
fi

EXPERIMENT="${EXPERIMENTS[${EXPERIMENT_IDX}]}"
MODEL="${MODELS[${MODEL_IDX}]}"

echo "=========================================================================="
echo "Task Assignment"
echo "=========================================================================="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Experiment Index: ${EXPERIMENT_IDX} -> ${EXPERIMENT}"
echo "Model Index: ${MODEL_IDX} -> ${MODEL}"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPO_ROOT}/logs/seg_experiments"

# Modify config for cluster paths
# Create a job-specific config directory to avoid conflicts between array tasks
CONFIG_DIR="${REPO_ROOT}/src/segmentation/config"
MASTER_CONFIG="${CONFIG_DIR}/master.yaml"
JOB_CONFIG_DIR="${OUTPUT_DIR}/config_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"

echo "=========================================================================="
echo "Config Modification"
echo "=========================================================================="
echo "Original config dir: ${CONFIG_DIR}"
echo "Job-specific config dir: ${JOB_CONFIG_DIR}"

# Remove any existing job config directory (from failed previous runs)
rm -rf "${JOB_CONFIG_DIR}"

# Copy entire config directory structure for this job
mkdir -p "${JOB_CONFIG_DIR}"
cp -r "${CONFIG_DIR}/experiments" "${JOB_CONFIG_DIR}/" 2>/dev/null || true
cp -r "${CONFIG_DIR}/models" "${JOB_CONFIG_DIR}/" 2>/dev/null || true
cp -r "${CONFIG_DIR}/replicas" "${JOB_CONFIG_DIR}/" 2>/dev/null || true
cp "${MASTER_CONFIG}" "${JOB_CONFIG_DIR}/master.yaml"

# Update paths in ALL yaml files in the job config directory
# Note: Using sed with careful patterns to match YAML structure
find "${JOB_CONFIG_DIR}" -name "*.yaml" -exec sed -i "s|output_dir:.*|output_dir: \"${OUTPUT_DIR}\"|g" {} \;
find "${JOB_CONFIG_DIR}" -name "*.yaml" -exec sed -i "s|cache_dir:.*slice_cache.*|cache_dir: \"${CACHE_DIR}\"|g" {} \;
find "${JOB_CONFIG_DIR}" -name "*.yaml" -exec sed -i "s|samples_dir:.*replicas.*|samples_dir: \"${SAMPLES_DIR}\"|g" {} \;

# Ensure use_negative_cases is set to any (lesion-only mode for fair comparison)
# This filters at SLICE level - only slices with actual lesion masks are used
find "${JOB_CONFIG_DIR}" -name "*.yaml" -exec sed -i "s|use_negative_cases:.*|use_negative_cases: ${USE_NEGATIVE_CASES}|g" {} \;

echo "Modified paths in all configs:"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  cache_dir: ${CACHE_DIR}"
echo "  samples_dir: ${SAMPLES_DIR}"
echo "  use_negative_cases: ${USE_NEGATIVE_CASES} (lesion-only mode)"

echo "=========================================================================="
echo "GPU Allocation"
echo "=========================================================================="
# SLURM automatically sets this variable.
# If SLURM uses cgroups (common), you might only see one GPU (device 0) available inside the job anyway.
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not Set (will default to all)}"
nvidia-smi

# Load conda module and activate environment
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>&1 | grep -qi "^${m}[[:space:]]"; then
        module load "$m" && module_loaded=1 && break
    fi
done

if [ "$module_loaded" -eq 0 ]; then
    echo "[env] no conda module loaded; assuming conda already in PATH."
fi

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

# Verify environment
echo "=========================================================================="
echo "Environment"
echo "=========================================================================="
echo "Python: $(which python)"
echo "Python version: $(python -c 'import sys; print(sys.version.split()[0])')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

# Navigate to repository root
cd "${REPO_ROOT}"
echo "Working directory: $(pwd)"

echo "=========================================================================="
echo "Experiment Configuration"
echo "=========================================================================="
echo "Experiment: ${EXPERIMENT}"
echo "Model: ${MODEL}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Config dir: ${JOB_CONFIG_DIR}"
if [ -n "${FOLDS}" ]; then
    echo "Folds: ${FOLDS}"
else
    echo "Folds: (using experiment config default)"
fi

echo "=========================================================================="
echo "Starting Experiment"
echo "=========================================================================="

# Build command - use job-specific config directory
CMD="python -m src.segmentation.cli.experiment_orchestrator \
    --experiments ${EXPERIMENT} \
    --models ${MODEL} \
    --output-dir ${OUTPUT_DIR} \
    --config-dir ${JOB_CONFIG_DIR} \
    --device 0 \
    --sequential"

# Add folds argument if specified
if [ -n "${FOLDS}" ]; then
    CMD="${CMD} --folds ${FOLDS}"
fi

echo "Command: ${CMD}"
eval "${CMD}"

# Clean up job-specific config directory
rm -rf "${JOB_CONFIG_DIR}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=========================================================================="
echo "Experiment Complete!"
echo "=========================================================================="
echo "Experiment: ${EXPERIMENT}"
echo "Model: ${MODEL}"
echo "End time: $(date)"
echo "Elapsed: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Output: ${OUTPUT_DIR}/${EXPERIMENT}_${MODEL}"
