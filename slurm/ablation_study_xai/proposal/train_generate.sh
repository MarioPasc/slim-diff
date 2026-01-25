#!/usr/bin/env bash
#SBATCH -J xai_proposal_full
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:2
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

START_TIME=$(date +%s)
echo "Job started at: $(date)"

# ========================================================================
# EXPERIMENT CONFIGURATION
# ========================================================================
EXPERIMENT_NAME="xai_proposal_full"
CONDA_ENV_NAME="jsddpm"

REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/js-ddpm-epilepsy"
DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy"
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/${EXPERIMENT_NAME}"
CONFIG_FILE="${REPO_SRC}/slurm/ablation_study_xai/proposal/proposal.yaml"

# ========================================================================
# GENERATION CONFIGURATION
# ========================================================================
NUM_REPLICAS=5
N_SAMPLES_PER_MODE=50
GEN_BATCH_SIZE=32
SEED_BASE=42
GEN_DTYPE="float32"

# ========================================================================
# CLASSIFICATION & DIAGNOSTICS CONFIGURATION
# ========================================================================
CLASSIFICATION_DIR="${RESULTS_DST}/classification"
DIAGNOSTICS_DIR="${RESULTS_DST}/diagnostics"
RESULTS_BASE_DIR=$(dirname "${RESULTS_DST}")

# ---------- Load conda module and activate prebuilt env ----------
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
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch, os; print('CUDA', torch.cuda.is_available())"
echo "[torch] $(python -c 'import torch; print(torch.__version__)')"
echo "[cuda devices] $(python -c 'import torch; print(torch.cuda.device_count())')"

GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "[gpu] Available GPUs: ${GPU_COUNT}"

# ========================================================================
# PHASE 1: TRAINING
# ========================================================================
echo ""
echo "=========================================================================="
echo "PHASE 1: Training — ${EXPERIMENT_NAME}"
echo "=========================================================================="

mkdir -p "${RESULTS_DST}"

CONFIG_BASENAME=$(basename "${CONFIG_FILE}")
MODIFIED_CONFIG="${RESULTS_DST}/${CONFIG_BASENAME}"

echo "Copying config file to: ${MODIFIED_CONFIG}"
cp "${CONFIG_FILE}" "${MODIFIED_CONFIG}"

echo "Modifying configuration file paths for cluster..."
sed -i "s|  cache_dir: .*|  cache_dir: \"${DATA_SRC}/slice_cache\"|" "${MODIFIED_CONFIG}"
sed -i "s|  output_dir: .*|  output_dir: \"${RESULTS_DST}\"|" "${MODIFIED_CONFIG}"

cd "${REPO_SRC}"
echo "Working directory: $(pwd)"

CACHE_DIR="${DATA_SRC}/slice_cache"
echo "Cache directory: ${CACHE_DIR}"

if [ ! -d "${CACHE_DIR}" ]; then
    echo "Cache directory not found. Running caching step..."
    CACHE_CONFIG_SRC="${REPO_SRC}/src/diffusion/config/cache/epilepsy.yaml"
    CACHE_CONFIG="${RESULTS_DST}/cache_epilepsy.yaml"
    cp "${CACHE_CONFIG_SRC}" "${CACHE_CONFIG}"
    sed -i "s|cache_dir: .*|cache_dir: \"${CACHE_DIR}\"|" "${CACHE_CONFIG}"
    sed -i "s|root_dir: .*|root_dir: \"${DATA_SRC}\"|" "${CACHE_CONFIG}"
    jsddpm-cache --config "${CACHE_CONFIG}"
    echo "Caching completed."
else
    echo "Cache directory exists. Skipping caching step."
fi

echo "Starting training with config: ${MODIFIED_CONFIG}"
jsddpm-train --config "${MODIFIED_CONFIG}"

TRAIN_END_TIME=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END_TIME - START_TIME))
echo "Training completed in $(($TRAIN_ELAPSED / 3600))h $((($TRAIN_ELAPSED / 60) % 60))m $(($TRAIN_ELAPSED % 60))s"

# ========================================================================
# PHASE 2: GENERATION
# ========================================================================
echo ""
echo "=========================================================================="
echo "PHASE 2: Replica Generation"
echo "=========================================================================="

CKPT_DIR="${RESULTS_DST}/checkpoints"
echo "Looking for checkpoint in: ${CKPT_DIR}"

if [ ! -d "${CKPT_DIR}" ]; then
    echo "ERROR: Checkpoint directory not found: ${CKPT_DIR}"
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

GEN_START_TIME=$(date +%s)

for REPLICA_ID in $(seq 0 $((NUM_REPLICAS - 1))); do
    echo "Generating replica ${REPLICA_ID} / $((NUM_REPLICAS - 1)) ..."

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

    echo "  Replica ${REPLICA_ID} done."
done

GEN_END_TIME=$(date +%s)
GEN_ELAPSED=$((GEN_END_TIME - GEN_START_TIME))
echo "Generation time: $(($GEN_ELAPSED / 3600))h $((($GEN_ELAPSED / 60) % 60))m $(($GEN_ELAPSED % 60))s"

# ========================================================================
# PHASE 3: PATCH EXTRACTION & CLASSIFICATION
# ========================================================================
echo ""
echo "=========================================================================="
echo "PHASE 3: Patch Extraction & Classification"
echo "=========================================================================="

REPLICA_IDS_YAML="[$(seq -s ', ' 0 $((NUM_REPLICAS - 1)))]"

CLASS_CONFIG="${RESULTS_DST}/classification_task.yaml"
mkdir -p "${CLASSIFICATION_DIR}"

cat > "${CLASS_CONFIG}" << CLASSEOF
experiment:
  name: "${EXPERIMENT_NAME}_classification"
  seed: 42

data:
  real:
    cache_dir: "${DATA_SRC}/slice_cache"
    csv_files: ["train.csv", "val.csv"]
    slices_subdir: "slices"

  synthetic:
    results_base_dir: "${RESULTS_BASE_DIR}"
    experiments:
      - name: "${EXPERIMENT_NAME}"
        replicas_subdir: "replicas"
        replica_ids: ${REPLICA_IDS_YAML}

  patch_extraction:
    method: "dynamic"
    padding: 8
    min_patch_size: 32
    max_patch_size: 96
    fixed_size: 64

  kfold:
    n_folds: 3
    stratify_by: "z_bin"
    seed: 42

  input_modes: ["joint"]

model:
  config_path: "models/simple_cnn.yaml"

training:
  batch_size: 8
  max_epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  optimizer: "adam"

  scheduler:
    type: "reduce_on_plateau"
    factor: 0.5
    patience: 5
    min_lr: 1.0e-6

  early_stopping:
    monitor: "val/auc"
    mode: "max"
    patience: 3
    min_delta: 0.001

  precision: "32-true"
  num_workers: 4
  pin_memory: true

evaluation:
  bootstrap:
    n_iterations: 2000
    confidence_level: 0.95
    seed: 42

  permutation_test:
    n_permutations: 10000
    alpha: 0.05
    seed: 42

  per_zbin:
    enabled: true
    min_samples: 10

  control:
    enabled: false
    n_repeats: 5

output:
  base_dir: "${CLASSIFICATION_DIR}"
  patches_subdir: "patches"
  full_images_subdir: "full_images"
  results_subdir: "results"
  checkpoints_subdir: "checkpoints"
  figures_subdir: "figures"
  tables_subdir: "tables"
CLASSEOF

echo "Extracting patches..."
EXTRACT_START=$(date +%s)

python -m src.classification extract \
    --config "${CLASS_CONFIG}" \
    --experiment "${EXPERIMENT_NAME}"

EXTRACT_END=$(date +%s)
echo "Patch extraction done in $(( (EXTRACT_END - EXTRACT_START) / 60 ))m $(( (EXTRACT_END - EXTRACT_START) % 60 ))s"

echo "Running classification (real vs synthetic)..."
CLASS_START=$(date +%s)

python -m src.classification run \
    --config "${CLASS_CONFIG}" \
    --experiment "${EXPERIMENT_NAME}" \
    --input-mode joint

CLASS_END=$(date +%s)
CLASS_ELAPSED=$((CLASS_END - CLASS_START))
echo "Classification done in $(($CLASS_ELAPSED / 60))m $(($CLASS_ELAPSED % 60))s"

# ========================================================================
# PHASE 4: DIAGNOSTICS
# ========================================================================
echo ""
echo "=========================================================================="
echo "PHASE 4: Diagnostics"
echo "=========================================================================="

DIAG_CONFIG="${RESULTS_DST}/diagnostics.yaml"
mkdir -p "${DIAGNOSTICS_DIR}"

cat > "${DIAG_CONFIG}" << DIAGEOF
data:
  patches_base_dir: "${CLASSIFICATION_DIR}/patches"
  checkpoints_base_dir: "${CLASSIFICATION_DIR}/checkpoints"
  replicas_base_dir: "${RESULTS_BASE_DIR}"
  real_cache_dir: "${DATA_SRC}/slice_cache"
  full_image_replica_ids: ${REPLICA_IDS_YAML}

dithering:
  seed: 42
  clip_range: [-1.0, 1.0]
  reclassification:
    n_folds: 5
    max_epochs: 50
    batch_size: 16
    learning_rate: 1.0e-4
    weight_decay: 1.0e-5
    early_stopping_patience: 10
    input_modes: ["joint"]
  metrics:
    bootstrap_n: 2000
    confidence_level: 0.95

gradcam:
  target_class: 1
  input_modes: ["joint"]
  batch_size: 16
  aggregation:
    per_zbin: true
    per_class: true
    radial_profile_bins: 20
    n_sample_visualizations: 16

feature_probes:
  spectral:
    channels: [0, 1]
    n_frequency_bands: 5
  texture:
    glcm_distances: [1, 2, 4]
    glcm_angles: [0.0, 0.7854, 1.5708, 2.3562]
    glcm_levels: 64
    lbp_radii: [1, 2, 3]
    lbp_points: [8, 16, 24]
    gradient_n_bins: 100
  frequency_bands:
    n_bands: 5
    channels: [0, 1]

statistical:
  alpha: 0.05
  correction: "fdr_bh"
  channels: [0, 1]
  per_zbin: true
  distributions:
    n_bins: 200
    tissue_segmentation: true
    lesion_threshold: 0.3
    brain_threshold: -0.8
    background_threshold: -0.95
  boundary:
    n_radii: 15
    max_distance: 10
  wavelet:
    wavelet: "db4"
    levels: 4

full_image:
  background:
    threshold: -0.95
    noise_threshold: 0.01
  spatial_correlation:
    max_lag: 20
  global_frequency:
    channels: [0]

xai:
  channel_decomposition:
    ablation_baseline: -1.0
  spectral_attribution:
    source: "gradcam"
    n_bands: 5
  feature_space:
    pca_components: 10
    tsne_perplexity: 30
    tsne_seed: 42
    top_features: 5
  integrated_gradients:
    n_steps: 50
    baseline: "zeros"
    max_samples: 100

output:
  base_dir: "${DIAGNOSTICS_DIR}"
  plot_format: ["png"]
  plot_dpi: 150
  save_data: true

experiment:
  seed: 42
  device: "cuda"
DIAGEOF

echo "Running diagnostic analyses..."
DIAG_START=$(date +%s)

python -m src.classification.diagnostics run-all \
    --config "${DIAG_CONFIG}" \
    --experiment "${EXPERIMENT_NAME}" \
    --gpu 0

DIAG_END=$(date +%s)
DIAG_ELAPSED=$((DIAG_END - DIAG_START))
echo "Diagnostics done in $(($DIAG_ELAPSED / 60))m $(($DIAG_ELAPSED % 60))s"

echo "Running aggregate report..."
python -m src.classification.diagnostics aggregate --config "${DIAG_CONFIG}"

# ========================================================================
# FINAL SUMMARY
# ========================================================================
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================================================="
echo "  XAI ABLATION: ${EXPERIMENT_NAME} — COMPLETE"
echo "=========================================================================="
echo ""
echo "Timing:"
echo "  Training:       $(($TRAIN_ELAPSED / 3600))h $((($TRAIN_ELAPSED / 60) % 60))m $(($TRAIN_ELAPSED % 60))s"
echo "  Generation:     $(($GEN_ELAPSED / 3600))h $((($GEN_ELAPSED / 60) % 60))m $(($GEN_ELAPSED % 60))s"
echo "  Classification: $(($CLASS_ELAPSED / 60))m $(($CLASS_ELAPSED % 60))s"
echo "  Diagnostics:    $(($DIAG_ELAPSED / 60))m $(($DIAG_ELAPSED % 60))s"
echo "  Total:          $(($TOTAL_ELAPSED / 3600))h $((($TOTAL_ELAPSED / 60) % 60))m $(($TOTAL_ELAPSED % 60))s"
echo ""
echo "Outputs:"
echo "  Config:         ${MODIFIED_CONFIG}"
echo "  Checkpoint:     ${CHECKPOINT}"
echo "  Replicas:       ${REPLICAS_OUT_DIR}/"
echo "  Classification: ${CLASSIFICATION_DIR}/results/${EXPERIMENT_NAME}/"
echo "  Diagnostics:    ${DIAGNOSTICS_DIR}/${EXPERIMENT_NAME}/"
echo ""

# Print diagnostic summary
DIAG_CSV="${DIAGNOSTICS_DIR}/cross_experiment/diagnostic_report.csv"
if [ -f "${DIAG_CSV}" ]; then
    echo "=========================================================================="
    echo "  DIAGNOSTIC RESULTS"
    echo "=========================================================================="
    python -c "
import csv
with open('${DIAG_CSV}') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['experiment'] == '${EXPERIMENT_NAME}':
            print(f\"  Experiment: {row['experiment']}\")
            print(f\"  Overall Artifact Severity: {float(row['overall_artifact_severity']):.4f}\")
            print(f\"    (baseline x0_lp_1.5: 0.0253)\")
            print()
            print('  Key Metrics:')
            for k in ['high_freq_excess','wavelet_energy_deviation','spectral_js_div',
                      'spectral_slope_diff','texture_mean_ks','boundary_sharpness_deficit',
                      'background_noise','xai_fisher_separability','xai_tsne_silhouette',
                      'xai_attribution_hf_fraction','xai_ig_concentration']:
                if k in row and row[k]:
                    print(f'    {k:30s}: {float(row[k]):.6f}')
            break
"
fi

echo ""
echo "=========================================================================="
echo "Experiment completed at: $(date)"
echo "=========================================================================="
