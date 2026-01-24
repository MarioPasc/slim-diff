#!/usr/bin/env bash
#SBATCH -J ablation_step1_ffl
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

START_TIME=$(date +%s)
echo "Job started at: $(date)"

# ========================================================================
# EXPERIMENT CONFIGURATION
# ========================================================================
EXPERIMENT_NAME="ablation_step1_x0_ffl"
CONDA_ENV_NAME="jsddpm"

REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/js-ddpm-epilepsy"
DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy"
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/${EXPERIMENT_NAME}"
CONFIG_FILE="${REPO_SRC}/slurm/ablation_study_hf/proposal.yaml"

# ========================================================================
# GENERATION CONFIGURATION
# ========================================================================
NUM_REPLICAS=5
N_SAMPLES_PER_MODE=50
GEN_BATCH_SIZE=32
SEED_BASE=42
# CRITICAL: Use float32 to avoid quantization artifacts that trivially
# separate real from synthetic in downstream classification (AUC=1.0 artifact).
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

# Activate your existing env named ${CONDA_ENV_NAME}
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

# Verify
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch, os; print('CUDA', torch.cuda.is_available())"
echo "[torch] $(python -c 'import torch; print(torch.__version__)')"
echo "[cuda devices] $(python -c 'import torch; print(torch.cuda.device_count())')"

# ========================================================================
# DDP VERIFICATION
# ========================================================================
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "[ddp] Available GPUs: ${GPU_COUNT}"
echo "[ddp] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Requested 2 GPUs but only ${GPU_COUNT} available"
    echo "Training will proceed but may not use DDP as expected"
fi

# ========================================================================
# PHASE 1: TRAINING
# ========================================================================
echo ""
echo "=========================================================================="
echo "PHASE 1: Training"
echo "=========================================================================="

# 1. Create the results directory
mkdir -p "${RESULTS_DST}"

# 2. Copy and modify the configuration file
CONFIG_BASENAME=$(basename "${CONFIG_FILE}")
MODIFIED_CONFIG="${RESULTS_DST}/${CONFIG_BASENAME}"

echo "Copying config file to: ${MODIFIED_CONFIG}"
cp "${CONFIG_FILE}" "${MODIFIED_CONFIG}"

echo "Modifying configuration file paths for cluster..."
sed -i "s|  cache_dir: .*|  cache_dir: \"${DATA_SRC}/slice_cache\"|" "${MODIFIED_CONFIG}"
sed -i "s|  output_dir: .*|  output_dir: \"${RESULTS_DST}\"|" "${MODIFIED_CONFIG}"

# 3. Handle data caching
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

# 4. Run training
echo "Starting DDP training with config: ${MODIFIED_CONFIG}"
echo "Using ${GPU_COUNT} GPUs with DDP strategy"

jsddpm-train --config "${MODIFIED_CONFIG}"

TRAIN_END_TIME=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END_TIME - START_TIME))
echo "=========================================================================="
echo "Training completed at: $(date)"
echo "Training time: $(($TRAIN_ELAPSED / 3600))h $((($TRAIN_ELAPSED / 60) % 60))m $(($TRAIN_ELAPSED % 60))s"
echo "=========================================================================="

# ========================================================================
# PHASE 2: GENERATION
# ========================================================================
echo ""
echo "=========================================================================="
echo "PHASE 2: Replica Generation"
echo "=========================================================================="

# Find the checkpoint
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

# Create replicas directory
REPLICAS_OUT_DIR="${RESULTS_DST}/replicas"
mkdir -p "${REPLICAS_OUT_DIR}"

echo ""
echo "Generation Configuration:"
echo "  - Number of replicas: ${NUM_REPLICAS}"
echo "  - Samples per (zbin, domain): ${N_SAMPLES_PER_MODE}"
echo "  - Batch size: ${GEN_BATCH_SIZE}"
echo "  - Seed base: ${SEED_BASE}"
echo "  - Output dtype: ${GEN_DTYPE}"
echo ""

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

# Build replica_ids list for YAML: [0, 1, 2, 3, 4]
REPLICA_IDS_YAML="[$(seq -s ', ' 0 $((NUM_REPLICAS - 1)))]"

# Generate classification config on the fly
CLASS_CONFIG="${RESULTS_DST}/classification_task.yaml"
mkdir -p "${CLASSIFICATION_DIR}"

echo "Generating classification config: ${CLASS_CONFIG}"
cat > "${CLASS_CONFIG}" << CLASSEOF
experiment:
  name: "ablation_classification"
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

# 3a. Extract patches
echo "Extracting patches..."
EXTRACT_START=$(date +%s)

python -m src.classification extract \
    --config "${CLASS_CONFIG}" \
    --experiment "${EXPERIMENT_NAME}"

EXTRACT_END=$(date +%s)
echo "Patch extraction done in $(( (EXTRACT_END - EXTRACT_START) / 60 ))m $(( (EXTRACT_END - EXTRACT_START) % 60 ))s"

# 3b. Run classification (k-fold)
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

# Generate diagnostics config on the fly
DIAG_CONFIG="${RESULTS_DST}/diagnostics.yaml"
mkdir -p "${DIAGNOSTICS_DIR}"

echo "Generating diagnostics config: ${DIAG_CONFIG}"
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

output:
  base_dir: "${DIAGNOSTICS_DIR}"
  plot_format: ["png"]
  plot_dpi: 150
  save_data: true

experiment:
  seed: 42
  device: "cuda"
DIAGEOF

# 4a. Run full diagnostics
echo "Running diagnostic analyses..."
DIAG_START=$(date +%s)

python -m src.classification.diagnostics run-all \
    --config "${DIAG_CONFIG}" \
    --experiment "${EXPERIMENT_NAME}" \
    --gpu 0

DIAG_END=$(date +%s)
DIAG_ELAPSED=$((DIAG_END - DIAG_START))
echo "Diagnostics done in $(($DIAG_ELAPSED / 60))m $(($DIAG_ELAPSED % 60))s"

# 4b. Run aggregate (cross-experiment report)
echo "Running aggregate report..."
python -m src.classification.diagnostics aggregate --config "${DIAG_CONFIG}"

# ========================================================================
# FINAL SUMMARY
# ========================================================================
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================================================="
echo "=========================================================================="
echo "  ABLATION STEP 1: x0_lp_1.5 + FFL  —  COMPLETE"
echo "=========================================================================="
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
echo "  Patches:        ${CLASSIFICATION_DIR}/patches/${EXPERIMENT_NAME}/"
echo "  Classification: ${CLASSIFICATION_DIR}/results/${EXPERIMENT_NAME}/"
echo "  Diagnostics:    ${DIAGNOSTICS_DIR}/${EXPERIMENT_NAME}/"
echo ""

# Print diagnostic summary if CSV exists
DIAG_CSV="${DIAGNOSTICS_DIR}/cross_experiment/diagnostic_report.csv"
if [ -f "${DIAG_CSV}" ]; then
    echo "=========================================================================="
    echo "  DIAGNOSTIC RESULTS"
    echo "=========================================================================="
    echo ""
    python -c "
import csv
with open('${DIAG_CSV}') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['experiment'] == '${EXPERIMENT_NAME}':
            print(f\"  Experiment: {row['experiment']}\")
            print(f\"  Prediction type: {row['prediction_type']}\")
            print(f\"  Lp norm: {row['lp_norm']}\")
            print()
            print(f\"  Overall Artifact Severity: {float(row['overall_artifact_severity']):.4f}\")
            print(f\"    (baseline x0_lp_1.5: 0.0253)\")
            print()
            print('  Key Metrics:')
            print(f\"    High-freq excess:        {float(row['high_freq_excess']):.4f}  (norm: {float(row['high_freq_excess_norm']):.4f})\")
            print(f\"    Wavelet energy dev:      {float(row['wavelet_energy_deviation']):.4f}  (norm: {float(row['wavelet_energy_deviation_norm']):.4f})\")
            print(f\"    Spectral JS divergence:  {float(row['spectral_js_div']):.6f}  (norm: {float(row['spectral_js_div_norm']):.4f})\")
            print(f\"    Spectral slope diff:     {float(row['spectral_slope_diff']):.4f}  (norm: {float(row['spectral_slope_diff_norm']):.4f})\")
            print(f\"    Texture mean KS:         {float(row['texture_mean_ks']):.4f}  (norm: {float(row['texture_mean_ks_norm']):.4f})\")
            print(f\"    Boundary sharpness:      {float(row['boundary_sharpness_deficit']):.4f}  (norm: {float(row['boundary_sharpness_deficit_norm']):.4f})\")
            print(f\"    Background noise:        {float(row['background_noise']):.6f}  (norm: {float(row['background_noise_norm']):.4f})\")
            print()
            break
    else:
        print(f'  WARNING: Experiment ${EXPERIMENT_NAME} not found in diagnostic report.')
        print(f'  Available experiments:')
        f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            print(f\"    - {row['experiment']}\")
"
    echo ""
else
    echo "  NOTE: Diagnostic CSV not found at ${DIAG_CSV}"
    echo "  Individual results available in: ${DIAGNOSTICS_DIR}/${EXPERIMENT_NAME}/"
fi

# Print classification AUC if results exist
CLASS_RESULTS="${CLASSIFICATION_DIR}/results/${EXPERIMENT_NAME}"
if [ -d "${CLASS_RESULTS}" ]; then
    echo "=========================================================================="
    echo "  CLASSIFICATION RESULTS (Real vs. Synthetic)"
    echo "=========================================================================="
    echo ""
    python -c "
import json, glob, os
results_dir = '${CLASS_RESULTS}'
json_files = sorted(glob.glob(os.path.join(results_dir, '**/results.json'), recursive=True))
if not json_files:
    json_files = sorted(glob.glob(os.path.join(results_dir, '**/*.json'), recursive=True))

aucs = []
for jf in json_files:
    try:
        with open(jf) as f:
            data = json.load(f)
        if 'auc' in data:
            aucs.append(data['auc'])
        elif 'test_auc' in data:
            aucs.append(data['test_auc'])
        elif 'mean_auc' in data:
            aucs.append(data['mean_auc'])
    except:
        pass

if aucs:
    import statistics
    mean_auc = statistics.mean(aucs)
    print(f'  Mean AUC: {mean_auc:.4f} (across {len(aucs)} folds)')
    print(f'  Individual fold AUCs: {[round(a, 4) for a in aucs]}')
    print()
    if mean_auc > 0.9:
        print('  --> Classifier easily distinguishes real from synthetic.')
        print('      Significant model-level artifacts likely remain.')
    elif mean_auc > 0.7:
        print('  --> Moderate distinguishability. Some artifacts present.')
    elif mean_auc > 0.6:
        print('  --> Low distinguishability. Minor artifacts.')
    else:
        print('  --> Near-chance performance. Synthetic quality is high.')
else:
    print('  Could not parse AUC from classification results.')
    print(f'  Check results in: {results_dir}')
"
    echo ""
fi

echo "=========================================================================="
echo "Experiment completed successfully at: $(date)"
echo "=========================================================================="
