#!/usr/bin/env bash
# =============================================================================
# run_fold_eval.sh — Download pretrained weights + submit fold-eval SLURM job
# =============================================================================
# Run on the login node (which has internet access):
#
#   bash slurm/camera_ready/run_fold_eval.sh
#
# This script:
#   1. Downloads VGG16 (LPIPS) and InceptionV3 (KID) weights into TORCH_HOME
#      (only if not already cached).
#   2. Submits _fold_eval_job.sh via sbatch.
#
# The compute job reads TORCH_HOME from picasso_paths.yaml and uses the
# pre-downloaded weights offline.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="${REPO_SRC:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-slimdiff}"

# =============================================================================
# STEP 0 — CONDA ENV ACTIVATION (needs PyYAML + torch.hub for downloading)
# =============================================================================
# shellcheck disable=SC1091
source "${REPO_SRC}/slurm/camera_ready/_activate_conda.sh"

# =============================================================================
# STEP 1 — LOAD TORCH_HOME FROM YAML
# =============================================================================
PATHS_YAML="${PATHS_YAML:-${REPO_SRC}/slurm/camera_ready/picasso_paths.yaml}"
LOADER="${REPO_SRC}/slurm/camera_ready/_load_paths.py"

if [ ! -f "${PATHS_YAML}" ]; then
    echo "ERROR: ${PATHS_YAML} not found." >&2
    exit 1
fi

eval "$(python "${LOADER}" "${PATHS_YAML}")"
export TORCH_HOME

WEIGHTS_DIR="${TORCH_HOME}/hub/checkpoints"
mkdir -p "${WEIGHTS_DIR}"

echo "TORCH_HOME = ${TORCH_HOME}"
echo "Weights dir = ${WEIGHTS_DIR}"

# =============================================================================
# STEP 2 — DOWNLOAD PRETRAINED WEIGHTS (login node has internet)
# =============================================================================
echo ""
echo "Checking pretrained weights ..."
python3 << 'PYEOF'
import os
import sys
from pathlib import Path

cache = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"

WEIGHTS = {
    "vgg16-397923af.pth": (
        "https://download.pytorch.org/models/vgg16-397923af.pth",
        "VGG16 backbone (LPIPS)",
    ),
    "weights-inception-2015-12-05-6726825d.pth": (
        "https://github.com/toshas/torch-fidelity/releases/download/"
        "v0.2.0/weights-inception-2015-12-05-6726825d.pth",
        "InceptionV3 backbone (KID via torch-fidelity)",
    ),
}

missing = []
for fname, (url, desc) in WEIGHTS.items():
    path = cache / fname
    if path.exists():
        mb = path.stat().st_size / 1e6
        print(f"  [cached] {fname} ({mb:.1f} MB)")
    else:
        missing.append((fname, url, desc))

if not missing:
    print("All weights already cached — nothing to download.")
    sys.exit(0)

import torch.hub

for fname, url, desc in missing:
    dst = str(cache / fname)
    print(f"  Downloading {desc}: {fname} ...")
    torch.hub.download_url_to_file(url, dst)
    mb = Path(dst).stat().st_size / 1e6
    print(f"  Done ({mb:.1f} MB)")

print("All weights ready.")
PYEOF

echo ""

# =============================================================================
# STEP 3 — SUBMIT SLURM JOB
# =============================================================================
JOB_SCRIPT="${SCRIPT_DIR}/_fold_eval_job.sh"
if [ ! -f "${JOB_SCRIPT}" ]; then
    echo "ERROR: ${JOB_SCRIPT} not found." >&2
    exit 1
fi

echo "Submitting fold-eval SLURM job ..."
JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
echo "Submitted: job ${JOB_ID}"
echo "Monitor:   tail -f slimdiff_cr_fold_eval.${JOB_ID}.out"
