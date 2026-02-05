#!/usr/bin/env bash
# Submit all pMF-style loss ablation experiments to SLURM
#
# IWINAC 2026 Experiment: pMF Loss Ablation Study
# Total: 8 configurations x 5 replicas = 40 training runs
#
# Ablation Design:
#   B1: x0-pred + x0-loss (Lp p=1.5)
#   B2: x0-pred + v-loss (L2)
#   B3a/b/c: x0-pred + x0-loss + LPIPS (lambda=0.1, 0.5, 1.0)
#   B4a/b/c: x0-pred + v-loss + LPIPS (lambda=0.1, 0.5, 1.0)
#
# Usage:
#   ./submit_all.sh           # Submit all jobs
#   ./submit_all.sh --dry-run # Show what would be submitted

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - Not actually submitting jobs"
    echo ""
fi

echo "=========================================================================="
echo "IWINAC 2026 pMF-Style Loss Ablation Study"
echo "=========================================================================="
echo ""
echo "Experiment configurations:"
echo "  B1:  x0-pred + x0-loss (no LPIPS)"
echo "  B2:  x0-pred + v-loss (no LPIPS)"
echo "  B3a: x0-pred + x0-loss + LPIPS (lambda=0.1)"
echo "  B3b: x0-pred + x0-loss + LPIPS (lambda=0.5)"
echo "  B3c: x0-pred + x0-loss + LPIPS (lambda=1.0)"
echo "  B4a: x0-pred + v-loss + LPIPS (lambda=0.1)"
echo "  B4b: x0-pred + v-loss + LPIPS (lambda=0.5)"
echo "  B4c: x0-pred + v-loss + LPIPS (lambda=1.0)"
echo ""
echo "Total: 8 configurations"
echo ""

# List of all experiment directories
EXPERIMENTS=(
    "B1_x0_pred_x0_loss"
    "B2_x0_pred_v_loss"
    "B3_x0_pred_x0_loss_lpips/lambda_0.1"
    "B3_x0_pred_x0_loss_lpips/lambda_0.5"
    "B3_x0_pred_x0_loss_lpips/lambda_1.0"
    "B4_x0_pred_v_loss_lpips/lambda_0.1"
    "B4_x0_pred_v_loss_lpips/lambda_0.5"
    "B4_x0_pred_v_loss_lpips/lambda_1.0"
)

JOB_IDS=()

for exp in "${EXPERIMENTS[@]}"; do
    SCRIPT="${SCRIPT_DIR}/${exp}/train_generate.sh"

    if [[ ! -f "${SCRIPT}" ]]; then
        echo "WARNING: Script not found: ${SCRIPT}"
        continue
    fi

    echo "Submitting: ${exp}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  [DRY RUN] sbatch ${SCRIPT}"
    else
        JOB_ID=$(sbatch "${SCRIPT}" | awk '{print $4}')
        JOB_IDS+=("${JOB_ID}")
        echo "  Submitted job ID: ${JOB_ID}"
    fi
done

echo ""
echo "=========================================================================="
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY RUN COMPLETE - No jobs submitted"
else
    echo "All jobs submitted!"
    echo ""
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "Cancel all with: scancel ${JOB_IDS[*]}"
fi
echo "=========================================================================="
