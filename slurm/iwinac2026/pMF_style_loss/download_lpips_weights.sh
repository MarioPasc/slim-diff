#!/usr/bin/env bash
# Download LPIPS VGG weights for offline HPC usage
#
# This script downloads the pre-trained VGG16 weights used by LPIPS.
# Run this on a machine with internet access, then copy weights/ to HPC.
#
# Usage:
#   ./download_lpips_weights.sh
#
# Output:
#   weights/lpips/vgg16_features.pth - VGG16 backbone weights

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WEIGHTS_DIR="${REPO_ROOT}/weights/lpips"

mkdir -p "${WEIGHTS_DIR}"

echo "Downloading LPIPS weights to: ${WEIGHTS_DIR}"
echo ""

# Download VGG16 weights from torchvision
python3 << 'EOF'
import torch
from torchvision.models import vgg16, VGG16_Weights
import os

weights_dir = os.environ.get('WEIGHTS_DIR', 'weights/lpips')
os.makedirs(weights_dir, exist_ok=True)

print("Downloading VGG16 weights from torchvision...")
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# Save the full state dict
vgg_path = os.path.join(weights_dir, 'vgg16_features.pth')
torch.save(model.state_dict(), vgg_path)
print(f"Saved VGG16 weights to: {vgg_path}")

# Verify the file
size_mb = os.path.getsize(vgg_path) / (1024 * 1024)
print(f"File size: {size_mb:.1f} MB")
EOF

echo ""
echo "Done! Weights saved to: ${WEIGHTS_DIR}"
echo ""
echo "To use on HPC, copy the weights/ directory to your HPC storage:"
echo "  rsync -av ${WEIGHTS_DIR} <hpc_user>@<hpc_host>:<path_to_repo>/weights/"
echo ""
echo "Then set in your config:"
echo "  loss:"
echo "    perceptual:"
echo "      vgg_weights_path: \"weights/lpips/vgg16_features.pth\""
