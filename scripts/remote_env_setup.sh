#!/usr/bin/env bash
# EGSQ Remote Server Environment Setup Script
set -e

echo "=== [1/3] Updating system packages & Python environment ==="
apt-get update && apt-get install -y git wget curl rsync unzip libgl1-mesa-glx

echo "=== [2/3] Installing PyTorch & Core Libraries ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 || pip install torch torchvision
pip install numpy scipy pandas scikit-learn pyyaml albumentations opencv-python pydicom tqdm

echo "=== [3/3] Creating workspace directories on remote ==="
REMOTE_DIR="${REMOTE_WORKSPACE:-/root/autodl-tmp/egsq}"
mkdir -p "${REMOTE_DIR}/data"
mkdir -p "${REMOTE_DIR}/checkpoints"
mkdir -p "${REMOTE_DIR}/reports"

echo "=== Remote environment setup complete in ${REMOTE_DIR}! ==="
