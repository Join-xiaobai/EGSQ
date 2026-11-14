#!/usr/bin/env bash
# EGSQ Automated Remote Dataset Acquisition Script
set -e

REMOTE_DIR="${REMOTE_WORKSPACE:-/root/autodl-tmp/egsq}"
DATA_DIR="${REMOTE_DIR}/data"
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

echo "=== [1/4] Cloning Multicentre-Fetal-Biometry Repository ==="
if [ ! -d "Multicentre-Fetal-Biometry" ]; then
    git clone https://ghfast.top/https://github.com/surgical-vision/Multicentre-Fetal-Biometry.git
fi

echo "=== [2/4] Downloading ISBI 2015 Cephalometric Grand Challenge via KaggleHub ==="
/root/miniconda3/bin/python -c '
import os, shutil, kagglehub
print("Downloading ISBI 2015 Cephalometric...")
path = kagglehub.dataset_download("jiahongqian/cephalometric-landmarks")
print("Downloaded to:", path)
dest = "'"${DATA_DIR}"'/isbi2015_cephalometric"
os.makedirs(dest, exist_ok=True)
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(dest, item)
    if not os.path.exists(d):
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
print("ISBI 2015 deployed to:", dest)
'

echo "=== [3/4] Downloading Aariz Cephalometric Dataset via KaggleHub ==="
/root/miniconda3/bin/python -c '
import os, shutil, kagglehub
print("Downloading Aariz Cephalometric...")
path = kagglehub.dataset_download("ketanmann30/aariz1")
print("Downloaded to:", path)
dest = "'"${DATA_DIR}"'/aariz_cephalometric"
os.makedirs(dest, exist_ok=True)
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(dest, item)
    if not os.path.exists(d):
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
print("Aariz deployed to:", dest)
'

echo "=== [4/4] Summary of Dataset Files ==="
ls -lh "${DATA_DIR}"
