#!/bin/bash
set -e

# ─── Initialize conda ───────────────────────────────────────────────────────
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh."
    exit 1
fi

ENV_NAME="gnr_project_env"
PYTHON_VERSION="3.11"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── On Git Bash / MSYS2, convert /c/Users/... → C:/Users/... ───────────────
if command -v cygpath &> /dev/null; then
    WORKDIR="$(cygpath -w "$WORKDIR")"
fi

echo "  WORKDIR resolved to: $WORKDIR"

echo ""
echo "===== [1/5] Cloning repository ====="

if [ -d "GNR_Project_repo" ]; then
    rm -rf GNR_Project_repo
fi

git clone "https://github.com/aarush2612/GNR_Project" GNR_Project_repo
cp GNR_Project_repo/inference.py "$WORKDIR/inference.py"
echo "  Copied inference.py → $WORKDIR/inference.py"

echo ""
echo "===== [2/5] Creating conda environment (Python 3.11) ====="
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo ""
echo "===== [3/5] Installing dependencies ====="
conda run -n "$ENV_NAME" python -m pip install --upgrade pip -q

echo "  Installing PyTorch with CUDA 12.6..."
conda run -n "$ENV_NAME" python -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 \
    --no-cache-dir \
    -v 2>&1 | grep -E "Downloading|Installing|Successfully|error|Error" || true

echo "  Installing transformers, kagglehub and other packages..."
conda run -n "$ENV_NAME" python -m pip install \
    "transformers>=4.51.0" accelerate pillow pandas tqdm kagglehub \
    --no-cache-dir \
    -v 2>&1 | grep -E "Downloading|Installing|Successfully|error|Error" || true

echo ""
echo "===== [4/5] Downloading Gemma-4-E2B-IT → $WORKDIR/model_weights ====="

conda run -n "$ENV_NAME" --no-capture-output python - <<EOF
import kagglehub
import os
import shutil

os.environ["KAGGLE_USERNAME"] = "aarushtripathi01"
os.environ["KAGGLE_KEY"]      = "KGAT_c13aaaec8263a31709cb90b9ff96444a"

dest = os.path.join(r"$WORKDIR", "model_weights")

# Remove existing dir so output_dir receives a clean empty directory
if os.path.exists(dest):
    print(f"Removing existing directory: {dest}")
    shutil.rmtree(dest)
os.makedirs(dest, exist_ok=True)

print(f"Downloading to: {dest}")
downloaded_path = kagglehub.model_download(
    "google/gemma-4/transformers/gemma-4-e2b-it",
    output_dir=dest
)
print(f"Model weights downloaded directly to: {downloaded_path}")
EOF

echo ""
echo "===== [5/5] Setup complete! ====="
echo "  inference.py  → $WORKDIR/inference.py"
echo "  model_weights → $WORKDIR/model_weights"
echo ""
echo "Grader will now run:"
echo "  conda activate gnr_project_env"
echo "  python inference.py --test_dir <absolute_path_to_test_dir>"
