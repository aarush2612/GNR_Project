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
    echo "ERROR: Could not find conda.sh. Please check your conda installation."
    exit 1
fi

ENV_NAME="gnr_project_env"
PYTHON_VERSION="3.11"

echo ""
echo "===== [1/5] Cloning repository ====="
git clone "https://github.com/aarush2612/GNR_Project"
cd "GNR_Project"

echo ""
echo "===== [2/5] Creating conda environment (Python 3.11) ====="
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo ""
echo "===== [3/5] Installing dependencies ====="
echo "  [3a] Upgrading pip..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip -q

echo "  [3b] Installing PyTorch with CUDA 12.6..."
conda run -n "$ENV_NAME" python -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 \
    --no-cache-dir \
    -v 2>&1 | grep -E "Downloading|Installing|Successfully|error|Error" || true

echo "  [3c] Installing transformers and other packages..."
conda run -n "$ENV_NAME" python -m pip install \
    "transformers>=4.51.0" accelerate pillow pandas tqdm huggingface_hub \
    --no-cache-dir \
    -v 2>&1 | grep -E "Downloading|Installing|Successfully|error|Error" || true

echo ""
echo "===== [4/5] Downloading Gemma-4 model weights INTO ./model_weights ====="
echo "  This will take 10-20 minutes depending on your internet speed..."

conda run -n "$ENV_NAME" --no-capture-output python - <<'EOF'
from huggingface_hub import snapshot_download
import os

model_name = "google/gemma-4-E2B-it"
local_dir  = os.path.join(os.getcwd(), "model_weights")   # <── saves HERE, not in ~/.cache

print(f"\nDownloading {model_name} → {local_dir}")
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False,   # real files, no symlinks into cache
    ignore_patterns=["*.gguf"],
)
print(f"\nAll weights saved to: {local_dir}")
EOF

echo ""
echo "===== [5/5] Setup complete! ====="
echo ""
echo "Next steps:"
echo "  conda activate gnr_project_env"
echo "  python inference.py --test_dir <absolute_path_to_test_dir>"
