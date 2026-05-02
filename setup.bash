#!/bin/bash
set -e  # Exit immediately on error

# ─── Initialize conda ───────────────────────────────────────────────────────
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/c/Users/aarus/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/c/Users/aarus/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh. Please check your conda installation."
    exit 1
fi

ENV_NAME="gnr_project_env"
PYTHON_VERSION="3.11"

# ─── HuggingFace token (required for gated Gemma-4 model) ───────────────────
export HF_TOKEN="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"

echo "===== [1/5] Cloning repository ====="
git clone "https://github.com/aarush2612/GNR_Project"
cd "GNR_Project"

echo "===== [2/5] Creating conda environment ====="
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo "===== [3/5] Installing dependencies (PyTorch ~2.5GB, please wait) ====="
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
echo "  --> Installing PyTorch with CUDA 12.6 (this will take a few minutes)..."
conda run -n "$ENV_NAME" python -m pip install --progress-bar on torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
echo "  --> Installing transformers and other dependencies..."
conda run -n "$ENV_NAME" python -m pip install --progress-bar on "transformers>=4.51.0" accelerate pillow pandas tqdm huggingface_hub

echo "===== [4/5] Pre-downloading Gemma-4 model weights (5-10GB, please wait) ====="
conda run -n "$ENV_NAME" --no-capture-output python - <<EOF
import os
os.environ["HF_TOKEN"] = "${HF_TOKEN}"

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

model_name = "google/gemma-4-E2B-it"
print(f"Downloading model: {model_name} ...")

processor = AutoProcessor.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
print("Processor downloaded.")

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
    token=os.environ["HF_TOKEN"],
)
print("Model downloaded and cached successfully.")
EOF

echo "===== [5/5] Setup complete! ====="
echo ""
echo "To run inference:"
echo "  conda activate gnr_project_env"
echo "  python inference.py --test_dir <path_to_test_dir>"
