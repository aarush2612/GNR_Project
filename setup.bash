#!/bin/bash
set -e  # Exit immediately on error

# ─────────────────────────────────────────────
# CONFIGURATION — update before submission
REPO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO.git"   # <-- REPLACE THIS
REPO_DIR="YOUR_REPO"                                          # <-- folder name after clone
ENV_NAME="gnr_project_env"
PYTHON_VERSION="3.11"
HF_MODEL="google/gemma-4-e2b-it"
# ─────────────────────────────────────────────

echo "===== [1/5] Cloning repository ====="
git clone "$REPO_URL"
cd "$REPO_DIR"

echo "===== [2/5] Creating conda environment ====="
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo "===== [3/5] Activating environment and installing dependencies ====="
# Use conda run to install inside the env (works without interactive shell)
conda run -n "$ENV_NAME" pip install --upgrade pip
conda run -n "$ENV_NAME" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
conda run -n "$ENV_NAME" pip install transformers accelerate pillow pandas tqdm

echo "===== [4/5] Pre-downloading model weights (requires internet) ====="
conda run -n "$ENV_NAME" python - <<'EOF'
from transformers import AutoProcessor, AutoModelForImageTextToText
import os

model_name = "google/gemma-4-e2b-it"
print(f"Downloading model: {model_name}")

# Download and cache model + processor so inference works offline
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
)
print("Model downloaded and cached successfully.")
EOF

echo "===== [5/5] Setup complete! ====="
echo ""
echo "To run inference:"
echo "  conda activate gnr_project_env"
echo "  python inference.py --test_dir <path_to_test_dir>"
