import argparse
import os
import csv
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ─── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", required=True, help="Absolute path to test directory")
args = parser.parse_args()

TEST_DIR = Path(args.test_dir)
OUTPUT_CSV = Path("submission.csv")          # written to current working directory
MODEL_NAME = "google/gemma-4-e2b-it"

# ─── Load model (no internet; uses cached weights from setup.bash) ───────────
print("Loading model from cache...")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()
print(f"Model loaded on {device}")

# ─── Prompt ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a genius in the field of deep learning. "
    "Solve the MCQ question in the image. "
    "First, provide all of your reasoning inside <thought> tags. "
    "After closing the thought tags, provide your final answer. "
    "The final answer MUST be exactly one letter (A, B, C, or D) and nothing else. "
    "Last letter should be the correct option, A, B, C or D. "
    "Also if you are not confident about your answer, answer with E"
)

def get_answer(text: str) -> str:
    """Extract the last single-letter answer from the model output."""
    # Try splitting on the model's delimiter first
    if "<turn|>" in text:
        candidate = text.split("<turn|>")[0][-1].strip().upper()
        if candidate in "ABCDE":
            return candidate
    # Fallback: scan from end for the last A-E letter
    for ch in reversed(text.strip()):
        if ch.upper() in "ABCDE":
            return ch.upper()
    return "E"   # default if nothing found

def run_inference(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=4096)

    # Decode only newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return get_answer(generated)

# ─── Collect images ──────────────────────────────────────────────────────────
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
image_files = sorted([
    f for f in TEST_DIR.iterdir()
    if f.suffix.lower() in VALID_EXT
])

if not image_files:
    raise FileNotFoundError(f"No images found in {TEST_DIR}")

print(f"Found {len(image_files)} images in {TEST_DIR}")

# ─── Run and write CSV ───────────────────────────────────────────────────────
results = []
for idx, img_path in enumerate(image_files, 1):
    start = time.time()
    answer = run_inference(img_path)
    elapsed = time.time() - start
    print(f"[{idx}/{len(image_files)}] {img_path.name} → {answer}  ({elapsed:.1f}s)")
    results.append({"image": img_path.name, "answer": answer})

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "answer"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone! Saved {len(results)} predictions to {OUTPUT_CSV}")
