import argparse
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

TEST_DIR   = Path(args.test_dir)
OUTPUT_CSV = Path("submission.csv")   # written to current working directory

# ─── Locate model weights ────────────────────────────────────────────────────
# model_weights/ sits next to inference.py in the grading working directory.
# snapshot_download may lay files out as either:
#   model_weights/config.json               (flat)
#   model_weights/snapshots/<hash>/...      (snapshot layout)
#   model_weights/models--xxx/snapshots/<hash>/...  (full cache layout)
# The helper below finds whichever structure is actually present.

def find_model_dir(base: Path) -> Path:
    """Return the directory that directly contains config.json."""
    # Case 1: flat — files are directly in base
    if (base / "config.json").exists():
        return base

    # Case 2: base/snapshots/<single_hash>/
    snapshots_dir = base / "snapshots"
    if snapshots_dir.is_dir():
        children = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if children:
            return children[0]   # only one hash folder will exist

    # Case 3: base/models--<name>/snapshots/<single_hash>/
    for sub in base.iterdir():
        if sub.is_dir() and "snapshots" in [d.name for d in sub.iterdir() if d.is_dir()]:
            snap = sub / "snapshots"
            children = [d for d in snap.iterdir() if d.is_dir()]
            if children:
                return children[0]

    raise FileNotFoundError(
        f"Cannot locate config.json anywhere under {base}. "
        "Check that setup.bash completed successfully."
    )

BASE_WEIGHTS = Path(__file__).resolve().parent / "model_weights"

if not BASE_WEIGHTS.exists():
    raise FileNotFoundError(
        f"model_weights/ not found at {BASE_WEIGHTS}. "
        "Did setup.bash finish without errors?"
    )

MODEL_DIR = find_model_dir(BASE_WEIGHTS)
print(f"Using model weights at: {MODEL_DIR}")

# ─── Load model (fully offline) ──────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
model = AutoModelForImageTextToText.from_pretrained(
    str(MODEL_DIR),
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
    if "<turn|>" in text:
        candidate = text.split("<turn|>")[0][-1].strip().upper()
        if candidate in "ABCDE":
            return candidate
    for ch in reversed(text.strip()):
        if ch.upper() in "ABCDE":
            return ch.upper()
    return "E"

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
    start   = time.time()
    answer  = run_inference(img_path)
    elapsed = time.time() - start
    print(f"[{idx}/{len(image_files)}] {img_path.name} → {answer}  ({elapsed:.1f}s)")
    results.append({"image": img_path.name, "answer": answer})

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "answer"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone! Saved {len(results)} predictions to {OUTPUT_CSV}")
