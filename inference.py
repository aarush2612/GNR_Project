import argparse
import csv
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", required=True, help="Absolute path to test directory")
args = parser.parse_args()

TEST_DIR   = Path(args.test_dir)
OUTPUT_CSV = Path("submission.csv")

# ─── Locate cached model ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
_snapshots = BASE_DIR / "models" / "models--google--gemma-4-E2B-it" / "snapshots"
_snap_dirs = [d for d in _snapshots.iterdir() if d.is_dir()]
if not _snap_dirs:
    raise FileNotFoundError(f"No snapshot found in {_snapshots}")
MODEL_PATH = str(_snap_dirs[0])
print(f"Loading model from: {MODEL_PATH}")

# ─── Load model ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16,
    local_files_only=True,
)
model.eval()
print(f"Model loaded — CUDA available: {torch.cuda.is_available()}")

# ─── Prompt ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a deep learning expert. "
    "Solve the MCQ question in the image. "
    "Reason step by step inside <thought> tags. "
    "After </thought>, output only a single letter: A, B, C, or D. "
    "If unsure, output E. No punctuation, no explanation — just the letter."
)

ANSWER_MAP = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}

def get_answer(text: str) -> str:
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
                {"type": "text",  "text": SYSTEM_PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE, dtype=torch.bfloat16)   # use DEVICE, not model.device (which is "meta" with device_map="auto")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=4096)

    input_len = inputs["input_ids"].shape[1]
    generated = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return get_answer(generated)

# ─── Read test.csv ────────────────────────────────────────────────────────────
test_csv = TEST_DIR / "test.csv"
if not test_csv.exists():
    raise FileNotFoundError(f"test.csv not found in {TEST_DIR}")

with open(test_csv, "r") as f:
    image_names = [row["image_name"] for row in csv.DictReader(f)]

print(f"Found {len(image_names)} images in test.csv")

# ─── Pre-fill submission.csv with default 5 ──────────────────────────────────
results = {name: "5" for name in image_names}

def save_csv():
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "option"])
        writer.writeheader()
        for name in image_names:
            writer.writerow({"image_name": name, "option": results[name]})

save_csv()

# ─── Inference loop ───────────────────────────────────────────────────────────
VALID_EXT = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

for idx, image_name in enumerate(image_names, 1):
    image_path = None
    for folder in ["images", "image", ""]:
        img_dir = TEST_DIR / folder if folder else TEST_DIR
        if not img_dir.exists():
            continue
        if (img_dir / image_name).exists():
            image_path = img_dir / image_name
            break
        for ext in VALID_EXT:
            if (img_dir / f"{image_name}{ext}").exists():
                image_path = img_dir / f"{image_name}{ext}"
                break
        if image_path:
            break

    if image_path is None:
        print(f"[{idx}/{len(image_names)}] WARNING: '{image_name}' not found — keeping 5")
        continue

    try:
        t      = time.time()
        letter = run_inference(image_path)
        elapsed = time.time() - t
        number = ANSWER_MAP.get(letter, "5")
        results[image_name] = number
        save_csv()
        print(f"[{idx}/{len(image_names)}] {image_name} → {letter} ({number})  ({elapsed:.1f}s)")
    except Exception as e:
        print(f"[{idx}/{len(image_names)}] ERROR on {image_name}: {e} — keeping 5")
        save_csv()

print(f"\nDone! Predictions saved to {OUTPUT_CSV}")
