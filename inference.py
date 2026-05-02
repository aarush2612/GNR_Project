import argparse
import csv
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", required=True, help="Absolute path to test directory")
args = parser.parse_args()

TEST_DIR   = Path(args.test_dir.replace("\\", "/"))
OUTPUT_CSV = Path("submission.csv")

# ─── Locate model ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_weights"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"model_weights directory not found at: {MODEL_PATH}")
else:
    print(f"Model weights directory found at: {MODEL_PATH}")

def find_model_root(base: Path) -> Path:
    if (base / "config.json").exists():
        return base
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    for sub in subdirs:
        if (sub / "config.json").exists():
            return sub
    raise FileNotFoundError(
        f"Could not find config.json under {base}. Contents: {list(base.iterdir())}"
    )

MODEL_PATH = find_model_root(MODEL_PATH)
print(f"Loading model from: {MODEL_PATH}")

# ─── Device setup ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {vram_gb:.1f} GB")
    if vram_gb >= 8:
        DEVICE = torch.device("cuda:0")
        print("VRAM sufficient — using GPU (cuda:0)")
    else:
        DEVICE = torch.device("cpu")
        print(f"VRAM too low ({vram_gb:.1f} GB < 8 GB) — falling back to CPU")
else:
    DEVICE = torch.device("cpu")
    print("CUDA not available — using CPU")

# ─── Load model ──────────────────────────────────────────────────────────────
print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    padding_side="left",
)

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    str(MODEL_PATH),
    dtype=torch.bfloat16,
    local_files_only=True,
).to(DEVICE)
model.eval()

print(f"Model loaded on: {DEVICE}")
if torch.cuda.is_available():
    print(f"VRAM used: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
    print(f"VRAM free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.1f} GB")

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
    ).to(DEVICE)

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
    rows = list(csv.DictReader(f))

image_names = [row.get("image_name") or row.get("image_id") for row in rows]
image_names = [n for n in image_names if n]
print(f"Found {len(image_names)} images in test.csv")

# ─── Image lookup ─────────────────────────────────────────────────────────────
SEARCH_FOLDERS = ["image", "images", ""]
VALID_EXT      = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

def find_image(image_name: str) -> Path | None:
    for folder in SEARCH_FOLDERS:
        img_dir = TEST_DIR / folder if folder else TEST_DIR
        if not img_dir.exists():
            continue
        if (img_dir / image_name).exists():
            return img_dir / image_name
        for ext in VALID_EXT:
            if (img_dir / f"{image_name}{ext}").exists():
                return img_dir / f"{image_name}{ext}"
    return None

# ─── submission.csv ───────────────────────────────────────────────────────────
results = {name: "5" for name in image_names}

def save_csv():
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image_name", "option"])
        writer.writeheader()
        for name in image_names:
            writer.writerow({"id": name, "image_name": name, "option": results[name]})

save_csv()

# ─── Inference loop ───────────────────────────────────────────────────────────
total_start = time.time()
timings     = []

pbar = tqdm(image_names, desc="Inference", unit="img", dynamic_ncols=True)

for image_name in pbar:
    image_path = find_image(image_name)

    if image_path is None:
        pbar.write(f"  WARNING: '{image_name}' not found in {TEST_DIR} — keeping 5")
        continue

    try:
        t_start = time.time()
        letter  = run_inference(image_path)
        elapsed = time.time() - t_start
        timings.append(elapsed)

        number = ANSWER_MAP.get(letter, "5")
        results[image_name] = number
        save_csv()

        avg = sum(timings) / len(timings)
        pbar.set_postfix({"last": f"{elapsed:.1f}s", "avg": f"{avg:.1f}s", "ans": f"{letter}({number})"})
        pbar.write(f"  [{image_name}] → {letter} ({number})  time={elapsed:.1f}s  avg={avg:.1f}s")

    except Exception as e:
        pbar.write(f"  ERROR on '{image_name}': {e} — keeping 5")
        save_csv()

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"Done! {len(image_names)} images processed in {total_elapsed:.1f}s")
print(f"Avg per image: {(total_elapsed / len(image_names)):.1f}s" if image_names else "")
print(f"Predictions saved to: {OUTPUT_CSV}")
print(f"{'='*60}")
