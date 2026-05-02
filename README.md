# GNR Project — MCQ Inference Pipeline

A deep learning inference pipeline that solves multiple-choice questions (MCQs) from images using **Google Gemma-4-E2B-IT**, a multimodal vision-language model.

---

## How It Works

1. `setup.bash` creates a conda environment, installs all dependencies, downloads the model via `kagglehub`, and clones `inference.py` from the repository.
2. `inference.py` loads the model, reads image paths from `test.csv`, runs inference on each image, and saves predictions to `submission.csv`.

---

## Evaluator Instructions

The grader should run the following commands **in order**:

```bash
cd ./your_directory
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
python <grading_script> --submission_file submission.csv
conda remove --name gnr_project_env --all -y
```

> **Important:** When passing `<absolute_path_to_test_dir>`, make sure to use **forward slashes `/`** and not backslashes `\`.
> For example:
> ```bash
> # ✅ Correct
> python inference.py --test_dir /home/user/dataset
>
> # ❌ Wrong (will break path resolution)
> python inference.py --test_dir \home\user\dataset
> ```
> If you copy-paste the path from a Windows file explorer or terminal, it will likely use `\` — **manually replace all `\` with `/` before running.**

---

## Requirements

- Linux system with conda installed
- CUDA 12.6 compatible GPU (recommended, minimum 8 GB VRAM)
- Internet access during `setup.bash` (for model download and pip installs)

> If VRAM is less than 8 GB, the script will automatically fall back to CPU inference. Expect significantly longer runtimes on CPU.

---

## Expected Test Directory Structure

The test directory passed via `--test_dir` should contain:

```
<test_dir>/
├── test.csv               # CSV with either 'image_name' or 'image_id' column
└── images/                # (or 'image/' or flat in root)
    ├── img_001.png
    ├── img_002.png
    └── ...
```

The `test.csv` should have one of the following column formats:

| image_name     | or | image_id       |
|----------------|----|----------------|
| img_001.png    |    | img_001.png    |
| img_002.png    |    | img_002.png    |

---

## Output

After inference, `submission.csv` will be saved in the **same directory as `inference.py`** with the following format:

| id          | image_name     | option |
|-------------|----------------|--------|
| img_001.png | img_001.png    | 3      |
| img_002.png | img_002.png    | 1      |

Options are mapped as: `A→1, B→2, C→3, D→4, E→5` (E/5 is the fallback if unsure).

---

## Model

- **Model:** `google/gemma-4/transformers/gemma-4-e2b-it`
- **Source:** Downloaded via `kagglehub` during `setup.bash`
- **Precision:** `bfloat16` for memory efficiency

---

## Note on Kaggle API Key

The `setup.bash` file contains a Kaggle API key (`KAGGLE_KEY`) in plaintext. **This has been done knowingly and intentionally** — Hugging Face model downloads were throwing errors during testing, so `kagglehub` was used as a reliable alternative. The key is included directly to ensure the model downloads without requiring any additional configuration on the evaluator's machine.

---

## Repository

Model weights are downloaded automatically. `inference.py` is pulled from:
👉 [https://github.com/aarush2612/GNR_Project](https://github.com/aarush2612/GNR_Project)
