# LoRA Dataset Curator

Interactive toolchain for automatic curation of LoRA training data from image folders and videos.
The curator combines local filters (sharpness, resolution, pHash), MediaPipe, CLIP and an OpenAI-powered image audit to produce a small, high-quality dataset. It can also build a centralized subject profile from audited images and use it to normalize captions for more consistent identity and trait tagging across the dataset.

> Note: This project only ships the **code**. The use of external models and APIs (e.g. InsightFace models, OpenAI API) is subject to their own license terms.

---

## Reason

This project started from a very practical frustration: manually sorting, preparing, and captioning datasets had become unnecessarily time-consuming. What began as an experiment in automating that workflow through ChatGPT evolved, after roughly 400 prompts and input from 4 different LLMs, into the tool you see here. In the most literal sense, it was 100% vibe-coded.
Please be also aware that this is still a work in progress. Every iteration makes it better, more detailed and harder to use. 

---

## Requirements

To use the OpenAI-assisted features of this project, you need your own OpenAI API key as well as sufficient API credit or available free tokens on your account. Also you need python 3.10.

## Features

Many checks and review steps are optional or configurable in the UI. The main features are:

### Dataset Curator

- Web UI with saved settings and English/German language switching
- Local pre-filtering and duplicate detection before expensive API calls
- OpenAI-assisted image review and automatic captioning
- Centralized subject profile generation from audited images
- Profile-guided caption normalization for more consistent dataset-wide captions
- Optional smart crop, subject checks and diversity balancing
- Structured outputs for train-ready, review and manual cleanup workflows
- Export of captions, CSV, JSONL and a markdown dataset report

### Video Extractor

- Extracts frames of a target person from videos using a reference image
- Samples videos efficiently and keeps the sharpest pose-diverse frames
- Saves extracted frames directly for use in the Dataset Curator

---

## Installation

### Quick start (Windows)

1. Clone the repository and change into the folder:

```bash
git clone https://github.com/Arona1812/DatasetCurator.git
cd <your-repo-folder>
```

2. Double-click `start_curator.bat`.

The launcher will:
- create the `curator_env` virtual environment,
- verify and install the required core packages (requests, pillow, numpy, scipy, mediapipe, torch, torchvision, open_clip_torch, opencv-python, onnxruntime-gpu, scikit-learn, gradio),
- try to install optional InsightFace support for the Video Processor and ArcFace identity check,
- start the Gradio UI in your browser.

> The Windows quick start installs the **CUDA 13.0 builds** of PyTorch and ONNX Runtime by default (pinned in the .bat). The tool still runs without an NVIDIA GPU — it simply falls back to CPU execution — but you will install several hundred MB of CUDA wheels you will not actually use. If you do not have a CUDA-capable GPU, prefer the manual installation below with the CPU-only commands.

> InsightFace is optional for the image curator but **required for the Video Processor** and the ArcFace identity check. On Windows, installing InsightFace may require Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Manual installation (example Linux/macOS)

Adjust versions to your preferred CUDA/PyTorch setup. The commands below use CPU-friendly defaults where possible:

```bash
python3.10 -m venv curator_env
source curator_env/bin/activate
pip install --upgrade pip setuptools wheel

pip install requests pillow numpy scipy
pip install mediapipe==0.10.33

# Choose the PyTorch command that matches your system.
# CPU example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install open_clip_torch
pip install opencv-python scikit-learn gradio

# ONNX Runtime: install exactly one of these variants.
# CPU/default:
pip install onnxruntime
# GPU alternative for suitable NVIDIA/CUDA setups:
# pip install onnxruntime-gpu

# Optional: required for the Video Processor and ArcFace identity check.
pip install insightface

python dataset_curator_ui.py
```

> For NVIDIA/CUDA acceleration, replace the PyTorch and ONNX Runtime commands with versions matching your driver/CUDA setup. See https://pytorch.org/get-started/locally/. On Windows, `insightface` may require Microsoft C++ Build Tools.

---

## Usage

### 1. Dataset Curator (images)

1. Start the UI:
   - Windows: run `start_curator.bat`
   - Other platforms: `python dataset_curator_ui.py` inside the virtual environment

2. In the **Dataset Curator** tab:
   - `Trigger Word`: unique token for your subject (e.g. `aronaLora09`).
   - `Input folder images`: folder with your source images (no subfolder recursion).
   - `Target dataset size`: desired number of final training images.
   - `OpenAI API Key`: your own OpenAI API key.
   - Tune quality scores, shot ratios, pre-filters, duplicate detection, smart-crop, clustering and caption options.
   - Choose the pipeline mode:
     - `Single Pass`: the subject profile is built and applied automatically during the run.
     - `Profile then Caption`: the run pauses after profile creation so you can review or edit it in the `🧬 Subject Profile` tab before starting captioning.

3. The curator writes temporary config files (`_ui_config.json`) and uses them to start `dataset_curator_v2.py` in the background.

4. During profile-based workflows, the curator also writes a `_subject_profile.json`, which stores the normalized subject information used for caption generation.

5. Results are written into `curated_<trigger>/` with folders such as `01_train_ready`, `02_keep_unused`, `03_caption_remove`, `04_review`, `05_reject`, `06_needs_manual_review`, `_cache` and `08_smart_crop_pairs`.

6. Use the `01_train_ready` files and selected pictures from `04_review` for your LoRA training. Also check `02_keep_unused`, `03_caption_remove` and `06_needs_manual_review` for shots that may only need minor manual cleanup, manual selection or recaptioning.

### 2. Video Processor

1. In the **Video Processor** tab:
   - `Video folder`: path with your video files (mp4, mov, mkv, avi).
   - `Output folder`: destination for extracted frames (e.g. `r.00_input`).
   - `Reference image target person`: clear reference photo of the target person.

2. The video processor:
   - detects the target person using InsightFace (`buffalo_l`),
   - samples frames at a configurable FPS,
   - clusters frames per minute by pose (yaw/pitch) and selects the sharpest candidates.

The extracted frames can be fed directly into the image curator.

---

## OpenAI API

This project can optionally use the OpenAI API to score images and generate structured metadata.

- You need your **own OpenAI account** and API key.
- The API key is either:
  - provided via the UI field `OpenAI API Key`, or
  - read from the `OPENAI_API_KEY` environment variable.
- The key is only used locally by the UI/subprocess workflow and is **never** stored in the repository.
  For convenience, saved UI settings may persist it in local runtime files such as `_ui_settings.json`; transient run configs such as `_ui_config.json` pass it to the curator process.
  These runtime config files (`_ui_config.json`, `_ui_video_config.json`, `_ui_settings.json`) are excluded via `.gitignore` and should not be shared.

By using the OpenAI API you agree to the OpenAI Terms of Use and Services Agreement.

---

## InsightFace models

The video processor uses InsightFace for face recognition, in particular the `buffalo_l` model.

- The **InsightFace Python library** is licensed under MIT.
- The **pretrained models** from the InsightFace model zoo (including `buffalo_l`) are released for **non-commercial research purposes only**.
- For **commercial use** of these models you must obtain a separate license directly from InsightFace.

This repository does **not** ship any pretrained InsightFace model files. They are downloaded by the InsightFace library or need to be obtained separately.

---

## Third-party licenses

The original code in this repository is licensed under the **MIT License**, see `LICENSE`.

Major dependencies and their licenses include:

- Gradio – Apache-2.0
- MediaPipe – Apache-2.0
- PyTorch – BSD-3-Clause
- OpenCV / opencv-python – Apache-2.0 (OpenCV), MIT (wrapper)
- Pillow – HPND
- NumPy – BSD-3-Clause
- scikit-learn – BSD-3-Clause
- open_clip_torch / OpenCLIP – Apache-2.0 / MIT (depending on version)
- InsightFace (code) – MIT; models non-commercial
- onnxruntime – MIT

See `thirdparty-lic.md` for more details.

---

## Donations

If this project is useful to you and you want to support development, you can optionally donate a coffee:

- Buy me a coffee: https://buymeacoffee.com/arona1812

Donations are entirely optional and do **not** change any license terms or third-party usage restrictions.

---

## Disclaimer

This project is provided "AS IS", without warranty of any kind.

You are responsible for:
- complying with the license terms of InsightFace models (non-commercial, separate commercial model licensing),
- complying with the license and usage terms of the OpenAI API and any other external services.

The author assumes no liability for the use of this tool in production or commercial environments.
