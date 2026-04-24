# LoRA Dataset Curator

Interactive toolchain for automatic curation of LoRA training data from image folders and videos.
The curator combines local filters (sharpness, resolution, pHash), MediaPipe, CLIP and an OpenAI-powered image audit to produce a small, high-quality dataset with consistent captions.

> Note: This project only ships the **code**. The use of external models and APIs (e.g. InsightFace models, OpenAI API) is subject to their own license terms.

---

## Reason

This project started from a very practical frustration: manually sorting, preparing, and captioning datasets had become unnecessarily time-consuming. What began as an experiment in automating that workflow through ChatGPT evolved, after roughly 300 prompts and input from 4 different LLMs, into the tool you see here. In the most literal sense, it was 100% vibe-coded.

---

## Requirements

To use the OpenAI-assisted features of this project, you need your own OpenAI API key as well as sufficient API credit or available free tokens on your account. Also you need python.

## Features

- Gradio web UI for the dataset curator and video processor
- Persistent UI settings with automatic restore plus switchable English/German UI language
- Video frame extraction using an InsightFace reference image (`buffalo_l`)
- Local pre-filters for resolution, file size, blur, exposure and early pHash dedup before API calls
- Optional subject-sanity / limb filter based on torso landmark detection
- Duplicate detection using pHash and OpenCLIP / CLIP semantic similarity
- OpenAI-powered image auditing for quality, shot type, subject clarity, attributes, text/watermarks and caption metadata
- Optional stronger-model escalation for difficult review cases, status conflicts and close smart-crop decisions
- Optional AI trigger-word check
- Automatic captioning with configurable caption profiles and optional caption rule overrides
- Text/watermark images can be routed automatically into `caption_remove` for manual cleanup instead of training-ready output
- Session/outfit clustering and diversity penalties for better dataset variety
- Smart pre-crop for headshots from wider images, including original-vs-crop comparison export
- Instagram/UI frame border removal for screenshots and social-media captures
- Bucket-friendly crop profiles to reduce training bucket spread
- Structured output folders for train-ready, caption-remove, review, reject and manual-review images
- Audit/embedding cache plus retry/resume logic to save time and API costs
- Integrated result viewer in the UI with image gallery, captions and dataset report preview
- Result export with captions, CSV, JSONL and a markdown dataset report

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
- install all required packages (requests, pillow, numpy, mediapipe, torch, torchvision, torchaudio, open_clip_torch, opencv-python, insightface, onnxruntime, scikit-learn, gradio),
- start the Gradio UI in your browser.

### Manual installation (example Linux/macOS)

Adjust versions to your preferred CUDA/PyTorch setup:

```bash
python3.10 -m venv curator_env
source curator_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install requests pillow numpy
pip install mediapipe==0.10.33
pip install "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" --index-url https://download.pytorch.org/whl/cu130
pip install open_clip_torch
pip install opencv-python insightface onnxruntime scikit-learn
pip install gradio
python dataset_curator_ui.py
```

> Please adapt the concrete versions to match your `start_curator.bat` setup and your CUDA driver.

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

3. The curator writes temporary config files (`_ui_config.json`) and uses them to start `dataset_curator_v2.py` in the background.

4. Results are written into `curated_<trigger>/` with folders such as `01_train_ready`, `02_caption_remove`, `03_review`, `04_reject`, `05_needs_manual_review`, `_cache` and `07_smart_crop_pairs`.

5. Use the `01_train_ready` files and selected pictures from `03_review` for your LoRA training. Also check `02_caption_remove` and `05_needs_manual_review` for shots that may only need minor manual cleanup or recaptioning.

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
- The key is only used locally in the process environment and is **never** stored in the repository.
  Runtime config files (like `_ui_config.json`, `_ui_video_config.json`, `_ui_settings.json`) are excluded via `.gitignore`.

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
