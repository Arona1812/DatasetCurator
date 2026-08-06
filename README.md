# LoRA Dataset Curator

[Deutsch](README_DE.md) · [MIT License](LICENSE) · [Third-party licenses](thirdparty-lic.md)

**LoRA Dataset Curator** is a local web UI for turning an image folder into a curated, captioned training dataset for **character LoRAs**. It is optimized for consistent person/character datasets, combining duplicate detection, image-quality checks, optional local frame cleanup, OpenAI-assisted image audits, identity-consistency checks, and dataset-aware captioning. A separate video helper can extract useful frames from source videos.

> The repository contains code only. External APIs and model weights are subject to their own terms and licenses.

## What it does

- Finds unreadable files, low-quality images, and near duplicates before paid API requests.
- Detects screenshot borders and other layouts locally; originals are never overwritten.
- Audits images, selects a diverse dataset, and separates ready, review, and reject results.
- Builds an optional **Subject Profile** to keep identity traits and captions consistent across the dataset.
- Supports training-target presets for **ERNIE Image**, **Z-Image Base**, and **Krea 2**.
- Exports images, captions, CSV, JSONL, and a Markdown dataset report.
- Extracts sharp, pose-diverse frames of a reference person from videos (optional Video Processor).

## Workflow

The image curator follows the processing order in the UI:

1. **Start / Project** — set the image folder, trigger word, and API key.
2. **Preflight** — run local file and pHash duplicate checks. No OpenAI request is made.
3. **Frames** *(optional)* — inspect local crop suggestions, keep the original, or define a manual crop.
4. **Audit & Selection** — run quality, identity, diversity, and caption processing.
5. **Subject Profile** — review the profile when using `Profile then Caption`, or let `Single Pass` continue automatically.
6. **Results** — inspect the exported training, review, and cleanup buckets.

## Requirements

- **Python 3.10**
- An **OpenAI API key and available API credit** for the audit and captioning pipeline. Local preflight and frame analysis work without it.
- NVIDIA GPU acceleration is optional. The application can run on CPU.
- **InsightFace** is optional for image curation, but required for Video Processor face matching and the ArcFace identity check.

The interface displays the installed source version from `git describe --tags --always --dirty`. It therefore reflects the checked-out Git tag, commits after that tag, and uncommitted changes rather than a manually maintained version string.

## Quick start on Windows

1. Clone the repository:

   ```bash
   git clone https://github.com/Arona1812/DatasetCurator.git
   cd DatasetCurator
   ```

2. Double-click `start_curator.bat`.

The launcher creates `curator_env`, installs core packages, tries to install optional InsightFace support, and opens the Gradio UI in your browser.

> The Windows launcher currently installs CUDA 13.0 PyTorch and ONNX Runtime builds. It still works without an NVIDIA GPU, but a CPU-only manual installation is smaller.

## Manual installation (Linux, macOS, or CPU-focused setup)

```bash
python3.10 -m venv curator_env
source curator_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional: required for Video Processor and ArcFace identity checking.
pip install insightface

python dataset_curator_ui.py
```

`requirements.txt` uses portable default packages. For NVIDIA/CUDA acceleration, install the PyTorch and ONNX Runtime variants matching your driver and platform; see [PyTorch installation instructions](https://pytorch.org/get-started/locally/). InsightFace can require [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) on Windows.

## Using the image curator

1. Start the UI and enter the **input image folder**, a unique **trigger word**, and your **OpenAI API key** on **Start / Project**.
2. Click **Run preflight**. Later modules unlock after the local checks finish.
3. Optionally use **Frames** to choose a local crop suggestion, retain the original, or mark a rectangular manual crop with two clicks on diagonally opposite corners.
4. In **Audit & Selection**, choose the training target and configure the desired audit, identity, diversity, and caption settings.
   - **Single Pass** creates the profile and completes the workflow.
   - **Profile then Caption** pauses after profile generation so you can review canonical traits, clusters, priority images, and caption policies.
5. Review the result in `curated_<trigger>/`. Key folders include:
   - `01_train_ready` — exported images and captions for training
   - `02_keep_unused` — good images not selected for the target size
   - `03_caption_remove` — images needing caption work
   - `04_review`, `05_reject`, `06_needs_manual_review` — images needing a decision or excluded by the workflow

The project workspace also stores reports and caches below `curated_<trigger>/`; source images remain unchanged.

## Training targets

| Target | Caption approach |
| --- | --- |
| **ERNIE Image** | Structured captions with visible identity anchors. |
| **Z-Image Base** | Compact structured captions; the trigger token carries stable identity. |
| **Krea 2** | Dataset-aware natural-language captions after final image selection. |

Target presets provide starting values. Settings changed in the UI remain authoritative for the run.

## Optional Video Processor

Open **Video Processor** in the UI, provide a video folder, an output folder, and a clear reference image of the target person. With InsightFace installed, it samples supported videos (`mp4`, `mov`, `mkv`, `avi`), matches the reference person, and saves sharp, pose-diverse frames for the image curator.

## API, privacy, and licenses

- The project uses the OpenAI Responses API through `requests`; no `openai` Python package is required.
- Provide the key in the UI or through `OPENAI_API_KEY`. It is used locally and is not committed to the repository. Local UI settings can retain it in ignored runtime files; do not share those files.
- The project does not include a youth-protection or NSFW suitability filter. Review your data and outputs before training.
- This project's code is licensed under the [MIT License](LICENSE). See [thirdparty-lic.md](thirdparty-lic.md) for dependency details.
- InsightFace code is MIT, but its pretrained model weights, including `buffalo_l`, have separate restrictions and are described as non-commercial research use. Obtain the required license directly from InsightFace before commercial use.

## Status and support

This is a work in progress. Please verify exported images and captions before using them for training.

If the project helps you, you can optionally support development at [Buy Me a Coffee](https://buymeacoffee.com/arona1812). Donations do not change any license terms.

## Disclaimer

The software is provided **as is**, without warranty. You are responsible for complying with the terms of OpenAI, InsightFace, and every other service or model you use.