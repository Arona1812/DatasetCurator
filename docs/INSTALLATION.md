# Installation notes

The main application uses Python 3.13. The Windows launcher is the maintained setup path; the manual route below is useful for Linux, macOS, and customized Python environments. Local Florence 2 and JoyCaption captioners require additional CUDA environments and model weights.

## Windows launcher

Run `start_curator.bat` from the repository root, either by double-clicking it or from PowerShell:

```powershell
.\start_curator.bat
```

The launcher creates `curator_env_py313`, verifies/install core packages, tries to install optional InsightFace, and starts the UI. It installs CUDA-enabled PyTorch and ONNX Runtime packages by default. To install portable CPU packages instead:

```powershell
.\start_curator.bat --cpu
```

If optional InsightFace cannot be installed, the UI can still start. ArcFace identity matching and Video Processor face matching then remain unavailable. On Windows, InsightFace may require Microsoft C++ Build Tools.

## Manual Python setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python dataset_curator_ui.py
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`. The portable `requirements.txt` installs CPU-oriented defaults. GPU acceleration needs PyTorch and ONNX Runtime builds that match the computer's driver and platform. The launcher-specific CUDA packages are not installed by this command.

## First-caption models

The Windows launcher installs the **main app**, not Florence 2 or JoyCaption model weights. Those local captioners need a CUDA GPU, their own Python environments, and pinned local weight snapshots. The required paths and metadata files are described in [CAPTION_PIPELINE.md](../CAPTION_PIPELINE.md#lokale-einrichtung). The app checks for those resources before starting a paid image audit.

To use the app without local captioner setup, select **GPT-6 Luna** as the first-description model in *Audit & Selection*. This sends the selected images to OpenAI and uses API credit. An OpenAI key is also required for image auditing, even when the first-description model runs locally. Local preflight and frame suggestions do not require an API key.

## Data and settings

Choose an existing image folder in *Start / Project*. The curator writes its output to a separate `curated_<trigger>/` folder. Original images are not overwritten. The API key can be saved in `_ui_settings.json`, which Git ignores; keep that file private.
