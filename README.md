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
- Persistent local-analysis, IG-frame decision, file-hash, CLIP and API caches for fast repeated runs
- OpenAI-assisted image review and automatic captioning
- Top-level training targets for `ERNIE Image`, `Z-Image Base` and `Krea 2`, each with its own prompt/caption engine defaults
- Dedicated Krea 2 workflow with dataset-aware natural-language captions
- Centralized subject profile generation from audited images
- Profile-guided caption normalization for more consistent dataset-wide captions
- Separate IG-frame cleanup, headshot smart crop and medium-shot rescue crop
- Optional controlled export buckets; natural image composition is preserved by default
- Subject checks, identity consistency and diversity balancing
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
   - `Training target / base model`: choose `ERNIE Image`, `Z-Image Base` or `Krea 2`. This setting selects the prompt family and caption engine and is independent from later caption-field customizations.
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

### Training targets and caption policies

The training target is selected at the top of the UI and is the single source of truth for the workflow:

- **ERNIE Image** uses explicit structured captions with visible identity anchors.
- **Z-Image Base** uses compact structured captions and keeps stable identity mainly on the trigger token.
- **Krea 2** uses a dedicated natural-language GPT caption pass after final image selection.

Changing individual caption fields no longer changes the training target or disables its caption engine. The UI only marks the target rules as **individually customized**.

### Krea 2 workflow

Selecting **`Krea 2`** applies the recommended starting configuration:

- target size: **20 train-ready images**,
- shot distribution: **40% headshot / 35% medium / 25% full body**,
- primary audit: **`gpt-5.6-luna`**,
- subject-profile normalization: **`gpt-5.6-terra`**,
- final natural-language captions: **`gpt-5.6-luna`**.

The profile stores stable identity and body information for consistency checks. Stable traits such as body build, tattoos, canonical piercings, scars and canonical facial features are not repeated in Krea captions; the trigger token carries that identity. Captions instead focus on visible, image-specific information such as framing, pose, action, expression, gaze, clothing, temporary accessories, background, lighting, camera angle and composition.

#### Automatic Krea caption repair

Krea 2 can perform one automatic repair attempt when the primary caption is empty, the API call fails, or the caption violates the confirmed Subject Profile / caption policy.

The validator now also enforces the tattoo checkbox. When `include_tattoos` is disabled, any tattoo wording in a GPT caption is treated as a policy violation and sent to the repair model. When enabled, only tattoos visible in the exported image may be described.

UI settings:

- `Use automatic caption repair attempt`: enabled by default
- `Krea 2 caption repair model`: default `gpt-5.6-terra`
- `Reasoning effort – caption repair`: default `low`

The repair call receives the exported image, visible audit facts, the first caption (when available), and the exact validation errors. It rewrites only that one caption; audit, selection and Subject Profile are not repeated. A deterministic local caption is used only when both the primary and repair attempts fail.

The audit exports now include:

- `caption_source`: `gpt_primary`, `gpt_repair`, `local_fallback`, or cache source
- `caption_model`
- `caption_retry_count`
- `caption_validation_error`

The dataset report also shows counts for primary captions, repaired captions and remaining local fallbacks.

Hair color/form, eye color, beard state/color and glasses are handled by a shared profile policy. The profile first normalizes each feature and stores a canonical baseline. The UI then offers two caption strategies:

#### Soft canon representation during selection

When continuing from a confirmed Subject Profile, the final selector can softly promote images matching the user-confirmed canonical hair color. The default target is three canon images with a diminishing `+6 / +4 / +2` bonus. The bonus is only applied when the candidate is no more than five `quality_total` points behind the best alternative in the same selection step.

This does not change the headshot/medium/full-body quotas, never promotes review or reject rows automatically through the canon bonus, and never turns the target into a hard minimum. An explicit user Priority override remains independent of this rule. The report records the canonical color, selected count, eligible keep candidates and canon candidates still located in review or reject.

Black and dark brown remain distinct values. Only selected close blonde variants receive a reduced match strength for a blonde canon.

- **Only deviations from the canonical appearance** (default): the baseline belongs to the trigger token. For example, if the subject is canonically blonde, blonde images omit hair color while red/copper images state the deviation.
- **Caption every visible value when genuine variation exists**: once repeated variation is detected, every visible state is captioned, including the baseline state.

Hair base color and color modifiers are stored separately, so `brown hair with blonde highlights` remains canonically brown while the highlights can still be captioned. Eye-color statistics only use images where the eyes are sufficiently visible and not distorted by grayscale, strong tints, sunglasses or cosmetic-lens signals.

Glasses use structured frame normalization and a position field (`on_face`, `on_head`, `held`, `hanging_from_clothing`). Equivalent wording can resolve to one canonical frame, while genuinely different frames, sunglasses, no-glasses images or a different wearing position remain distinct states.

The Subject Profile stores two piercing layers:

- `piercing_inventory`: every repeatedly observed body piercing and item of ear jewelry, editable in the UI with the roles `canonical`, `variable`, `accessory` or `ignore`.
- `piercing_baseline`: only entries explicitly classified as canonical.

Krea/Z-Image captions omit canonical piercings, but include visible `variable` piercings and `accessory` ear jewelry using normalized location-aware wording such as `septum ring`, `lower-lip stud` or `hoop earring`.

After final exclusions, the curator backfills from clean keep candidates so the configured target refers to actual files in `01_train_ready`, not to images later moved to `caption_remove` or removed by identity checks.

For the most controllable result, use `Profile then Caption`, review the canonical appearance and piercing roles in the Subject Profile tab, and then start the final caption pass.

### Crop and export mechanisms

The curator treats three mechanisms separately:

1. **IG-frame cleanup** removes detected social-media borders before audit and selection.
2. **Rescue crops** create additional candidates without changing the original image classification:
   - Smart Pre-Crop can recover a headshot from a large image with a small face.
   - Medium Rescue Crop can recover a usable torso/hip composition from a weak full-body image.
3. **Controlled buckets** affect only the final export and are **off by default**:
   - off: preserve the selected image's natural composition and let the trainer bucket it,
   - on: export headshots at `1024×1024` and medium/full-body images at `832×1216`.

There is no legacy mode that forces every image into a square crop.

### Performance and cache behavior

All caches remain inside `curated_<trigger>/_cache`; the curator does not write a second cache tree to `%LOCALAPPDATA%`.

Repeated runs now avoid the main sources of unnecessary work:

- the optional one-second request delay is applied only after a real successful OpenAI request, never after a cache hit,
- the IG-frame remover caches both positive and negative decisions,
- IG-frame detection runs on a preview with a maximum edge of 1024 px while the final crop is taken from the original image,
- local blur/color/pHash/face/pose metrics are cached per image,
- quick-reject metrics are cached separately so repeatedly rejected images do not need to be decoded again,
- singleton pHash groups are not scored with face/pose analysis,
- CLIP, MediaPipe and Haar models are initialized lazily and are not loaded during fully cached runs,
- a single `file_hash_index.json` reuses SHA-1 hashes while file size and nanosecond modification time remain unchanged.

The Markdown dataset report contains a `Performance` section with stage timings, artificial API wait time and cache counters such as audit, IG, local-analysis, CLIP, ArcFace and file-hash hits/misses.

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

### Configurable reasoning effort

The UI exposes separate `reasoning.effort` selectors for:

- regular image audits
- trigger-word checks
- review escalation
- subject-profile normalization
- final Krea 2 captions

Recommended Krea 2 defaults are `none` for routine audits, trigger checks and final captions, and `low` for Terra-based escalation and subject-profile reconciliation. GPT-5.6 also supports `medium`, `high`, `xhigh` and `max`; higher settings increase latency and token use. Selecting the Krea 2 preset restores these recommended stage-specific values.

### Automatic Subject Profile normalizer repair

The Subject Profile stage now treats malformed or incomplete structured JSON as a recoverable model error instead of immediately replacing the whole profile with local fallback values.

- all `output_text` parts from the Responses API are concatenated before JSON parsing;
- harmless Markdown fences or trailing text are removed locally;
- the profile output budget is increased because reasoning tokens count against `max_output_tokens`;
- an incomplete or invalid primary response triggers one automatic retry with the configured Subject Profile model and reasoning effort;
- only if both model responses fail is `fallback_local` used;
- fallback profiles are not silently reused from the profile cache;
- profile schema `v13` invalidates earlier fallback caches;
- the Profile UI shows a prominent warning and the original parsing error when a local fallback profile is loaded.

The profile JSON records:

- `normalizer_source`: `gpt_primary`, `gpt_retry`, or `local_fallback`
- `normalizer_retry_count`
- `normalizer_primary_error`

A retry rebuilds only the Subject Profile response. It does not repeat the individual image audits.

### Immediate and process-safe cancellation

The Curator, Subject Profile caption continuation, and Video Processor now use an immediate cancellation action that is not queued behind the active Gradio generator.

- Cancel callbacks use `queue=False` and cancel the associated Gradio run event.
- The active subprocess is stored and accessed under a thread lock.
- On Windows, cancellation terminates the complete process tree with `taskkill /T /F`; on Unix-like systems it terminates the dedicated process group.
- The log streamer keeps its own subprocess reference, avoiding races when the UI requests cancellation.
- Cancelled runs retain their current progress instead of being shown as 100% complete.
- Temporary UI configuration files are cleaned up after cancellation.
- A second run cannot start while an active process is still running.

### Cooperative and registry-backed cancellation

Cancellation no longer relies only on Gradio cancelling the active generator or on an in-memory `Popen` reference.

- Each run receives its own cancellation marker and persistent PID registry.
- The Stop button is executed immediately and is no longer configured with Gradio's `cancels=[run_event]`, which could detach the UI generator before the subprocess was stopped.
- The UI writes the cooperative marker first, then terminates the complete process tree. It can recover the PID from disk even if the in-memory process reference is unavailable.
- The curator checks the marker between images, exports, retry waits and API phases.
- Blocking OpenAI requests run in a daemon worker and are polled for cancellation, so the run does not have to wait for the HTTP timeout.
- Windows still uses `taskkill /T /F`, with direct process and PowerShell fallbacks. The video processor uses the same cooperative marker.
- Cancelled runs exit with code 130 and are shown as cancelled rather than failed or completed.

## Fix 2026-08-01: Continue-from-Profile-Export

- Behebt `TypeError: argument of type 'int' is not iterable` beim Captioning aus einem bestaetigten Subject Profile.
- Die zentrale Row-Map wird nicht mehr durch den numerischen Export-Fortschrittsindex ueberschrieben.
- Die Synchronisierung von `selected`, `output_bucket`, `new_basename` und `final_caption` ist zusaetzlich defensiv gegen falsche Argumenttypen abgesichert.


## Profile buckets, audited rejects, and Priority images

The Subject Profile identity-clustering view supports image-level decisions:

- Click an image in the selected bucket to enable individual actions.
- **Detach image from bucket** moves only the exact selected source file into a new manual singleton bucket. Selection now uses filename/path plus a unique member key, so duplicate-content images in the large reject bucket can be detached safely. A normal image inherits its prior role; an image detached from the audited-reject bucket starts as `exclude` and can then be deliberately promoted.
- Fully audited rejects and ArcFace hard flags appear in an always-last `audited_rejects` bucket with the fixed default role `exclude`. Early size rejects, script errors, NSFW blocks, and first-pass pHash duplicates are not shown because they did not receive a useful semantic audit.
- The reject reason is visible below every thumbnail, in the selected-image details, and as an aggregate reason summary for the bucket.
- A rejected image can be recovered by detaching it and assigning the new bucket a training role, or by marking the image as **Priority**.
- **Priority** is an explicit user override. It forces the image into `01_train_ready` despite quality score, ArcFace result, bucket role, quota, duplicate, diversity, or caption-remove logic. Only a missing or unreadable source file can prevent export. If the number of Priority images exceeds the configured target, all Priority images are kept and the final dataset grows beyond the target.

The profile persists the image-level overrides in:

```json
{
  "priority_image_ids": ["..."],
  "priority_image_filenames": ["..."]
}
```

### Detach UI refresh fix (2026-08-01)

After **Detach image from bucket**, the cluster panel is now rebuilt from the just-saved `_subject_profile.json`. The interactive cluster table and gallery receive explicit Gradio updates, the new singleton bucket becomes the active bucket immediately, and the previous image selection is cleared. This avoids a Gradio 6 behavior where the profile changed on disk but the visible Dataframe/Gallery could continue showing their previous values.


### Audited-reject detach hardening and reason display (2026-08-01)

- Reject-bucket members use a stable per-source member ID in addition to the content hash. Exact duplicate files no longer make Detach target the wrong row.
- The persisted filename and source path are used as additional exact-match safeguards.
- The audited-reject bucket is normalized to `exclude`; detached reject images also start as `exclude`.
- `short_reason`, ArcFace hard flags, local override reasons and audit issues are surfaced in the gallery and selected-image panel.
- Reject reasons are stored parallel to the reject bucket members so they remain visible even when the stage file is temporarily unavailable.

### Large reject-bucket paging and stable selection (2026-08-01)

The audited-reject gallery is now loaded in deterministic pages of 36 records.

- A **Gallery page** selector appears above the preview when a bucket spans multiple pages.
- The record list is paged before images are decoded, so large reject buckets no longer try to load hundreds of thumbnails in one callback.
- Each visible tile retains the original absolute member index, member ID, filename and source path. Detach therefore resolves the exact source record even on later pages or when files have identical image content.
- Missing or unreadable previews are represented by a neutral placeholder instead of being silently skipped. Such records remain selectable and can still be detached or marked Priority.
- Changing page or cluster clears the previous image selection, preventing a stale selection from detaching a different record.
- After Detach, the new singleton bucket opens on page 1 and the persisted profile is reloaded as before.


### Very large local images and Pillow safety (2026-08-02)

The curator accepts trusted local high-resolution photographs and panoramas without emitting Pillow's default `DecompressionBombWarning` at roughly 89 megapixels.

- A finite hard ceiling of 350 megapixels remains in place; files above it are treated as unreadable instead of being decoded without limit.
- UI thumbnails use JPEG decoder-level draft scaling where supported and are then bounded to the requested preview size.
- API input, pHash, exposure/color analysis, CLIP and ArcFace use bounded working copies; the original file is never resized in place.
- Instagram-frame detection continues to analyze a small preview while taking any accepted crop from the original full-resolution source.
- Final export and requested full-resolution crops still use the original image.

This removes repeated warning spam for legitimate 108-megapixel images and substantially reduces memory use during preview and model analysis.

## Smart Frame Cleanup / Rahmenprüfung

The former IG-frame and secondary nav-bar workflows are unified in a local four-side detector.
It analyses left, right, top and bottom independently on a bounded preview and never uses an LLM.

- High-confidence candidates can be applied automatically before audit.
- Medium-confidence candidates stay reversible and appear in the **Frame Review / Rahmenprüfung** tab.
- The review tab shows original and proposed crop side by side and supports:
  - accept suggestion,
  - keep original,
  - restore automatic behavior,
  - enter and preview manual crop bounds.
- Source and effective image paths are stored separately (`source_original_path`,
  `effective_image_path`, `frame_crop_path`). Original files are never modified.
- Detector cache can be recalculated independently without deleting audit, CLIP, ArcFace or profile caches.
- Manual decisions are stored in `curated_<trigger>/_frame_cleanup_decisions.json`.
- Early pHash duplicates are excluded from the audited-reject profile bucket.

Default mode: automatically apply only high-confidence local crops and send medium-confidence
cases to Frame Review. The optional aggressive mode can also apply medium-confidence candidates;
`suggest_only` never crops automatically.
