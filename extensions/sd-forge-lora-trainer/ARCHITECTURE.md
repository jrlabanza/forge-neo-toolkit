# sd-forge-lora-trainer — Architecture & Reference

This document describes every feature and function in the LoRA Trainer extension. Read top-to-bottom for an overview, or jump to a section to look up a specific piece of code.

---

## 1. File layout

```
sd-forge-lora-trainer/
├── README.md                   Quick-start usage docs
├── ARCHITECTURE.md             This file
├── install.py                  Forge-startup stub (lightweight; defers heavy work)
├── setup_venv.bat              Bootstrap launcher — picks the right Python interpreter
├── setup_bootstrap.py          Bootstrap logic — clones kohya, creates sandboxed venv, pip-installs
├── scripts/
│   └── lora_trainer.py         Main extension (UI tab + entire pipeline)
│
└── (runtime, created on first use, .gitignored)
    ├── sd-scripts/             Cloned kohya-ss/sd-scripts checkout
    ├── sd-scripts-venv/        Sandboxed Python venv with kohya's deps
    ├── projects/<name>/        Per-project working dir (images, config, logs)
    └── downloads/<slug>/       Booru-downloaded image+caption sets
```

---

## 2. Feature overview

### 2.1 Training pipeline

End-to-end SDXL LoRA training inside a Forge UI tab. The user picks images, a base model, and presets, then clicks Start training. Output `.safetensors` lands in `models/Lora/` automatically.

| Feature                              | Where                                                                    |
|--------------------------------------|--------------------------------------------------------------------------|
| Sandboxed venv for kohya             | `setup_bootstrap.py` + `setup_venv.bat`                                  |
| SDXL bucket resizing                 | `_resize_to_bucket`, `prepare_dataset`                                   |
| WD14 / BLIP autocaption              | `autocaption_wd14`, `autocaption_blip`                                   |
| Trigger word injection               | `inject_trigger_word`                                                    |
| kohya TOML config generation         | `write_training_toml`                                                    |
| `accelerate launch` subprocess       | `build_train_command`, `_run_subprocess`                                  |
| UTF-8 subprocess env                 | `_run_subprocess` (PYTHONIOENCODING + PYTHONUTF8)                        |
| Output copy to `models/Lora/`        | End of `run_training_job`                                                |

### 2.2 Booru integration

Two distinct workflows, both backed by Danbooru API with Gelbooru fallback:

| Workflow              | Source of training images | Source of captions      | Function           |
|-----------------------|---------------------------|-------------------------|--------------------|
| **Search & download** | Booru (filtered top-N)    | Booru tags per-image    | `search_booru`     |
| **Tag MY uploads**    | User's uploaded files     | Booru tags as reference | `tag_my_uploads`   |

Both:
- Auto-derive trigger from booru tag (`seele_(honkai_impact)` → `seele`)
- Bake the trigger into every caption as the first tag
- Detect defining features (≥60% frequency) and promote them to caption front
- Strip multi-character tags in solo mode
- Skip animated/video files (gif/webm/mp4)

### 2.3 Filters

| Filter                | UI control                | Implementation                                                              |
|-----------------------|---------------------------|-----------------------------------------------------------------------------|
| Rating                | dropdown                  | `DANBOORU_RATING_LETTER`, `GELBOORU_RATING_NAME`                            |
| Solo / Multiple / Any | dropdown                  | `SOLO_INDICATORS`, `MULTI_INDICATORS` (strict client-side verification)     |
| Official art only     | checkbox                  | `official_art` tag filter (server-side on Gelbooru, client-side on Danbooru)|
| No gifs/videos        | always on                 | `STILL_IMG_EXTS` whitelist applied 3 layers deep                            |
| Boilerplate strip     | always on                 | `CAPTION_BLACKLIST`                                                         |
| Style tags excluded   | always on (feature dect.) | `STYLE_TAGS` — chibi, sketch, monochrome, 3d, etc.                          |
| Multi-char tag strip  | only in solo mode         | `_filter_tags(..., solo_mode='solo')`                                       |

### 2.4 UX features

- Required project name + trigger word with friendly error messages
- "Previously downloaded" dropdown for re-using past searches
- Per-image caption editor (preview + edit + save per file)
- Real-time log auto-refresh (gr.Timer 2s tick + autoscroll)
- Diagnostic logging — tag-frequency table, per-filter drop counts
- Status box reports save paths, detected features, trigger derivation

---

## 3. Constants

Defined at the top of `scripts/lora_trainer.py` and used throughout.

| Constant                    | Purpose                                                                  |
|-----------------------------|--------------------------------------------------------------------------|
| `PRESETS`                   | Character/Style/Concept hyperparameters (dim, alpha, lr, epochs, repeats)|
| `CAPTION_MODES`             | UI dropdown choices for autocaption: WD14, BLIP, trigger-only            |
| `TARGET_BASE = 1024`        | SDXL training resolution (long-edge target before bucketing)             |
| `WD14_MODEL_REPO`           | HuggingFace repo for the WD14 ONNX tagger                                |
| `CAPTION_BLACKLIST`         | Tags stripped from every caption (commentary, watermark, signature…)     |
| `GENERIC_TAGS`              | Tags that aren't character-defining features (1girl, solo, looking_at_viewer…) |
| `STYLE_TAGS`                | Art-style tags excluded from feature detection (chibi, sketch, 3d…)      |
| `STILL_IMG_EXTS`            | `{jpg, jpeg, png, webp}` — animated/video files rejected                 |
| `SOLO_INDICATORS`           | Tags that prove a single-character image (1girl, 1boy, solo…)            |
| `MULTI_INDICATORS`          | Tags that prove a multi-character image (2girls, multiple_girls, duo…)   |
| `DANBOORU_RATING_LETTER`    | Maps UI rating to Danbooru rating letters (g/s/q/e)                      |
| `GELBOORU_RATING_NAME`      | Maps UI rating to Gelbooru rating names (general/sensitive/…)            |
| `DANBOORU_BASE`/`GELBOORU_BASE` | API base URLs                                                        |
| `USER_AGENT`                | HTTP user-agent string sent to the boorus                                |

---

## 4. Function reference

### 4.1 Path helpers

These return absolute `pathlib.Path` objects pointing at locations the extension uses. They centralise path resolution so the rest of the code doesn't hardcode anything.

| Function                | Returns                                                        |
|-------------------------|----------------------------------------------------------------|
| `_forge_root()`         | Forge install root (two parents up from the extension folder)  |
| `_models_path()`        | Forge's `models/` folder                                       |
| `_lora_output_dir()`    | `models/Lora/` — created if missing; where final LoRAs are copied to |
| `_checkpoints_dir()`    | `models/Stable-diffusion/` — listed for the base-model dropdown|
| `_venv_python()`        | Path to `sd-scripts-venv/Scripts/python.exe`                   |

### 4.2 String helpers

| Function                       | Purpose                                                                              |
|--------------------------------|--------------------------------------------------------------------------------------|
| `_slugify(s)`                  | Filesystem-safe slug. Strips punctuation, joins spaces with `_`. Empty input → `""`. |
| `_tag_to_trigger(tag)`         | Derive trigger word from booru tag. `seele_(honkai_impact)` → `seele`. Returns `""` if input is empty. |

### 4.3 HTTP helpers

| Function                          | Purpose                                                                   |
|-----------------------------------|---------------------------------------------------------------------------|
| `_http_get_json(url, timeout=30)` | GET a JSON response. Raises on HTTP errors. Used for Danbooru/Gelbooru API calls. |
| `_http_download(url, dest, timeout=60)` | Download a single file to `dest`. Returns `True` on success, `False` on any exception (so a single bad image doesn't break the batch). |

### 4.4 Booru search (low-level)

These query the booru APIs directly. They return lists of dicts with `{url, tags, ext, score, source}`.

| Function                                                                   | Purpose                                                                                            |
|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `_search_danbooru(tag, rating, count, log, official_only=False, solo_mode='solo')` | Query Danbooru. Anonymous accounts have a 2-tag server-side limit, so the function dynamically picks the best 2-tag combo: `<char> solo`, `<char> -solo`, or `<char> official_art` depending on flags. Filters file-extension, rating, solo-vs-multi, and (when needed) official_art client-side. |
| `_search_gelbooru(tag, rating, count, log, official_only=False, solo_mode='solo')` | Query Gelbooru. No tag limit, so all constraints go server-side. Still does client-side file-ext and final solo verification using SOLO_INDICATORS/MULTI_INDICATORS. |

Both functions log diagnostic counters: `kept N (dropped X non-still, Y wrong rating, Z fan-art, W solo-check)`.

### 4.5 Booru orchestration (high-level)

| Function                                                                                       | Purpose                                                                                  |
|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `search_booru(tag, rating, count, log, trigger="", official_only=True, solo_mode='solo')`     | Full Search & Download. Queries Danbooru first; tops up from Gelbooru if needed. Downloads top-N by score to `downloads/<slug>/`. Writes per-image `.txt` captions with trigger + features baked in. Returns the download dir. |
| `tag_my_uploads(tag, image_paths, count, log, trigger="", official_only=True)`                | Tag MY uploads. Uses booru only for tag DATA, not images. Copies user's uploaded files to `downloads/<slug>_custom/`. Writes per-image varied captions (each image draws a different booru post's tag set). Returns the download dir. |
| `list_downloaded_sets()`                                                                       | Enumerate `downloads/*/` folders that contain images. Returns labelled entries sorted by mtime descending. Backs the "Previously downloaded" dropdown. |

### 4.6 Caption helpers

| Function                                                              | Purpose                                                                                                            |
|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `_filter_tags(tag_string, solo_mode='any')`                           | Clean a booru tag string for caption use. Drops `CAPTION_BLACKLIST` tags always. Drops `MULTI_INDICATORS` when `solo_mode='solo'`. Converts underscores to spaces for Illustrious compatibility. |
| `_compute_character_features(caption_dir, top_n=8, min_frequency=0.6)`| Scan all `.txt` captions in a folder. Find tags appearing in ≥60% of them (excluding `GENERIC_TAGS`, `CAPTION_BLACKLIST`, `STYLE_TAGS`, `MULTI_INDICATORS`). Returns space-form feature list, most-common first. Used by `search_booru` (download path). |
| `inject_trigger_word(bucket_dir, trigger, log)`                       | Prepend the trigger word to every `.txt` file in `bucket_dir`. Idempotent — strips leading duplicates before re-prepending. |

### 4.7 Image preparation

| Function                                                              | Purpose                                                                                                            |
|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `_resize_to_bucket(im, base=1024)`                                    | Resize a PIL image so its long edge equals `base`, then snap both dims to multiples of 64 (kohya bucket step). Uses LANCZOS resampling. |
| `prepare_dataset(project_name, image_paths, repeats, class_token, log)` | Copy + resize images into the kohya folder layout `projects/<name>/images/<repeats>_<class>/`. Also copies any colocated `.txt` caption file alongside each image. Returns the parent dir (kohya wants `train_data_dir` to point here). |

### 4.8 Captioning (subprocess-based)

These run kohya's captioning scripts as subprocesses inside the sandboxed venv. Used only when the user picked WD14 or BLIP autocaption (booru paths already produce captions).

| Function                                                              | Purpose                                                                                                            |
|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `autocaption_wd14(bucket_dir, log)`                                   | Run `finetune/tag_images_by_wd14_tagger.py` with `--onnx` flag, threshold 0.35, batch 4, and a curated `--undesired_tags` list. |
| `autocaption_blip(bucket_dir, log)`                                   | Run `finetune/make_captions.py` (BLIP captioner) with `max_length=75`. |

### 4.9 Subprocess helper

| Function                              | Purpose                                                                                |
|---------------------------------------|----------------------------------------------------------------------------------------|
| `_run_subprocess(cmd, cwd, log)`      | Launch a subprocess and stream stdout/stderr line-by-line to `log`. Sets `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1` so Windows cp1252 doesn't choke on kohya's Japanese log strings. |

### 4.10 Training config + launch

| Function                                                              | Purpose                                                                                                            |
|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `write_training_toml(project_root, train_data_dir, output_dir, base_model, preset, project_name, resolution, batch_size, log)` | Build kohya's TOML config. Handles all the subtle type requirements: `resolution="WIDTH,HEIGHT"` (string, not int), `caption_extension=".txt"`, `network_train_unet_only=True` (mandatory when `cache_text_encoder_outputs=True`), `no_half_vae=True` (avoids NaN), `mixed_precision="fp16"`, `optimizer_type="AdamW8bit"`, `gradient_checkpointing=True`, `sdpa=True`, multires noise + offset for richer LoRAs. Falls back to a hand-rolled TOML emitter when `tomli_w` isn't available. |
| `build_train_command(toml_path, train_data_dir)`                      | Build the `accelerate launch` argv. Passes `--num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no` to suppress launcher warnings, then `sdxl_train_network.py --config_file <toml> --train_data_dir <dir>`. |

### 4.11 Bootstrap check

| Function                              | Purpose                                                                                |
|---------------------------------------|----------------------------------------------------------------------------------------|
| `ensure_ready(log)`                   | Verify sd-scripts repo + sandboxed venv exist before training. Returns `True` if ready, `False` (with helpful log message pointing at `setup_venv.bat`) if not. Called at the start of `run_training_job`. |

### 4.12 Job orchestration

| Function                                                              | Purpose                                                                                                            |
|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `run_training_job(project_name, trigger_word, base_model_name, preset_name, caption_mode, image_paths, network_dim, network_alpha, learning_rate, num_epochs, repeats, resolution, batch_size, log)` | End-to-end. Validates project name + base model. Builds preset with optional UI overrides. Calls `ensure_ready`, `prepare_dataset`, autocaption (skipping if `.txt` files already exist), `inject_trigger_word`, `write_training_toml`, `build_train_command`, `_run_subprocess`. Copies the final `.safetensors` to `models/Lora/`. |

### 4.13 UI helpers

| Function                              | Purpose                                                                                |
|---------------------------------------|----------------------------------------------------------------------------------------|
| `_list_base_models()`                 | Enumerate `*.safetensors` + `*.ckpt` files under `models/Stable-diffusion/` for the base-model dropdown. Returns names sorted alphabetically. |
| `_job_log(s)`                         | Append a line to the shared `JOB["log_buf"]` ring used by all UI callbacks. Also writes to Python `logging` so it surfaces in Forge's console too. |

### 4.14 UI builder

| Function                              | Purpose                                                                                |
|---------------------------------------|----------------------------------------------------------------------------------------|
| `_build_tab()`                        | Builds the entire Gradio `Blocks` tree. Defines all input widgets (project name, trigger, model, preset, caption mode, image uploader, booru accordion, advanced accordion, caption editor accordion). Defines all inner handler functions (`_start`, `_tail`, `_do_booru_search`, `_do_tag_my_uploads`, `_do_refresh_prev`, `_do_use_prev`, `_do_refresh_captions`, `_do_load_caption`, `_do_save_caption`, `_do_refresh_prev`). Wires every button/dropdown to the right handler. Returns the root Blocks instance. |
| `_on_ui_tabs()`                       | Forge's `script_callbacks.on_ui_tabs` callback. Wraps `_build_tab` in a `try`/`except` so any construction error produces a fallback stub tab with the traceback visible (rather than the tab silently failing to appear). Returns `[(ui, "LoRA Trainer", "lora_trainer")]`. |

### 4.15 Internal nested handlers (defined inside `_build_tab`)

| Handler                       | Triggered by                              | Effect                                                                          |
|-------------------------------|-------------------------------------------|---------------------------------------------------------------------------------|
| `_start`                      | "Start training" button                   | Validates required fields, kicks off `run_training_job` in a background thread  |
| `_tail`                       | `gr.Timer` (every 2s) + Refresh log button| Returns the last 500 log lines as a string                                      |
| `_do_booru_search`            | "Search & download from Booru" button     | Calls `search_booru`, returns status text + updated trigger dropdown            |
| `_do_tag_my_uploads`          | "Tag MY uploads from this tag" button     | Calls `tag_my_uploads`, returns status text + updated trigger dropdown          |
| `_do_refresh_prev`            | "Refresh list" button (previously-downloaded) | Re-scans `downloads/`, updates the dropdown choices                          |
| `_do_use_prev`                | "Use this folder" button                  | Sets `JOB["searched_dir"]` to the selected folder                               |
| `_do_refresh_captions`        | "Refresh image list" (caption editor)     | Lists `*.png/jpg/jpeg/webp` with paired `.txt` in `JOB["searched_dir"]`         |
| `_do_load_caption`            | Caption-editor image dropdown change      | Loads the `.txt` content into the editor textbox + shows the image as preview  |
| `_do_save_caption`            | "Save caption" button                     | Writes the editor textbox back to the `.txt` file on disk                       |

---

## 5. Shared mutable state

A single module-level dict, `JOB`, holds runtime state that survives across Gradio callback invocations:

```python
JOB = {
    "running": False,         # True while a training thread is active
    "log_buf": [],            # Ring buffer of log lines (capped to last 500 via _tail)
    "searched_dir": None,     # Path to the active booru download folder (used by _start fallback)
}
```

Everything else is either a constant or a local variable inside `_build_tab`.

---

## 6. Data flow diagrams

### 6.1 Search & download → train

```
User clicks "Search & download"
  ↓
_do_booru_search()
  ↓
search_booru(tag, rating, count, trigger, official_only, solo_mode)
  ├─ _search_danbooru()  ──→ Danbooru API
  ├─ _search_gelbooru()  ──→ Gelbooru API  (if Danbooru under-delivered)
  ├─ for each result: _http_download() → downloads/<slug>/NNN.ext
  ├─ for each result: write .txt with _filter_tags(post.tags, solo_mode) + trigger
  └─ _compute_character_features(downloads/<slug>/) → promote features to caption fronts
  ↓
JOB["searched_dir"] = downloads/<slug>/

User clicks "Start training"
  ↓
_start() → spawns worker thread → run_training_job()
  ├─ ensure_ready()
  ├─ prepare_dataset()           projects/<name>/images/<repeats>_<class>/
  ├─ inject_trigger_word()       (idempotent — captions already have trigger)
  ├─ write_training_toml()       projects/<name>/config.toml
  ├─ build_train_command()
  ├─ _run_subprocess() with PYTHONIOENCODING=utf-8
  └─ shutil.copy2(final_safetensors → models/Lora/<name>.safetensors)
```

### 6.2 Tag MY uploads → train

```
User uploads images via gr.Files
User types booru tag (or auto-derives from "seele_(honkai_impact)")
User clicks "Tag MY uploads"
  ↓
_do_tag_my_uploads()
  ↓
tag_my_uploads(tag, image_paths, count, trigger, official_only)
  ├─ _search_danbooru(solo_mode="any")    → tag data only, no downloads
  ├─ _search_gelbooru(solo_mode="any")    → backup tag data
  ├─ Compute defining features (≥60% across booru sample, excluding STYLE_TAGS etc.)
  ├─ for each user upload:
  │    pick a different booru post from the sample (seeded random)
  │    write caption: <trigger>, <features...>, <that post's other tags>
  └─ Copy uploads + captions to downloads/<slug>_custom/
  ↓
JOB["searched_dir"] = downloads/<slug>_custom/

User clicks "Start training" → same path as 6.1
```

### 6.3 Manual caption editing

```
User opens "Edit individual captions" accordion
  ↓
clicks "Refresh image list" → _do_refresh_captions() → dropdown populated
  ↓
selects image from dropdown → _do_load_caption() → preview + textbox loaded
  ↓
user edits caption text → clicks "Save caption" → _do_save_caption() → .txt written
  ↓
(repeat per image)
  ↓
User clicks "Start training" → uses the edited captions
```

---

## 7. Critical kohya-correctness notes

These are non-obvious requirements that kohya enforces and which the trainer handles. If you fork or tweak the TOML generator, keep these:

| Setting                          | Requirement                                                                       |
|----------------------------------|-----------------------------------------------------------------------------------|
| `resolution`                     | Must be a **string** like `"1024,1024"` (kohya does `.split(",")` on it)          |
| `caption_extension`              | Default is `.caption`; must set `.txt` to use the captions we write               |
| `network_train_unet_only`        | Must be `True` when `cache_text_encoder_outputs=True` (kohya asserts this)        |
| `no_half_vae`                    | Must be `True` for SDXL or VAE produces NaN latents on some checkpoints           |
| accelerate `--mixed_precision`   | Must be passed on the launch command, not just in the TOML                        |
| Subprocess env                   | `PYTHONIOENCODING=utf-8` to handle kohya's Japanese log strings on Windows        |
| WD14 tagger                      | `--onnx` flag required because we install `onnxruntime-gpu`, not `tensorflow`     |

---

## 8. Files you can safely tweak

| File                              | When to edit                                                                  |
|-----------------------------------|-------------------------------------------------------------------------------|
| `PRESETS` dict at top of lora_trainer.py | Tune hyperparameters per preset                                        |
| `GENERIC_TAGS`, `STYLE_TAGS`, `MULTI_INDICATORS`, `SOLO_INDICATORS` | Add/remove tags from the filter sets         |
| `CAPTION_BLACKLIST`               | Add tags you never want in any caption                                         |
| `min_frequency` arg in feature detection | Change the 60% threshold for "defining features"                       |
| `setup_bootstrap.py` torch version | Bump `TORCH_PACKAGES` if you want a newer torch                              |

Don't edit:
- The TOML generator schema (kohya's expected types matter)
- The `caption_extension` value (must match what the booru/WD14/BLIP paths produce)
- `STILL_IMG_EXTS` (removing `webp` would break newer Danbooru uploads)
