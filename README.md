# forge-neo-toolkit

Custom extensions, launchers and docs for my Stable Diffusion WebUI **Forge Neo** install
(RTX 3070 8 GB · SDXL/Illustrious workflows · built with Claude, 2026).

## Extensions

| Extension | UI | Purpose |
|---|---|---|
| `sd-forge-tag-translator` | **Describe** tab | Plain-English paragraph → correctly ordered Illustrious tag prompt (offline matching against the Danbooru tag dictionary) + 🤖 local AI prompt writer (DanTagGen 400M) + scene suggestions |
| `sd-forge-gallery` | **Gallery** tab | Browse/search all outputs by embedded prompt text, date filter, favorites, send params back to txt2img |
| `sd-forge-model-presets` | **Model Presets** tab | Per-family recommended settings + quality-tag dialects (Illustrious/NoobAI/Animagine/Pony/SD1.5) |
| `sd-forge-styles-manager` | **Styles** tab | Search/edit/add/delete styles.csv with automatic backups; sample image per style |
| `sd-forge-civitai-helper` | **Civitai** tab | Hash-match LoRAs → trigger words into cards, preview thumbnails, update checks |
| `sd-forge-job-runner` | **Job Runner** tab | Generation queue, batch re-process (upscale/refine/ADetailer) of old outputs, checkpoint test-card contact sheets (needs `--api`) |
| `sd-forge-notify` | accordion in txt2img/img2img | Windows toast / Discord webhook when long generations finish |
| `sd-forge-config-backup` | none | Snapshots config.json / ui-config.json / styles.csv on every launch, keeps 10 |
| `sd-forge-prompt-enhancer` | **Prompt Enhancer** tab | Guided dropdown prompt builder for Illustrious + WD14 image analyzer + metadata tools |
| `sd-forge-lora-trainer` | **LoRA Trainer** tab | Drop images + trigger word → kohya sd-scripts training (sd-scripts + its venv are fetched at first run, not stored here) |
| `sd-forge-reference-image` | accordion in txt2img | NovelAI-style "Vibe / Precise" reference image via IPAdapter/InstantID |
| `sd-forge-ipadapter-compat` | none | Runtime patches: IPAdapter dict crash fix + Sage↔SDP attention auto-swap |
| `sd-forge-naitosd` | **NAI Converter** tab | NovelAI PNG metadata → A1111/Illustrious format |

## Install on a fresh Forge Neo

1. Copy each folder from `extensions/` into your Forge `extensions/` directory
2. Copy `launchers/webui-user.bat` (and the faststart variant) into the Forge root
3. Launch once with `webui-user.bat` (installs any extension deps, registers tabs)
4. Optional: read `docs/FORGE_NEO_OPTIMIZATION_NOTES.md` for the full setup/tuning history

Requirements: Forge Neo (gradio 4.40 era), `a1111-sd-webui-tagcomplete` for the Describe tab's dictionary, `--api` flag for the Job Runner (both launchers set it).

## What is deliberately NOT in this repo

Personal/runtime data: caches, favorites, queue state, webhook URLs, prompt history,
downloaded models (WD14, DanTagGen, kohya sd-scripts). Fresh installs regenerate all of it.

## Updating this repo

Double-click `sync_and_push.bat` — it re-copies the live files from the install,
commits, and pushes.
