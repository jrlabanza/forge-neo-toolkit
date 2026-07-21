# Forge Neo — Optimization Notes & Workflow Cheat Sheet

_System audited 2026-05-26 for Jorelle's setup (RTX 3070 8 GB / 40 GB RAM)._
_Follow-up session 2026-07-20 — see section 0 below._

---

## 0. Changes applied 2026-07-20

### New extensions (all self-contained; delete a folder to remove that feature)

| Extension | Where | What it does |
|---|---|---|
| `sd-forge-gallery` | **Gallery** tab | Browse all output folders newest-first, full-text search over embedded prompts/parameters, date filter, favorites (★), click any image → params + Send to txt2img/img2img. Read-only. |
| `sd-forge-model-presets` | **Model Presets** tab | Auto-detects the loaded checkpoint's family and one-click loads the right sampler/steps/CFG/size/clip-skip + quality-tag dialect + negative prompt (Illustrious vs Pony `score_9…` vs Animagine vs SD1.5 `RAW photo…`). The load button REPLACES the txt2img prompt+settings. |
| `sd-forge-styles-manager` | **Styles** tab | Search/edit/add/delete your 398 styles.csv entries; auto-backup before every write (`_attic\styles-backups\`, newest 20); shows a sample output that used the style; native dropdown refreshes after save. |
| `sd-forge-civitai-helper` | **Civitai** tab | Hash-matches your LoRAs to Civitai → writes trigger words into each card ("activation text" auto-inserts on click), downloads preview thumbnails, checks for newer versions. Merge-safe: never overwrites your edits or existing previews. |
| `sd-forge-job-runner` | **Job Runner** tab | Queue (stack infotext jobs, run unattended, survives restarts) + Batch re-process (upscale / img2img-refine / +ADetailer over old outputs or Gallery favorites) + Test cards (same prompt+seed across checkpoints → labeled contact sheet). Uses the local API. |
| `sd-forge-notify` | 🔔 accordion in txt2img/img2img | Windows toast and/or Discord webhook when a generation ran longer than N seconds. |
| `sd-forge-tag-translator` | **Describe** tab | Write a plain-English paragraph → correctly ordered Illustrious tag prompt (matched against the 140k-tag Danbooru dictionary, offline). Reports unmapped words, suggests tags for what the description left open (lighting, framing, mood…) + "what next" scene ideas, one-click send with quality tags + settings. Also contains the **🤖 AI prompt writer**: DanTagGen (400M local model, ~800 MB download on first click, CPU by default) that invents fitting detail tags and merges them in canonical order. |
| `sd-forge-config-backup` | (no UI) | Snapshots config.json / ui-config.json / styles.csv on every launch to `_attic\config-autobackups\`, keeps newest 10. |

### Launch args change
`--api` added to both launchers — enables the local API on 127.0.0.1 only (not exposed to your network). Required by the Job Runner tab. If you launch through Stability Matrix instead of the bats, add `--api` there too.

### 2026-07-21 — upstream update applied + standalone launch fixes
- Updated to upstream neo `1fe0cb44` (75 commits: Krea2/PiD/Qwen3-VL support, faster LoRA loading, int8, hires/refiner work). Rollback: `git reset --hard backup/pre-update-20260720`.
- `PortableGit\` now lives in this folder (official git-for-windows 2.55, checksum-verified). All bats + Forge's GitPython use it automatically.
- Launchers now set `PYTHON` to the venv and `GIT` to PortableGit — this machine has no system python/git, which used to make standalone (non-Stability-Matrix) launches fail.
- `fix_and_launch.bat` — repair launcher: kills stale python processes that lock venv DLLs, then starts normally. Use it if a launch ever dies mid-update.
- **Local core patch**: `modules/images.py` wraps `import pillow_jxl` in try/except — Windows Application Control blocks the new JXL plugin DLL on this machine. Only JPEG-XL format support is affected. If a future update conflicts on that file, re-apply the guard or accept upstream and re-patch.
- Console notes on startup: the "PyTorch 2.10 outdated" banner is informational (our torch/sage combo is intentional); adetailer suggests migrating to ADetailer-Neo eventually — current fork works.

### New launcher: `webui-user-faststart.bat`
Same GPU args as `webui-user.bat` plus `--skip-prepare-environment --skip-torch-cuda-test --skip-version-check`. Saves ~15–40 s per start. Use the normal `webui-user.bat` once after installing/updating extensions or Forge itself. NOTE: first launch after adding the two new extensions must be via `webui-user.bat`.

### Housekeeping
- 9 crash-era `ui-config.json.*` backups, `webui-user.bat.pre-opt-bak` and `extensions_broken_backups\` moved to `_attic\` (see its README; safe to delete whenever).
- `config.json` and `ui-config.json` validated OK; ADetailer eye-fix defaults still in place.
- Prompt Enhancer: removed the two child-subject presets from the Subject dropdown; fixed a leftover debug ternary in the Actions list.

### Disk space you can reclaim (~21–28 GB, your call — nothing deleted)
Superseded checkpoint versions in `models\Stable-diffusion\`:
- `waiNSFWIllustrious_v140.safetensors` (6.5 GB) — superseded by v170
- `waiNSFWIllustrious_v150.safetensors` (6.5 GB) — superseded by v170
- `waiIllustriousSDXL_v160.safetensors` (6.5 GB) — superseded by v170
- `animagineXLV31_v31.safetensors` (6.5 GB) — superseded by animagineXL40 (optional; 3.1 has a different look some prefer)

### Optional VRAM lever (not applied)
Settings → `forge_unet_storage_dtype_xl` can store the UNet in fp8 — roughly halves UNet VRAM at a small quality cost. Worth trying only if you start hitting OOM with multi-ControlNet + ADetailer stacks; otherwise leave on Automatic.

This file documents every change applied to your Forge Neo install, the reasoning behind each, and ready-to-use workflows for your four main use cases.

---

## 1. What changed in this audit

### Config (`config.json`)

| Setting | Before | After | Why |
|---|---|---|---|
| `setting_allocated_vram` | `1.0` (100% to model weights → 0 MB reserved) | `0.65` (65% to weights, ~2.8 GB reserved for sampling + IPAdapter) | The previous value left almost no headroom for sampling activations, which is what was causing the "GPU free 660 MB" warnings and contributed to NaN crashes when IPAdapter loaded the extra CLIP-ViT-H encoder. |
| `cross_attention_optimization` | `Automatic` | `Automatic` (unchanged — Forge Neo locks this) | Forge Neo doesn't expose this for user editing; it auto-picks SDP on modern GPUs. |
| `sd_vae_decode_method` | `Full` | `Full` (unchanged) | The only other option in Neo is TAESD which degrades quality. Stay on Full. |

### Launch args (`webui-user.bat`)

```
set COMMANDLINE_ARGS=--no-half-vae --pin-shared-memory --cuda-malloc
```

| Flag | Why |
|---|---|
| `--no-half-vae` | Runs the VAE in FP32 instead of FP16. Prevents NaN during the final VAE decode step. Tiny speed cost. |
| `--pin-shared-memory` | Makes CPU↔GPU model offload fast instead of stalling. With your 40 GB RAM there's lots of room for pinned buffers. |
| `--cuda-malloc` | Uses CUDA's native allocator instead of PyTorch's default. Reduces fragmentation when extensions (IPAdapter, ControlNet, ADetailer) load/unload models during a run. |

> SageAttention 2 stays **enabled** for normal generations (full speed). The `sd-forge-ipadapter-compat` extension automatically swaps it for PyTorch attention only during IPAdapter generations, so you don't pay the speed cost when not using IPAdapter.

### ADetailer two-pass eye fix (pre-configured as default)

The `ui-config.json` now ships with two ADetailer tabs preconfigured for every txt2img generation:

| Tab | Detector | Denoise | Padding | Purpose |
|---|---|---|---|---|
| 1 | `face_yolov8n.pt` | 0.4 | 32 px | Re-render whole face at higher quality (catches general eye structure) |
| 2 | `mediapipe_face_mesh_eyes_only` | 0.35 | 64 px | Eye-only laser pass with `(perfect eyes, detailed iris, sharp pupils, clean color separation:1.2)` prompt and `(blurry eyes, washed out eyes, color bleeding, asymmetric eyes:1.2)` negative |

These eliminate the SDXL anime eye-bleeding problem on Illustrious / Pony / NoobAI. Cost: roughly +20–30 seconds per generation. To disable temporarily, untick "Enable this tab" on either ADetailer panel.

> **Warning:** clicking Forge's "Save UI defaults" button in Settings will overwrite these values with whatever your UI currently shows. If you do that, the eye-fix pre-config is gone until I re-apply it.

### New preprocessor: "IP-Adapter Face (Auto-Crop)"

Added by the `sd-forge-ipadapter-compat` extension. Appears in the ControlNet panel's Preprocessor dropdown when you select the IP-Adapter radio.

**What it does:** Runs InsightFace on your reference image, finds the largest face, crops to that face with ~35% padding, then encodes the crop with CLIP-ViT-H. Forge logs e.g. `face-crop: cropped to 312x312 from 1024x1024` so you can see it firing.

**Why it matters:** `ip-adapter-plus-face_sdxl_vit-h` is trained on tight face crops. If you give it a full-body image with a small face, the model wastes capacity on the background. Auto-cropping puts the face at the center of the embedding and dramatically improves identity transfer.

**How to use:**

- ControlNet Unit 0
- Drop a reference image (face must be visible)
- ✅ Enable, IP-Adapter radio
- Preprocessor: **IP-Adapter Face (Auto-Crop)** ← new entry
- Model: **ip-adapter-plus-face_sdxl_vit-h**
- Control Weight: 0.5–0.7

### Custom extension: `sd-forge-ipadapter-compat`

I built this as a proper Forge extension at `extensions/sd-forge-ipadapter-compat/` so it survives Forge updates and is easy to remove. It does two things at runtime:

1. **Patches `ControlNet.get_control` / `T2IAdapter.get_control`** to early-return when `cond_hint_original` is a dict (IPAdapter case). Without this, IPAdapter crashes with `AttributeError: 'dict' object has no attribute 'shape'`.

2. **Auto-swaps `attention_function` from Sage → PyTorch SDP** during IPAdapter generations, then restores Sage afterward. Without this, Sage + IPAdapter produces black-image NaN latents on Illustrious-family checkpoints.

Both fixes are fully automatic — no UI, no config. To temporarily disable, rename the extension folder.

### Files moved into place

- 6 SDXL IPAdapter models → `F:\Data\Models\IpAdapter\`
- 1 FaceID LoRA → `F:\Data\Models\Lora\`
- InsightFace antelopev2 (5 ONNX files) → `models\insightface\models\antelopev2\` — required for any FaceID workflow

### Extensions installed

- **adetailer** — auto-detects and inpaints faces/hands at the end of every generation. The single biggest quality-of-life win for anime + portrait work. Configure once, forget it exists.
- **a1111-sd-webui-tagcomplete** — autocomplete for Danbooru/E621 tags as you type. Speeds up anime prompting massively.

### Extensions already installed (kept)

- `sd-forge-controlnet`, `sd-forge-ipadapter` (built-in, working)
- `sd-dynamic-prompts` — wildcards + random variations
- `sd-forge-couple` + `sd-webui-regional-prompter` — regional prompting (different prompts for different parts of the image)
- `WAI-NSFW-illustrious-character-select` — your Chinese character picker for Illustrious

---

## 2. Still on your wishlist (auto-downloads when needed)

Two pieces will be auto-fetched by Forge the first time you select the matching preprocessor; no action required.

1. **CLIP-ViT-bigG.safetensors** (~3.7 GB) — only used by `ip-adapter_sdxl.safetensors` (the non-vit-h variant). Auto-downloads to `models\ControlNetPreprocessor\` when you pick the `CLIP-ViT-bigG (IPAdapter)` preprocessor. If you don't plan to use that specific model, you can skip it forever.
2. You already have `CLIP-ViT-H-14.safetensors` for everything else. ✅

---

## 3. The black-image (NaN) issue — likely fix

Your generation produced a black image because the IPAdapter Plus + Illustrious SDXL combination is overflowing in FP16. The launch-arg fixes above will help, but the cleanest fix is:

**Lower the IPAdapter Control Weight from 1.0 → 0.5.**

Plus variants are roughly 2× stronger than base variants, so weight 0.5 on `ip-adapter-plus_sdxl_vit-h` is similar in strength to weight 1.0 on `ip-adapter_sdxl_vit-h`. If 0.5 still produces black, try `ip-adapter_sdxl_vit-h` (non-Plus) at weight 1.0.

---

## 4. Workflows for your four use cases

### Use case A — Anime / illustration (your current main flow)

**Setup:**
- Checkpoint: `waiIllustriousSDXL_v170` or any Illustrious / Pony / NoobAI / Animagine
- Sampler: Euler a, 28–32 steps, CFG 5–6, Clip Skip 2
- Size: 832×1216 (portrait) or 1024×1024 (square) — these are SDXL's "native" anime sizes and run faster than 1024×1344
- **Enable ADetailer** (new!) → in the ADetailer panel, pick `face_yolov8n.pt` as the first detector. Leave defaults. Faces will be auto-fixed every generation.
- Hires fix: optional. If used, set to `R-ESRGAN 4x+ Anime6B`, denoise 0.35, upscale 1.5×.

### Use case B — Style transfer / fan art (the IPAdapter workflow)

**Setup:**
- Same checkpoint as use case A (Illustrious works great)
- ControlNet Unit 0:
  - Drop your style-reference image
  - ✅ Enable
  - Preprocessor: `CLIP-ViT-H (IPAdapter)`
  - Model: `ip-adapter-plus_sdxl_vit-h` (Plus = stronger style transfer)
  - **Control Weight: 0.5** (do NOT use 1.0 with Plus on Illustrious — NaN)
  - Control Mode: Balanced
  - Guidance End: 0.8 (stop IPAdapter slightly early; lets the model "free-style" the last 20% for detail)
- Same Sampler / Steps / Size as use case A
- Generate. Your output adopts the reference image's style/colour/composition without copying it.

**Tip:** for character-style transfer (not just art style), use `ip-adapter-plus-face_sdxl_vit-h` instead — it focuses the attention on the subject's face/features.

### Use case C-easy — NovelAI-style "Reference Image" accordion in txt2img (Recommended path)

Built as `sd-forge-reference-image` extension. Adds an accordion **inside the txt2img tab** (between Hi-res Fix and ControlNet), so the reference flow lives side-by-side with your prompt — not in a separate tab. This is the user-friendly path that hides all the preprocessor/model footguns.

**Two modes, picked via radio button:**

| Mode | What it does | Behind the scenes | When to use |
|---|---|---|---|
| **Vibe** | Style / mood / composition transfer | Single ControlNet unit with `CLIP-ViT-H (IPAdapter)` + `ip-adapter-plus_sdxl_vit-h` | "I want this aesthetic / colour palette / vibe, but a different subject." |
| **Precise** | Character / face cloning | Dual ControlNet units: `InstantID_ControlNet` (landmarks) + `InstantID_ip-adapter` (identity), with InsightFace preprocessor | "I want this exact character/person in a new scene." |

**Workflow:**
1. Open the **Reference Image** tab
2. Drop your reference (drag or click to upload)
3. Pick Vibe or Precise
4. Set Reference strength: 0.5–0.7 for Vibe, 0.7–1.0 for Precise
5. Type prompt + negative prompt
6. Click **Generate with reference**

No preprocessor dropdown, no model dropdown, no manual unit-1 setup. The extension wires everything correctly behind the scenes using Forge's existing ControlNet pipeline.

### Use case C-alt — Character face cloning (InstantID, SDXL state-of-the-art)

This is the most accurate way to lock in a specific character's face across generations. Much more reliable than IPAdapter Plus Face on Illustrious. Verified working at ~37 sec/gen on your 3070.

**One-time files installed at `models/controlnet/`:**
- `Instant-ID_ControlNet.safetensors` (2.4 GB) — the ControlNet half (face landmarks)
- `Instant-ID_ip-adapter.bin` (1.6 GB) — the IPAdapter half (face identity)
- Plus the InsightFace `antelopev2` we already installed

**Setup (dual-unit ControlNet, repeat each session unless you Save UI defaults):**
- ControlNet Unit 0:
  - Drop a clear face reference image
  - ✅ Enable, **Instant-ID** radio
  - Preprocessor: `InsightFace (InstantID)`
  - Model: `Instant-ID_ControlNet`
  - Control Weight: **0.8**
- ControlNet Unit 1:
  - Drop the SAME face reference image
  - ✅ Enable, **Instant-ID** radio
  - Preprocessor: `InsightFace (InstantID)`
  - Model: `Instant-ID_ip-adapter`
  - Control Weight: **0.7**
- Generate

The two units work as a pair: Unit 0 enforces the face landmark structure (so the face is in the right place / right pose / right shape), Unit 1 enforces the identity (so it actually looks like that person).

### Use case C — Realistic portraits / people

**Setup:**
- Checkpoint: `cyberrealisticPony_v110` (your current realistic Pony) or `realisticVisionV51_v51VAE` (SD1.5 but excellent for faces)
- Sampler: DPM++ 2M Karras, 25–30 steps, CFG 4–5
- Size: 832×1216
- **ADetailer:** add a SECOND tab inside ADetailer for hands using `hand_yolov8n.pt`. Realistic photos break on hands more than anywhere else.
- **For face cloning** (your own face or any reference person):
  - ControlNet Unit 0 → reference photo with the person
  - Preprocessor: `InsightFace+CLIP-H (IPAdapter)`
  - Model: `ip-adapter-faceid-plusv2_sdxl`  ← uses InsightFace (now installed)
  - Weight: 0.7 (start here; lower if face distorts the body, higher if face identity is too weak)
  - The `ip-adapter-faceid-plusv2_sdxl_lora.safetensors` LoRA is auto-applied — no manual loading needed

### Use case D — Concept / environment / mixed

**Setup:**
- Checkpoint: any SDXL, but NoobAI-XL is a great generalist
- Sampler: DPM++ 2M SDE Karras, 30 steps, CFG 6
- Size: 1216×832 (landscape) for environments, 1024×1024 for concept props
- Use ControlNet with `diffusion_pytorch_model_promax.safetensors` (the ControlNet Union you already have) for composition control — it handles canny/depth/pose/scribble in a single model
- Regional prompter (already installed) helps for "foreground X / background Y" compositions

---

## 5. VRAM management for 8 GB

Your card is sufficient for everything above, but it's tight. Quick rules:

- **Resolution:** prefer 832×1216 or 1024×1024 for initial generation. Use Hires Fix to scale up, not direct large generations.
- **ControlNet units:** try not to enable more than 2 simultaneously with SDXL. Each one adds VRAM cost.
- **IPAdapter + ControlNet Pose** is the most demanding combo (CLIP-ViT-H + ControlNet model + UNet). Drop other units when using both together.
- **If you see "Encountered NaN" again:** the first fix is always "lower the IPAdapter weight by 0.2". Second: switch from Plus → base. Third: lower resolution.
- **Watch the console** during your first runs with the new config — you should now see "Manually Reserving 2867 MB VRAM" near startup, and "free memory" should hover around 2.5–3 GB during sampling instead of 660 MB.

---

## 6. ADetailer quick-config (one-time)

In txt2img, scroll to the ADetailer panel:

1. **Tab 1** — `Ad-model: face_yolov8n.pt`, Detection confidence 0.3, Mask blur 4, Inpaint denoising 0.4
2. **Tab 2** (click "+" to add) — `Ad-model: hand_yolov8n.pt`, same settings
3. (Optional Tab 3) — `person_yolov8n-seg.pt` for full-body cleanup at low denoise (0.2)

Tick the master "Enable ADetailer" checkbox at the top of the panel. Leave it on forever — it adds ~10 seconds per generation but eliminates 90% of face/hand artefacts.

---

## 7. Tag autocomplete quick-config (one-time)

After restart, open Settings tab → Tag Autocomplete:
- **Tag source:** `Danbooru` (for anime checkpoints like Illustrious/Pony)
- **Translation file:** none (leave blank)
- Apply settings + reload UI

Now in any prompt box, start typing a tag and hit Tab to autocomplete. Massive speedup for anime prompting.

---

## 8. If things break

- Black image → lower IPAdapter weight, check console for "Encountered NaN"
- OOM (out of memory) → drop resolution, disable extra ControlNet units, set GPU Weights slider to 0.55
- New extension fails to load → check the Stability Matrix console for the error, usually a missing pip dependency. Forge will auto-install on next restart.
- Want to roll back any change → all original values are listed in section 1 above.
