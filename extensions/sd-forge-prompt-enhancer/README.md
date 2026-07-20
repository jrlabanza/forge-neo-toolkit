# sd-forge-prompt-enhancer

A **guided prompt builder + reference-image tagger** for Illustrious-XL and
NoobAI XL on Forge. You either pick from human-readable dropdowns (Pose:
"sitting", Shot: "upper body", Style: "soft watercolor") and let the extension
translate every choice into the right booru tags — or you upload a reference
image and we reverse-engineer the tags directly.

## How to use

1. Open the **Prompt Enhancer** tab in Forge.
2. (Optional) Pick a **Scenario template** at the top — "Beach vacation",
   "School day", "Cyberpunk night" — and click *Apply scenario*. Every section
   below fills with a sensible starting point.
3. Tweak any section: subject, pose, framing, outfit, scene, mood, art style.
4. Click **Build prompt →**.
5. Click **Send to txt2img** (or img2img) to paste both prompts into that tab.

Or for **1:1 image replication**:

1. Open the **🖼️ Image to Prompt** accordion at the top.
2. Upload a reference image, set thresholds.
3. Click **🔍 Analyze image** — first use downloads the WD14 model (~360 MB).
4. Click **🎯 Replicate (send to txt2img)** — the extracted tags + quality
   boilerplate are pushed straight into the txt2img tab, ready to generate.

## What's inside

### Image-to-Prompt (WD14 ViT-v3 tagger)

- **WD14 ViT-v3** by SmilingWolf — same Danbooru tagger family that trained
  Illustrious / NoobAI, so its output drops straight into a 1:1 replication
  prompt.
- Lazy-downloads the ONNX model + tag list on first use (~360 MB total,
  cached forever after).
- Tag-confidence thresholds for general tags and character tags.
- Outputs: detected character(s), general tags (composition / style /
  details), and the Danbooru content rating.
- Buttons: insert into your *Extra subject details* field, or replicate as
  a full prompt and send straight to txt2img.

### 11 builder sections (mostly dropdowns)

| # | Section | Approximate option count |
|---|---|---|
| 1 | Subject / Character | 17 subjects + per-character traits + force-female-only + yuri |
| 2 | Pose & Action | 64 poses · 66 actions |
| 3 | Framing | 17 shot types · 8 camera angles |
| 4 | Outfit | 91 presets (safe + [NSFW]) |
| 5 | Scene / Time / Lighting | 81 locations · 10 times · 21 lighting moods |
| 6 | Expression & Mood | 47 expressions (safe + [NSFW]) |
| 7 | Art style preset | 33 visual style presets |
| 8 | Quality, Negative, Booru meta | Best/Standard/Minimal tiers · year/source/rating |
| 9 | Body / Physique | 26 body types (safe + [NSFW]) |
| 9b | Artist style | Searchable dropdown over **33,719 artists** + Danbooru preview + strength slider |
| 10 | Intimacy / NSFW level | 10 tiers from safe to explicit / fetish |
| 11 | Custom extras | Free-text positive / negative additions |

### Scenario templates (one-click presets)

39 ready-made scenes including Character portrait, School day, Beach vacation,
Combat scene, Magical girl, Idol performance, Cafe outing, Library study,
Karaoke night, Sushi date, Sunset rooftop, Camping fireside, New Year fireworks,
plus 7 [NSFW] scenarios.

### Canonical Illustrious tag order (what we emit)

```
subject -> character_name -> per-char traits -> outfit -> pose/action
       -> framing -> location -> time/lighting -> mood -> body
       -> artist -> style -> intimacy -> year/source/rating -> quality
       -> custom extras
```

## Supported model families

- **Illustrious-XL** (and any merge with `illustrious` in the filename)
- **NoobAI XL** (v-pred and eps)

Detection reads `shared.opts.sd_model_checkpoint`. Override in the dropdown
if auto-detect picks wrong.

## Install

Already installed at `extensions/sd-forge-prompt-enhancer`. Restart Forge
once after first install — `install.py` will install `onnxruntime` (GPU
build if available, CPU fallback) so the WD14 tagger can run.

## Files

- `install.py` — installs onnxruntime + Pillow if missing.
- `artists.txt` — bundled list of 33,719 Illustrious / NoobAI compatible
  Danbooru artist tags (from
  [ThetaCursed/Illustrious-NoobAI-Style-Explorer](https://github.com/ThetaCursed/Illustrious-NoobAI-Style-Explorer)).
- `scripts/prompt_enhancer.py` — builder UI + tag dictionaries + scenario
  templates + WD14 tagger + Danbooru preview + paste-params wiring.
- `wd14_model/` — lazy-downloaded on first analyze (gitignored).

## Credits

- **SmilingWolf** — WD14 ViT-v3 tagger model.
- **OnomaAI Research** — Illustrious-XL.
- **Laxhar Lab** — NoobAI-XL.
- **ThetaCursed** — bundled artist list / style references.
