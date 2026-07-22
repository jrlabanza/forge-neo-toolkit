# sd-forge-autopilot

"Auto Pilot" tab — generate → judge → retry until you have the keepers you asked for.

Every image passes four local quality gates:
1. **Blank/black frame** detection (catches NaN collapses)
2. **Sharpness** (Laplacian variance — catches blur/mush)
3. **Face gate** (optional): a clear, sharp face must be detected (YOLO on CPU, uses the ADetailer face model already installed)
4. **Anime aesthetic score** (cafeai/cafe_aesthetic ViT, CPU) — must beat your threshold

Only passing images are kept (saved to `output/autopilot/<date>/` with full metadata + score in the filename); everything else is discarded (or kept in `/rejected` if toggled). Fresh seeds are rolled batch after batch until the keeper quota is met or the safety cap stops it honestly. Ends with a pass-rate report and the best seeds for reuse.

**Anatomy gates** (optional, on by default): rejects extra hands (>2 detected), extra people in solo prompts, and broken head/figure proportions. A **Deep tag scan** (off by default, slower) additionally rejects images the WD14 tagger flags as `bad_hands` / `extra_digits` / `bad_anatomy` / `deformed` etc.

**LoRA selector**: pick any of your LoRAs (multi-select, weight slider) — they're appended as `<lora:name:weight>`, and if the Civitai helper has fetched trigger words, those are auto-added too.

- All judging on CPU — zero VRAM stolen from generation
- First-ever run downloads the ~350 MB judge model (cached forever)
- Needs `--api` (webui-user.bat sets it)
- Threshold slider = your strictness dial: 0.72 balanced, 0.85+ brutal
