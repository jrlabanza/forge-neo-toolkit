# sd-forge-director

"Director" tab — NovelAI-style Director Tools, local and unlimited. Drop any image:

| Tool | What it does |
|---|---|
| ✨ Enhance | 1.5× upscale + low-denoise refine using the image's own embedded prompt |
| 🎲 Variations | N re-rolls at chosen strength (0.15 subtle → 0.7 loose) |
| 😊 Change emotion | Redraws ONLY the face with the chosen expression (ADetailer-targeted inpaint) — 10 emotions |
| 🎨 Recolor | Palette/mood presets: warm sunset, cool night, pastel, vivid, monochrome, autumn |
| 🪄 Remove background | Transparent cutout (rembg, auto-installed on next normal launch) |

Results → `output/director/<date>/` with metadata. Needs `--api` (launchers set it). Images generated anywhere work — txt2img, Auto Pilot keepers, old Gallery images; embedded prompts are reused automatically.
