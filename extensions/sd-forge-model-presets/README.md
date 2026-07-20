# sd-forge-model-presets

"Model Presets" tab: one-click recommended settings per model family, matched to the checkpoints installed on this machine.

Each family = sampler + steps + CFG + size + clip skip + quality-tag dialect + negative prompt:

| Family | Sampler | Size | Quality dialect |
|---|---|---|---|
| Illustrious / WAI | Euler a, 28, CFG 5.5 | 832×1216 | masterpiece, best quality… |
| NoobAI-XL | Euler a, 28, CFG 5 | 832×1216 | …newest, absurdres |
| Animagine 3.1/4.0 | Euler a, 27, CFG 5.5 | 832×1216 | …high score, great score |
| CyberRealistic Pony | DPM++ 2M Karras, 30, CFG 5 | 832×1216 | score_9, score_8_up… |
| SD1.5 realistic | DPM++ SDE Karras, 28, CFG 4.5 | **512×768** | RAW photo, 8k uhd… |

## Usage
1. Open **Model Presets** → **Detect** (reads the loaded checkpoint) or pick a family manually
2. Optionally type your subject — it gets composed in front of the quality tags
3. Everything is editable in place
4. **Load into txt2img** — note: this replaces the current prompt and settings in that tab

No files are written; presets are code-defined.
