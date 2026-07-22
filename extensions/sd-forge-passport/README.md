# sd-forge-passport

"Passports" tab — one saved identity per character: canonical tags + LoRA & weight + trigger words (✨ auto-filled from the Civitai helper's card data) + reference image & strength + character-specific negatives.

Passports live as plain `json`+`png` pairs in `<install>\passports\`, so other tools read them with zero coupling:

- **Characters tab** — per-character 🪪 passport dropdowns; Insert composes `trigger, tags, <lora:name:weight>` into that character's line
- **Auto Pilot** — 🪪 passport row inserts the fragment into the prompt

Build a character once, use them forever.
