# sd-forge-tag-translator

"Describe" tab — write a plain-English paragraph (character, outfit, action, place), get a correctly ordered Illustrious prompt.

Example in → out:

> A young woman with long silver hair and blue eyes wearing a school uniform with a pleated skirt. She is sitting by the window of a classroom at sunset, smiling softly at the viewer while holding a book.

> `1girl, solo, grey_hair, blue_eyes, long_hair, school_uniform, pleated_skirt, sitting, holding_book, looking_at_viewer, window, classroom, sunset, smile` + quality tags + negative + recommended settings

How: greedy phrase-matching against Tag Autocomplete's 140k-tag Danbooru dictionary (incl. aliases + a synonym layer for renamed tags like silver→grey hair), a skip-gram pass for stacked adjectives ("long silver hair" → long_hair + silver/grey_hair), subject inference from pronouns, then bucket-sorting into canonical Illustrious order. Fully offline, no AI service.

Also shows what it couldn't map (leftovers are often fine to keep — Illustrious understands some English), and a "Scene ideas" box that suggests tags for whatever your description left open (framing, lighting, mood…) plus "what next" scene variations. One click sends everything to txt2img.

Requires the a1111-sd-webui-tagcomplete extension (installed) for its danbooru.csv.

## 🤖 AI prompt writer (in the same tab)

The Describe tab also has a local AI that writes prompts for you: **DanTagGen-delta-rev2** (400M-param LLaMA trained on 7.2M Danbooru posts, CC-BY-SA-4.0). Put a few tags in the prompt box (or translate a paragraph first), pick a content rating and how much to add, click **AI-complete my prompt** — it invents fitting detail tags (outfit details, background elements, lighting, composition) and merges them in canonical order.

- Fully local: first click downloads ~800 MB from HuggingFace, cached forever after
- CPU mode (default): a few seconds per run, zero VRAM taken from SDXL
- GPU mode: near-instant, uses ~0.8 GB of the 8 GB
- "Unload AI model" frees the memory when you're done
- Uses the transformers+torch already in the venv — no new dependencies
