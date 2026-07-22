"""
sd-forge-tag-translator
=======================

"Describe" tab: write a normal paragraph — who the character is, what they
wear, what they're doing, where — and get a properly formatted
Illustrious-XL prompt back.

How it works (fully offline, no AI service):
- Your text is matched against the 140k-tag Danbooru dictionary that ships
  with the Tag Autocomplete extension (tags + all their aliases), using
  greedy longest-phrase matching. "wearing a school uniform and looking at
  the viewer" -> school_uniform, looking_at_viewer.
- Matched tags are sorted into Illustrious's canonical order:
  subject -> character -> series -> body traits -> outfit -> pose/action ->
  framing -> location -> time/lighting -> mood -> details -> quality.
- Anything it couldn't match is listed so you can decide (Illustrious
  understands some plain English, so leftovers are often fine to keep).
- The Scene ideas box suggests tags for the categories your description
  left empty (lighting, camera angle, framing…) and "what next" variations.
  Tick the ones you like and Add them.

Then send it to txt2img with quality tags + recommended settings included.

Author: built by Claude on 2026-07-20.
"""
from __future__ import annotations

import csv
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

try:
    from modules import script_callbacks
except ImportError:
    script_callbacks = None  # type: ignore

logger = logging.getLogger(__name__)
TAG = "[tag-translator]"

EXT_ROOT = Path(__file__).resolve().parents[1]

MIN_COUNT = 100          # ignore tags rarer than this on Danbooru
USE_CATEGORIES = {0, 3, 4}  # general, copyright, character
MAX_NGRAM = 4

# Not offered by this tool (installation policy): minor-age subject tags.
EXCLUDED_TAGS = {
    "child", "children", "loli", "shota", "toddler", "baby", "infant",
    "little_girl", "little_boy", "aged_down", "kindergarten_uniform",
    "randoseru", "child_on_child",
}

# words removed before phrase matching (glue that never appears in tags)
GLUE = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
    "she", "he", "her", "his", "hers", "him", "they", "their", "its", "it",
    "wearing", "wears", "wore", "dressed", "has", "have", "had", "then",
    "very", "really", "quite", "also", "just", "some", "that", "this",
    "there", "i", "want", "wants", "of",
}

# Natural-language phrases Danbooru renamed away or never had. Each maps to
# a real tag; applied only when the dictionary lacks the phrase AND has the
# target (so a csv update can never break this).
SYNONYMS = {
    "silver hair": "grey_hair",
    "silver eyes": "grey_eyes",
    "golden hair": "blonde_hair",
    "blond hair": "blonde_hair",
    "golden eyes": "yellow_eyes",
    "scarlet hair": "red_hair",
    "crimson hair": "red_hair",
    "crimson eyes": "red_eyes",
    "violet hair": "purple_hair",
    "violet eyes": "purple_eyes",
    "turquoise hair": "aqua_hair",
    "teal hair": "aqua_hair",
    "teal eyes": "aqua_eyes",
    "emerald eyes": "green_eyes",
    "azure eyes": "blue_eyes",
    "pigtails": "twintails",
    "soft smile": "light_smile",
    "gentle smile": "light_smile",
    "night time": "night",
    "nighttime": "night",
}

QUALITY_TAGS = "masterpiece, best quality, amazing quality, very aesthetic, absurdres"
NEGATIVE_TAGS = ("bad quality, worst quality, worst detail, sketch, censor, "
                 "signature, watermark, username, jpeg artifacts")
SETTINGS_LINE = ("Steps: 28, Sampler: Euler a, CFG scale: 5.5, "
                 "Size: 832x1216, Clip skip: 2")


# ---------------------------------------------------------------------------
# Dictionary loading (lazy, cached)
# ---------------------------------------------------------------------------

_LOOKUP: Optional[Dict[str, Tuple[str, int, int]]] = None  # phrase -> (tag, count, category)


def _danbooru_csv() -> Optional[Path]:
    p = (EXT_ROOT.parent / "a1111-sd-webui-tagcomplete" / "tags" / "danbooru.csv")
    return p if p.is_file() else None


def _load_lookup() -> Dict[str, Tuple[str, int, int]]:
    global _LOOKUP
    if _LOOKUP is not None:
        return _LOOKUP
    lookup: Dict[str, Tuple[str, int, int]] = {}
    path = _danbooru_csv()
    if path is None:
        logger.warning("%s danbooru.csv not found (is Tag Autocomplete installed?)", TAG)
        _LOOKUP = {}
        return _LOOKUP
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 3:
                    continue
                tag, cat, count = row[0], row[1], row[2]
                try:
                    cat_i, count_i = int(cat), int(count)
                except ValueError:
                    continue
                if cat_i not in USE_CATEGORIES or count_i < MIN_COUNT:
                    continue
                if tag in EXCLUDED_TAGS:
                    continue
                phrases = [tag]
                if len(row) > 3 and row[3]:
                    phrases += [a for a in row[3].split(",") if a]
                for ph in phrases:
                    key = ph.replace("_", " ").replace("-", " ").strip().lower()
                    if not key or key in GLUE:
                        continue
                    old = lookup.get(key)
                    if old is None or count_i > old[1]:
                        lookup[key] = (tag, count_i, cat_i)
    except Exception:
        logger.exception("%s failed to load danbooru.csv", TAG)
    for phrase, target in SYNONYMS.items():
        if phrase not in lookup:
            ent = lookup.get(target.replace("_", " "))
            if ent is not None:
                lookup[phrase] = ent
    _LOOKUP = lookup
    logger.info("%s dictionary ready: %d phrases", TAG, len(lookup))
    return lookup


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

_SUBJECT_RES = [
    (re.compile(r"\b(?:two|2)\s+girls?\b"), "2girls"),
    (re.compile(r"\b(?:three|3)\s+girls?\b"), "3girls"),
    (re.compile(r"\b(?:two|2)\s+boys?\b"), "2boys"),
    (re.compile(r"\b(?:three|3)\s+boys?\b"), "3boys"),
]
_FEMALE_RE = re.compile(r"\b(girl|woman|lady|female|she|her)\b")
_MALE_RE = re.compile(r"\b(boy|man|guy|male|he|his)\b")


def _tokenize(text: str) -> List[str]:
    text = re.sub(r"[^\w\s'-]", " ", (text or "").lower())
    text = text.replace("-", " ")
    return [t for t in text.split() if t and t not in GLUE]


def translate(text: str) -> Tuple[List[Tuple[str, int]], List[str], List[str]]:
    """-> (matched [(tag, category)], leftovers, subject_tags)"""
    lookup = _load_lookup()
    raw = (text or "").lower()

    subjects: List[str] = []
    for rex, tag in _SUBJECT_RES:
        if rex.search(raw):
            subjects.append(tag)
    tokens = _tokenize(text)
    matched: List[Tuple[str, int]] = []
    seen = set()
    consumed = [False] * len(tokens)
    i = 0
    while i < len(tokens):
        hit = None
        for n in range(min(MAX_NGRAM, len(tokens) - i), 0, -1):
            phrase = " ".join(tokens[i:i + n])
            ent = lookup.get(phrase)
            if ent is not None:
                hit = ent
                for k in range(i, i + n):
                    consumed[k] = True
                i += n
                break
        if hit is None:
            i += 1
        else:
            tag, _count, cat = hit
            if tag not in seen and tag not in EXCLUDED_TAGS:
                seen.add(tag)
                matched.append((tag, cat))

    # skip-gram pass: stacked adjectives before a head noun, e.g.
    # "long silver hair" -> long_hair + silver_hair ("silver" blocks the
    # adjacency, so pair each unconsumed word with the next few tokens).
    for i, tok in enumerate(tokens):
        if consumed[i]:
            continue
        for j in range(i + 1, min(i + 4, len(tokens))):
            ent = lookup.get(f"{tok} {tokens[j]}")
            if ent is not None:
                tag, _count, cat = ent
                if tag not in seen and tag not in EXCLUDED_TAGS:
                    seen.add(tag)
                    matched.append((tag, cat))
                consumed[i] = True
                break

    # "smiling at the viewer" and similar: viewer present but no gaze tag
    if "viewer" in tokens and not any(t.endswith("at_viewer") for t, _ in matched):
        ent = lookup.get("looking at viewer")
        if ent is not None:
            matched.append((ent[0], ent[2]))
        for i, tok in enumerate(tokens):
            if tok == "viewer":
                consumed[i] = True

    leftovers: List[str] = [tokens[i] for i in range(len(tokens))
                            if not consumed[i]]

    # subject inference when no explicit Ngirls/Nboys matched
    tag_names = {t for t, _ in matched} | set(subjects)
    if not any(re.match(r"^\d(girl|boy)s$", t) for t in tag_names):
        female = bool(_FEMALE_RE.search(raw)) or "1girl" in tag_names
        male = bool(_MALE_RE.search(raw)) or "1boy" in tag_names
        if female and "1girl" not in tag_names:
            subjects.insert(0, "1girl")
        if male and "1boy" not in tag_names:
            subjects.append("1boy")
    if len(subjects) + sum(1 for t in tag_names if re.match(r"^\d(girl|boy)s?$", t)) == 1:
        if "solo" not in tag_names:
            subjects.append("solo")

    # de-noise leftovers
    junk = {"and", "with", "in", "on", "at", "to", "as", "for", "into",
            "while", "next", "by", "from", "up", "down", "out", "her", "who",
            "woman", "man", "girl", "boy", "lady", "guy", "female", "male",
            "person", "young"}
    leftovers = [w for w in leftovers if w not in junk and len(w) > 2]
    return matched, leftovers, subjects


# ---------------------------------------------------------------------------
# Canonical ordering
# ---------------------------------------------------------------------------

BUCKETS = ["subject", "character", "series", "traits", "outfit", "pose",
           "framing", "location", "light", "mood", "details"]

_BUCKET_PATTERNS = [
    ("subject", r"^\d(girl|boy)s?$|^solo$|^multiple_(girls|boys)$"),
    ("framing", r"portrait|upper_body|full_body|cowboy_shot|close_?-?up|wide_shot"
                r"|^pov$|from_(above|below|side|behind)|dutch_angle|^profile$"
                r"|^selfie$|_focus$|^scenery$"),
    ("traits", r"hair|eyes$|_eye$|breasts|skin|^tail$|_tail$|ears$|horn|wing"
               r"|ahoge|bangs|ponytail|twintails|braid|mole_|freckles|fang"
               r"|_pupils$|eyelashes|^tall$|^short$|muscular|curvy|slim"),
    ("outfit", r"uniform|dress|skirt|shirt|jacket|coat|bikini|swimsuit|clothes"
               r"|thighhighs|pantyhose|gloves|hat$|_hat|cap$|shoes|boots|socks"
               r"|panties|bra$|lingerie|apron|hoodie|sweater|kimono|yukata"
               r"|maid|leotard|necktie|ribbon|^bow$|hair_ornament|jewelry"
               r"|necklace|earrings|glasses|scarf|belt|shorts$|jeans|pants$"
               r"|top$|vest|cape|armor|costume|nude|topless|bottomless|underwear"),
    ("pose", r"standing|sitting|lying|kneeling|squatting|crouching|walking"
             r"|running|jumping|holding|looking_|arms?_|legs?_|hands?_|^v$"
             r"|pose$|hug|stretch|leaning|bent_|spread_|crossed_|_up$|_down$"
             r"|pointing|waving|reading|eating|drinking|sleeping|dancing"
             r"|fighting|riding|carrying|head_tilt|on_(back|stomach|side|bed|chair|floor)"),
    ("location", r"indoors|outdoors|room$|bedroom|kitchen|bathroom|classroom"
                 r"|school|library|cafe|office|beach|ocean|forest|city|street"
                 r"|park$|garden|pool|bath|onsen|shrine|castle|rooftop|sky$"
                 r"|_background|mountain|field|snow$|rain$|window|bed$|couch"
                 r"|desk$|train|station|festival|stage$|underwater|cave"),
    ("light", r"night$|^day$|daytime|morning|evening|sunset|sunrise|dusk|dawn"
              r"|moonlight|sunlight|lighting|backlighting|golden_hour|lens_flare"
              r"|^dark$|light_rays|sunbeam|neon|candlelight|lamp|glow"),
    ("mood", r"smile|smiling|blush|crying|tears|angry|happy|^sad$|embarrassed"
             r"|expression|open_mouth|closed_eyes|closed_mouth|wink|laughing"
             r"|pout|serious|shy|surprised|scared|frown|grin|seductive|sleepy"),
]


def _bucket_of(tag: str, category: int) -> str:
    if category == 4:
        return "character"
    if category == 3:
        return "series"
    for name, pattern in _BUCKET_PATTERNS:
        if re.search(pattern, tag):
            return name
    return "details"


def _fmt_tag(tag: str) -> str:
    """Escape parens so tags like fu_hua_(honkai) don't become attention syntax."""
    return tag.replace("(", r"\(").replace(")", r"\)")


def ordered_prompt(matched: List[Tuple[str, int]], subjects: List[str]) -> str:
    groups: Dict[str, List[str]] = {b: [] for b in BUCKETS}
    for s in subjects:
        if s not in groups["subject"]:
            groups["subject"].append(s)
    for tag, cat in matched:
        b = _bucket_of(tag, cat)
        if tag not in groups[b] and tag not in groups["subject"]:
            groups[b].append(tag)
    parts: List[str] = []
    for b in BUCKETS:
        parts.extend(_fmt_tag(t) for t in groups[b])
    return ", ".join(parts)


def empty_buckets(matched: List[Tuple[str, int]]) -> List[str]:
    filled = {_bucket_of(t, c) for t, c in matched}
    return [b for b in ("framing", "pose", "location", "light", "mood")
            if b not in filled]


# ---------------------------------------------------------------------------
# Scene suggestions
# ---------------------------------------------------------------------------

SUGGESTIONS = {
    "framing": ["upper body", "cowboy shot", "full body", "portrait",
                "from above", "from below", "from side", "dutch angle"],
    "pose": ["looking at viewer", "hand on own chest", "arms behind back",
             "head tilt", "leaning forward", "hands up", "waving",
             "holding own arm", "peace sign"],
    "location": ["simple background", "bedroom", "classroom", "city street",
                 "beach", "forest", "rooftop", "cafe", "night sky background"],
    "light": ["soft lighting", "golden hour", "backlighting", "sunlight",
              "moonlight", "neon lights", "window light", "dappled sunlight"],
    "mood": ["smile", "light blush", "gentle expression", "serious",
             "surprised", "shy", "laughing", "closed eyes"],
}

NEXT_IDEAS = [
    "same scene, {subj} now sitting and looking back over shoulder",
    "closer shot: portrait, wind lifting hair, soft backlight",
    "night version: moonlight through the window, dim lighting",
    "action beat: {subj} walking toward viewer, motion blur background",
    "quiet beat: {subj} eyes closed, small smile, sunbeam",
    "dramatic angle: from below, dutch angle, serious expression",
    "golden hour rooftop version of the same outfit and pose",
    "rainy version: wet hair, reflections, umbrella nearby",
]


def make_suggestions(matched: List[Tuple[str, int]], subjects: List[str]):
    lookup = _load_lookup()
    chips: List[str] = []
    for bucket in empty_buckets(matched):
        pool = SUGGESTIONS.get(bucket, [])
        for phrase in random.sample(pool, min(3, len(pool))):
            ent = lookup.get(phrase)
            chips.append(ent[0] if ent else phrase.replace(" ", "_"))
    subj = "1girl"
    for s in subjects:
        subj = s
        break
    ideas = random.sample(NEXT_IDEAS, 3)
    ideas = [i.format(subj=subj.replace("1girl", "the girl").replace("1boy", "the boy"))
             for i in ideas]
    return chips, "**What next?** Try describing:\n" + "\n".join(f"- {i}" for i in ideas)


# ---------------------------------------------------------------------------
# Local AI prompt writer (DanTagGen — 400M LLaMA trained on Danbooru tags)
# https://huggingface.co/KBlueLeaf/DanTagGen-delta-rev2   (CC-BY-SA-4.0)
# Runs via the transformers+torch already in the venv. First use downloads
# ~800 MB to the HF cache. CPU by default so SDXL keeps the VRAM.
# ---------------------------------------------------------------------------

AI_MODELS = ["KBlueLeaf/DanTagGen-delta-rev2", "KBlueLeaf/DanTagGen-beta"]
_AI = {"model": None, "tokenizer": None, "name": None, "device": None}


def _ai_split_categories(tags: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """-> (special 1girl-likes, characters, copyrights, general)"""
    lookup = _load_lookup()
    special, chars, copy, general = [], [], [], []
    for t in tags:
        t = t.strip().replace("\\(", "(").replace("\\)", ")")
        if not t:
            continue
        if re.match(r"^\d(girl|boy)s?$|^solo$", t.replace(" ", "_")):
            special.append(t.replace(" ", "_"))
            continue
        ent = lookup.get(t.replace("_", " ").lower())
        cat = ent[2] if ent else 0
        (chars if cat == 4 else copy if cat == 3 else general).append(
            t.replace(" ", "_"))
    return special, chars, copy, general


def _ai_build_input(tags: List[str], rating: str, target: str) -> str:
    special, chars, copy, general = _ai_split_categories(tags)
    head = ", ".join(special + general) or "1girl"
    return (
        f"quality: masterpiece\n"
        f"rating: {rating or 'safe'}\n"
        f"artist: <|empty|>\n"
        f"characters: {', '.join(chars) or '<|empty|>'}\n"
        f"copyrights: {', '.join(copy) or '<|empty|>'}\n"
        f"aspect ratio: 0.7\n"
        f"target: <|{target or 'long'}|>\n"
        f"general: {head}<|input_end|>"
    )


def _ai_parse_output(full_text: str, existing: List[str]) -> List[str]:
    """Tags the model added after <|input_end|>, cleaned + filtered."""
    if "<|input_end|>" in full_text:
        gen = full_text.split("<|input_end|>", 1)[1]
    else:
        gen = full_text
    gen = gen.split("<|")[0]  # stop at any trailing special token
    have = {t.strip().replace(" ", "_").lower() for t in existing}
    out: List[str] = []
    for raw in gen.split(","):
        t = raw.strip().strip(".").replace(" ", "_").lower()
        if not t or len(t) > 40 or not re.match(r"^[\w()\-'./:!?]+$", t):
            continue
        if t in EXCLUDED_TAGS or t in have or t in out:
            continue
        out.append(t)
    return out


def _ai_load(model_name: str, device: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if _AI["model"] is not None and _AI["name"] == model_name and _AI["device"] == device:
        return "already loaded"
    _ai_unload()
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device).eval()
    _AI.update(model=model, tokenizer=tok, name=model_name, device=device)
    return "loaded"


def _ai_unload() -> str:
    if _AI["model"] is None:
        return "AI model not loaded."
    _AI.update(model=None, tokenizer=None, name=None, device=None)
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return "AI model unloaded, memory freed."


def ai_complete(prompt: str, rating: str, target: str, temperature: float,
                device_label: str, model_name: str) -> Tuple[str, str]:
    """-> (new_prompt, status)"""
    tags = [t for t in (prompt or "").split(",") if t.strip()]
    if not tags:
        return prompt, "Write or translate some tags first — the AI extends them."
    try:
        import torch
    except Exception:
        return prompt, "⚠ torch/transformers not available in this environment."
    device = "cuda" if "GPU" in (device_label or "") and torch.cuda.is_available() else "cpu"
    try:
        _ai_load(model_name or AI_MODELS[0], device)
    except Exception as exc:
        return prompt, (f"⚠ Could not load the AI model ({exc}). First use needs "
                        f"internet for a ~800 MB download.")
    try:
        text = _ai_build_input(tags, rating, target)
        tok, model = _AI["tokenizer"], _AI["model"]
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=192, do_sample=True,
                temperature=max(0.1, float(temperature or 1.0)),
                top_p=0.95, top_k=100,
                pad_token_id=tok.eos_token_id)
        full = tok.decode(out[0], skip_special_tokens=False)
        new_tags = _ai_parse_output(full, tags)
        if not new_tags:
            return prompt, "AI returned nothing new — try again (it samples randomly) or raise temperature."
        lookup = _load_lookup()
        merged = [(t.strip(), 0) for t in tags]
        for t in new_tags:
            ent = lookup.get(t.replace("_", " "))
            merged.append((t, ent[2] if ent else 0))
        # rebuild in canonical order, preserving subject tags
        subjects = [t for t, _ in merged if re.match(r"^\d(girl|boy)s?$|^solo$", t.replace(" ", "_"))]
        rest = [(t.replace(" ", "_"), c) for t, c in merged
                if t.replace(" ", "_") not in subjects]
        new_prompt = ordered_prompt(rest, [s.replace(" ", "_") for s in subjects])
        return new_prompt, f"🤖 AI added {len(new_tags)} tags: {', '.join(new_tags[:12])}" + ("…" if len(new_tags) > 12 else "")
    except Exception as exc:
        logger.exception("%s AI generation failed", TAG)
        return prompt, f"⚠ AI generation failed: {exc}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

EXAMPLE = ("A young woman with long silver hair and blue eyes wearing a "
           "school uniform with a pleated skirt. She is sitting by the "
           "window of a classroom at sunset, smiling softly at the viewer "
           "while holding a book.")


def _on_ui_tabs():
    try:
        return _build_tab()
    except Exception:
        logger.exception("%s failed to build tab", TAG)
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown("Describe tab failed to load - see console log.")
        return [(ui, "Describe", "forge_tag_translator")]


def _build_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## Describe → Illustrious prompt\n"
            "Write what you see in your head — character, outfit, action, "
            "place. I'll turn it into correctly ordered booru tags, tell you "
            "what I couldn't map, and suggest what the scene still needs."
        )
        para_box = gr.Textbox(label="Your description", lines=5, value=EXAMPLE,
                              placeholder="A woman with long red hair wearing…")
        with gr.Row():
            go_btn = gr.Button("✨ Translate to tags", variant="primary", scale=0)
            with_quality = gr.Checkbox(label="Add quality tags", value=True, scale=0)
            with_settings = gr.Checkbox(label="Include recommended settings "
                                        "(Euler a, 28, CFG 5.5, 832×1216)",
                                        value=True, scale=0)

        prompt_box = gr.Textbox(label="Illustrious prompt (editable)", lines=4,
                                show_copy_button=True, elem_classes=["prompt"])
        unmatched_md = gr.Markdown("")

        with gr.Row():
            with gr.Column():
                chips_group = gr.CheckboxGroup(
                    label="Scene ideas — tags for what your description "
                          "left open (tick and Add)", choices=[])
                with gr.Row():
                    add_btn = gr.Button("➕ Add ticked ideas", scale=0)
                    reroll_btn = gr.Button("🎲 Different ideas", scale=0)
            with gr.Column():
                ideas_md = gr.Markdown("")

        with gr.Accordion("🤖 AI prompt writer (local model, offline after first "
                          "download)", open=False):
            gr.Markdown(
                "Extends whatever is in the prompt box with fitting detail tags — "
                "a 400M-parameter model (DanTagGen) trained on 7M Danbooru posts, "
                "running locally. First click downloads ~800 MB. CPU mode leaves "
                "all VRAM to SDXL (a few seconds per run); GPU mode is instant "
                "but shares the 8 GB."
            )
            with gr.Row():
                ai_model_dd = gr.Dropdown(label="Model", choices=AI_MODELS,
                                          value=AI_MODELS[0], scale=2)
                ai_device = gr.Radio(["CPU (recommended)", "GPU"],
                                     value="CPU (recommended)", label="Run on",
                                     scale=1)
            with gr.Row():
                ai_rating = gr.Dropdown(label="Content rating",
                                        choices=["safe", "sensitive", "nsfw",
                                                 "explicit"],
                                        value="safe", scale=1)
                ai_target = gr.Dropdown(label="How much to add",
                                        choices=["short", "long", "very_long"],
                                        value="long", scale=1)
                ai_temp = gr.Slider(0.5, 1.5, value=1.0, step=0.05,
                                    label="Creativity", scale=2)
            with gr.Row():
                ai_go_btn = gr.Button("🤖 AI-complete my prompt",
                                      variant="primary", scale=0)
                ai_unload_btn = gr.Button("Unload AI model (free memory)",
                                          scale=0)
            ai_status = gr.Markdown("")

        negative_box = gr.Textbox(label="Negative prompt", value=NEGATIVE_TAGS,
                                  lines=2, show_copy_button=True,
                                  elem_classes=["prompt"])
        composed_box = gr.Textbox(visible=False)
        with gr.Row():
            send_t2i = gr.Button("Send to txt2img (replaces prompt + settings)",
                                 variant="primary")
            send_i2i = gr.Button("Send to img2img")
        status = gr.Markdown("")

        st_matched = gr.State([])
        st_subjects = gr.State([])

        # ------------------------------------------------------------------
        def _compose(prompt, negative, use_quality, use_settings):
            p = (prompt or "").strip().rstrip(",")
            if use_quality and p:
                p = f"{p}, {QUALITY_TAGS}"
            elif use_quality:
                p = QUALITY_TAGS
            out = p
            if (negative or "").strip():
                out += f"\nNegative prompt: {negative.strip()}"
            if use_settings:
                out += f"\n{SETTINGS_LINE}"
            return out

        def do_translate(text, use_quality, use_settings):
            if not _load_lookup():
                return ("", "⚠ danbooru.csv not found — is the Tag "
                        "Autocomplete extension still installed?",
                        gr.update(choices=[]), "", "", [], [])
            matched, leftovers, subjects = translate(text)
            prompt = ordered_prompt(matched, subjects)
            if leftovers:
                un = ("Couldn't map (often fine to leave in, or reword): "
                      + ", ".join(f"`{w}`" for w in leftovers[:25]))
            else:
                un = "Everything in your description was mapped to tags. ✓"
            chips, ideas = make_suggestions(matched, subjects)
            composed = _compose(prompt, NEGATIVE_TAGS, use_quality, use_settings)
            return (prompt, un, gr.update(choices=chips, value=[]), ideas,
                    composed, matched, subjects)

        def do_add(prompt, ticked, negative, use_quality, use_settings):
            extra = [t for t in (ticked or [])
                     if t not in (prompt or "")]
            new_prompt = ", ".join([x for x in [(prompt or "").rstrip(", ")] + extra if x])
            return new_prompt, _compose(new_prompt, negative, use_quality, use_settings)

        def do_reroll(matched, subjects):
            chips, ideas = make_suggestions(matched, subjects)
            return gr.update(choices=chips, value=[]), ideas

        def do_recompose(prompt, negative, use_quality, use_settings):
            return _compose(prompt, negative, use_quality, use_settings)

        go_btn.click(do_translate, [para_box, with_quality, with_settings],
                     [prompt_box, unmatched_md, chips_group, ideas_md,
                      composed_box, st_matched, st_subjects])
        add_btn.click(do_add,
                      [prompt_box, chips_group, negative_box, with_quality,
                       with_settings],
                      [prompt_box, composed_box])
        reroll_btn.click(do_reroll, [st_matched, st_subjects],
                         [chips_group, ideas_md])

        def do_ai(prompt, rating, target, temp, device, model_name,
                  negative, use_quality, use_settings):
            new_prompt, msg = ai_complete(prompt, rating, target, temp,
                                          device, model_name)
            return (new_prompt, msg,
                    _compose(new_prompt, negative, use_quality, use_settings))

        ai_go_btn.click(do_ai,
                        [prompt_box, ai_rating, ai_target, ai_temp, ai_device,
                         ai_model_dd, negative_box, with_quality, with_settings],
                        [prompt_box, ai_status, composed_box])
        ai_unload_btn.click(lambda: _ai_unload(), [], [ai_status])
        for comp in (prompt_box, negative_box, with_quality, with_settings):
            comp.change(do_recompose,
                        [prompt_box, negative_box, with_quality, with_settings],
                        [composed_box])

        # paste wiring
        import importlib
        paste_mod = None
        for mod_name in ("modules.infotext_utils",
                         "modules.generation_parameters_copypaste"):
            try:
                paste_mod = importlib.import_module(mod_name)
                break
            except ImportError:
                continue
        if paste_mod is not None and hasattr(paste_mod, "register_paste_params_button"):
            for btn, tab in ((send_t2i, "txt2img"), (send_i2i, "img2img")):
                paste_mod.register_paste_params_button(paste_mod.ParamBinding(
                    paste_button=btn, tabname=tab,
                    source_text_component=composed_box,
                ))
        else:
            msg = "Paste API not found — copy the prompt manually."
            send_t2i.click(lambda: msg, None, [status])
            send_i2i.click(lambda: msg, None, [status])

    return [(ui, "Describe", "forge_tag_translator")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
