"""
sd-forge-naitosd / lib_naitosd / converter.py
=============================================

Pure-logic conversion functions for NovelAI -> Illustrious XL A1111 metadata.
Lifted from the standalone nai_to_sd.py (no Tkinter, no GUI bits) so the Forge
extension can import them directly.

Functions exposed:
  - extract_novelai_metadata(pil_image)  -> dict | None
  - build_a1111_parameters(meta)         -> str
  - convert_file(src_path)               -> Path  (writes <name>_sd.png)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from PIL import Image, PngImagePlugin

ILLUSTRIOUS_MODEL_LABEL = "Illustrious XL"

SAMPLER_MAP = {
    "k_euler":               "Euler",
    "k_euler_ancestral":     "Euler a",
    "k_heun":                "Heun",
    "k_lms":                 "LMS",
    "k_dpm_2":               "DPM2",
    "k_dpm_2_ancestral":     "DPM2 a",
    "k_dpm_fast":            "DPM fast",
    "k_dpm_adaptive":        "DPM adaptive",
    "k_dpmpp_sde":           "DPM++ SDE",
    "k_dpmpp_2s_ancestral":  "DPM++ 2S a",
    "k_dpmpp_2m":            "DPM++ 2M",
    "k_dpmpp_2m_sde":        "DPM++ 2M SDE",
    "k_dpmpp_3m_sde":        "DPM++ 3M SDE",
    "ddim":                  "DDIM",
    "ddim_v3":               "DDIM",
    "plms":                  "PLMS",
}

# Illustrious XL recommended quality tags.
# These get prepended to every converted prompt so the output is ready to
# generate well on Illustrious checkpoints. We skip tags that are already
# present in the source prompt to avoid duplication.
ILLUSTRIOUS_QUALITY_TAGS = [
    "masterpiece",
    "best quality",
    "amazing quality",
    "very aesthetic",
    "absurdres",
    "newest",
]

ILLUSTRIOUS_NEGATIVE_TAGS = [
    "worst quality",
    "low quality",
    "normal quality",
    "lowres",
    "bad anatomy",
    "bad hands",
    "watermark",
    "signature",
    "artist name",
    "text",
    "error",
    "blurry",
    "jpeg artifacts",
]

ILLUSTRIOUS_MODEL_LABEL = "Illustrious XL"


def _existing_tags(body):
    """Return a set of lowercase comma-separated tokens already present in body.
    Treats newlines and the A1111 `BREAK` keyword as chunk separators so that
    e.g. `blurry\nBREAK\nextra fingers` yields {`blurry`, `extra fingers`}."""
    if not body:
        return set()
    import re
    # Normalize BREAK (case-insensitive, whole word) and newlines to commas.
    normalized = re.sub(r"\bBREAK\b", ",", body, flags=re.IGNORECASE)
    normalized = normalized.replace("\n", ",")
    return {t.strip().lower() for t in normalized.split(",") if t.strip()}


def _prepend_unique_tags(tags, body):
    """Return a tag-string of `tags` not already present in `body`, joined by ', '."""
    existing = _existing_tags(body)
    new = [t for t in tags if t.lower() not in existing]
    return ", ".join(new)


def _apply_illustrious_to_chunks(chunks, illu_tags):
    """Prepend Illustrious tags to the first chunk. Skip tags already present
    in ANY chunk to avoid duplication."""
    if not illu_tags:
        return chunks
    existing = set()
    for ch in chunks:
        for t in ch.split(","):
            t = t.strip().lower()
            if t and t != "break":
                existing.add(t)
    to_add = [t for t in illu_tags if t.lower() not in existing]
    if not to_add:
        return chunks
    addition = ", ".join(to_add)
    if not chunks:
        return [addition]
    return [addition + ", " + chunks[0]] + chunks[1:]


SCHEDULE_MAP = {
    "karras":          "Karras",
    "exponential":     "Exponential",
    "polyexponential": "Polyexponential",
    "sgm_uniform":     "SGM Uniform",
}


def map_sampler(nai_sampler):
    if not nai_sampler:
        return "Euler"
    base = nai_sampler.replace("_ka", "")
    return SAMPLER_MAP.get(base, "Euler")


def map_schedule(noise_schedule):
    if not noise_schedule:
        return None
    return SCHEDULE_MAP.get(noise_schedule.lower())


# ---------------------------------------------------------------------------
# Metadata extraction / A1111 builder
# ---------------------------------------------------------------------------

def extract_novelai_metadata(img):
    info = img.info or {}
    comment_raw = info.get("Comment") or "{}"
    try:
        comment = json.loads(comment_raw)
    except (json.JSONDecodeError, TypeError):
        comment = {}
    return {
        "software":    (info.get("Software") or "").strip(),
        "source":      (info.get("Source") or "").strip(),
        "title":       (info.get("Title") or "").strip(),
        "description": (info.get("Description") or "").strip(),
        "gen_time":    (info.get("Generation time") or "").strip(),
        "comment":     comment,
        "width":       img.width,
        "height":      img.height,
    }


# --------------------------------------------------------------------------
# NAI -> A1111 prompt syntax translator
# --------------------------------------------------------------------------

# NAI V4 weight syntax:  "1.5::content::"  ->  "(content:1.5)"
# We restrict `content` to not contain `::` to avoid greedy matches across
# multiple weights. Repeat until the string is stable to handle adjacent
# weight markers.
_NAI_V4_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)::\s*([^:]+?)\s*::")

# NAI legacy weight syntax (pre-V4). Only matched when braces are BALANCED
# and contain no nested braces, so we don\'t mangle Dynamic Prompts wildcard
# braces or unrelated text.
_NAI_DOUBLE_BRACE_RE = re.compile(r"\{\{([^{}]+?)\}\}")  # 1.10x
_NAI_SINGLE_BRACE_RE = re.compile(r"\{([^{}]+?)\}")        # 1.05x


def _translate_nai_prompt(text):
    """Translate NAI-specific prompt syntax to A1111 equivalents and clean up
    malformed artifacts.

    Conversions (in order):
      1. NAI V4 weight  N::content::          -> (content:N)
      2. NAI legacy {{tag}} / {tag}           -> (tag:1.10) / (tag:1.05)
      3. Unbalanced {{ ... ::  (NAI legacy double-brace mixed with V4 :: end)
                                              -> (content:1.10)
      4. Unbalanced [  ... ::                  -> (content:0.95)
      5. Strip orphan numeric weight markers   ( "2::," with no closing :: )
      6. Strip stray :: that survived
      7. Strip ":)" smiley artifacts left behind by malformed weights
      8. Embedded newlines -> ", "
      9. Collapse whitespace and repeated commas
    """
    if not text:
        return ""

    # 1. V4 balanced weight, apply repeatedly until stable.
    prev = None
    while prev != text:
        prev = text
        text = _NAI_V4_WEIGHT_RE.sub(
            lambda m: f"({m.group(2).strip()}:{m.group(1)})", text
        )

    # 2. Legacy balanced braces (double first so {{x}} doesn\'t match single).
    text = _NAI_DOUBLE_BRACE_RE.sub(r"(\1:1.10)", text)
    text = _NAI_SINGLE_BRACE_RE.sub(r"(\1:1.05)", text)

    # 3. Unbalanced {{ ... :: -> (...:1.10)
    #    Matches "{{ <content> [optional trailing ,/space] ::"
    text = re.sub(
        r"\{\{\s*([^{}]+?)[,\s]*::",
        lambda m: f"({m.group(1).strip().rstrip(',').strip()}:1.10)",
        text,
    )
    # Strip any leftover lone {{ or }} that didn\'t pair up.
    text = text.replace("{{", "").replace("}}", "")
    text = text.replace("{", "").replace("}", "")

    # 4. Unbalanced [ ... :: -> (...:0.95)
    #    We MUST keep balanced A1111 brackets like [tag] (which means 0.9x) intact,
    #    so only convert when followed by a :: terminator within the bracket region.
    text = re.sub(
        r"\[\s*([^\[\]]+?)[,\s]*::",
        lambda m: f"({m.group(1).strip().rstrip(',').strip()}:0.95)",
        text,
    )

    # 5. Strip orphan numeric weight markers like "2::," (no closing ::).
    text = re.sub(r"\b\d+(?:\.\d+)?::(?=\s*,|\s*$|\s+[^:])", "", text)

    # 6. Strip any remaining stray :: not part of a successful conversion.
    text = re.sub(r"::", "", text)

    # 7. Strip leftover ":)" artifacts that came from malformed NAI markers.
    text = text.replace(":)", "")

    # 8. Newlines embedded inside a prompt -> tag separators.
    text = text.replace("\r", "").replace("\n", ", ")

    # 9. Collapse internal whitespace runs.
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse runs of consecutive commas (with optional whitespace).
    text = re.sub(r"(?:\s*,\s*){2,}", ", ", text)

    # Tidy leading / trailing commas + spaces.
    text = text.strip().strip(",").strip()
    return text


def _dedupe_chunk(text):
    """Dedupe comma-separated tokens within a single chunk.
    Case-insensitive match; first occurrence wins (preserves original casing)."""
    if not text:
        return ""
    seen = set()
    out = []
    for raw in text.split(","):
        tok = raw.strip()
        if not tok:
            continue
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tok)
    return ", ".join(out)


def _flatten_v4_caption(block):
    """
    NovelAI V4 prompt structure:
        { "caption": { "base_caption": "...setting/scene...",
                       "char_captions": [ {"char_caption": "..."}, ... ] } }

    Returns a *list* of cleaned chunks (base first, then each non-empty
    character caption). Each chunk has been put through the NAI->A1111
    syntax translator and per-chunk dedup. Returns [] when *block* is not
    a V4 caption structure.
    """
    if not isinstance(block, dict):
        return []
    cap = block.get("caption")
    if not isinstance(cap, dict):
        return []

    chunks = []
    base = _dedupe_chunk(_translate_nai_prompt((cap.get("base_caption") or "").strip()))
    if base:
        chunks.append(base)
    for ch in cap.get("char_captions") or []:
        if not isinstance(ch, dict):
            continue
        text = _dedupe_chunk(_translate_nai_prompt((ch.get("char_caption") or "").strip()))
        if text:
            chunks.append(text)
    return chunks


def _collect_positive_chunks(c, meta):
    """List of cleaned positive-prompt chunks (V4 base + characters, or legacy single)."""
    chunks = _flatten_v4_caption(c.get("v4_prompt"))
    if chunks:
        return chunks
    raw = (c.get("prompt") or meta.get("description") or "").strip()
    cleaned = _dedupe_chunk(_translate_nai_prompt(raw))
    return [cleaned] if cleaned else []


def _collect_negative_chunks(c):
    """List of cleaned negative-prompt chunks."""
    chunks = _flatten_v4_caption(c.get("v4_negative_prompt"))
    if chunks:
        return chunks
    raw = (c.get("uc") or c.get("negative_prompt") or "").strip()
    cleaned = _dedupe_chunk(_translate_nai_prompt(raw))
    return [cleaned] if cleaned else []


def _merge_chunks_chars_first(chunks):
    """Reorder + merge chunks Gemini-style:
        [base, char1, char2, ...]  ->  [merged_chars, base]
    Returns a list of 0, 1, or 2 cleaned chunks ready for BREAK-joining.
    Characters are merged into ONE chunk (no BREAK between them) and the
    merged chunk is deduped against itself."""
    if not chunks:
        return []
    base = chunks[0].strip() if chunks else ""
    char_chunks = [c.strip() for c in chunks[1:] if c.strip()]

    out = []
    if char_chunks:
        merged_chars = _dedupe_chunk(", ".join(char_chunks))
        if merged_chars:
            out.append(merged_chars)
    if base:
        out.append(base)
    return out


def build_a1111_parameters(meta):
    c = meta["comment"]

    # 1. Collect CLEAN chunks (NAI syntax translated, per-chunk deduped).
    pos_chunks = _collect_positive_chunks(c, meta)
    neg_chunks = _collect_negative_chunks(c)

    # 2. Reorder Gemini-style: characters merged first, base/setting after.
    pos_chunks = _merge_chunks_chars_first(pos_chunks)
    neg_chunks = _merge_chunks_chars_first(neg_chunks)

    # 3. Join with BREAK on its own line (only when there\'s >1 chunk).
    def _join(chunks):
        if not chunks:
            return ""
        if len(chunks) == 1:
            return chunks[0]
        return "\nBREAK\n".join(chunks)
    positive = _join(pos_chunks)
    negative = _join(neg_chunks)

    # 4. Build the params line in A1111 canonical order.
    steps    = c.get("steps")
    sampler  = map_sampler(c.get("sampler", ""))
    cfg      = c.get("scale")
    seed     = c.get("seed")
    width    = c.get("width")  or meta["width"]
    height   = c.get("height") or meta["height"]
    schedule = map_schedule(c.get("noise_schedule"))

    params = []
    if steps is not None:    params.append(("Steps", steps))
    params.append(("Sampler", sampler))
    if schedule:             params.append(("Schedule type", schedule))
    if cfg is not None:      params.append(("CFG scale", cfg))

    # cfg_rescale: A1111\'s Dynamic Thresholding extension reads "CFG Rescale".
    # Only emit when meaningful (nonzero) so default behaviour isn\'t altered.
    cfg_rescale = c.get("cfg_rescale")
    try:
        if cfg_rescale is not None and float(cfg_rescale) != 0.0:
            params.append(("CFG Rescale", cfg_rescale))
    except (TypeError, ValueError):
        pass

    if seed is not None:     params.append(("Seed", seed))
    params.append(("Size", f"{width}x{height}"))
    params.append(("Model", ILLUSTRIOUS_MODEL_LABEL))

    strength = c.get("strength")
    if strength is not None:
        params.append(("Denoising strength", strength))

    param_line = ", ".join(f"{k}: {v}" for k, v in params)

    # 5. Assemble the canonical A1111 chunk.
    sections = [positive.strip()] if positive.strip() else [""]
    if negative.strip():
        sections.append(f"Negative prompt: {negative.strip()}")
    sections.append(param_line)
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_file(src):
    """Read NovelAI PNG at *src*, write `<stem>_sd.png` next to it. Return Path."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")
    if src.suffix.lower() != ".png":
        raise ValueError(f"Not a PNG: {src.name}")

    with Image.open(src) as img:
        img.load()
        meta = extract_novelai_metadata(img)
        params_string = build_a1111_parameters(meta)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", params_string)

        dst = src.with_name(f"{src.stem}_sd.png").resolve()
        img.save(dst, format="PNG", pnginfo=pnginfo, optimize=False)

    if not dst.exists():
        raise RuntimeError(f"Save succeeded but output file is missing: {dst}")
    return dst


# ---------------------------------------------------------------------------
# Dropped-path sanitizer
# ---------------------------------------------------------------------------

_FILE_URL_RE = re.compile(r"^file:/+", re.IGNORECASE)

