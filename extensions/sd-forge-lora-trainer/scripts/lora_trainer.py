"""
sd-forge-lora-trainer
=====================

Adds a "LoRA Trainer" tab to Forge Neo. Drop 5-15 reference images, type a
trigger word, pick a preset (Character / Style / Concept) and a base SDXL
checkpoint, then click Start.

This is the rewritten "v2" — same orchestration logic, but the Gradio UI
is intentionally minimal (no gr.Timer auto-refresh, no fancy info= params
that vary across Gradio versions). Anything that *could* fail at tab-build
time has been removed or wrapped in try/except, so the tab always
registers and we surface errors in the visible log textbox at the bottom.

Author: built by Claude on 2026-05-26.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import traceback
from pathlib import Path
from typing import List, Optional

import gradio as gr

try:
    from modules import script_callbacks
except ImportError:
    script_callbacks = None  # type: ignore

try:
    from modules import paths as _paths_mod
except ImportError:
    _paths_mod = None  # type: ignore

logger = logging.getLogger(__name__)
TAG = "[lora-trainer]"


# ===========================================================================
# Paths
# ===========================================================================

EXT_ROOT = Path(__file__).resolve().parents[1]
SD_SCRIPTS_DIR = EXT_ROOT / "sd-scripts"
SD_SCRIPTS_VENV = EXT_ROOT / "sd-scripts-venv"
PROJECTS_DIR = EXT_ROOT / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR = EXT_ROOT / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _forge_root() -> Path:
    return EXT_ROOT.parents[1]


def _models_path() -> Path:
    if _paths_mod is not None and getattr(_paths_mod, "models_path", None):
        return Path(_paths_mod.models_path)
    return _forge_root() / "models"


def _lora_output_dir() -> Path:
    p = _models_path() / "Lora"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _checkpoints_dir() -> Path:
    return _models_path() / "Stable-diffusion"


def _venv_python() -> Path:
    if os.name == "nt":
        return SD_SCRIPTS_VENV / "Scripts" / "python.exe"
    return SD_SCRIPTS_VENV / "bin" / "python"


# ===========================================================================
# Presets - tuned for SDXL Illustrious-family on 8-12 GB cards
# ===========================================================================

PRESETS = {
    "Character": dict(
        network_dim=16, network_alpha=8,
        learning_rate=1e-4, num_epochs=15, repeats=10,
        lr_scheduler="cosine", optimizer="AdamW8bit",
    ),
    "Style": dict(
        network_dim=32, network_alpha=16,
        learning_rate=8e-5, num_epochs=20, repeats=6,
        lr_scheduler="cosine", optimizer="AdamW8bit",
    ),
    "Concept": dict(
        network_dim=8, network_alpha=4,
        learning_rate=1e-4, num_epochs=15, repeats=10,
        lr_scheduler="cosine", optimizer="AdamW8bit",
    ),
}

CAPTION_MODES = [
    "WD14 tagger (anime / Illustrious)",
    "BLIP (photo / realistic)",
    "Trigger word only (no autocaption)",
]




# ===========================================================================
# Booru search & download (Danbooru -> Gelbooru fallback)
# ===========================================================================
# Both APIs return tag-rich posts that are already human-labeled. We use that
# as caption text directly - higher quality than WD14 because real users wrote
# the tags. Anonymous Danbooru access supports up to 2 tags per query, which
# is enough for "character + rating".

import json as _json
import urllib.parse as _urlparse
import urllib.request as _urlreq

USER_AGENT = "sd-forge-lora-trainer/1.0 (educational; local LoRA training)"
DANBOORU_BASE = "https://danbooru.donmai.us"
GELBOORU_BASE = "https://gelbooru.com"

# Tags to strip from caption strings - common boilerplate that hurts training.
CAPTION_BLACKLIST = {
    "highres", "absurdres", "lowres", "scan", "official_art", "translated",
    "translation_request", "commentary", "commentary_request",
    "english_commentary", "japanese_commentary",
    "bad_anatomy", "bad_hands", "bad_id", "bad_pixiv_id",
    "watermark", "signature", "artist_name", "logo", "copyright_name",
    "text", "censored", "uncensored", "mosaic_censoring",
    "multiple_views", "reference_sheet",
}

# Tags that show up on basically every image but aren't actually distinguishing
# character features. Skip them when picking out "defining features" - including
# them would just teach the LoRA that the character is a girl, which it already
# knows from the rest of the caption.
GENERIC_TAGS = {
    "1girl", "1boy", "2girls", "2boys", "solo", "solo_focus",
    "looking_at_viewer", "looking_away", "looking_back", "looking_to_the_side",
    "looking_up", "looking_down",
    "simple_background", "white_background", "grey_background", "black_background",
    "transparent_background", "gradient_background", "outdoors", "indoors",
    "standing", "sitting", "walking", "running",
    "full_body", "upper_body", "cowboy_shot", "portrait", "close-up",
    "from_above", "from_below", "from_side", "from_behind", "facing_viewer",
    "smile", "open_mouth", "closed_mouth", "blush", "expressionless",
    "day", "night", "sky", "cloud", "tree", "grass", "flower",
}

# Art-style / medium / context tags. These describe HOW an image was drawn,
# not WHAT the character looks like. They show up frequently enough across
# fan art that they can cross the feature-detection threshold (e.g. chibi
# appears on 40% of some characters' booru posts), but they are NOT defining
# features of the character. Strip from feature detection on both paths.
STYLE_TAGS = {
    # Body proportion / art style
    "chibi", "minigirl", "miniboy", "deformed", "sd_(super_deformed)",
    # Rendering style
    "realistic", "semi-realistic", "photorealistic", "3d", "3d_(artwork)",
    "2d", "anime_style", "anime", "cel_shading", "flat_color", "flat_chest",
    # Medium / technique
    "traditional_media", "watercolor_(medium)", "marker_(medium)", "pencil_(medium)",
    "ink_(medium)", "sketch", "rough_sketch", "lineart", "line_art",
    "monochrome", "greyscale", "limited_palette", "pixel_art", "pixelated",
    "vector_trace", "screencap", "official_art", "concept_art",
    # Misc context
    "parody", "meme", "fanart", "crossover", "what_if",
    "alternate_costume", "alternate_hairstyle", "alternate_universe",
    "no_humans",  # for some reason booru tags this on chibis sometimes
}



def _compute_character_features(caption_dir, top_n=8, min_frequency=0.6):
    """Scan all .txt captions in caption_dir and identify tags that appear
    in at least `min_frequency` of them. Those are the character's defining
    features (hair color, eye color, signature outfit, etc.) that should be
    promoted to the front of every caption.

    Returns a list of feature strings (most-common first), or [] on error.
    The strings are space-form (as written in the captions: 'white hair'),
    not underscore-form.
    """
    import collections
    counts = collections.Counter()
    total = 0
    try:
        for txt in Path(caption_dir).glob("*.txt"):
            content = txt.read_text(encoding="utf-8").strip()
            if not content:
                continue
            total += 1
            seen_here = set()
            for raw in content.split(","):
                tag = raw.strip().lower()
                if not tag:
                    continue
                # Convert back to underscore form for blacklist matching
                tag_us = tag.replace(" ", "_")
                if (tag_us in GENERIC_TAGS or tag_us in CAPTION_BLACKLIST
                        or tag_us in STYLE_TAGS or tag_us in MULTI_INDICATORS):
                    continue
                # de-dupe within a single caption (don't count "1girl, 1girl" as 2)
                if tag in seen_here:
                    continue
                seen_here.add(tag)
                counts[tag] += 1
    except Exception:
        return []

    if total < 3:
        return []

    threshold = max(2, int(total * min_frequency))
    features = [t for t, c in counts.most_common() if c >= threshold]
    return features[:top_n]


def _http_get_json(url: str, timeout: int = 30):
    req = _urlreq.Request(url, headers={"User-Agent": USER_AGENT})
    with _urlreq.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read().decode("utf-8"))


def _http_download(url: str, dest: Path, timeout: int = 60) -> bool:
    try:
        req = _urlreq.Request(url, headers={"User-Agent": USER_AGENT})
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as e:
        return False


def _filter_tags(tag_string: str, solo_mode: str = "any") -> str:
    """Clean a booru tag string for caption use. Drops boilerplate tags,
    converts underscores to spaces (Illustrious is trained on both styles
    but spaces interact better with prompt parsing).

    When solo_mode == "solo", also strips multi-character indicators
    (2girls, multiple_girls, duo, group, etc.) since those would teach the
    LoRA something that contradicts the rest of the dataset.
    """
    if not tag_string:
        return ""
    # Build the strip set based on context
    strip_set = set(CAPTION_BLACKLIST)
    if solo_mode == "solo":
        strip_set |= MULTI_INDICATORS
    tags = [t.strip() for t in tag_string.split() if t.strip()]
    tags = [t for t in tags if t.lower() not in strip_set]
    # Convert underscores to spaces in tags (keep multi-word tags joined)
    return ", ".join(t.replace("_", " ") for t in tags)


STILL_IMG_EXTS = {"jpg", "jpeg", "png", "webp"}

# Tag sets used by the booru filter to verify "actually solo".
# The booru 'solo' tag alone is unreliable (background characters slip through).
# We additionally require one of SOLO_INDICATORS to be present AND none of
# MULTI_INDICATORS.
SOLO_INDICATORS = {"solo", "1girl", "1boy", "1other", "1futa"}
MULTI_INDICATORS = {
    "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls",
    "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple_boys",
    "multiple_others", "multiple_persons",
    "duo", "trio", "group", "couple", "harem",
}  # excludes gif, mp4, webm, zip
DANBOORU_RATING_LETTER = {
    "safe": {"g"},
    "safe+sensitive": {"g", "s"},
    "questionable": {"q"},
    "explicit": {"e"},
    "all": {"g", "s", "q", "e"},
}


def _search_danbooru(tag: str, rating: str, count: int, log,
                     official_only: bool = False, solo_mode: str = "solo") -> list:
    """Returns list of {url, tags, source} dicts. Anonymous accounts are
    limited to 2 tags server-side.

    solo_mode:
      "solo"     - require single character (default for training)
      "multiple" - exclude solo (group shots, duets, scenes)
      "any"      - no constraint
    """
    # Pick the best 2-tag combo for the constraints requested
    if solo_mode == "solo":
        solo_part = "solo"
    elif solo_mode == "multiple":
        solo_part = "-solo"
    else:
        solo_part = ""

    if solo_part and official_only:
        # 2 tags max - solo/-solo server-side, filter official_art client-side
        tags_part = "{} {}".format(tag, solo_part)
        filter_official_client = True
    elif solo_part:
        tags_part = "{} {}".format(tag, solo_part)
        filter_official_client = False
    elif official_only:
        # No solo constraint - put official_art server-side for max efficiency
        tags_part = "{} official_art".format(tag)
        filter_official_client = False
    else:
        tags_part = tag
        filter_official_client = False

    # Over-fetch more aggressively (8x) when client-side filtering will drop posts
    overshoot = 8 if filter_official_client else 4
    over_count = min(int(count) * overshoot, 200)
    url = "{}/posts.json?tags={}&limit={}".format(
        DANBOORU_BASE, _urlparse.quote(tags_part), over_count)
    log("{} querying Danbooru: {}".format(TAG, url))
    try:
        data = _http_get_json(url, timeout=30)
    except Exception as e:
        log("{} Danbooru query failed: {}".format(TAG, e))
        return []

    allowed_ratings = DANBOORU_RATING_LETTER.get(rating, {"g"})
    out = []
    skipped_ext = 0
    skipped_rating = 0
    skipped_fanart = 0
    skipped_not_solo = 0
    for p in data:
        file_url = p.get("file_url") or p.get("large_file_url")
        if not file_url:
            continue
        ext = (p.get("file_ext") or "").lower()
        if ext not in STILL_IMG_EXTS:
            skipped_ext += 1
            continue
        r = (p.get("rating") or "").lower()
        if r and r not in allowed_ratings:
            skipped_rating += 1
            continue
        tag_str = p.get("tag_string_general", "") + " " + p.get("tag_string_character", "")
        # Official-art filter (only if not already handled server-side).
        tag_words = set(tag_str.split())
        if filter_official_client and "official_art" not in tag_words:
            skipped_fanart += 1
            continue
        # Strict solo verification when in solo mode: the post must have one
        # of the solo indicators AND none of the multi-character indicators.
        # Catches misleading posts that have 'solo' tag but actually contain
        # background characters or duos.
        if solo_mode == "solo":
            if not (tag_words & SOLO_INDICATORS):
                skipped_not_solo += 1
                continue
            if tag_words & MULTI_INDICATORS:
                skipped_not_solo += 1
                continue
        elif solo_mode == "multiple":
            # In "multiple" mode, require an explicit multi indicator
            if not (tag_words & MULTI_INDICATORS):
                skipped_not_solo += 1
                continue
        out.append({
            "url": file_url,
            "tags": tag_str.strip(),
            "ext": ext or "jpg",
            "score": p.get("score", 0),
            "source": "danbooru",
        })
    log("{} Danbooru: kept {} (dropped {} non-still, {} wrong rating, {} fan-art, {} solo-check)".format(
        TAG, len(out), skipped_ext, skipped_rating, skipped_fanart, skipped_not_solo))
    return out


GELBOORU_RATING_NAME = {
    "safe": {"general"},
    "safe+sensitive": {"general", "sensitive"},
    "questionable": {"questionable"},
    "explicit": {"explicit"},
    "all": {"general", "sensitive", "questionable", "explicit"},
}


def _search_gelbooru(tag: str, rating: str, count: int, log,
                     official_only: bool = False, solo_mode: str = "solo") -> list:
    """Gelbooru has no 2-tag limit, so we can layer constraints freely.
    solo_mode: 'solo' / 'multiple' / 'any'.
    """
    parts = [tag]
    if solo_mode == "solo":
        parts.append("solo")
    elif solo_mode == "multiple":
        parts.append("-solo")
    if official_only:
        parts.append("official_art")
    tags_part = " ".join(parts)
    # Server-side rating filter (Gelbooru supports it) for the easy cases.
    if rating in ("safe", "questionable", "explicit"):
        rating_q = {"safe": "general"}.get(rating, rating)
        tags_part = "{} rating:{}".format(tags_part, rating_q)

    overshoot = 8 if official_only else 4
    over_count = min(int(count) * overshoot, 200)
    url = ("{}/index.php?page=dapi&s=post&q=index&json=1"
           "&tags={}&limit={}&sort=score:desc").format(
        GELBOORU_BASE, _urlparse.quote(tags_part), over_count)
    log("{} querying Gelbooru: {}".format(TAG, url))
    try:
        data = _http_get_json(url, timeout=30)
    except Exception as e:
        log("{} Gelbooru query failed: {}".format(TAG, e))
        return []
    posts = data.get("post", []) if isinstance(data, dict) else []
    if not isinstance(posts, list):
        posts = []

    allowed = GELBOORU_RATING_NAME.get(rating, {"general"})
    out = []
    skipped_ext = 0
    skipped_rating = 0
    skipped_not_solo = 0
    for p in posts:
        file_url = p.get("file_url")
        if not file_url:
            continue
        ext_raw = file_url.rsplit(".", 1)[-1].split("?")[0].lower() if "." in file_url else ""
        if ext_raw not in STILL_IMG_EXTS:
            skipped_ext += 1
            continue
        r = (p.get("rating") or "").lower()
        if r and r not in allowed:
            skipped_rating += 1
            continue
        tag_words = set((p.get("tags") or "").split())
        if solo_mode == "solo":
            if not (tag_words & SOLO_INDICATORS):
                skipped_not_solo += 1
                continue
            if tag_words & MULTI_INDICATORS:
                skipped_not_solo += 1
                continue
        elif solo_mode == "multiple":
            if not (tag_words & MULTI_INDICATORS):
                skipped_not_solo += 1
                continue
        out.append({
            "url": file_url,
            "tags": p.get("tags", ""),
            "ext": ext_raw,
            "score": p.get("score", 0),
            "source": "gelbooru",
        })
    log("{} Gelbooru: kept {} (dropped {} non-still, {} wrong rating, {} solo-check)".format(
        TAG, len(out), skipped_ext, skipped_rating, skipped_not_solo))
    return out


def list_downloaded_sets() -> list:
    """Return [(label, path)] of existing booru download folders, newest first.
    Label format: '<name> (<image_count> images)'."""
    if not DOWNLOADS_DIR.exists():
        return []
    out = []
    for d in DOWNLOADS_DIR.iterdir():
        if not d.is_dir():
            continue
        imgs = (list(d.glob("*.png")) + list(d.glob("*.jpg")) +
                list(d.glob("*.jpeg")) + list(d.glob("*.webp")))
        if not imgs:
            continue
        out.append({
            "label": "{} ({} images)".format(d.name, len(imgs)),
            "path": str(d),
            "mtime": d.stat().st_mtime,
        })
    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out


def tag_my_uploads(tag: str, image_paths: list, count: int, log,
                   trigger: str = "", official_only: bool = True) -> Path:
    """Use booru as a TAG SOURCE for your own uploaded images.

    Same downstream effect as search_booru but the images come from the user.
    We hit the booru APIs only to learn the character's typical tag set
    (no image downloads from booru), then:
      1. Copy each uploaded image to DOWNLOADS_DIR/<slug>_custom/
      2. Detect the character's defining features from booru tag frequencies
      3. Write a caption .txt for each image: "<trigger>, <feature1>, ..."
      4. Return the directory so the trainer picks it up just like a booru download.
    """
    tag = (tag or "").strip()
    if not tag:
        raise ValueError("No character tag provided.")
    if not image_paths:
        raise ValueError("No uploaded images.")

    slug = _slugify(tag) + "_custom"
    dest = DOWNLOADS_DIR / slug
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    log("{} === tagging {} uploads from booru reference '{}' ===".format(
        TAG, len(image_paths), tag))

    # Step 1: Query booru for tag data only (no image download).
    # We use rating="all" for the tag query because we only care about tags,
    # not the images. Larger sample = more accurate feature detection.
    sample_count = max(int(count) * 2, 30)
    # solo_mode="any": tag_my_uploads is using booru as tag reference only,
    # not pulling images from it. Group shots and scene art are fine sources
    # of tag data - the user's own uploaded images are what's used for training.
    results = _search_danbooru(tag, "all", sample_count, log,
                               official_only=official_only, solo_mode="any")
    if len(results) < 20:
        more = _search_gelbooru(tag, "all", sample_count - len(results), log,
                                official_only=official_only, solo_mode="any")
        seen = {r["url"] for r in results}
        for r in more:
            if r["url"] not in seen:
                results.append(r)
                seen.add(r["url"])

    if not results:
        hint = ("Check the tag spelling - Danbooru format with underscores like "
                "'seele_(honkai_impact)'.")
        if official_only:
            hint += ("  Also try unchecking 'Official art only' - some "
                     "characters only have fan art on these boorus.")
        raise RuntimeError("No booru posts found for '{}'. {}".format(tag, hint))

    log("{} sampled {} posts from booru for tag-frequency analysis".format(TAG, len(results)))

    # Step 2: Compute tag frequencies from the booru sample.
    # Also exclude the character tag itself - it'd be counted as a 'feature'
    # otherwise, but it's already the trigger.
    import collections
    counts = collections.Counter()
    char_tag_lower = tag.lower()
    # Pre-build a set of forms to skip (tag with/without parens, slugified, etc.)
    skip_tags = {
        char_tag_lower,
        char_tag_lower.replace("_", " "),
        _tag_to_trigger(tag).lower(),
    }
    for r in results:
        seen_here = set()
        for raw in r.get("tags", "").split():
            t = raw.strip().lower()
            if not t or t in seen_here:
                continue
            seen_here.add(t)
            if (t in GENERIC_TAGS or t in CAPTION_BLACKLIST
                    or t in STYLE_TAGS or t in MULTI_INDICATORS):
                continue
            if t in skip_tags:
                continue
            counts[t] += 1

    # Diagnostics: top 15 most-common tags with their counts
    top15 = counts.most_common(15)
    log("{} top tag counts in sample (out of {} posts):".format(TAG, len(results)))
    for tname, c in top15:
        pct = (c * 100.0) / max(1, len(results))
        log("{}   {:<30} {:>3}/{:<3} ({:.0f}%)".format(TAG, tname, c, len(results), pct))

    # Step 2b: Tag selection. 60% frequency threshold - matches what the user
    # observed as the natural break between defining features and incidental tags.
    # If nothing meets threshold, fall back to top-N regardless - we'd rather
    # have noisy features than a caption with just the trigger.
    threshold = max(2, int(len(results) * 0.6))
    features = [t for t, c in top15 if c >= threshold]
    if not features:
        # Fallback: just take the top 8 most-common tags. Anything is better
        # than zero features when training.
        features = [t for t, _ in top15[:8]]
        log("{} no tags met 60% threshold; using top-8 most-common as fallback".format(TAG))
    else:
        features = features[:12]

    # Convert to space-form for caption use
    features = [t.replace("_", " ") for t in features]

    log("{} final character features ({}): {}".format(
        TAG, len(features), ", ".join(features) if features else "(none)"))

    # Step 3: Copy uploaded images to dest + write per-image varied captions.
    # Each uploaded image gets:
    #   <trigger>, <character_features...>, <unique per-post tags from booru>
    # The trigger+features stay pinned at the front so the LoRA learns them
    # as core character markers; the trailing portion varies per image so the
    # LoRA learns diverse activation contexts (pose, outfit, scene).
    import random
    trig_clean = (trigger or "").strip().strip(",")
    feature_set_lower = {f.lower() for f in features}
    feat_str = ", ".join(features)

    def _per_image_caption(post_tags_raw: str) -> str:
        """Build caption: trigger + features + this post's other tags."""
        # Parse this booru post's tags. Strip features (no dups), generics,
        # blacklist. Keep order from the post.
        seen = set()
        extra = []
        for raw in post_tags_raw.split():
            t = raw.strip().lower()
            if not t or t in seen:
                continue
            seen.add(t)
            if t in GENERIC_TAGS or t in CAPTION_BLACKLIST:
                continue
            # tag_my_uploads always uses solo_mode='any' currently, but be
            # defensive: if a future call passes solo_mode='solo', also strip
            # multi-character indicators.
            if t in MULTI_INDICATORS:
                continue
            if t in skip_tags:  # the character tag itself
                continue
            t_space = t.replace("_", " ")
            if t_space.lower() in feature_set_lower:
                continue  # already in the features section, don't repeat
            extra.append(t_space)

        parts = []
        if trig_clean:
            parts.append(trig_clean)
        if feat_str:
            parts.append(feat_str)
        if extra:
            parts.append(", ".join(extra))
        return ", ".join(parts) if parts else trig_clean

    # If we have more uploads than booru results, sample with replacement.
    # If fewer, sample without replacement so each upload gets a distinct post.
    rng = random.Random(42)  # deterministic so re-runs are reproducible
    if len(results) >= len(image_paths):
        chosen = rng.sample(results, len(image_paths))
    else:
        chosen = [rng.choice(results) for _ in image_paths]

    copied = 0
    for i, src in enumerate(image_paths):
        src_path = Path(src)
        if not src_path.exists():
            log("{} skip missing upload: {}".format(TAG, src_path))
            continue
        ext = src_path.suffix.lower().lstrip(".")
        if ext not in STILL_IMG_EXTS:
            log("{} skip non-still upload: {}".format(TAG, src_path))
            continue
        dst = dest / "{:03d}_{}.{}".format(i, src_path.stem[:40], ext)
        try:
            shutil.copy2(src_path, dst)
            caption = _per_image_caption(chosen[i].get("tags", ""))
            dst.with_suffix(".txt").write_text(caption, encoding="utf-8")
            copied += 1
        except Exception as e:
            log("{} failed to copy {}: {}".format(TAG, src_path, e))

    if copied == 0:
        raise RuntimeError("All uploads failed to copy.")

    log("{} copied {} uploads + wrote per-image captions to {}".format(TAG, copied, dest))
    return dest


def search_booru(tag: str, rating: str, count: int, log, trigger: str = "",
                 official_only: bool = True, solo_mode: str = "solo") -> Path:
    """Search Danbooru first; if fewer than count results, top up from Gelbooru.
    Downloads images + writes .txt caption files into DOWNLOADS_DIR/<slug>/.

    If `trigger` is non-empty, it's prepended to every caption file as the
    first tag - this is the LoRA's activation token, so it MUST appear
    consistently at the front of every training caption. Saved into the
    download folder so the trigger is baked in even if you reuse this
    folder later.

    Returns the download directory."""
    tag = (tag or "").strip()
    if not tag:
        raise ValueError("No tag provided.")

    slug = _slugify(tag)
    dest = DOWNLOADS_DIR / slug
    # Wipe and re-create so re-searches don't pile up stale images
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    log("{} === searching for '{}' (rating={}, want={}) ===".format(TAG, tag, rating, count))
    results = _search_danbooru(tag, rating, count, log,
                                official_only=official_only, solo_mode=solo_mode)
    if len(results) < count:
        more = _search_gelbooru(tag, rating, count - len(results), log,
                                official_only=official_only, solo_mode=solo_mode)
        # de-dupe by URL
        seen = {r["url"] for r in results}
        for r in more:
            if r["url"] not in seen:
                results.append(r)
                seen.add(r["url"])

    # Sort by score desc, take top N
    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    results = results[:int(count)]

    if not results:
        official_hint = ""
        if official_only:
            official_hint = ("\nIf this is a niche character, try unchecking "
                             "'Official art only' - official art is rare for "
                             "some characters.")
        raise RuntimeError("No images found for tag '{}'. Check the tag spelling - "
                           "use Danbooru format with underscores, e.g. 'seele_(honkai_impact)'.{}".format(tag, official_hint))
    if official_only and len(results) < count // 2:
        log("{} WARNING: only found {} official-art posts (asked for {}). "
            "Consider unchecking 'Official art only' if you want more variety.".format(
            TAG, len(results), count))

    downloaded = 0
    for i, r in enumerate(results):
        ext = (r.get("ext") or "jpg").lower()
        # Hard guard: skip animated/video posts even if they slipped through.
        if ext not in STILL_IMG_EXTS:
            log("{} [{:>3}] skipping {}: animated/video".format(TAG, i + 1, r.get("url", "?")))
            continue
        img_path = dest / "{:03d}.{}".format(i, ext)
        if _http_download(r["url"], img_path):
            cap = _filter_tags(r.get("tags", ""), solo_mode=solo_mode)
            # Bake in the trigger word as the first tag if provided. Kohya
            # uses tag order as a soft signal during training, so putting the
            # trigger first reinforces it as the LoRA's activation token.
            trig = (trigger or "").strip().strip(",")
            if trig:
                # Already-present trigger (re-download or manual edit) is
                # tolerated - strip a leading duplicate.
                if cap.lower() == trig.lower():
                    cap = trig
                elif cap.lower().startswith(trig.lower() + ","):
                    cap = "{}, {}".format(trig, cap[len(trig) + 1:].lstrip())
                else:
                    cap = "{}, {}".format(trig, cap) if cap else trig
            img_path.with_suffix(".txt").write_text(cap, encoding="utf-8")
            downloaded += 1
            log("{} [{:>3}] downloaded {} (score {}, src {})".format(
                TAG, i + 1, img_path.name, r.get("score", "?"), r.get("source", "?")))
        else:
            log("{} [{:>3}] FAILED to download {}".format(TAG, i + 1, r["url"]))

    if downloaded == 0:
        raise RuntimeError("All {} downloads failed. Network issue?".format(len(results)))

    log("{} downloaded {} images + captions to {}".format(TAG, downloaded, dest))

    # ----------------------------------------------------------------------
    # Identify the character's defining features (tags present in >=50% of
    # the downloads, excluding generic ones) and promote them to the front
    # of every caption right after the trigger. This gives the LoRA stable
    # "always-on" markers AND makes those tags available for fine-grained
    # control when generating.
    # ----------------------------------------------------------------------
    features = _compute_character_features(dest, top_n=8, min_frequency=0.5)
    if features:
        log("{} identified defining features ({} tags): {}".format(
            TAG, len(features), ", ".join(features)))
        feat_str = ", ".join(features)
        feat_lower = [f.lower() for f in features]
        trig_clean = (trigger or "").strip().strip(",")
        for txt in dest.glob("*.txt"):
            try:
                existing = txt.read_text(encoding="utf-8").strip()
                # Strip the trigger off the front (we'll re-add at the very
                # end so it stays first), then strip any feature tags from
                # the rest so they only appear once after promotion.
                rest = existing
                if trig_clean and rest.lower().startswith(trig_clean.lower() + ","):
                    rest = rest[len(trig_clean) + 1:].lstrip()
                elif trig_clean and rest.lower() == trig_clean.lower():
                    rest = ""
                # Filter remaining tags to drop features (avoid duplicates)
                remaining = []
                for raw in rest.split(","):
                    t = raw.strip()
                    if t and t.lower() not in feat_lower:
                        remaining.append(t)
                parts = []
                if trig_clean:
                    parts.append(trig_clean)
                parts.append(feat_str)
                if remaining:
                    parts.append(", ".join(remaining))
                txt.write_text(", ".join(parts), encoding="utf-8")
            except Exception:
                pass
        log("{} promoted features to front of {} captions".format(
            TAG, len(list(dest.glob("*.txt")))))
    else:
        log("{} not enough images to identify stable defining features (need >=3)".format(TAG))

    return dest


# ===========================================================================
# Image preparation
# ===========================================================================

TARGET_BASE = 1024


def _resize_to_bucket(im, base: int = TARGET_BASE):
    from PIL import Image
    w, h = im.size
    if w >= h:
        new_w = base
        new_h = int(round(h * base / w))
    else:
        new_h = base
        new_w = int(round(w * base / h))
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)
    return im.resize((new_w, new_h), Image.LANCZOS)


def prepare_dataset(project_name: str, image_paths: List[str], repeats: int,
                    class_token: str, log) -> Path:
    from PIL import Image, ImageOps

    project_root = PROJECTS_DIR / project_name
    if project_root.exists():
        shutil.rmtree(project_root)

    images_parent = project_root / "images"
    bucket_dir = images_parent / "{}_{}".format(repeats, class_token)
    bucket_dir.mkdir(parents=True, exist_ok=True)

    out_count = 0
    for src in image_paths:
        src_path = Path(src)
        if not src_path.exists():
            log("{} skip missing: {}".format(TAG, src_path))
            continue
        try:
            with Image.open(src_path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im = _resize_to_bucket(im)
                out_path = bucket_dir / "{:03d}_{}.png".format(out_count, src_path.stem)
                im.save(out_path, format="PNG")
                # If a .txt caption file lives next to the source image, copy
                # it across so we keep human-written tags (e.g. booru tags).
                src_txt = src_path.with_suffix(".txt")
                if src_txt.exists():
                    shutil.copy2(src_txt, out_path.with_suffix(".txt"))
                out_count += 1
        except Exception as e:
            log("{} failed to load {}: {}".format(TAG, src_path, e))

    if out_count == 0:
        raise RuntimeError("No usable input images.")
    log("{} prepared {} images at {}".format(TAG, out_count, bucket_dir))
    return images_parent


# ===========================================================================
# Captioning
# ===========================================================================

WD14_MODEL_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"


def _run_subprocess(cmd, cwd, log):
    log("{} $ {}".format(TAG, " ".join(str(c) for c in cmd)))
    # Force the child Python to write stdout/stderr as UTF-8. Without this,
    # Windows defaults to cp1252 and kohya's Japanese log strings cause
    # UnicodeEncodeError mid-training. Also set PYTHONUTF8=1 as a belt-and-
    # suspenders for older interpreters that ignore PYTHONIOENCODING.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:  # type: ignore
        log(line.rstrip("\n"))
    proc.wait()
    return proc.returncode


def autocaption_wd14(bucket_dir: Path, log) -> bool:
    tagger_script = SD_SCRIPTS_DIR / "finetune" / "tag_images_by_wd14_tagger.py"
    if not tagger_script.exists():
        log("{} WD14 script not found, skipping autocaption.".format(TAG))
        return False
    rc = _run_subprocess(
        [str(_venv_python()), str(tagger_script), str(bucket_dir),
         "--repo_id", WD14_MODEL_REPO,
         "--onnx",
         "--thresh", "0.35", "--batch_size", "4",
         "--caption_extension", ".txt",
         "--remove_underscore",
         "--undesired_tags",
         "lowres,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry"],
        cwd=SD_SCRIPTS_DIR, log=log)
    return rc == 0


def autocaption_blip(bucket_dir: Path, log) -> bool:
    script = SD_SCRIPTS_DIR / "finetune" / "make_captions.py"
    if not script.exists():
        log("{} BLIP script not found, skipping autocaption.".format(TAG))
        return False
    rc = _run_subprocess(
        [str(_venv_python()), str(script), str(bucket_dir),
         "--batch_size", "4", "--max_length", "75",
         "--caption_extension", ".txt"],
        cwd=SD_SCRIPTS_DIR, log=log)
    return rc == 0


def inject_trigger_word(bucket_dir: Path, trigger: str, log):
    trigger = (trigger or "").strip().strip(",")
    if not trigger:
        return
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    n = 0
    for img in bucket_dir.iterdir():
        if img.suffix.lower() not in image_exts:
            continue
        caption_path = img.with_suffix(".txt")
        existing = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else ""
        if existing.lower() == trigger.lower():
            existing = ""
        elif existing.lower().startswith(trigger.lower() + ","):
            existing = existing[len(trigger) + 1:].lstrip()
        merged = "{}, {}".format(trigger, existing) if existing else trigger
        caption_path.write_text(merged, encoding="utf-8")
        n += 1
    log("{} injected trigger '{}' into {} caption(s).".format(TAG, trigger, n))


# ===========================================================================
# Training config + launch
# ===========================================================================

def write_training_toml(project_root, train_data_dir, output_dir, base_model,
                        preset, project_name, resolution, batch_size, log):
    """Write kohya TOML config. Falls back to hand-rolled emitter if tomli_w
    isn't available in Forge's venv (it usually isn't)."""
    config = {
        "model_arguments": {
            "pretrained_model_name_or_path": str(base_model),
            "v2": False, "v_parameterization": False,
        },
        "additional_network_arguments": {
            "no_metadata": False,
            "network_module": "networks.lora",
            "network_dim": int(preset["network_dim"]),
            "network_alpha": float(preset["network_alpha"]),
            # MUST be True when cache_text_encoder_outputs is True - kohya asserts this.
            # For 8-12GB cards on SDXL Illustrious this is the right trade-off:
            # huge VRAM/speed win, mild quality loss on text-side activations.
            "network_train_unet_only": True,
            "network_train_text_encoder_only": False,
        },
        "optimizer_arguments": {
            "optimizer_type": preset["optimizer"],
            "learning_rate": float(preset["learning_rate"]),
            "max_grad_norm": 1.0,
            "lr_scheduler": preset["lr_scheduler"],
            "lr_warmup_steps": 0,
        },
        "dataset_arguments": {
            "cache_latents": True, "debug_dataset": False,
            "vae_batch_size": 1,
            # IMPORTANT: kohya defaults this to ".caption" but our code (booru
            # downloader + WD14 tagger + manual users) writes ".txt" files.
            # Without this line kohya silently ignores all captions.
            "caption_extension": ".txt",
            "resolution": "{},{}".format(int(resolution), int(resolution)),
            "enable_bucket": True,
            "min_bucket_reso": 320, "max_bucket_reso": 1536,
            "bucket_reso_steps": 64, "bucket_no_upscale": False,
        },
        "training_arguments": {
            "output_dir": str(output_dir),
            "output_name": project_name,
            "save_precision": "fp16",
            "save_every_n_epochs": max(1, int(preset["num_epochs"]) // 3),
            "save_model_as": "safetensors",
            "max_train_epochs": int(preset["num_epochs"]),
            "train_batch_size": int(batch_size),
            "max_token_length": 225,
            "sdpa": True, "xformers": False,
            "max_data_loader_n_workers": 2,
            "persistent_data_loader_workers": True,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "fp16",
            "logging_dir": str(project_root / "logs"),
            "log_prefix": project_name,
            "noise_offset": 0.0357,
            "adaptive_noise_scale": 0.00357,
            "multires_noise_iterations": 6,
            "multires_noise_discount": 0.3,
            "cache_text_encoder_outputs": True,
            "cache_text_encoder_outputs_to_disk": True,
            "no_half_vae": True,
            "min_snr_gamma": 5,
        },
        "saving_arguments": {"save_state": False},
    }

    def emit_value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        return '"{}"'.format(str(v).replace('\\', '\\\\').replace('"', '\\"'))

    lines = []
    for section, vals in config.items():
        lines.append("[{}]".format(section))
        for k, v in vals.items():
            lines.append("{} = {}".format(k, emit_value(v)))
        lines.append("")

    toml_path = project_root / "config.toml"
    toml_path.write_text("\n".join(lines), encoding="utf-8")
    log("{} wrote config to {}".format(TAG, toml_path))
    return toml_path


def build_train_command(toml_path: Path, train_data_dir: Path) -> list:
    return [
        str(_venv_python()), "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", "2",
        "--num_processes", "1",
        "--num_machines", "1",
        "--mixed_precision", "fp16",
        "--dynamo_backend", "no",
        str(SD_SCRIPTS_DIR / "sdxl_train_network.py"),
        "--config_file", str(toml_path),
        "--train_data_dir", str(train_data_dir),
    ]


# ===========================================================================
# Bootstrap check
# ===========================================================================

def ensure_ready(log) -> bool:
    """Check that the user has run setup_venv.bat. Don't try to install from
    inside the WebUI - that's setup_venv.bat's job, with proper interpreter
    selection."""
    venv_python = _venv_python()
    train_script = SD_SCRIPTS_DIR / "sdxl_train_network.py"
    if not train_script.exists() or not venv_python.exists():
        log("{} sd-scripts or sandboxed venv missing.".format(TAG))
        log("{} run setup_venv.bat in the extension folder to install:".format(TAG))
        log("{}   {}".format(TAG, EXT_ROOT / "setup_venv.bat"))
        return False
    return True


# ===========================================================================
# Orchestration (called from the UI button)
# ===========================================================================

def _slugify(s: str) -> str:
    """Filename-safe slug. Returns empty string for empty input (callers
    that need a default must supply their own fallback - we no longer hide
    a missing project name behind a silent 'lora' default)."""
    import re
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _tag_to_trigger(tag: str) -> str:
    """Derive a sensible trigger word from a booru tag.
    Examples:
      seele_(honkai_impact)        -> seele
      kafka_(honkai:_star_rail)    -> kafka
      nahida_(genshin_impact)      -> nahida
      raiden_shogun                -> raiden_shogun
      hatsune_miku                 -> hatsune_miku
    Strips parenthesized suffixes (which are usually disambiguators), then
    slugifies whatever's left."""
    import re
    if not tag:
        return ""
    # Cut off at the first '(' or '_(' so we keep the core character name
    s = re.split(r"\s*\(", tag, maxsplit=1)[0]
    s = re.split(r"_+\(", s, maxsplit=1)[0]
    return _slugify(s)


def run_training_job(project_name, trigger_word, base_model_name, preset_name,
                     caption_mode, image_paths, network_dim, network_alpha,
                     learning_rate, num_epochs, repeats, resolution,
                     batch_size, log):
    project_name = _slugify(project_name)
    if not project_name:
        raise ValueError("Project name is required.")
    if not image_paths:
        raise ValueError("No images uploaded.")
    base_model_path = _checkpoints_dir() / base_model_name
    if not base_model_path.exists():
        raise FileNotFoundError("Base model not found: {}".format(base_model_path))

    log("{} ==== job '{}' ====".format(TAG, project_name))
    log("{} trigger: {!r}  preset: {}  images: {}".format(
        TAG, trigger_word, preset_name, len(image_paths)))

    if not ensure_ready(log):
        raise RuntimeError("Setup not complete - run setup_venv.bat first.")

    preset = dict(PRESETS[preset_name])
    if network_dim:
        preset["network_dim"] = int(network_dim)
    if network_alpha:
        preset["network_alpha"] = float(network_alpha)
    if learning_rate:
        preset["learning_rate"] = float(learning_rate)
    if num_epochs:
        preset["num_epochs"] = int(num_epochs)
    repeats_eff = int(repeats) if repeats else int(preset["repeats"])

    class_token = _slugify(trigger_word) or _slugify(project_name) or "lora"
    train_data_dir = prepare_dataset(project_name, image_paths, repeats_eff,
                                     class_token, log)
    bucket_dir = train_data_dir / "{}_{}".format(repeats_eff, class_token)

    # If captions already exist (e.g. from booru search), skip autocaption.
    existing_caps = list(bucket_dir.glob("*.txt"))
    if existing_caps:
        log("{} found {} existing caption file(s), skipping autocaption.".format(TAG, len(existing_caps)))
    elif caption_mode == "WD14 tagger (anime / Illustrious)":
        log("{} running WD14 autocaption...".format(TAG))
        autocaption_wd14(bucket_dir, log)
    elif caption_mode == "BLIP (photo / realistic)":
        log("{} running BLIP autocaption...".format(TAG))
        autocaption_blip(bucket_dir, log)
    else:
        log("{} skipping autocaption (trigger-only mode).".format(TAG))

    inject_trigger_word(bucket_dir, trigger_word, log)

    project_root = PROJECTS_DIR / project_name
    out_dir = project_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    toml_path = write_training_toml(
        project_root, train_data_dir, out_dir, base_model_path,
        preset, project_name, resolution, batch_size, log)

    cmd = build_train_command(toml_path, train_data_dir)
    rc = _run_subprocess(cmd, cwd=SD_SCRIPTS_DIR, log=log)
    if rc != 0:
        raise RuntimeError("training exited with code {}".format(rc))

    final = out_dir / "{}.safetensors".format(project_name)
    if not final.exists():
        cands = sorted(out_dir.glob("{}*.safetensors".format(project_name)),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            raise FileNotFoundError("No output LoRA in {}".format(out_dir))
        final = cands[-1]

    dest = _lora_output_dir() / "{}.safetensors".format(project_name)
    shutil.copy2(final, dest)
    log("{} DONE. LoRA at: {}".format(TAG, dest))
    log("{} use it as: <lora:{}:1>".format(TAG, project_name))
    return dest


def _list_base_models() -> list:
    d = _checkpoints_dir()
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.safetensors")):
        out.append(p.name)
    for p in sorted(d.glob("*.ckpt")):
        out.append(p.name)
    return out


# ===========================================================================
# Gradio UI tab
# ===========================================================================

# Module-level job state - shared by start/poll callbacks.
JOB = {"running": False, "log_buf": []}


def _job_log(s: str):
    JOB["log_buf"].append(s)
    logger.info("{} {}".format(TAG, s))


def _build_tab():
    """Build the tab. Wrapped so any error during construction is loud."""
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(textwrap.dedent("""
        ## LoRA Trainer
        Drop 5-15 reference images, type a trigger word, pick a preset and a base SDXL model, then click **Start training**.

        - Before the first training run, double-click `extensions/sd-forge-lora-trainer/setup_venv.bat` to install kohya-ss/sd-scripts in a sandboxed venv (~5-7 GB, one time, 5-15 min).
        - Training a small character LoRA on SDXL typically takes 20-60 min on a 3070/3080 GPU.
        - The output LoRA is copied to `models/Lora/` automatically.
        """).strip())

        with gr.Row():
            with gr.Column(scale=1):
                project_name = gr.Textbox(label="Project name",
                                          placeholder="e.g. mychar_v1")
                trigger_word = gr.Textbox(label="Trigger word",
                                          placeholder="e.g. mychar1, sks_dog")
                base_model = gr.Dropdown(label="Base model (from models/Stable-diffusion)",
                                         choices=_list_base_models(), value=None,
                                         interactive=True)
                refresh_btn = gr.Button("Refresh model list")
                preset = gr.Radio(label="Preset",
                                  choices=list(PRESETS.keys()), value="Character")
                caption_mode = gr.Radio(label="Captioning",
                                        choices=CAPTION_MODES, value=CAPTION_MODES[0])

            with gr.Column(scale=1):
                images = gr.Files(label="Training images (drag 5-15 here)",
                                  file_count="multiple", file_types=["image"])

                with gr.Accordion("Or: find images on Booru (Danbooru + Gelbooru)", open=False):
                    gr.Markdown(
                        "Type a character tag in Danbooru format "
                        "(e.g. `seele_(honkai_impact)`). Searches Danbooru first, "
                        "tops up from Gelbooru if needed, downloads the top-N by score "
                        "(solo-only, no gifs/videos), and saves the existing tags as "
                        "captions with the trigger word baked in as the first tag.\n\n"
                        "If you leave the **Trigger word** field at the top blank, "
                        "it gets auto-derived from the tag "
                        "(e.g. `seele_(honkai_impact)` → `seele`). Override it by typing "
                        "your own (e.g. `my_seele`) before clicking Search."
                    )
                    booru_tag = gr.Textbox(label="Character tag",
                                           placeholder="e.g. seele_(honkai_impact)")
                    booru_rating = gr.Dropdown(label="Rating",
                                               choices=["safe", "safe+sensitive",
                                                        "questionable", "explicit", "all"],
                                               value="safe")
                    booru_count = gr.Slider(label="How many images",
                                            minimum=5, maximum=50, step=5, value=15)
                    booru_official = gr.Checkbox(
                        label="Official art only (no fan art)",
                        value=True,
                        info="Filters to posts tagged 'official_art'. Best quality "
                             "for major characters; uncheck for niche characters "
                             "where official art is sparse.",
                    )
                    booru_solo_mode = gr.Dropdown(
                        label="Composition (Search & download only)",
                        choices=[
                            "Solo only (single character)",
                            "Multiple only (group shots, no solo)",
                            "Any (solo + groups)",
                        ],
                        value="Solo only (single character)",
                        info="Tag MY uploads ignores this - it uses 'any' "
                             "since you're providing the images.",
                    )
                    with gr.Row():
                        booru_search_btn = gr.Button("Search & download from Booru",
                                                     variant="primary")
                        booru_tag_uploads_btn = gr.Button("Tag MY uploads from this tag")
                    gr.Markdown(
                        "_**Search & download** pulls images from Booru. "
                        "**Tag MY uploads** uses Booru only for tag reference - "
                        "it'll auto-caption the images you already dragged into "
                        "the upload box above._"
                    )
                    booru_status = gr.Textbox(label="Search / tag status", value="",
                                              interactive=False, lines=2)

                with gr.Accordion("Edit individual captions before training", open=False):
                    gr.Markdown(
                        "After Search & download OR Tag MY uploads finishes, you can "
                        "review/edit each image's caption here. Pick an image from the "
                        "dropdown, edit the text, click **Save caption**.  Click "
                        "**Refresh image list** if the dropdown is empty after a search."
                    )
                    with gr.Row():
                        cap_image_dropdown = gr.Dropdown(
                            label="Image", choices=[],
                            value=None, interactive=True,
                        )
                        cap_refresh_btn = gr.Button("Refresh image list")
                    cap_image_preview = gr.Image(label="Preview", interactive=False,
                                                 height=240)
                    cap_textbox = gr.Textbox(
                        label="Caption (comma-separated tags)",
                        value="", lines=4, interactive=True,
                    )
                    cap_save_btn = gr.Button("Save caption", variant="primary")
                    cap_save_status = gr.Textbox(label="Editor status", value="",
                                                 interactive=False, lines=1)

                    gr.Markdown("**Or pick a folder you downloaded before:**")
                    PREV_PLACEHOLDER = "(no downloads yet — use the search above)"
                    _existing = list_downloaded_sets()
                    _existing_labels = [e["label"] for e in _existing]
                    # IMPORTANT: never start with choices=[] - Gradio drops the
                    # component value from click payloads in that state and the
                    # handler raises a ValueError on input validation.
                    if not _existing_labels:
                        _existing_labels = [PREV_PLACEHOLDER]
                    prev_sets = gr.Dropdown(
                        label="Previously downloaded",
                        choices=_existing_labels,
                        value=_existing_labels[0],
                        interactive=True,
                    )
                    with gr.Row():
                        prev_refresh_btn = gr.Button("Refresh list")
                        prev_use_btn = gr.Button("Use this folder", variant="primary")
                with gr.Accordion("Advanced settings", open=False):
                    gr.Markdown("Leave blank to use preset defaults.")
                    a_dim = gr.Number(label="network_dim (rank)", value=None, precision=0)
                    a_alpha = gr.Number(label="network_alpha", value=None, precision=2)
                    a_lr = gr.Number(label="learning_rate", value=None)
                    a_epochs = gr.Number(label="epochs", value=None, precision=0)
                    a_repeats = gr.Number(label="repeats per image", value=None, precision=0)
                    a_res = gr.Slider(label="resolution",
                                      minimum=512, maximum=1280, step=64, value=1024)
                    a_batch = gr.Slider(label="train_batch_size",
                                        minimum=1, maximum=4, step=1, value=1)

                start_btn = gr.Button("Start training", variant="primary")
                refresh_log_btn = gr.Button("Refresh log")
                status_box = gr.Textbox(label="Status", value="Idle.",
                                        interactive=False, lines=2)

        log_box = gr.Textbox(label="Training log",
                             value="", interactive=False, lines=24,
                             max_lines=200, show_copy_button=True,
                             # autoscroll=True keeps the textbox pinned to the
                             # bottom (tail -f style) instead of jumping to the
                             # top on every auto-refresh tick. Wrapped in a
                             # try/except via a separate `try:` block below
                             # would be over-engineering - autoscroll is
                             # standard in Gradio 4.x.
                             autoscroll=True)

        refresh_btn.click(
            fn=lambda: gr.update(choices=_list_base_models()),
            outputs=[base_model],
        )

        def _start(project_name, trigger_word, base_model_name, preset_name,
                   caption_mode, images_list, a_dim, a_alpha, a_lr, a_epochs,
                   a_repeats, a_res, a_batch):
            if JOB["running"]:
                return ("A training job is already running.",
                        "\n".join(JOB["log_buf"]))
            if not project_name or not project_name.strip():
                return ("Project name is required - pick something descriptive "
                        "like 'seele_v1'. It'll be the output filename.", "")
            if not trigger_word or not trigger_word.strip():
                return ("Trigger word is required - this is what you'll type in "
                        "prompts to activate the LoRA (e.g. 'seele1', 'sks_dog'). "
                        "Without it the LoRA has no unique activation tag.", "")
            if not base_model_name:
                return ("Pick a base model first.", "")
            # If user didn't upload, but a booru search produced a directory, use it.
            if not images_list:
                searched = JOB.get("searched_dir")
                if searched and Path(searched).exists():
                    # IMPORTANT: pass full-path STRINGS, not Path objects.
                    # The downstream extractor calls `.name` on items, and on
                    # Path objects that returns just the basename.
                    found = sorted(
                        list(Path(searched).glob("*.png")) +
                        list(Path(searched).glob("*.jpg")) +
                        list(Path(searched).glob("*.jpeg")) +
                        list(Path(searched).glob("*.webp"))
                    )
                    images_list = [str(p) for p in found]
                    _job_log("{} no manual upload - using {} booru-downloaded image(s) from {}".format(
                        TAG, len(images_list), searched))
                else:
                    return ("Drop training images OR use the Booru search first.", "")

            image_paths = []
            for f in images_list:
                if isinstance(f, str):
                    image_paths.append(f)
                elif isinstance(f, Path):
                    image_paths.append(str(f))
                elif isinstance(f, dict):
                    image_paths.append(f.get("name") or f.get("path"))
                elif hasattr(f, "name"):
                    # gr.File hands back NamedTemporaryFile-likes whose `.name`
                    # is the absolute path. Path objects are handled above.
                    image_paths.append(f.name)
                else:
                    image_paths.append(str(f))

            JOB["log_buf"] = []
            JOB["running"] = True

            def worker():
                try:
                    run_training_job(
                        project_name=project_name, trigger_word=trigger_word,
                        base_model_name=base_model_name, preset_name=preset_name,
                        caption_mode=caption_mode, image_paths=image_paths,
                        network_dim=int(a_dim) if a_dim else None,
                        network_alpha=float(a_alpha) if a_alpha else None,
                        learning_rate=float(a_lr) if a_lr else None,
                        num_epochs=int(a_epochs) if a_epochs else None,
                        repeats=int(a_repeats) if a_repeats else None,
                        resolution=int(a_res or 1024),
                        batch_size=int(a_batch or 1),
                        log=_job_log,
                    )
                    _job_log("{} ====== SUCCESS ======".format(TAG))
                except Exception as e:
                    _job_log("{} ERROR: {}: {}".format(TAG, type(e).__name__, e))
                    _job_log(traceback.format_exc())
                finally:
                    JOB["running"] = False

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            return ("Training started - click 'Refresh log' to update.",
                    "\n".join(JOB["log_buf"]))

        def _tail():
            return "\n".join(JOB["log_buf"][-500:])

        # --- Booru search handler ---
        def _do_booru_search(tag, rating, count, trig, official, solo_mode_label):
            if not tag or not tag.strip():
                return ("Type a character tag first.",
                        gr.update())  # leave trigger field unchanged

            # If user didn't fill in a trigger, derive one from the tag
            # automatically. seele_(honkai_impact) -> seele
            auto_derived = False
            trig_clean = (trig or "").strip()
            if not trig_clean:
                trig_clean = _tag_to_trigger(tag)
                auto_derived = True
                if not trig_clean:
                    return ("Couldn't derive a trigger from this tag - type one "
                            "manually in the Trigger word field at the top.",
                            gr.update())

            try:
                JOB["log_buf"] = []  # fresh log for the search
                # Map UI label to internal solo_mode value
                _solo_map = {
                    "Solo only (single character)": "solo",
                    "Multiple only (group shots, no solo)": "multiple",
                    "Any (solo + groups)": "any",
                }
                solo_mode = _solo_map.get(solo_mode_label, "solo")
                dest = search_booru(tag.strip(), rating, int(count), _job_log,
                                    trigger=trig_clean, official_only=bool(official),
                                    solo_mode=solo_mode)
                imgs = list(dest.glob("*.png")) + list(dest.glob("*.jpg")) + list(dest.glob("*.jpeg")) + list(dest.glob("*.webp"))
                JOB["searched_dir"] = str(dest)
                # Read features back from one of the captions to surface them in UI.
                feat_preview = ""
                try:
                    sample = next(iter(dest.glob("*.txt")), None)
                    if sample:
                        first_line = sample.read_text(encoding="utf-8").strip()
                        # Skip the trigger (first item) - the rest leading edge
                        # contains the promoted feature tags
                        parts = [p.strip() for p in first_line.split(",")]
                        if parts and trig_clean and parts[0].lower() == trig_clean.lower():
                            parts = parts[1:]
                        feat_preview = ", ".join(parts[:8])
                except Exception:
                    pass

                msg = ("Downloaded {} images (trigger '{}'{} baked in) to {}.\n"
                       "Detected character features (promoted to front of every caption): {}\n"
                       "Click Start training.").format(
                    len(imgs), trig_clean,
                    " - auto-derived from the tag" if auto_derived else "",
                    dest,
                    feat_preview or "(too few images to detect features)")
                # Write the trigger back into the UI field so the user sees
                # what got picked and can edit it before training.
                trig_update = gr.update(value=trig_clean) if auto_derived else gr.update()
                return (msg, trig_update)
            except Exception as e:
                _job_log("{} search error: {}: {}".format(TAG, type(e).__name__, e))
                return ("ERROR: {}: {}".format(type(e).__name__, e), gr.update())

        booru_search_btn.click(
            fn=_do_booru_search,
            inputs=[booru_tag, booru_rating, booru_count, trigger_word,
                    booru_official, booru_solo_mode],
            outputs=[booru_status, trigger_word],
        )

        # --- Tag-my-uploads handler ---
        def _do_tag_my_uploads(tag, count, trig, images_list, official):
            if not tag or not tag.strip():
                return ("Type a character tag first (e.g. seele_(honkai_impact)).",
                        gr.update())
            if not images_list:
                return ("Drag your own images into the 'Training images' upload "
                        "box above before clicking this.", gr.update())

            # Auto-derive trigger if blank
            auto_derived = False
            trig_clean = (trig or "").strip()
            if not trig_clean:
                trig_clean = _tag_to_trigger(tag)
                auto_derived = True
                if not trig_clean:
                    return ("Couldn't derive a trigger - type one in the "
                            "Trigger word field at the top.", gr.update())

            # Extract real paths from Gradio's file objects
            image_paths = []
            for f in images_list:
                if isinstance(f, str):
                    image_paths.append(f)
                elif isinstance(f, Path):
                    image_paths.append(str(f))
                elif isinstance(f, dict):
                    image_paths.append(f.get("name") or f.get("path"))
                elif hasattr(f, "name"):
                    image_paths.append(f.name)
                else:
                    image_paths.append(str(f))

            try:
                JOB["log_buf"] = []
                dest = tag_my_uploads(tag.strip(), image_paths, int(count),
                                      _job_log, trigger=trig_clean,
                                      official_only=bool(official))
                imgs = list(dest.glob("*.png")) + list(dest.glob("*.jpg")) + list(dest.glob("*.jpeg")) + list(dest.glob("*.webp"))
                JOB["searched_dir"] = str(dest)

                # Preview the caption that got written
                preview = ""
                sample = next(iter(dest.glob("*.txt")), None)
                if sample:
                    preview = sample.read_text(encoding="utf-8").strip()

                msg = ("Tagged {} of your uploaded images using '{}' as booru reference.\n"
                       "Saved to: {}\n"
                       "Trigger: '{}'{}.  Caption: \"{}\"\n"
                       "Click Start training - your uploads will be used.").format(
                    len(imgs), tag.strip(), dest, trig_clean,
                    " (auto-derived)" if auto_derived else "",
                    preview[:200] + ("..." if len(preview) > 200 else ""))

                trig_update = gr.update(value=trig_clean) if auto_derived else gr.update()
                return (msg, trig_update)
            except Exception as e:
                _job_log("{} tag_my_uploads error: {}: {}".format(TAG, type(e).__name__, e))
                return ("ERROR: {}: {}".format(type(e).__name__, e), gr.update())

        booru_tag_uploads_btn.click(
            fn=_do_tag_my_uploads,
            inputs=[booru_tag, booru_count, trigger_word, images, booru_official],
            outputs=[booru_status, trigger_word],
        )

        # --- Previously-downloaded folder handlers ---
        def _do_refresh_prev():
            existing = list_downloaded_sets()
            labels = [e["label"] for e in existing]
            if not labels:
                labels = [PREV_PLACEHOLDER]
            return gr.update(choices=labels, value=labels[0])

        def _do_use_prev(label):
            if not label or label == PREV_PLACEHOLDER:
                return "Pick a real folder first (or run a Booru search above)."
            for e in list_downloaded_sets():
                if e["label"] == label:
                    JOB["searched_dir"] = e["path"]
                    return ("Selected: {} - click Start training to use these images.".format(e["label"]))
            return "Could not find the selected folder. Try Refresh list."

        prev_refresh_btn.click(fn=_do_refresh_prev, outputs=[prev_sets])
        prev_use_btn.click(fn=_do_use_prev, inputs=[prev_sets], outputs=[booru_status])

        # --- Caption editor handlers ---
        def _list_caption_images():
            """Return [(label, value)] of all images in the current
            searched_dir that have a paired .txt caption file."""
            d = JOB.get("searched_dir")
            if not d or not Path(d).exists():
                return []
            out = []
            for p in sorted(Path(d).iterdir()):
                if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                if p.with_suffix(".txt").exists():
                    out.append(p.name)
            return out

        def _do_refresh_captions():
            choices = _list_caption_images()
            return gr.update(choices=choices, value=choices[0] if choices else None)

        def _do_load_caption(image_name):
            if not image_name:
                return "", None, ""
            d = JOB.get("searched_dir")
            if not d:
                return "", None, "No search/tag run yet. Use the controls above first."
            img_path = Path(d) / image_name
            txt_path = img_path.with_suffix(".txt")
            if not img_path.exists() or not txt_path.exists():
                return "", None, "File not found - try Refresh."
            try:
                caption = txt_path.read_text(encoding="utf-8")
            except Exception as e:
                caption = ""
            return caption, str(img_path), "Loaded {}".format(image_name)

        def _do_save_caption(image_name, caption):
            if not image_name:
                return "Pick an image from the dropdown first."
            d = JOB.get("searched_dir")
            if not d:
                return "No search/tag run yet."
            txt_path = (Path(d) / image_name).with_suffix(".txt")
            try:
                txt_path.write_text((caption or "").strip(), encoding="utf-8")
                return "Saved {} ({} chars).".format(txt_path.name, len(caption or ""))
            except Exception as e:
                return "ERROR saving: {}".format(e)

        cap_refresh_btn.click(fn=_do_refresh_captions, outputs=[cap_image_dropdown])
        cap_image_dropdown.change(
            fn=_do_load_caption,
            inputs=[cap_image_dropdown],
            outputs=[cap_textbox, cap_image_preview, cap_save_status],
        )
        cap_save_btn.click(
            fn=_do_save_caption,
            inputs=[cap_image_dropdown, cap_textbox],
            outputs=[cap_save_status],
        )

        start_btn.click(
            fn=_start,
            inputs=[project_name, trigger_word, base_model, preset, caption_mode,
                    images, a_dim, a_alpha, a_lr, a_epochs, a_repeats, a_res, a_batch],
            outputs=[status_box, log_box],
        )
        refresh_log_btn.click(fn=_tail, outputs=[log_box])

        # Auto-refresh the log every 2 seconds. gr.Timer is in Gradio 4.x;
        # if it's somehow missing or the constructor signature differs we
        # silently fall back to manual refresh (button still works).
        try:
            if hasattr(gr, "Timer"):
                _log_timer = gr.Timer(2.0, active=True)
                _log_timer.tick(fn=_tail, outputs=[log_box])
                # Tell the user where to look so they know auto-refresh is on.
                gr.Markdown("_Log auto-refreshes every 2 seconds; use the "
                            "**Refresh log** button if you want to force an "
                            "immediate update._")
        except Exception as _timer_err:
            # Any oddity here is non-fatal - the manual button covers it.
            logger.warning("{} gr.Timer setup skipped: {}".format(TAG, _timer_err))

    return ui


def _on_ui_tabs():
    """Forge callback. Any failure here is logged and a fallback stub tab is
    shown so the user can SEE that the extension partly loaded."""
    try:
        ui = _build_tab()
        return [(ui, "LoRA Trainer", "lora_trainer")]
    except Exception as e:
        tb = traceback.format_exc()
        print("=" * 60, flush=True)
        print("{} FAILED to build LoRA Trainer tab: {}: {}".format(TAG, type(e).__name__, e), flush=True)
        print(tb, flush=True)
        print("=" * 60, flush=True)
        with gr.Blocks(analytics_enabled=False) as stub:
            gr.Markdown("# LoRA Trainer (failed to load)")
            gr.Markdown("Exception: `{}: {}`".format(type(e).__name__, e))
            gr.Code(value=tb, language="python")
        return [(stub, "LoRA Trainer", "lora_trainer")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
