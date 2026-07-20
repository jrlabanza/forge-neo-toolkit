"""Heuristic ControlNet unit suggester.

Given the analyzed WD14 result and the source image, propose which CN units
to enable. Not magic — just sensible defaults based on what's in the image.

Rules:
  - Always recommend Canny for any image (cheapest structural anchor)
  - Recommend OpenPose if the WD14 result has 1+ character tags or pose tags
  - Recommend Depth if the scene has 3D-ish context (architecture, multiple
    people, vehicles)
  - Recommend Tile if image > 1MP (helps with upscale-style replication)
  - Recommend IP-Adapter when no character tag was confidently detected
    (style transfer makes more sense than character transfer)

Public API:
    suggest(wd14_result, source_size) -> dict
        {canny: bool, depth: bool, pose: bool, tile: bool, ipadapter: bool,
         reasoning: list[str]}
"""

_POSE_HINTS = {
    "standing", "sitting", "lying down", "kneeling", "running", "walking",
    "jumping", "crouching", "dancing", "fighting", "hugging", "kissing",
    "arms up", "arms crossed", "hands on hips", "leaning",
}

_DEPTH_HINTS = {
    "architecture", "building", "interior", "city", "street", "room",
    "bedroom", "kitchen", "vehicle", "car", "ship", "airplane", "train",
    "landscape", "mountain", "forest", "multiple girls", "multiple boys",
    "2girls", "3girls", "2boys", "crowd",
}


def _tags_only(tag_list):
    return {t[0].lower() if isinstance(t, tuple) else str(t).lower()
            for t in (tag_list or [])}


def suggest(wd14_result, source_size=None):
    out = {
        "canny": False, "depth": False, "pose": False,
        "tile": False, "ipadapter": False, "reasoning": [],
    }
    if not wd14_result:
        return out

    general    = _tags_only(wd14_result.get("general"))
    characters = wd14_result.get("characters") or []

    # Canny is almost always useful
    out["canny"] = True
    out["reasoning"].append("Canny: structural anchor for any source image")

    # OpenPose
    pose_hits = general & _POSE_HINTS
    if pose_hits or len(characters) >= 1:
        out["pose"] = True
        out["reasoning"].append(
            "OpenPose: detected pose cues or {} character(s)".format(len(characters)))

    # Depth
    depth_hits = general & _DEPTH_HINTS
    if depth_hits:
        out["depth"] = True
        out["reasoning"].append(
            "Depth: scene has 3D context ({})".format(", ".join(sorted(depth_hits)[:3])))

    # Tile (helps with high-res source replication)
    if source_size:
        sw, sh = source_size
        if sw * sh >= 1024 * 1024:
            out["tile"] = True
            out["reasoning"].append(
                "Tile: source is high-res ({}x{}), tile model preserves detail".format(sw, sh))

    # IP-Adapter: style transfer when no recognized character
    if not characters:
        out["ipadapter"] = True
        out["reasoning"].append(
            "IP-Adapter: no specific character recognized — fall back to style transfer")

    return out
