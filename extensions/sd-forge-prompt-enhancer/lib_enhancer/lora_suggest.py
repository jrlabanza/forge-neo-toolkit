"""Auto-suggest LoRAs from local models/Lora folder based on detected tags.

Scans for .safetensors/.pt files, indexes them by lowercased basename, and
fuzzy-matches against character/artist/style tags discovered by WD14 or
typed by the user.

Public API:
    refresh_index(lora_root_paths) -> int           # count
    suggest(query_tags, top_k=8) -> list[dict]
        each dict: {name, path, score, match_reason}
    inject_syntax(name, weight=0.8) -> str          # "<lora:name:0.8>"
"""
import os
import re

_LORA_INDEX = []  # list of {name, path, lower}
_INDEX_BUILT = False


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def refresh_index(lora_root_paths):
    """Walk every path, register .safetensors / .pt files."""
    global _LORA_INDEX, _INDEX_BUILT
    seen = set()
    out = []
    for root in lora_root_paths or []:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not (f.endswith(".safetensors") or f.endswith(".pt")):
                    continue
                full = os.path.join(dirpath, f)
                if full in seen:
                    continue
                seen.add(full)
                name = os.path.splitext(f)[0]
                out.append({
                    "name":  name,
                    "path":  full,
                    "lower": _norm(name),
                })
    _LORA_INDEX = out
    _INDEX_BUILT = True
    return len(out)


def _score(query_norm, lora_norm):
    """Substring match scoring. 1.0 = exact, 0 = no overlap."""
    if not query_norm or not lora_norm:
        return 0.0
    if query_norm == lora_norm:
        return 1.0
    if query_norm in lora_norm:
        return 0.85 + 0.10 * (len(query_norm) / len(lora_norm))
    if lora_norm in query_norm:
        return 0.75
    # Token-level overlap
    q_toks = set(re.findall(r"[a-z0-9]+", query_norm))
    l_toks = set(re.findall(r"[a-z0-9]+", lora_norm))
    if not q_toks or not l_toks:
        return 0.0
    overlap = len(q_toks & l_toks) / max(len(q_toks), len(l_toks))
    return overlap * 0.6  # cap fuzzy matches below 0.6


def suggest(query_tags, top_k=8):
    """query_tags is a list of strings (e.g. ['saber (fate)', 'monochrome'])."""
    if not _INDEX_BUILT:
        return []
    out = []
    for tag in (query_tags or []):
        if not tag or not tag.strip():
            continue
        qn = _norm(tag)
        for entry in _LORA_INDEX:
            sc = _score(qn, entry["lower"])
            if sc >= 0.4:
                out.append({
                    "name":  entry["name"],
                    "path":  entry["path"],
                    "score": sc,
                    "match_reason": "matches `{}`".format(tag),
                })
    # Dedupe by lora name, keep best score
    seen = {}
    for it in out:
        if it["name"] not in seen or seen[it["name"]]["score"] < it["score"]:
            seen[it["name"]] = it
    ranked = sorted(seen.values(), key=lambda x: -x["score"])
    return ranked[:top_k]


def inject_syntax(name, weight=0.8):
    return "<lora:{}:{}>".format(name, round(float(weight), 2))


def index_size():
    return len(_LORA_INDEX)
