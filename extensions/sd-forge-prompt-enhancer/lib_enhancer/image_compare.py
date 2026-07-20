"""Side-by-side comparison of two images' embedded metadata.

Returns a markdown table contrasting positive/negative/settings/wd14-tags
so the user can see what differs.

Public API:
    compare(meta_a, wd14_a, meta_b, wd14_b) -> str (markdown)
"""


def _short(s, n=80):
    if not s:
        return "_(empty)_"
    s = str(s).replace("\n", " ")
    if len(s) <= n:
        return s
    return s[:n] + "…"


def _tag_str(wd):
    if not wd:
        return "_(no tags)_"
    g = (wd.get("general") or [])[:10]
    return ", ".join(t[0] for t in g) or "_(no tags)_"


def compare(meta_a, wd14_a, meta_b, wd14_b):
    a = meta_a or {}
    b = meta_b or {}
    rows = [
        ("Source",            a.get("source", "?"),    b.get("source", "?")),
        ("Positive",          _short(a.get("positive")), _short(b.get("positive"))),
        ("Negative",          _short(a.get("negative")), _short(b.get("negative"))),
        ("WD14 top general",  _tag_str(wd14_a),        _tag_str(wd14_b)),
    ]
    # Settings comparison
    sa = a.get("settings") or {}
    sb = b.get("settings") or {}
    for k in ("Steps", "Sampler", "CFG scale", "Seed", "Size", "Model",
              "Schedule type"):
        rows.append(("Setting · " + k, sa.get(k, "—"), sb.get(k, "—")))

    out = ["| Field | Image A | Image B |", "|---|---|---|"]
    for label, av, bv in rows:
        out.append("| **{}** | {} | {} |".format(
            label, str(av).replace("|", "\\|"), str(bv).replace("|", "\\|")))
    return "\n".join(out)
