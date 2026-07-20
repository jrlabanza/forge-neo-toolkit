"""Prompt linter. Flags common mistakes BEFORE you waste a generation.

Checks:
  - Unbalanced parens
  - Mal-formed weights (tag::1.5)
  - Contradictory tags
  - Conflicting body/eye states
  - Weight nest depth > 3 (visually unreadable)
  - NSFW tags present when "force SFW" intended

Public API:
    lint(positive, negative=None, sfw_mode=False) -> list[dict]
        each dict: {"severity": "warn"|"error"|"info", "msg": str}
"""
import re


# Pairs that contradict each other if both present in positive
_CONTRADICTIONS = [
    ({"closed eyes"},          {"looking at viewer", "eye contact"}),
    ({"open mouth"},           {"closed mouth", "lips pursed"}),
    ({"smiling", "happy"},     {"sad", "crying", "angry", "frown"}),
    ({"standing"},             {"sitting", "lying down", "kneeling"}),
    ({"long hair"},            {"short hair", "bald"}),
    ({"daytime", "morning", "afternoon"},
                               {"night", "midnight"}),
    ({"indoors"},              {"outdoors"}),
    ({"clothed"},              {"nude", "naked", "completely nude"}),
    ({"facing viewer"},        {"from behind", "facing away"}),
]

# Tags that are commonly explicit/NSFW
_NSFW_HINTS = {
    "nude", "naked", "completely nude", "topless", "bottomless",
    "nipples", "pussy", "vagina", "penis", "sex", "explicit",
    "questionable", "ass focus", "pussy focus",
}

# Pattern for malformed double-colon weights like "tag::1.5"
_MALFORMED_WEIGHT = re.compile(r"\b[\w\s]+::\s*\d+(?:\.\d+)?\s*\b")
# A1111 weight syntax: (tag:1.5) — count parens to spot unbalanced.
# Detect "(tag:1.5))" (extra close) and "((tag:1.5)" (extra open).


def _clean_tag(s):
    """Strip A1111 weight syntax and brackets from a single tag.
    `(closed eyes:1.2)` -> `closed eyes`."""
    s = s.strip().lower()
    while s and s[0] in "([{":
        s = s[1:].strip()
    while s and s[-1] in ")]}":
        s = s[:-1].strip()
    if ":" in s:
        head, _, tail = s.rpartition(":")
        try:
            float(tail)
            s = head.strip()
        except ValueError:
            pass
    return s


def _tagset(text):
    return {_clean_tag(t) for t in (text or "").replace("\n", ",").split(",")
            if t.strip()}


def lint(positive, negative=None, sfw_mode=False):
    out = []
    pos = positive or ""
    neg = negative or ""

    # 1. Unbalanced parens (the #1 cause of "weird gen" prompts)
    if pos.count("(") != pos.count(")"):
        out.append({"severity": "error",
                    "msg": "Positive prompt has unbalanced parentheses "
                           "(`{}` open, `{}` close).".format(
                               pos.count("("), pos.count(")"))})
    if pos.count("[") != pos.count("]"):
        out.append({"severity": "error",
                    "msg": "Positive prompt has unbalanced brackets."})

    # 2. Malformed double-colon weights (mistaken NAI->A1111 conversion residue)
    if _MALFORMED_WEIGHT.search(pos):
        out.append({"severity": "warn",
                    "msg": "Found `tag::weight` syntax. A1111 uses `(tag:weight)` "
                           "instead — the `::` form will be treated as literal text."})

    # 3. Contradictions
    pos_tags = _tagset(pos)
    for set_a, set_b in _CONTRADICTIONS:
        hits_a = pos_tags & set_a
        hits_b = pos_tags & set_b
        if hits_a and hits_b:
            out.append({"severity": "warn",
                        "msg": "Contradiction: `{}` and `{}` both present.".format(
                            ", ".join(sorted(hits_a)), ", ".join(sorted(hits_b)))})

    # 4. Weight nesting depth — deeply nested weights are usually a typo
    depth = 0
    max_depth = 0
    for ch in pos:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    if max_depth > 3:
        out.append({"severity": "info",
                    "msg": "Weight nesting depth = {} — usually you meant one (tag:x.x) "
                           "wrapper, not multiple.".format(max_depth)})

    # 5. SFW intent vs NSFW tags
    if sfw_mode:
        leaked = pos_tags & _NSFW_HINTS
        if leaked:
            out.append({"severity": "error",
                        "msg": "SFW mode requested but explicit tags found: "
                               "`{}`".format(", ".join(sorted(leaked)))})

    # 6. Quality reminder
    quality_set = {"masterpiece", "best quality", "amazing quality",
                   "very aesthetic", "absurdres"}
    if pos.strip() and not (pos_tags & quality_set):
        out.append({"severity": "info",
                    "msg": "No quality tags detected (masterpiece, best quality, "
                           "etc.). Most Illustrious/Noob checkpoints expect them."})

    # 7. Empty negative warning
    if pos.strip() and not neg.strip():
        out.append({"severity": "info",
                    "msg": "Negative prompt is empty. Use the preset dropdown "
                           "to start from a recommended baseline."})

    return out


def format_lint_md(issues):
    """Render lint output as markdown."""
    if not issues:
        return "✅ **No issues detected.**"
    icons = {"error": "❌", "warn": "⚠️", "info": "💡"}
    lines = []
    for it in issues:
        icon = icons.get(it["severity"], "·")
        lines.append("{} {}".format(icon, it["msg"]))
    return "\n\n".join(lines)
