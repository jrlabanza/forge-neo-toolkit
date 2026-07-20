"""Wildcard expansion: {red|blue|green} dress -> picks one randomly per call.

Nested expressions like {happy {girl|boy}|sad cat} work. The format intentionally
mirrors dynamic-prompts' "variant" syntax so users already familiar with it
don't have to relearn anything.

Public API:
    expand(text, rng=None) -> str
    expand_n(text, n, rng=None) -> list[str]   # generate N variants
    has_wildcards(text) -> bool
"""
import random
import re

_WILDCARD_RE = re.compile(r"\{([^{}]+)\}")


def has_wildcards(text):
    return bool(text) and "{" in text and "|" in text


def expand(text, rng=None):
    """Expand all {a|b|c} groups in `text`. Inner groups are resolved first."""
    if not text:
        return text
    rng = rng or random
    # Innermost-first: repeatedly substitute every {...} that has no nested braces.
    safety = 0
    while True:
        m = _WILDCARD_RE.search(text)
        if not m:
            return text
        options = [o.strip() for o in m.group(1).split("|")]
        text = text[:m.start()] + rng.choice(options) + text[m.end():]
        safety += 1
        if safety > 200:
            # Pathological input — bail with partial result
            return text


def expand_n(text, n, rng=None):
    rng = rng or random.Random()
    return [expand(text, rng) for _ in range(n)]
