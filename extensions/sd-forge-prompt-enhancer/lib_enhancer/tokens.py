"""CLIP token counting + chunk indicator.

SDXL uses CLIP-L which has a 77-token context window (75 usable + 2 BOS/EOS).
Long prompts get chunked. This module counts tokens with a graceful fallback
if transformers isn't installed yet.

Public API:
    count_tokens(text) -> int
    chunk_info(text) -> dict   # {tokens, chunks, used_in_last, percent}
    chunk_indicator(text) -> str   # markdown bar like "[##......] 47/75"
"""

_TOKENIZER = None


def _get_tokenizer():
    """Lazy-load CLIPTokenizer. Cached globally."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        from transformers import CLIPTokenizer  # type: ignore
        # openai/clip-vit-large-patch14 is what SDXL CLIP-L uses
        _TOKENIZER = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        return _TOKENIZER
    except Exception:
        # Fallback: rough estimate (~1 token per word, plus punctuation)
        _TOKENIZER = "fallback"
        return _TOKENIZER


def _approx_token_count(text):
    """Word-based approximation when CLIP tokenizer isn't available.
    Booru tags often tokenize at ~1.3 tokens per word due to underscores."""
    if not text:
        return 0
    # Split on commas + whitespace, count
    parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
    # Each tag averages ~2 tokens (BPE), single words ~1
    tot = 0
    for p in parts:
        words = p.split()
        tot += sum(1 if len(w) < 5 else 2 for w in words)
    return tot


def count_tokens(text):
    if not text:
        return 0
    tok = _get_tokenizer()
    if tok == "fallback":
        return _approx_token_count(text)
    try:
        # CLIP tokenizer adds BOS+EOS — subtract them for a clean tag count
        ids = tok.encode(text)
        return max(0, len(ids) - 2)
    except Exception:
        return _approx_token_count(text)


def chunk_info(text):
    """Return {tokens, chunks, used_in_last, percent_last}.

    A chunk is 75 usable tokens. Used_in_last = how many tokens of the last
    chunk are filled. Percent = used_in_last / 75."""
    n = count_tokens(text)
    if n == 0:
        return {"tokens": 0, "chunks": 0, "used_in_last": 0, "percent": 0.0,
                "is_fallback": _get_tokenizer() == "fallback"}
    chunks = (n + 74) // 75   # ceil
    used_in_last = n - 75 * (chunks - 1)
    return {
        "tokens": n,
        "chunks": chunks,
        "used_in_last": used_in_last,
        "percent": used_in_last / 75.0,
        "is_fallback": _get_tokenizer() == "fallback",
    }


def chunk_indicator(text):
    """Return a markdown-formatted indicator suitable for a gr.Markdown box."""
    info = chunk_info(text)
    n = info["tokens"]
    chunks = info["chunks"]
    if n == 0:
        return "*(empty)*"
    used = info["used_in_last"]
    bar_w = 20
    filled = max(0, min(bar_w, int(round(info["percent"] * bar_w))))
    bar = "▰" * filled + "▱" * (bar_w - filled)
    suffix = ""
    if info["is_fallback"]:
        suffix = "  *(approx — install `transformers` for exact)*"
    color = "green"
    if chunks > 1: color = "orange"
    if used > 70 and chunks == 1: color = "orange"
    return (
        "**{tokens} tokens** · **{chunks}** chunk(s) · last chunk: "
        "{used}/75 `{bar}`{suffix}"
    ).format(tokens=n, chunks=chunks, used=used, bar=bar, suffix=suffix)
