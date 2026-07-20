"""Render a "recipe card" PNG showing the prompt + thumbnail + key settings.

Useful for sharing a generation result + its recipe in one image.

Public API:
    build_card(thumbnail_pil, positive, negative, settings,
               width=1200) -> PIL.Image
"""
from PIL import Image, ImageDraw, ImageFont


def _safe_font(size):
    """Try a few font paths. Falls back to PIL's default if all fail."""
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(text, font, max_w, draw):
    """Word-wrap to fit max_w pixels."""
    if not text:
        return ["(empty)"]
    words = text.replace("\n", " ").split()
    if not words:
        return ["(empty)"]
    lines = []
    cur = []
    for w in words:
        trial = " ".join(cur + [w])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if (bbox[2] - bbox[0]) > max_w and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def build_card(thumbnail_pil, positive, negative, settings,
               width=1200, bg="#1e1e23", fg="#e8e8ec",
               accent="#7aa2f7"):
    settings = settings or {}
    pad = 32
    thumb_h = 400
    # Thumbnail letterbox
    if thumbnail_pil:
        t = thumbnail_pil.copy()
        t.thumbnail((width - 2 * pad, thumb_h))
        thumb_w, thumb_h_actual = t.size
    else:
        t = None
        thumb_w, thumb_h_actual = 0, 0

    # Fonts
    f_title = _safe_font(36)
    f_label = _safe_font(20)
    f_body  = _safe_font(18)
    f_meta  = _safe_font(16)

    # Compute height by laying out text
    canvas = Image.new("RGB", (width, 100), bg)
    draw = ImageDraw.Draw(canvas)
    body_w = width - 2 * pad

    pos_lines = _wrap(positive, f_body, body_w, draw)
    neg_lines = _wrap(negative or "_(no negative)_", f_body, body_w, draw)
    settings_str = " · ".join("{}: {}".format(k, v) for k, v in settings.items())
    set_lines = _wrap(settings_str or "(no settings)", f_meta, body_w, draw)

    line_h_body  = 24
    line_h_meta  = 20
    h = pad
    h += 50                                          # title
    if t is not None:
        h += thumb_h_actual + pad
    h += 30 + len(pos_lines) * line_h_body + pad     # POSITIVE label + body
    h += 30 + len(neg_lines) * line_h_body + pad     # NEGATIVE
    h += 30 + len(set_lines) * line_h_meta + pad     # SETTINGS

    canvas = Image.new("RGB", (width, h), bg)
    draw = ImageDraw.Draw(canvas)

    y = pad
    draw.text((pad, y), "Prompt Recipe", fill=accent, font=f_title)
    y += 50
    if t is not None:
        canvas.paste(t, ((width - thumb_w) // 2, y))
        y += thumb_h_actual + pad

    def section(label, lines, font, lh, color=fg):
        nonlocal y
        draw.text((pad, y), label, fill=accent, font=f_label)
        y += 30
        for ln in lines:
            draw.text((pad, y), ln, fill=color, font=font)
            y += lh
        y += pad

    section("POSITIVE", pos_lines, f_body, line_h_body)
    section("NEGATIVE", neg_lines, f_body, line_h_body, color="#aab1bf")
    section("SETTINGS", set_lines, f_meta, line_h_meta, color="#aab1bf")

    return canvas
