"""
sd-forge-director
=================

"Director" tab — NovelAI-style Director Tools for Forge. Drop any image
(generate it anywhere: txt2img, Auto Pilot, an old keeper from the Gallery)
and apply one-click operations:

- ✨ Enhance        — 1.5× upscale + low-denoise refine (sharper, richer detail)
- 🎲 Variations     — N re-rolls of the same image at chosen strength
- 😊 Emotion        — change the character's expression only (face-targeted
                      inpaint via ADetailer; body/background untouched)
- 🎨 Recolor        — palette/mood presets (warm, cool, pastel, vivid, mono)
- 🪄 Remove BG      — transparent-background cutout (rembg, auto-installed)

Enhance/Variations/Emotion/Recolor reuse the image's own embedded prompt
when present (fallback: a neutral prompt), and run through Forge's local
API — same pipeline as the Generate button. Results are saved to
output/director/<date>/ with metadata and shown in the results gallery.

Requires --api (webui-user.bat sets it).

Author: built by Claude on 2026-07-21.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image, PngImagePlugin

try:
    from modules import script_callbacks, shared
except ImportError:
    script_callbacks = None  # type: ignore
    shared = None            # type: ignore

logger = logging.getLogger(__name__)
TAG = "[director]"

EXT_ROOT = Path(__file__).resolve().parents[1]

FALLBACK_PROMPT = "masterpiece, best quality, amazing quality, very aesthetic"
FALLBACK_NEGATIVE = ("bad quality, worst quality, sketch, censor, signature, "
                     "watermark, jpeg artifacts")

EMOTIONS = {
    "Happy":       "smile, happy, open mouth",
    "Gentle":      "light smile, gentle expression, soft eyes",
    "Sad":         "sad, teary eyes, frown",
    "Angry":       "angry, furrowed brow, gritted teeth",
    "Surprised":   "surprised, wide eyes, open mouth, blush",
    "Embarrassed": "embarrassed, blush, averted eyes",
    "Serious":     "serious, closed mouth, intense eyes",
    "Smug":        "smug, smirk, half-closed eyes",
    "Crying":      "crying, tears, sobbing",
    "Sleepy":      "sleepy, half-closed eyes, yawning",
}

RECOLORS = {
    "Warm sunset":  "warm colors, golden hour lighting, orange tones",
    "Cool night":   "cool colors, blue tones, night ambience, moonlight",
    "Pastel":       "pastel colors, soft palette, airy, light",
    "Vivid":        "vivid colors, high saturation, vibrant palette",
    "Monochrome":   "monochrome, greyscale, high contrast ink",
    "Autumn":       "autumn palette, warm browns, amber light, falling leaves",
}


# ---------------------------------------------------------------------------
# Local API client
# ---------------------------------------------------------------------------

def _port() -> int:
    try:
        return int(getattr(shared.cmd_opts, "port", None) or 7860)
    except Exception:
        return 7860


def _api_post(path: str, payload: dict, timeout: int = 3600):
    import requests
    r = requests.post(f"http://127.0.0.1:{_port()}{path}", json=payload,
                      timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_ok() -> bool:
    try:
        import requests
        return requests.get(f"http://127.0.0.1:{_port()}/sdapi/v1/samplers",
                            timeout=5).ok
    except Exception:
        return False


def _api_get(path: str, timeout: int = 15):
    import requests
    r = requests.get(f"http://127.0.0.1:{_port()}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


API_HINT = "⚠ Forge API not reachable — launch with webui-user.bat (it sets --api)."


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _from_b64(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s)))


def _embedded_params(im: Image.Image) -> Tuple[str, str]:
    """(prompt, negative) recovered from PNG metadata, with fallbacks."""
    try:
        txt = (getattr(im, "text", None) or {}).get("parameters", "")
        if txt:
            lines = txt.splitlines()
            neg = ""
            prompt_lines: List[str] = []
            for ln in lines:
                if ln.startswith("Negative prompt:"):
                    neg = ln[len("Negative prompt:"):].strip()
                    break
                if ln.strip().startswith("Steps:"):
                    break
                prompt_lines.append(ln)
            prompt = "\n".join(prompt_lines).strip()
            if prompt:
                return prompt, (neg or FALLBACK_NEGATIVE)
    except Exception:
        pass
    return FALLBACK_PROMPT, FALLBACK_NEGATIVE


def _out_dir() -> Path:
    try:
        from modules import paths
        base = Path(paths.data_path)
    except Exception:
        base = EXT_ROOT.parents[1]
    d = base / "output" / "director" / datetime.now().strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(im: Image.Image, label: str, infotext: str = "") -> Path:
    info = PngImagePlugin.PngInfo()
    if infotext:
        info.add_text("parameters", infotext)
    path = _out_dir() / f"{label}_{datetime.now().strftime('%H%M%S_%f')[:13]}.png"
    im.save(path, pnginfo=info)
    return path


# ---------------------------------------------------------------------------
# Payload builders (pure — unit tested)
# ---------------------------------------------------------------------------

def build_enhance_payload(im_size: Tuple[int, int], prompt: str, negative: str,
                          scale: float, denoise: float) -> dict:
    w, h = im_size
    nw, nh = int(w * scale / 8) * 8, int(h * scale / 8) * 8
    return {
        "prompt": prompt, "negative_prompt": negative,
        "denoising_strength": denoise, "steps": 24,
        "sampler_name": "Euler a", "cfg_scale": 5.5,
        "width": nw, "height": nh, "seed": -1,
        "send_images": True, "save_images": False,
    }


def build_variation_payload(im_size: Tuple[int, int], prompt: str,
                            negative: str, strength: float, count: int) -> dict:
    w, h = im_size
    return {
        "prompt": prompt, "negative_prompt": negative,
        "denoising_strength": strength, "steps": 24,
        "sampler_name": "Euler a", "cfg_scale": 5.5,
        "width": int(w / 8) * 8, "height": int(h / 8) * 8, "seed": -1,
        "n_iter": max(1, int(count)), "batch_size": 1,
        "send_images": True, "save_images": False,
    }


def build_emotion_payload(im_size: Tuple[int, int], prompt: str, negative: str,
                          emotion_tags: str) -> dict:
    w, h = im_size
    return {
        "prompt": prompt, "negative_prompt": negative,
        "denoising_strength": 0.06,   # base pass barely touches the image…
        "steps": 24, "sampler_name": "Euler a", "cfg_scale": 5.5,
        "width": int(w / 8) * 8, "height": int(h / 8) * 8, "seed": -1,
        "send_images": True, "save_images": False,
        "alwayson_scripts": {
            "ADetailer": {
                "args": [
                    True, False,
                    {   # …ADetailer redraws ONLY the face with the emotion
                        "ad_model": "face_yolov8n.pt",
                        "ad_prompt": emotion_tags,
                        "ad_denoising_strength": 0.55,
                        "ad_inpaint_only_masked": True,
                        "ad_dilate_erode": 8,
                    },
                ]
            }
        },
    }


def build_recolor_payload(im_size: Tuple[int, int], prompt: str, negative: str,
                          palette_tags: str, strength: float) -> dict:
    w, h = im_size
    return {
        "prompt": f"{palette_tags}, {prompt}",
        "negative_prompt": negative,
        "denoising_strength": strength, "steps": 24,
        "sampler_name": "Euler a", "cfg_scale": 6.0,
        "width": int(w / 8) * 8, "height": int(h / 8) * 8, "seed": -1,
        "send_images": True, "save_images": False,
    }


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _run_img2img(im: Image.Image, payload: dict, label: str) -> Tuple[List[Image.Image], str]:
    payload = dict(payload)
    payload["init_images"] = [_b64(im)]
    res = _api_post("/sdapi/v1/img2img", payload)
    images = [_from_b64(s) for s in (res.get("images") or [])]
    try:
        infotexts = json.loads(res.get("info") or "{}").get("infotexts") or []
    except Exception:
        infotexts = []
    out = []
    for i, out_im in enumerate(images):
        info = infotexts[i] if i < len(infotexts) else ""
        _save(out_im, label, info)
        out.append(out_im)
    return out, f"{len(out)} image(s) → output/director/"


def op_enhance(im, scale, denoise):
    if im is None:
        return [], "Drop an image first."
    if not api_ok():
        return [], API_HINT
    p, n = _embedded_params(im)
    t0 = time.monotonic()
    outs, msg = _run_img2img(im, build_enhance_payload(im.size, p, n,
                                                       float(scale), float(denoise)),
                             "enhance")
    return outs, f"✨ Enhanced in {time.monotonic()-t0:.0f}s — {msg}"


def op_variations(im, strength, count):
    if im is None:
        return [], "Drop an image first."
    if not api_ok():
        return [], API_HINT
    p, n = _embedded_params(im)
    t0 = time.monotonic()
    outs, msg = _run_img2img(im, build_variation_payload(im.size, p, n,
                                                         float(strength), int(count)),
                             "variation")
    return outs, f"🎲 {len(outs)} variation(s) in {time.monotonic()-t0:.0f}s — {msg}"


def op_emotion(im, emotion):
    if im is None:
        return [], "Drop an image first."
    if not api_ok():
        return [], API_HINT
    tags = EMOTIONS.get(emotion, "smile")
    p, n = _embedded_params(im)
    t0 = time.monotonic()
    outs, msg = _run_img2img(im, build_emotion_payload(im.size, p, n, tags),
                             f"emotion_{emotion.lower()}")
    return outs, f"😊 Emotion → {emotion} in {time.monotonic()-t0:.0f}s — {msg}"


def op_recolor(im, palette, strength):
    if im is None:
        return [], "Drop an image first."
    if not api_ok():
        return [], API_HINT
    tags = RECOLORS.get(palette, "vivid colors")
    p, n = _embedded_params(im)
    t0 = time.monotonic()
    outs, msg = _run_img2img(im, build_recolor_payload(im.size, p, n, tags,
                                                       float(strength)),
                             f"recolor")
    return outs, f"🎨 Recolored ({palette}) in {time.monotonic()-t0:.0f}s — {msg}"


def extract_mask(editor_value):
    """gr.ImageEditor value -> (background RGB PIL, mask L PIL or None)."""
    try:
        import numpy as np
        if not isinstance(editor_value, dict):
            return None, None
        bg = editor_value.get("background")
        if bg is None:
            return None, None
        bg_im = bg if isinstance(bg, Image.Image) else \
            Image.fromarray(np.asarray(bg).astype("uint8"))
        acc = None
        for layer in editor_value.get("layers") or []:
            arr = np.asarray(layer)
            if arr.ndim == 3 and arr.shape[2] == 4:
                a = arr[:, :, 3]
            elif arr.ndim == 3:
                a = arr.max(axis=2)
            else:
                a = arr
            acc = a if acc is None else np.maximum(acc, a)
        if acc is None or int(acc.max()) == 0:
            return bg_im.convert("RGB"), None
        mask = Image.fromarray(((acc > 0).astype("uint8")) * 255, "L")
        return bg_im.convert("RGB"), mask
    except Exception:
        logger.exception("%s mask extraction failed", TAG)
        return None, None


def build_fix_payload(im_size: Tuple[int, int], prompt: str, negative: str,
                      denoise: float, mask_b64: str) -> dict:
    w, h = im_size
    return {
        "prompt": prompt, "negative_prompt": negative,
        "denoising_strength": float(denoise), "steps": 28,
        "sampler_name": "Euler a", "cfg_scale": 5.5,
        "width": int(w / 8) * 8, "height": int(h / 8) * 8, "seed": -1,
        "mask": mask_b64, "mask_blur": 4,
        "inpainting_fill": 1,           # start from original content
        "inpaint_full_res": True,       # only masked region, high detail
        "inpaint_full_res_padding": 32,
        "inpainting_mask_invert": 0,
        "send_images": True, "save_images": False,
    }


def op_fix_region(editor_value, instruction, denoise):
    bg, mask = extract_mask(editor_value)
    if bg is None:
        return [], "Drop an image into the Fix Region editor first."
    if mask is None:
        return [], "🖌 Paint over the area you want fixed, then click again."
    if not api_ok():
        return [], API_HINT
    instruction = (instruction or "").strip().rstrip(",")
    prompt = f"{instruction}, {FALLBACK_PROMPT}" if instruction else FALLBACK_PROMPT
    t0 = time.monotonic()
    payload = build_fix_payload(bg.size, prompt, FALLBACK_NEGATIVE,
                                denoise, _b64(mask.convert("RGB")))
    outs, msg = _run_img2img(bg, payload, "fixregion")
    return outs, f"🖌 Region fixed in {time.monotonic()-t0:.0f}s — {msg}"


# ---------------------------------------------------------------------------
# Line Art / Sketch / Colorize (NAI Director parity via ControlNet API)
# ---------------------------------------------------------------------------

LINEART_CANDIDATES = ["lineart_anime", "lineart_anime_denoise",
                      "lineart_realistic", "lineart_standard", "lineart"]
SKETCH_CANDIDATES = ["sketch_t2ia", "t2ia_sketch_pidi", "softedge_pidinet",
                     "pidinet_sketch", "scribble_pidinet", "softedge_hed"]
INVERT_CANDIDATES = ["invert (from white bg & black line)", "invert"]

_CN_CACHE: Dict[str, list] = {}


def pick_first(available: List[str], candidates: List[str]) -> Optional[str]:
    """First candidate present in available (case-insensitive). Pure."""
    low = {a.lower(): a for a in available or []}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _cn_modules() -> List[str]:
    if "modules" not in _CN_CACHE:
        try:
            _CN_CACHE["modules"] = _api_get("/controlnet/module_list").get(
                "module_list", [])
        except Exception as exc:
            logger.warning("%s module list failed: %s", TAG, exc)
            return []
    return _CN_CACHE["modules"]


def _cn_models() -> List[str]:
    if "models" not in _CN_CACHE:
        try:
            _CN_CACHE["models"] = _api_get("/controlnet/model_list").get(
                "model_list", [])
        except Exception as exc:
            logger.warning("%s model list failed: %s", TAG, exc)
            return []
    return _CN_CACHE["models"]


def _detect(im: Image.Image, module: str, res: int = 512) -> Optional[Image.Image]:
    try:
        r = _api_post("/controlnet/detect", {
            "controlnet_module": module,
            "controlnet_input_images": [_b64(im)],
            "controlnet_processor_res": res,
        }, timeout=300)
        imgs = r.get("images") or []
        return _from_b64(imgs[0]) if imgs else None
    except Exception as exc:
        logger.warning("%s detect(%s) failed: %s", TAG, module, exc)
        return None


def _op_extract(im, candidates, label, pretty):
    if im is None:
        return [], "Drop an image first."
    if not api_ok():
        return [], API_HINT
    module = pick_first(_cn_modules(), candidates)
    if not module:
        return [], f"{pretty}: no matching preprocessor found on this install."
    t0 = time.monotonic()
    res = max(im.size)
    res = min(int(res / 64) * 64, 1024) or 512
    out = _detect(im, module, res)
    if out is None:
        return [], f"{pretty} failed — see console."
    path = _save(out, label)
    return [out], (f"{pretty} ({module}) in {time.monotonic()-t0:.0f}s "
                   f"→ {path.name}")


def op_lineart(im):
    return _op_extract(im, LINEART_CANDIDATES, "lineart", "✒ Line art")


def op_sketch(im):
    return _op_extract(im, SKETCH_CANDIDATES, "sketch", "✏ Sketch")


def build_colorize_payload(im_size: Tuple[int, int], prompt: str,
                           negative: str, denoise: float,
                           cn_model: Optional[str], cn_module: Optional[str],
                           lineart_b64: str) -> dict:
    w, h = im_size
    payload = {
        "prompt": prompt, "negative_prompt": negative,
        "denoising_strength": float(denoise), "steps": 28,
        "sampler_name": "Euler a", "cfg_scale": 6.0,
        "width": int(w / 8) * 8, "height": int(h / 8) * 8, "seed": -1,
        "send_images": True, "save_images": False,
    }
    if cn_model:
        payload["alwayson_scripts"] = {"ControlNet": {"args": [{
            "enabled": True,
            "image": lineart_b64,
            "module": cn_module or "None",
            "model": cn_model,
            "weight": 0.75,
            "guidance_start": 0.0,
            "guidance_end": 0.8,
        }]}}
    return payload


def op_colorize(im, color_prompt, denoise):
    if im is None:
        return [], "Drop a line art / sketch first."
    if not api_ok():
        return [], API_HINT
    prompt = (color_prompt or "").strip().rstrip(",")
    prompt = f"{prompt}, {FALLBACK_PROMPT}" if prompt else \
        f"colored, vibrant colors, {FALLBACK_PROMPT}"
    cn_model = next((m for m in _cn_models() if "promax" in m.lower()), None)
    cn_module = pick_first(_cn_modules(), INVERT_CANDIDATES) if cn_model else None
    t0 = time.monotonic()
    payload = build_colorize_payload(im.size, prompt, FALLBACK_NEGATIVE,
                                     denoise, cn_model, cn_module, _b64(im))
    outs, msg = _run_img2img(im, payload, "colorize")
    held = "lines held by ControlNet union" if cn_model else \
        "no union ControlNet found — colorized freely from the init image"
    return outs, f"🖍 Colorized in {time.monotonic()-t0:.0f}s ({held}) — {msg}"


def op_load_handoff():
    """Load the image the Gallery's '→ Director' button pointed at."""
    try:
        pointer = EXT_ROOT.parent / "sd-forge-gallery" / "director_handoff.txt"
        path = pointer.read_text("utf-8").strip()
        im = Image.open(path)
        im.load()
        return im, f"📥 Loaded from Gallery: {Path(path).name}"
    except Exception:
        return None, "No Gallery handoff found — use '→ Director' in the Gallery first."


def op_finalize(im, scale, denoise, cutout):
    """Macro: Enhance → (optional Remove BG) → save to output/final/."""
    if im is None:
        return [], "Drop an image first."
    if not api_ok():
        return [], API_HINT
    p, n = _embedded_params(im)
    t0 = time.monotonic()
    outs, _ = _run_img2img(im, build_enhance_payload(im.size, p, n,
                                                     float(scale), float(denoise)),
                           "final_enhance")
    if not outs:
        return [], "Finalize failed at the enhance step."
    result = outs[-1]
    steps = ["enhanced"]
    if cutout:
        try:
            from rembg import remove
            result = remove(result.convert("RGBA"))
            steps.append("background removed")
        except Exception as exc:
            steps.append(f"cutout skipped ({exc})")
    try:
        from modules import paths
        base = Path(paths.data_path)
    except Exception:
        base = EXT_ROOT.parents[1]
    final_dir = base / "output" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    out_path = final_dir / f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    result.save(out_path)
    return [result], (f"🏁 Finalized in {time.monotonic()-t0:.0f}s "
                      f"({' → '.join(steps)}) → {out_path}")


def op_removebg(im):
    if im is None:
        return [], "Drop an image first."
    try:
        from rembg import remove
    except Exception:
        return [], ("🪄 rembg not installed yet — restart Forge once with "
                    "webui-user.bat (the extension installs it on launch).")
    try:
        t0 = time.monotonic()
        cut = remove(im.convert("RGBA"))
        path = _save(cut, "removebg")
        return [cut], f"🪄 Background removed in {time.monotonic()-t0:.0f}s → {path.name}"
    except Exception as exc:
        logger.exception("%s removebg failed", TAG)
        return [], f"🪄 Remove BG failed: {exc}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _on_ui_tabs():
    try:
        return _build_tab()
    except Exception:
        logger.exception("%s failed to build tab", TAG)
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown("Director failed to load - see console log.")
        return [(ui, "Director", "forge_director")]


def _build_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## 🎬 Director — one-click tools for any image\n"
            "Drop an image (its embedded prompt is reused automatically when "
            "present). Results save to `output/director/` and appear below."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(label="Image", type="pil", height=420)
                handoff_btn = gr.Button("📥 From Gallery")
                status = gr.Markdown("")
            with gr.Column(scale=2):
                with gr.Group():
                    with gr.Row():
                        enhance_btn = gr.Button("✨ Enhance", variant="primary")
                        enhance_scale = gr.Slider(1.1, 2.0, value=1.5, step=0.1,
                                                  label="Upscale ×")
                        enhance_denoise = gr.Slider(0.1, 0.6, value=0.35, step=0.05,
                                                    label="Refine strength")
                with gr.Group():
                    with gr.Row():
                        var_btn = gr.Button("🎲 Variations", variant="primary")
                        var_strength = gr.Slider(0.15, 0.7, value=0.4, step=0.05,
                                                 label="Variation strength")
                        var_count = gr.Slider(1, 8, value=4, step=1, label="Count")
                with gr.Group():
                    with gr.Row():
                        emo_btn = gr.Button("😊 Change emotion", variant="primary")
                        emo_dd = gr.Dropdown(label="Emotion",
                                             choices=list(EMOTIONS.keys()),
                                             value="Happy")
                with gr.Group():
                    with gr.Row():
                        rec_btn = gr.Button("🎨 Recolor", variant="primary")
                        rec_dd = gr.Dropdown(label="Palette",
                                             choices=list(RECOLORS.keys()),
                                             value="Warm sunset")
                        rec_strength = gr.Slider(0.2, 0.6, value=0.4, step=0.05,
                                                 label="Strength")
                with gr.Group():
                    rbg_btn = gr.Button("🪄 Remove background", variant="primary")
                with gr.Group():
                    with gr.Row():
                        fin_btn = gr.Button("🏁 Finalize (enhance → final/)",
                                            variant="primary")
                        fin_cutout = gr.Checkbox(label="+ background cutout",
                                                 value=False)
                with gr.Group():
                    with gr.Row():
                        la_btn = gr.Button("✒ Line art", variant="primary")
                        sk_btn = gr.Button("✏ Sketch", variant="primary")
                with gr.Group():
                    with gr.Row():
                        col_btn = gr.Button("🖍 Colorize", variant="primary")
                        col_prompt = gr.Textbox(
                            label="Colors / look", scale=2,
                            elem_classes=["prompt"],
                            placeholder="silver hair, red dress, golden hour")
                        col_denoise = gr.Slider(0.5, 0.95, value=0.8,
                                                step=0.05, label="Strength")
        with gr.Accordion("🖌 Fix Region — paint over a problem, describe the fix",
                          open=False):
            fix_editor = gr.ImageEditor(label="Paint the region to redo",
                                        type="numpy", height=460)
            with gr.Row():
                fix_prompt = gr.Textbox(
                    label="What should be there?", scale=3,
                    elem_classes=["prompt"],
                    placeholder="e.g.  perfect hand, five fingers  ·  "
                                "clean background  ·  blue eyes")
                fix_denoise = gr.Slider(0.2, 0.9, value=0.55, step=0.05,
                                        label="Fix strength")
                fix_btn = gr.Button("🖌 Fix it", variant="primary", scale=0)
        results = gr.Gallery(label="Results", columns=4, height=420,
                             object_fit="contain")

        enhance_btn.click(op_enhance, [image_in, enhance_scale, enhance_denoise],
                          [results, status])
        var_btn.click(op_variations, [image_in, var_strength, var_count],
                      [results, status])
        emo_btn.click(op_emotion, [image_in, emo_dd], [results, status])
        rec_btn.click(op_recolor, [image_in, rec_dd, rec_strength],
                      [results, status])
        rbg_btn.click(op_removebg, [image_in], [results, status])
        fix_btn.click(op_fix_region, [fix_editor, fix_prompt, fix_denoise],
                      [results, status])
        handoff_btn.click(op_load_handoff, [], [image_in, status])
        fin_btn.click(op_finalize,
                      [image_in, enhance_scale, enhance_denoise, fin_cutout],
                      [results, status])
        la_btn.click(op_lineart, [image_in], [results, status])
        sk_btn.click(op_sketch, [image_in], [results, status])
        col_btn.click(op_colorize, [image_in, col_prompt, col_denoise],
                      [results, status])

    return [(ui, "Director", "forge_director")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
