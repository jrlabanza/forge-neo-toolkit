"""
sd-forge-model-presets
======================

"Model Presets" tab for Forge Neo: one-click recommended settings per model
family. Switching between Illustrious, Pony, Animagine, NoobAI and realistic
SD1.5 checkpoints means juggling different samplers, CFG ranges, resolutions
and (most annoyingly) completely different quality-tag dialects. This tab:

1. Detects which family the currently loaded checkpoint belongs to.
2. Shows the recommended sampler / steps / CFG / size + quality tags +
   negative prompt for that family (all editable before applying).
3. Optionally takes your subject text and composes a complete prompt.
4. One click pastes everything into txt2img (or img2img) via Forge's
   paste-params API — same mechanism as "Send to txt2img" elsewhere.

Presets target the checkpoints actually installed on this machine:
waiIllustriousSDXL, waiNSFWIllustrious, NoobAI-XL, animagineXL 3.1/4.0,
cyberrealisticPony, cyberrealistic classic (SD1.5), realisticVision (SD1.5).

Author: built by Claude on 2026-07-20.
"""
from __future__ import annotations

import importlib
import logging

import gradio as gr

try:
    from modules import script_callbacks, shared
except ImportError:
    script_callbacks = None  # type: ignore
    shared = None            # type: ignore

logger = logging.getLogger(__name__)
TAG = "[model-presets]"


# ---------------------------------------------------------------------------
# Preset data
# ---------------------------------------------------------------------------

PRESETS = {
    "Illustrious / WAI (anime)": {
        "match": ("wai", "illustrious"),
        "settings": "Steps: 28, Sampler: Euler a, CFG scale: 5.5, Size: 832x1216, Clip skip: 2",
        "quality": "masterpiece, best quality, amazing quality, very aesthetic, absurdres",
        "negative": "bad quality, worst quality, worst detail, sketch, censor, "
                    "signature, watermark, username, jpeg artifacts",
        "notes": "Portrait default. Use 1216x832 for landscape, 1024x1024 square. "
                 "Hires fix: R-ESRGAN 4x+ Anime6B, denoise 0.35, upscale 1.5x. "
                 "CFG 4.5-7 is the usable range; higher saturates colors.",
    },
    "NoobAI-XL (anime)": {
        "match": ("noob",),
        "settings": "Steps: 28, Sampler: Euler a, CFG scale: 5, Size: 832x1216, Clip skip: 2",
        "quality": "masterpiece, best quality, newest, absurdres, highres",
        "negative": "worst quality, old, early, low quality, lowres, signature, "
                    "username, logo, bad hands, mutated hands",
        "notes": "For the eps-pred NoobAI-XL v1.1 you have installed. "
                 "Understands both Danbooru artist tags and 'year 2023'-style newest/oldest tags.",
    },
    "Animagine XL 3.1 / 4.0 (anime)": {
        "match": ("animagine",),
        "settings": "Steps: 27, Sampler: Euler a, CFG scale: 5.5, Size: 832x1216, Clip skip: 2",
        "quality": "masterpiece, high score, great score, absurdres",
        "negative": "lowres, bad anatomy, bad hands, text, error, missing finger, "
                    "extra digits, fewer digits, cropped, worst quality, low quality, "
                    "low score, bad score, average score, signature, watermark, username, blurry",
        "notes": "Quality tags above are the official Animagine 4.0 set. "
                 "For 3.1 swap quality to: masterpiece, best quality, very aesthetic, absurdres. "
                 "Animagine likes tag order: 1girl, character, series, everything else.",
    },
    "Pony realistic (CyberRealistic Pony)": {
        "match": ("pony",),
        "settings": "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 5, Size: 832x1216, Clip skip: 2",
        "quality": "score_9, score_8_up, score_7_up, photo, photorealistic, raw photo",
        "negative": "score_6, score_5, score_4, worst quality, low quality, bad anatomy, "
                    "bad hands, watermark, cartoon, anime, 3d, render, painting",
        "notes": "Pony models NEED the score_ prefix tags or quality craters. "
                 "Clip skip 2 is required for Pony-family. "
                 "Add 'source_anime' to quality for anime-style output from Pony.",
    },
    "Realistic SD1.5 (RealisticVision / CR Classic)": {
        "match": ("realisticvision", "classic"),
        "settings": "Steps: 28, Sampler: DPM++ SDE Karras, CFG scale: 4.5, Size: 512x768, Clip skip: 1",
        "quality": "RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        "negative": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, "
                    "sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, "
                    "worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, "
                    "mutilated, extra fingers, mutated hands, poorly drawn hands, poorly "
                    "drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, "
                    "extra limbs, cloned face, disfigured, gross proportions, malformed "
                    "limbs, missing arms, missing legs, fused fingers, too many fingers, long neck",
        "notes": "SD1.5 models — note the smaller 512x768 size (SDXL sizes produce "
                 "doubled bodies on SD1.5). Put your subject right after 'RAW photo, '. "
                 "These run much faster than SDXL on your 3070.",
    },
}

DETECT = "(auto-detect from loaded checkpoint)"


def _current_checkpoint() -> str:
    try:
        return str(shared.opts.sd_model_checkpoint or "")
    except Exception:
        return ""


def _detect_family(ckpt_name: str) -> str:
    low = (ckpt_name or "").lower()
    # order matters: "cyberrealisticPony" must hit Pony before the SD1.5 rule
    order = [
        "Pony realistic (CyberRealistic Pony)",
        "NoobAI-XL (anime)",
        "Animagine XL 3.1 / 4.0 (anime)",
        "Illustrious / WAI (anime)",
        "Realistic SD1.5 (RealisticVision / CR Classic)",
    ]
    for name in order:
        if any(m in low for m in PRESETS[name]["match"]):
            return name
    return "Illustrious / WAI (anime)"


def _compose(subject: str, quality: str, negative: str, settings: str) -> str:
    subject = (subject or "").strip().rstrip(",")
    prompt = f"{subject}, {quality}" if subject else quality
    return f"{prompt}\nNegative prompt: {negative}\n{settings}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _on_ui_tabs():
    try:
        return _build_tab()
    except Exception:
        logger.exception("%s failed to build tab", TAG)
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown("Model Presets failed to load - see console log.")
        return [(ui, "Model Presets", "forge_model_presets")]


def _build_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## Model Presets — right settings for the right checkpoint\n"
            "Pick a family (or auto-detect), optionally type your subject, then "
            "**Load into txt2img**. Sampler, steps, CFG, size, clip skip, quality "
            "tags and negative prompt all get filled in at once."
        )

        with gr.Row():
            family_dd = gr.Dropdown(label="Model family",
                                    choices=[DETECT] + list(PRESETS.keys()),
                                    value=DETECT, scale=2)
            detect_lbl = gr.Textbox(label="Loaded checkpoint", interactive=False,
                                    scale=2)
            refresh_btn = gr.Button("🔍 Detect", scale=0)

        subject_box = gr.Textbox(
            label="Your subject (optional — composed in front of the quality tags)",
            placeholder="e.g.  1girl, solo, fu hua, honkai (series), upper body, night sky",
            lines=2, elem_classes=["prompt"])

        with gr.Row():
            quality_box = gr.Textbox(label="Quality tags (editable)", lines=3,
                                     show_copy_button=True)
            negative_box = gr.Textbox(label="Negative prompt (editable)", lines=3,
                                      show_copy_button=True)
        settings_box = gr.Textbox(label="Generation settings (editable)",
                                  show_copy_button=True)
        notes_md = gr.Markdown("")

        composed_box = gr.Textbox(label="Composed parameters (what gets loaded)",
                                  lines=6, interactive=True, show_copy_button=True)

        with gr.Row():
            load_t2i = gr.Button("Load into txt2img  (replaces prompt + settings)",
                                 variant="primary")
            load_i2i = gr.Button("Load into img2img")
        status = gr.Markdown("")

        # ------------------------------------------------------------------
        def pick(family, subject):
            ckpt = _current_checkpoint()
            fam = _detect_family(ckpt) if family in (DETECT, None, "") else family
            p = PRESETS[fam]
            composed = _compose(subject, p["quality"], p["negative"], p["settings"])
            return (ckpt or "(none loaded yet)", p["quality"], p["negative"],
                    p["settings"], f"**{fam}** — {p['notes']}", composed)

        def recompose(subject, quality, negative, settings):
            return _compose(subject, quality, negative, settings)

        for comp, ev in ((family_dd, "input"), (refresh_btn, "click")):
            getattr(comp, ev)(pick, inputs=[family_dd, subject_box],
                              outputs=[detect_lbl, quality_box, negative_box,
                                       settings_box, notes_md, composed_box])
        for comp in (subject_box, quality_box, negative_box, settings_box):
            comp.change(recompose,
                        inputs=[subject_box, quality_box, negative_box, settings_box],
                        outputs=[composed_box])
        ui.load(pick, inputs=[family_dd, subject_box],
                outputs=[detect_lbl, quality_box, negative_box,
                         settings_box, notes_md, composed_box])

        # paste wiring
        paste_mod = None
        for mod_name in ("modules.infotext_utils", "modules.generation_parameters_copypaste"):
            try:
                paste_mod = importlib.import_module(mod_name)
                break
            except ImportError:
                continue
        if paste_mod is not None and hasattr(paste_mod, "register_paste_params_button"):
            for btn, tab in ((load_t2i, "txt2img"), (load_i2i, "img2img")):
                paste_mod.register_paste_params_button(paste_mod.ParamBinding(
                    paste_button=btn, tabname=tab,
                    source_text_component=composed_box,
                ))
        else:
            msg = "Paste API not found — copy the composed parameters manually."
            load_t2i.click(lambda: msg, None, [status])
            load_i2i.click(lambda: msg, None, [status])

    return [(ui, "Model Presets", "forge_model_presets")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
