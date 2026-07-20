"""
sd-forge-naitosd / scripts / naitosd.py
=======================================

Forge Neo extension: NovelAI -> Illustrious XL PNG metadata converter.
Adds a "NAI Converter" tab to the WebUI. Drop a NovelAI PNG, see the
A1111-formatted prompt block, and send it to txt2img / img2img with one click.

Conversion logic ported from the standalone nai_to_sd.py and lives in
../lib_naitosd/converter.py so the Gradio shim here stays small.
"""
from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image, PngImagePlugin

from modules import script_callbacks

# Make sibling lib_naitosd importable regardless of how Forge loads the extension.
_HERE = Path(__file__).resolve().parent
_EXT_ROOT = _HERE.parent
import sys
if str(_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXT_ROOT))

from lib_naitosd import (
    extract_novelai_metadata,
    build_a1111_parameters,
    ILLUSTRIOUS_MODEL_LABEL,
)


# =============================================================================
# Conversion glue (PIL image in -> (params_string, status_message) out)
# =============================================================================

def _convert_pil(image):
    """Take a PIL image (from gr.Image) and return (params_string, message)."""
    if image is None:
        return "", "Waiting for a NovelAI PNG"
    try:
        meta = extract_novelai_metadata(image)
        if not meta:
            return "", ("[!] No NovelAI metadata found in this PNG. "
                        "Either it's not from NovelAI, or the metadata was stripped.")
        params_string = build_a1111_parameters(meta)
        src_name = getattr(image, "filename", None) or "unknown"
        return params_string, "[OK] Converted. Source: " + str(src_name)
    except Exception as exc:
        return "", "[ERR] " + type(exc).__name__ + ": " + str(exc)


def _save_converted(image, params_string, save_dir):
    """Write a new <name>_sd.png. Returns (path_or_empty, message)."""
    if not params_string:
        return "", "[!] Nothing to save - convert an image first."
    try:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(
            save_dir,
            "nai_converted_" + os.urandom(4).hex() + "_sd.png",
        )
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", params_string)
        image.save(out_path, format="PNG", pnginfo=pnginfo, optimize=False)
        return out_path, "[OK] Saved: " + out_path
    except Exception as exc:
        return "", "[ERR] save failed: " + type(exc).__name__ + ": " + str(exc)


# =============================================================================
# Tab UI
# =============================================================================

def _on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## NAI → " + ILLUSTRIOUS_MODEL_LABEL + " converter\n\n"
            "Drop a NovelAI-generated PNG. The extension reads its embedded "
            "metadata, translates NovelAI weight syntax + V4 character "
            "captions into A1111 format, and prepends Illustrious's "
            "recommended quality tags."
        )

        with gr.Row():
            with gr.Column():
                image_in = gr.Image(
                    label="NovelAI PNG (drag here)",
                    image_mode="RGBA",
                    type="pil",
                )
                status = gr.Textbox(
                    label="Status",
                    value="Waiting for a NovelAI PNG",
                    interactive=False,
                    lines=2,
                )
            with gr.Column():
                params_box = gr.Textbox(
                    label="Converted A1111 parameters",
                    lines=14,
                    interactive=True,
                    show_copy_button=True,
                    placeholder="The converted prompt block will appear here.",
                )
                with gr.Row():
                    send_t2i_btn = gr.Button("Send to txt2img", variant="primary")
                    send_i2i_btn = gr.Button("Send to img2img")
                    save_btn = gr.Button("Save converted PNG")

        # Auto-convert on image change
        image_in.change(
            fn=_convert_pil,
            inputs=[image_in],
            outputs=[params_box, status],
        )

        # Save -> output/txt2img-images by default
        def _default_save_dir():
            try:
                from modules import shared
                return shared.opts.outdir_txt2img_samples or "output/txt2img-images"
            except Exception:
                return "output/txt2img-images"

        save_btn.click(
            fn=lambda img, params: _save_converted(img, params, _default_save_dir()),
            inputs=[image_in, params_box],
            outputs=[gr.State(), status],
        )

        # =====================================================================
        # Send-to-txt2img / img2img via Forge's paste-params API.
        # Forge Neo renamed generation_parameters_copypaste -> infotext_utils;
        # try the new name first, fall back to the old one.
        # =====================================================================
        _paste_mod = None
        for mod_name in ("modules.infotext_utils",
                         "modules.generation_parameters_copypaste"):
            try:
                _paste_mod = importlib.import_module(mod_name)
                break
            except ImportError:
                continue

        if _paste_mod is not None and hasattr(_paste_mod, "register_paste_params_button"):
            _paste_mod.register_paste_params_button(
                _paste_mod.ParamBinding(
                    paste_button=send_t2i_btn,
                    tabname="txt2img",
                    source_text_component=params_box,
                    source_image_component=image_in,
                )
            )
            _paste_mod.register_paste_params_button(
                _paste_mod.ParamBinding(
                    paste_button=send_i2i_btn,
                    tabname="img2img",
                    source_text_component=params_box,
                    source_image_component=image_in,
                )
            )
        else:
            # Surface the issue in the status box instead of silently no-op'ing
            send_t2i_btn.click(
                fn=lambda: "[!] Send-to-txt2img not wired (Forge paste API not found). "
                           "Copy-paste the parameters manually from the box above.",
                inputs=None,
                outputs=[status],
            )
            send_i2i_btn.click(
                fn=lambda: "[!] Send-to-img2img not wired. Copy-paste manually.",
                inputs=None,
                outputs=[status],
            )

    return [(ui, "NAI Converter", "nai_converter")]


script_callbacks.on_ui_tabs(_on_ui_tabs)
