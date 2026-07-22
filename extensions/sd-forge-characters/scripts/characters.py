"""
sd-forge-characters
===================

"Characters" tab — NovelAI V4.5-style multi-character generation without the
region math. Give each character their own prompt, pick a layout, generate.

Under the hood it drives the installed Forge Couple extension (Basic mode,
attention coupling) through Forge's local API: each character owns a column
of the canvas, the optional background line owns the whole frame. Forge
Couple errors (e.g. too few lines) are detected and reported cleanly.

Results save to output/characters/<date>/ with full metadata — so keepers
can go straight to the Gallery, Director, or Auto Pilot flows.

Requires --api (the launchers set it) and sd-forge-couple (installed).

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
from typing import List, Optional, Tuple

import gradio as gr
from PIL import Image, PngImagePlugin

try:
    from modules import script_callbacks, shared
except ImportError:
    script_callbacks = None  # type: ignore
    shared = None            # type: ignore

logger = logging.getLogger(__name__)
TAG = "[characters]"

EXT_ROOT = Path(__file__).resolve().parents[1]

QUALITY_TAGS = "masterpiece, best quality, amazing quality, very aesthetic, absurdres"
DEFAULT_NEGATIVE = ("bad quality, worst quality, worst detail, sketch, censor, "
                    "signature, watermark, username, jpeg artifacts, "
                    "bad anatomy, bad hands, extra digits")


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


# ---------------------------------------------------------------------------
# Prompt + Forge Couple payload assembly (pure — unit tested)
# ---------------------------------------------------------------------------

def compose_couple_prompt(chars: List[str], shared_tags: str, background: str,
                          add_quality: bool) -> Tuple[str, str]:
    """-> (joined_prompt, background_setting)

    Each character line gets the shared tags (and quality tags) appended so
    per-column conditioning stays complete. The background line, if any,
    is appended last and Forge Couple is told to treat the last line as
    background.
    """
    extras = ", ".join(x for x in [shared_tags.strip().rstrip(","),
                                   QUALITY_TAGS if add_quality else ""] if x)
    lines = []
    for c in chars:
        c = (c or "").strip().rstrip(",")
        if not c:
            continue
        lines.append(f"{c}, {extras}" if extras else c)
    bg = (background or "").strip().rstrip(",")
    bg_setting = "None"
    if bg:
        lines.append(f"{bg}, {extras}" if extras else bg)
        bg_setting = "Last Line"
    return "\n".join(lines), bg_setting


def build_couple_args(bg_setting: str, bg_weight: float) -> list:
    """Forge Couple v7 positional args (Basic mode, horizontal columns)."""
    return [
        True,          # enable
        False,         # disable_hr (compatibility)
        "Basic",       # mode
        "\n",          # separator
        "Horizontal",  # direction — characters side by side
        bg_setting,    # background: "None" | "Last Line"
        float(bg_weight),  # background_weight
        [],            # mapping (unused in Basic)
        "None",        # common_parser (off)
        False,         # common_debug
        False,         # def_in_prompt (doubles as tile-debug flag downstream)
    ]


def build_payload(prompt: str, negative: str, couple_args: list, steps: int,
                  cfg: float, width: int, height: int, sampler: str,
                  count: int) -> dict:
    return {
        "prompt": prompt,
        "negative_prompt": negative or DEFAULT_NEGATIVE,
        "steps": int(steps), "cfg_scale": float(cfg),
        "width": int(width), "height": int(height),
        "sampler_name": sampler or "Euler a", "seed": -1,
        "n_iter": max(1, int(count)), "batch_size": 1,
        "send_images": True, "save_images": False,
        "alwayson_scripts": {"forge couple": {"args": couple_args}},
    }


def couple_errored(info_json: str) -> bool:
    try:
        info = json.loads(info_json or "{}")
        for text in info.get("infotexts") or []:
            if "forge_couple: ERROR" in text or 'forge_couple": "ERROR"' in text:
                return True
        return "ERROR" in str(info.get("extra_generation_params", {}).get("forge_couple", ""))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _out_dir() -> Path:
    try:
        from modules import paths
        base = Path(paths.data_path)
    except Exception:
        base = EXT_ROOT.parents[1]
    d = base / "output" / "characters" / datetime.now().strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Character Passports (read from <data>/passports — no imports needed)
# ---------------------------------------------------------------------------

def _passports_dir() -> Path:
    try:
        from modules import paths
        return Path(paths.data_path) / "passports"
    except Exception:
        return EXT_ROOT.parents[1] / "passports"


def list_passports() -> List[str]:
    try:
        return [""] + sorted(p.stem for p in _passports_dir().glob("*.json"))
    except Exception:
        return [""]


def passport_fragment(name: str) -> str:
    if not name:
        return ""
    try:
        p = json.loads((_passports_dir() / f"{name}.json").read_text("utf-8"))
    except Exception:
        return ""
    parts = []
    for key in ("trigger", "tags"):
        v = (p.get(key) or "").strip().rstrip(",")
        if v:
            parts.append(v)
    lora = (p.get("lora") or "").strip()
    if lora:
        w = float(p.get("lora_weight") or 0.8)
        parts.append(f"<lora:{lora}:{w:g}>")
    return ", ".join(parts)


def insert_passport(name: str, current: str) -> str:
    frag = passport_fragment(name)
    if not frag:
        return current
    cur = (current or "").strip().rstrip(",")
    return f"{cur}, {frag}" if cur else frag


def run_characters(char_a, char_b, char_c, use_c, background, shared_tags,
                   add_quality, negative, bg_weight,
                   steps, cfg, width, height, sampler, count):
    chars = [char_a, char_b] + ([char_c] if use_c else [])
    chars = [c for c in chars if (c or "").strip()]
    if len(chars) < 2:
        yield [], "Give at least characters A and B a prompt each."
        return
    if not api_ok():
        yield [], "⚠ Forge API not reachable — launch with webui-user.bat (it sets --api)."
        return

    prompt, bg_setting = compose_couple_prompt(chars, shared_tags or "",
                                               background or "", bool(add_quality))
    args = build_couple_args(bg_setting, float(bg_weight))
    payload = build_payload(prompt, negative, args, steps, cfg, width, height,
                            sampler, count)
    n_chars = len(chars)
    yield [], (f"Generating {int(count)} image(s) — {n_chars} characters in "
               f"side-by-side columns"
               + (", background line active" if bg_setting != "None" else "")
               + "…")
    t0 = time.monotonic()
    try:
        res = _api_post("/sdapi/v1/txt2img", payload)
    except Exception as exc:
        yield [], f"✗ Generation failed: {exc}"
        return
    if couple_errored(res.get("info", "")):
        yield [], ("✗ Forge Couple rejected the layout (needs at least 2 "
                   "character lines; 3 when no background). Add prompts and retry.")
        return
    try:
        infotexts = json.loads(res.get("info") or "{}").get("infotexts") or []
    except Exception:
        infotexts = []
    images: List[Image.Image] = []
    out = _out_dir()
    for i, b64 in enumerate(res.get("images") or []):
        try:
            im = Image.open(io.BytesIO(base64.b64decode(b64)))
        except Exception:
            continue
        info = PngImagePlugin.PngInfo()
        if i < len(infotexts):
            info.add_text("parameters", infotexts[i])
        im.save(out / f"chars_{datetime.now().strftime('%H%M%S')}_{i}.png",
                pnginfo=info)
        images.append(im)
    yield images, (f"✅ {len(images)} image(s) in {time.monotonic()-t0:.0f}s "
                   f"→ output/characters/  (each character held their column)")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _on_ui_tabs():
    try:
        return _build_tab()
    except Exception:
        logger.exception("%s failed to build tab", TAG)
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown("Characters failed to load - see console log.")
        return [(ui, "Characters", "forge_characters")]


def _build_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## 👥 Characters — multi-character scenes without region math\n"
            "One prompt per character; each keeps their own column of the "
            "canvas (left → right). LoRA syntax works inside each line. "
            "Optional background line paints the whole frame."
        )
        with gr.Row():
            with gr.Column():
                char_a = gr.Textbox(label="Character A (left)", lines=3,
                                    placeholder="1girl, fu hua, white hair, "
                                                "school uniform, smile, <lora:…>")
                with gr.Row():
                    pass_a = gr.Dropdown(label="🪪 Passport A",
                                         choices=list_passports(), value="",
                                         scale=2)
                    ins_a = gr.Button("Insert", scale=0)
            with gr.Column():
                char_b = gr.Textbox(label="Character B (right)", lines=3,
                                    placeholder="1girl, vivian, purple hair, "
                                                "black dress, serious")
                with gr.Row():
                    pass_b = gr.Dropdown(label="🪪 Passport B",
                                         choices=list_passports(), value="",
                                         scale=2)
                    ins_b = gr.Button("Insert", scale=0)
            with gr.Column():
                use_c = gr.Checkbox(label="Add character C (middle-right)",
                                    value=False)
                char_c = gr.Textbox(label="Character C", lines=3)
                with gr.Row():
                    pass_c = gr.Dropdown(label="🪪 Passport C",
                                         choices=list_passports(), value="",
                                         scale=2)
                    ins_c = gr.Button("Insert", scale=0)
        pass_refresh = gr.Button("🔄 Refresh passports", scale=0)
        with gr.Row():
            background = gr.Textbox(
                label="Background / scene (optional, whole frame)",
                placeholder="classroom, sunset, window light", scale=2)
            bg_weight = gr.Slider(0.1, 1.2, value=0.5, step=0.05,
                                  label="Background weight")
        with gr.Row():
            shared_tags = gr.Textbox(
                label="Shared tags (applied to every character)",
                value="2girls, looking at viewer", scale=2)
            add_quality = gr.Checkbox(label="Add quality tags", value=True)
        negative = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE,
                              lines=2)
        with gr.Accordion("Generation settings", open=False):
            with gr.Row():
                steps = gr.Number(label="Steps", value=28, precision=0)
                cfg = gr.Number(label="CFG", value=5.5)
                width = gr.Number(label="Width", value=1216, precision=0)
                height = gr.Number(label="Height", value=832, precision=0)
                sampler = gr.Textbox(label="Sampler", value="Euler a")
        with gr.Row():
            count = gr.Slider(1, 8, value=2, step=1, label="Images")
            go_btn = gr.Button("👥 Generate scene", variant="primary")
        gallery = gr.Gallery(label="Results", columns=4, height=460,
                             object_fit="contain")
        status = gr.Markdown("")

        go_btn.click(run_characters,
                     [char_a, char_b, char_c, use_c, background, shared_tags,
                      add_quality, negative, bg_weight,
                      steps, cfg, width, height, sampler, count],
                     [gallery, status])
        ins_a.click(insert_passport, [pass_a, char_a], [char_a])
        ins_b.click(insert_passport, [pass_b, char_b], [char_b])
        ins_c.click(insert_passport, [pass_c, char_c], [char_c])
        pass_refresh.click(
            lambda: (gr.update(choices=list_passports()),) * 3, [],
            [pass_a, pass_b, pass_c])

    return [(ui, "Characters", "forge_characters")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
