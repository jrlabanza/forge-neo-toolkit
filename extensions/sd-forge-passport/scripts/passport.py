"""
sd-forge-passport
=================

"Passports" tab — one saved identity per character. A passport bundles:

- canonical tags        (hair, eyes, outfit, series…)
- LoRA + weight         (with trigger words auto-filled from the Civitai
                         helper's card data when available)
- reference image       (stored copy, for the Reference accordion)
- extra negative        (character-specific exclusions)

Passports are plain files in <data>/passports/ (json + png), so every other
tool can read them without imports: the Characters tab and Auto Pilot both
grow passport pickers that insert the composed fragment in one click.

The composed fragment looks like:
    fu hua, honkai (series), white hair, long hair, <lora:fu_hua_v2:0.8>

Author: built by Claude on 2026-07-21.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

try:
    from modules import script_callbacks, shared
except ImportError:
    script_callbacks = None  # type: ignore
    shared = None            # type: ignore

logger = logging.getLogger(__name__)
TAG = "[passport]"

EXT_ROOT = Path(__file__).resolve().parents[1]


def _data_path() -> Path:
    try:
        from modules import paths
        return Path(paths.data_path)
    except Exception:
        return EXT_ROOT.parents[1]


def passports_dir() -> Path:
    d = _data_path() / "passports"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Data model (pure functions — unit tested)
# ---------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    return "".join(c for c in (name or "").strip()
                   if c.isalnum() or c in " _-")[:60].strip()


def compose_fragment(p: dict) -> str:
    """Passport dict -> prompt fragment."""
    parts: List[str] = []
    for key in ("trigger", "tags"):
        v = (p.get(key) or "").strip().rstrip(",")
        if v:
            parts.append(v)
    lora = (p.get("lora") or "").strip()
    if lora:
        w = float(p.get("lora_weight") or 0.8)
        parts.append(f"<lora:{lora}:{w:g}>")
    return ", ".join(parts)


def list_passports() -> List[str]:
    try:
        return sorted(p.stem for p in passports_dir().glob("*.json"))
    except Exception:
        return []


def load_passport(name: str) -> Optional[dict]:
    try:
        return json.loads((passports_dir() / f"{name}.json").read_text("utf-8"))
    except Exception:
        return None


def save_passport(p: dict) -> str:
    name = sanitize_name(p.get("name", ""))
    if not name:
        raise ValueError("passport needs a name")
    p = dict(p, name=name)
    (passports_dir() / f"{name}.json").write_text(
        json.dumps(p, ensure_ascii=False, indent=2), "utf-8")
    return name


# ---------------------------------------------------------------------------
# LoRA + trigger helpers (self-contained copies, same conventions as elsewhere)
# ---------------------------------------------------------------------------

def _lora_dirs() -> List[Path]:
    dirs: List[Path] = []
    try:
        v = getattr(shared.cmd_opts, "lora_dirs", None)
        if isinstance(v, str) and v:
            dirs.append(Path(v))
        elif isinstance(v, (list, tuple)):
            dirs.extend(Path(x) for x in v if x)
    except Exception:
        pass
    base = _data_path()
    dirs += [base / "models" / "Lora", base / "models" / "LyCORIS"]
    seen, out = set(), []
    for d in dirs:
        k = str(d).lower()
        if k not in seen and d.is_dir():
            seen.add(k)
            out.append(d)
    return out


def list_loras() -> List[str]:
    names = set()
    for root in _lora_dirs():
        try:
            for f in root.rglob("*"):
                if f.suffix.lower() in (".safetensors", ".pt", ".ckpt") and f.is_file():
                    names.add(f.stem)
        except Exception:
            pass
    return [""] + sorted(names, key=str.lower)


def lora_trigger(stem: str) -> str:
    if not stem:
        return ""
    for root in _lora_dirs():
        try:
            for f in root.rglob(stem + ".json"):
                try:
                    card = json.loads(f.read_text("utf-8"))
                    return str(card.get("activation text") or "").strip()
                except Exception:
                    continue
        except Exception:
            pass
    return ""


# ---------------------------------------------------------------------------
# UI actions
# ---------------------------------------------------------------------------

def do_save(name, tags, lora, lora_weight, trigger, neg_extra, ref_img,
            ref_mode, ref_strength):
    try:
        p = {
            "name": name, "tags": tags or "", "lora": lora or "",
            "lora_weight": float(lora_weight or 0.8),
            "trigger": trigger or "", "negative_extra": neg_extra or "",
            "ref_mode": ref_mode or "Vibe (style + composition)",
            "ref_strength": float(ref_strength or 0.7),
        }
        clean = save_passport(p)
        if ref_img is not None:
            Image.fromarray(np.asarray(ref_img).astype("uint8")).save(
                passports_dir() / f"{clean}.png")
        frag = compose_fragment(p)
        return (gr.update(choices=list_passports(), value=clean),
                frag, f"✅ Saved passport '{clean}'.")
    except Exception as exc:
        logger.warning("%s save failed: %s", TAG, exc)
        return gr.update(choices=list_passports()), "", f"Save failed: {exc}"


def do_load(name):
    p = load_passport(name)
    if not p:
        return ("", "", "", 0.8, "", "", None, "Vibe (style + composition)",
                0.7, "", "Pick a passport first.")
    img = None
    try:
        f = passports_dir() / f"{name}.png"
        if f.is_file():
            img = np.asarray(Image.open(f).convert("RGB"))
    except Exception:
        pass
    return (p.get("name", name), p.get("tags", ""), p.get("lora", ""),
            float(p.get("lora_weight", 0.8)), p.get("trigger", ""),
            p.get("negative_extra", ""), img,
            p.get("ref_mode", "Vibe (style + composition)"),
            float(p.get("ref_strength", 0.7)),
            compose_fragment(p), f"Loaded '{name}'.")


def do_autofill_trigger(lora):
    t = lora_trigger(lora)
    return t, ("Trigger words filled from Civitai data."
               if t else "No trigger data — run the Civitai tab fetch first, "
                         "or type them manually.")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _on_ui_tabs():
    try:
        return _build_tab()
    except Exception:
        logger.exception("%s failed to build tab", TAG)
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown("Passports failed to load - see console log.")
        return [(ui, "Passports", "forge_passport")]


def _build_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## 🪪 Character Passports — save an identity once, use it everywhere\n"
            "A passport = tags + LoRA + trigger words + reference image. "
            "The **Characters** tab and **Auto Pilot** both have passport "
            "pickers that insert the composed fragment in one click."
        )
        with gr.Row():
            pass_dd = gr.Dropdown(label="Saved passports",
                                  choices=list_passports(), scale=2)
            load_btn = gr.Button("📥 Load", scale=0)
            refresh_btn = gr.Button("🔄", scale=0)
        with gr.Row():
            with gr.Column(scale=2):
                name_box = gr.Textbox(label="Name (e.g. Fu Hua)")
                tags_box = gr.Textbox(
                    label="Canonical tags", lines=3,
                    placeholder="white hair, long hair, side ponytail, "
                                "hair between eyes, blue eyes…")
                with gr.Row():
                    lora_dd = gr.Dropdown(label="LoRA", choices=list_loras(),
                                          value="", scale=2)
                    lora_w = gr.Slider(0.1, 1.5, value=0.8, step=0.05,
                                       label="Weight")
                with gr.Row():
                    trigger_box = gr.Textbox(label="Trigger words", scale=2)
                    trig_btn = gr.Button("✨ Auto-fill from Civitai data",
                                         scale=0)
                neg_box = gr.Textbox(
                    label="Extra negative (character-specific)",
                    placeholder="e.g. wrong hair color tags to suppress")
            with gr.Column(scale=1):
                ref_img = gr.Image(label="Reference image (for the Reference "
                                         "accordion)", type="numpy", height=260)
                ref_mode = gr.Radio(
                    choices=["Vibe (style + composition)",
                             "Precise (face / character)"],
                    value="Precise (face / character)", label="Reference mode")
                ref_strength = gr.Slider(0.1, 1.2, value=0.7, step=0.05,
                                         label="Reference strength")
        with gr.Row():
            save_btn = gr.Button("💾 Save passport", variant="primary")
        frag_box = gr.Textbox(label="Composed fragment (what gets inserted)",
                              interactive=False, show_copy_button=True)
        status = gr.Markdown("")

        save_btn.click(do_save,
                       [name_box, tags_box, lora_dd, lora_w, trigger_box,
                        neg_box, ref_img, ref_mode, ref_strength],
                       [pass_dd, frag_box, status])
        load_btn.click(do_load, [pass_dd],
                       [name_box, tags_box, lora_dd, lora_w, trigger_box,
                        neg_box, ref_img, ref_mode, ref_strength, frag_box,
                        status])
        refresh_btn.click(lambda: gr.update(choices=list_passports()), [],
                          [pass_dd])
        trig_btn.click(do_autofill_trigger, [lora_dd], [trigger_box, status])

    return [(ui, "Passports", "forge_passport")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
