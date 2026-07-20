"""
sd-forge-styles-manager
=======================

A proper manager for styles.csv (448 entries and counting). Adds a
"Styles" tab:

- live search over style names + prompt text (finds "which style was that
  ZZZ character again?")
- select a style -> edit name / prompt / negative in place
- Save / Save as new / Delete — every write first backs up styles.csv to
  _attic/styles-backups/<timestamp>.csv (keeps newest 20)
- Send to txt2img / img2img (composes prompt + negative via the paste API;
  REPLACES what's in the target tab)
- shows a sample image from your outputs that used the style's LoRA,
  found via the Gallery extension's parameter cache (if present)

After saving, the native styles dropdown refreshes automatically (Forge's
StyleDatabase.reload).

Author: built by Claude on 2026-07-20.
"""
from __future__ import annotations

import csv
import importlib
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from PIL import Image

try:
    from modules import script_callbacks, shared
except ImportError:
    script_callbacks = None  # type: ignore
    shared = None            # type: ignore

logger = logging.getLogger(__name__)
TAG = "[styles-manager]"

EXT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_KEEP = 20


# ---------------------------------------------------------------------------
# styles.csv IO
# ---------------------------------------------------------------------------

def _data_path() -> Path:
    try:
        from modules import paths
        return Path(paths.data_path)
    except Exception:
        return EXT_ROOT.parents[1]


def _styles_file() -> Path:
    # honor --styles-file if set to a single concrete path
    try:
        v = getattr(shared.cmd_opts, "styles_file", None)
        if isinstance(v, str) and v and "*" not in v:
            return Path(v)
        if isinstance(v, (list, tuple)) and len(v) == 1 and "*" not in str(v[0]):
            return Path(v[0])
    except Exception:
        pass
    return _data_path() / "styles.csv"


def load_styles() -> List[dict]:
    path = _styles_file()
    rows: List[dict] = []
    if not path.is_file():
        return rows
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                name = (r.get("name") or "").strip()
                if not name:
                    continue
                rows.append({
                    "name": name,
                    "prompt": r.get("prompt") or "",
                    "negative_prompt": r.get("negative_prompt") or "",
                })
    except Exception as exc:
        logger.warning("%s failed reading %s: %s", TAG, path, exc)
    return rows


def _backup_styles() -> None:
    path = _styles_file()
    if not path.is_file():
        return
    dest_dir = _data_path() / "_attic" / "styles-backups"
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    shutil.copy2(path, dest_dir / f"styles_{stamp}.csv")
    old = sorted(dest_dir.glob("styles_*.csv"), reverse=True)[BACKUP_KEEP:]
    for f in old:
        try:
            f.unlink()
        except OSError:
            pass


def save_styles(rows: List[dict]) -> None:
    _backup_styles()
    path = _styles_file()
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "prompt", "negative_prompt"])
        w.writeheader()
        for r in rows:
            w.writerow({"name": r["name"], "prompt": r["prompt"],
                        "negative_prompt": r["negative_prompt"]})
    # refresh Forge's own style registry so the native dropdown updates
    try:
        shared.prompt_styles.reload()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Search / selection helpers
# ---------------------------------------------------------------------------

def _label(idx: int, row: dict) -> str:
    return f"{idx:03d} · {row['name']}"


def _idx_from_label(label: str) -> Optional[int]:
    try:
        return int(label.split("·", 1)[0].strip())
    except Exception:
        return None


def _filter_labels(rows: List[dict], needle: str) -> List[str]:
    needle = (needle or "").strip().lower()
    out = []
    for i, r in enumerate(rows):
        if not needle or needle in r["name"].lower() or needle in r["prompt"].lower():
            out.append(_label(i, r))
    return out


# ---------------------------------------------------------------------------
# Sample image via the Gallery extension's cache
# ---------------------------------------------------------------------------

def _find_sample(row: dict) -> Tuple[Optional[Image.Image], str]:
    try:
        import json
        cache_file = (EXT_ROOT.parent / "sd-forge-gallery" / "param_cache.json")
        if not cache_file.is_file():
            return None, "No sample (Gallery cache not built yet — open the Gallery tab and search once)."
        with open(cache_file, "r", encoding="utf-8") as fh:
            cache = json.load(fh)
        # most distinctive token: the lora reference if present, else the name
        import re
        m = re.search(r"<lora:([^:>]+)", row["prompt"])
        token = (m.group(1) if m else row["name"]).lower()
        if not token:
            return None, ""
        for p, entry in cache.items():
            params = (entry or {}).get("p", "")
            if token in params.lower():
                try:
                    return Image.open(p).convert("RGB"), f"Sample: {Path(p).name}"
                except Exception:
                    continue
        return None, "No output found that used this style yet."
    except Exception as exc:
        logger.warning("%s sample lookup failed: %s", TAG, exc)
        return None, ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _on_ui_tabs():
    try:
        return _build_tab()
    except Exception:
        logger.exception("%s failed to build tab", TAG)
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown("Styles Manager failed to load - see console log.")
        return [(ui, "Styles", "forge_styles_manager")]


def _build_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown("## Styles Manager — search, edit and reuse styles.csv")

        st_rows = gr.State(load_styles())

        with gr.Row():
            search_box = gr.Textbox(label="Search styles (name or prompt text)",
                                    placeholder="e.g.  zzz, honkai, screencap…", scale=3)
            refresh_btn = gr.Button("🔄 Reload from disk", scale=0)
        count_md = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=2):
                pick_dd = gr.Dropdown(label="Style", choices=[], value=None)
                name_box = gr.Textbox(label="Name")
                prompt_box = gr.Textbox(label="Prompt", lines=6, show_copy_button=True)
                negative_box = gr.Textbox(label="Negative prompt", lines=3,
                                          show_copy_button=True)
                with gr.Row():
                    save_btn = gr.Button("💾 Save changes", variant="primary")
                    new_btn = gr.Button("➕ Save as new style")
                    del_btn = gr.Button("🗑 Delete style", variant="stop")
                with gr.Row():
                    send_t2i = gr.Button("Send to txt2img (replaces prompt)")
                    send_i2i = gr.Button("Send to img2img")
                status = gr.Markdown("")
                composed_box = gr.Textbox(visible=False)
            with gr.Column(scale=1):
                sample_img = gr.Image(label="Sample from your outputs", type="pil",
                                      interactive=False, height=380)
                sample_md = gr.Markdown("")

        # ------------------------------------------------------------------
        def _dd_update(rows, needle, keep_value=None):
            labels = _filter_labels(rows, needle)
            value = keep_value if keep_value in labels else None
            return gr.update(choices=labels, value=value), \
                f"{len(labels)} / {len(rows)} styles shown"

        def do_reload():
            rows = load_styles()
            dd, cnt = _dd_update(rows, "")
            return rows, dd, cnt, "Reloaded from disk."

        def do_search(rows, needle):
            dd, cnt = _dd_update(rows, needle)
            return dd, cnt

        def do_pick(rows, label):
            idx = _idx_from_label(label)
            if idx is None or idx >= len(rows):
                return "", "", "", None, "", ""
            r = rows[idx]
            img, cap = _find_sample(r)
            composed = f"{r['prompt']}\nNegative prompt: {r['negative_prompt']}" \
                if r["negative_prompt"] else r["prompt"]
            return r["name"], r["prompt"], r["negative_prompt"], img, cap, composed

        def do_save(rows, label, name, prompt, negative, needle):
            idx = _idx_from_label(label)
            name = (name or "").strip()
            if idx is None or idx >= len(rows):
                return rows, gr.update(), gr.update(), "Pick a style first."
            if not name:
                return rows, gr.update(), gr.update(), "Name cannot be empty."
            rows = list(rows)
            rows[idx] = {"name": name, "prompt": prompt or "",
                         "negative_prompt": negative or ""}
            save_styles(rows)
            dd, cnt = _dd_update(rows, needle, keep_value=_label(idx, rows[idx]))
            return rows, dd, cnt, f"Saved '{name}' (backup made)."

        def do_new(rows, name, prompt, negative, needle):
            name = (name or "").strip()
            if not name:
                return rows, gr.update(), gr.update(), "Name cannot be empty."
            rows = list(rows) + [{"name": name, "prompt": prompt or "",
                                  "negative_prompt": negative or ""}]
            save_styles(rows)
            dd, cnt = _dd_update(rows, needle, keep_value=_label(len(rows) - 1, rows[-1]))
            return rows, dd, cnt, f"Added '{name}' ({len(rows)} styles now)."

        def do_delete(rows, label, needle):
            idx = _idx_from_label(label)
            if idx is None or idx >= len(rows):
                return rows, gr.update(), gr.update(), "Pick a style first."
            rows = list(rows)
            gone = rows.pop(idx)
            save_styles(rows)
            dd, cnt = _dd_update(rows, needle)
            return rows, dd, cnt, f"Deleted '{gone['name']}' (backup made)."

        # ------------------------------------------------------------------
        refresh_btn.click(do_reload, [], [st_rows, pick_dd, count_md, status])
        search_box.change(do_search, [st_rows, search_box], [pick_dd, count_md])
        pick_dd.input(do_pick, [st_rows, pick_dd],
                      [name_box, prompt_box, negative_box, sample_img,
                       sample_md, composed_box])
        save_btn.click(do_save,
                       [st_rows, pick_dd, name_box, prompt_box, negative_box, search_box],
                       [st_rows, pick_dd, count_md, status])
        new_btn.click(do_new,
                      [st_rows, name_box, prompt_box, negative_box, search_box],
                      [st_rows, pick_dd, count_md, status])
        del_btn.click(do_delete, [st_rows, pick_dd, search_box],
                      [st_rows, pick_dd, count_md, status])

        # keep the hidden composed box current while editing
        for comp in (prompt_box, negative_box):
            comp.change(lambda p, n: (f"{p}\nNegative prompt: {n}" if n else p or ""),
                        [prompt_box, negative_box], [composed_box])

        ui.load(do_reload, [], [st_rows, pick_dd, count_md, status])

        # paste wiring
        paste_mod = None
        for mod_name in ("modules.infotext_utils", "modules.generation_parameters_copypaste"):
            try:
                paste_mod = importlib.import_module(mod_name)
                break
            except ImportError:
                continue
        if paste_mod is not None and hasattr(paste_mod, "register_paste_params_button"):
            for btn, tab in ((send_t2i, "txt2img"), (send_i2i, "img2img")):
                paste_mod.register_paste_params_button(paste_mod.ParamBinding(
                    paste_button=btn, tabname=tab,
                    source_text_component=composed_box,
                ))
        else:
            msg = "Paste API not found — copy the prompt manually."
            send_t2i.click(lambda: msg, None, [status])
            send_i2i.click(lambda: msg, None, [status])

    return [(ui, "Styles", "forge_styles_manager")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
