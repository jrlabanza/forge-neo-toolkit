"""
sd-forge-gallery
================

Output gallery browser for Forge Neo. Adds a "Gallery" tab:

- Browses every configured output folder (txt2img / img2img / extras / saved),
  newest first, including date subfolders.
- Full-text search over the embedded generation parameters (prompt, seed,
  model, anything in the infotext).
- Date filter, favorites (star images, filter to favorites only).
- Click an image -> see its full generation parameters + file path, and
  send them straight to txt2img / img2img with one click (same paste API
  the NAI Converter tab uses).

Everything is read-only: this extension never modifies or deletes images.

Design notes:
- Thumbnails are passed to Gradio as PIL objects, so the tab keeps working
  even if your output folders live outside Forge's gradio-allowed paths
  (e.g. redirected by Stability Matrix).
- Parameter text is cached in param_cache.json (path+mtime keyed), so only
  new images are opened on re-scan. Cache and favorites live inside this
  extension's folder.

Author: built by Claude on 2026-07-20.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

try:
    from modules import script_callbacks, shared
except ImportError:          # direct execution / static analysis
    script_callbacks = None  # type: ignore
    shared = None            # type: ignore

logger = logging.getLogger(__name__)
TAG = "[gallery]"

EXT_ROOT = Path(__file__).resolve().parents[1]
CACHE_FILE = EXT_ROOT / "param_cache.json"
FAV_FILE = EXT_ROOT / "favorites.json"
SETTINGS_FILE = EXT_ROOT / "gallery_settings.json"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SCAN_CAP = 8000          # newest N files considered
PAGE_SIZE = 60
THUMB_PX = 320
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Small JSON persistence helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _save_json(path: Path, data) -> None:
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("%s could not save %s: %s", TAG, path.name, exc)


# ---------------------------------------------------------------------------
# Output folder discovery
# ---------------------------------------------------------------------------

def _data_path() -> Path:
    try:
        from modules import paths
        return Path(paths.data_path)
    except Exception:
        return EXT_ROOT.parents[1]


def _output_roots() -> List[Path]:
    """All folders that may contain generated images, existing ones only."""
    roots: List[Path] = []
    opt_names = (
        "outdir_samples", "outdir_txt2img_samples", "outdir_img2img_samples",
        "outdir_extras_samples", "outdir_save", "outdir_init_images",
        "outdir_txt2img_grids", "outdir_img2img_grids", "outdir_grids",
    )
    base = _data_path()
    if shared is not None:
        for name in opt_names:
            try:
                val = getattr(shared.opts, name, "") or ""
            except Exception:
                val = ""
            if not val:
                continue
            p = Path(val)
            if not p.is_absolute():
                p = base / p
            roots.append(p)
    # user-added extra folders
    extra = _load_json(SETTINGS_FILE, {}).get("extra_roots", [])
    roots.extend(Path(x) for x in extra if x)
    # fallback so the tab is never empty-handed
    roots.append(base / "output")

    seen, unique = set(), []
    for r in roots:
        try:
            key = str(r.resolve()).lower()
        except Exception:
            key = str(r).lower()
        if key not in seen and r.is_dir():
            seen.add(key)
            unique.append(r)
    return unique


def _scan_images() -> List[Tuple[str, float]]:
    """(path, mtime) for all images under all roots, newest first.

    Roots may nest (e.g. the fallback 'output' contains 'output/txt2img-images'),
    so files are deduped by normalized path.
    """
    found: List[Tuple[str, float]] = []
    seen = set()
    for root in _output_roots():
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                        full = os.path.join(dirpath, fn)
                        key = os.path.normcase(os.path.normpath(full))
                        if key in seen:
                            continue
                        try:
                            found.append((full, os.path.getmtime(full)))
                            seen.add(key)
                        except OSError:
                            pass
        except Exception as exc:
            logger.warning("%s scan failed for %s: %s", TAG, root, exc)
    found.sort(key=lambda t: t[1], reverse=True)
    return found[:SCAN_CAP]


# ---------------------------------------------------------------------------
# Generation-parameter extraction + cache
# ---------------------------------------------------------------------------

def _read_params_from_file(path: str) -> str:
    """Embedded A1111 'parameters' infotext, or '' if none."""
    try:
        with Image.open(path) as im:
            # PNG text chunk
            txt = getattr(im, "text", None)
            if isinstance(txt, dict):
                for key in ("parameters", "Comment", "Description"):
                    if txt.get(key):
                        return str(txt[key])
            # JPEG/WebP EXIF UserComment
            try:
                exif = im.getexif()
                raw = exif.get(0x9286)  # UserComment
                if isinstance(raw, bytes):
                    for codec, skip in (("utf-16-be", 8), ("utf-8", 8), ("utf-8", 0)):
                        try:
                            s = raw[skip:].decode(codec, errors="ignore").strip("\x00 ")
                            if s:
                                return s
                        except Exception:
                            pass
                elif isinstance(raw, str) and raw.strip():
                    return raw
            except Exception:
                pass
    except Exception:
        pass
    return ""


def _ensure_index(files: List[Tuple[str, float]],
                  progress: Optional[gr.Progress] = None) -> Dict[str, Dict]:
    """Return {path: {'m': mtime, 'p': params}} reading only new/changed files."""
    with _lock:
        cache: Dict[str, Dict] = _load_json(CACHE_FILE, {})
        todo = [(p, m) for p, m in files
                if p not in cache or abs(cache[p].get("m", -1) - m) > 1e-6]
        if todo:
            it = enumerate(todo)
            if progress is not None:
                try:
                    it = progress.tqdm(list(it), desc="Indexing new images")
                except Exception:
                    it = enumerate(todo)
            for _, (p, m) in it:
                cache[p] = {"m": m, "p": _read_params_from_file(p)}
        # drop entries for files that vanished (keeps cache from growing forever)
        live = {p for p, _ in files}
        stale = [p for p in cache if p not in live and not os.path.exists(p)]
        for p in stale:
            del cache[p]
        if todo or stale:
            _save_json(CACHE_FILE, cache)
        return cache


# ---------------------------------------------------------------------------
# Trash (safe delete: move into output/_trash, never actually deletes)
# ---------------------------------------------------------------------------

def _trash_dir() -> Path:
    d = _data_path() / "output" / "_trash"
    d.mkdir(parents=True, exist_ok=True)
    return d


def trash_files(paths: List[str]) -> Tuple[int, str]:
    import shutil
    moved = 0
    for p in paths:
        try:
            src = Path(p)
            if src.is_file():
                dest = _trash_dir() / src.name
                i = 1
                while dest.exists():
                    dest = _trash_dir() / f"{src.stem}_{i}{src.suffix}"
                    i += 1
                shutil.move(str(src), str(dest))
                moved += 1
        except Exception as exc:
            logger.warning("%s trash failed for %s: %s", TAG, p, exc)
    return moved, f"🗑 {moved} file(s) moved to output/_trash (restore by moving back)."


# ---------------------------------------------------------------------------
# Director handoff (cross-tab image passing via a small pointer file)
# ---------------------------------------------------------------------------

HANDOFF_FILE = EXT_ROOT / "director_handoff.txt"


def send_to_director(path: str) -> str:
    if not path or not Path(path).is_file():
        return "Select an image first."
    try:
        HANDOFF_FILE.write_text(path, "utf-8")
        return ("→ Sent. Open the **Director** tab and press **📥 From "
                "Gallery**.")
    except Exception as exc:
        return f"Handoff failed: {exc}"


# ---------------------------------------------------------------------------
# Favorites
# ---------------------------------------------------------------------------

def _favs() -> set:
    return set(_load_json(FAV_FILE, []))


def _toggle_fav(path: str) -> bool:
    """Toggle; returns True if now a favorite."""
    with _lock:
        favs = _favs()
        if path in favs:
            favs.discard(path)
            now = False
        else:
            favs.add(path)
            now = True
        _save_json(FAV_FILE, sorted(favs))
        return now


# ---------------------------------------------------------------------------
# Filtering + page rendering
# ---------------------------------------------------------------------------

def _date_of(path: str, mtime: float) -> str:
    parent = os.path.basename(os.path.dirname(path))
    if DATE_RE.match(parent):
        return parent
    try:
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


def _filtered_list(search: str, date_pick: str, favs_only: bool,
                   progress: Optional[gr.Progress] = None) -> List[str]:
    files = _scan_images()
    needle = (search or "").strip().lower()
    cache = _ensure_index(files, progress) if needle else _load_json(CACHE_FILE, {})
    favs = _favs() if favs_only else None

    out: List[str] = []
    for path, mtime in files:
        if favs is not None and path not in favs:
            continue
        if date_pick and date_pick != "All dates" and _date_of(path, mtime) != date_pick:
            continue
        if needle:
            params = (cache.get(path) or {}).get("p", "")
            hay = (params + " " + os.path.basename(path)).lower()
            if needle not in hay:
                continue
        out.append(path)
    return out


def _thumb(path: str) -> Optional[Image.Image]:
    try:
        im = Image.open(path)
        im.draft("RGB", (THUMB_PX, THUMB_PX))
        im.thumbnail((THUMB_PX, THUMB_PX))
        return im.convert("RGB")
    except Exception:
        return None


def _page_items(paths: List[str], page: int):
    total_pages = max(1, (len(paths) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, total_pages))
    chunk = paths[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    favs = _favs()
    items, kept = [], []
    for p in chunk:
        t = _thumb(p)
        if t is None:
            continue
        star = "★ " if p in favs else ""
        items.append((t, star + os.path.basename(p)))
        kept.append(p)
    label = f"Page {page} / {total_pages}  ({len(paths)} images)"
    return items, kept, page, label


def _dates_for_dropdown() -> List[str]:
    dates = {_date_of(p, m) for p, m in _scan_images()}
    return ["All dates"] + sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _on_ui_tabs():
    try:
        return _build_tab()
    except Exception:
        logger.exception("%s failed to build tab", TAG)
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown("Gallery failed to load - see console log.")
        return [(ui, "Gallery", "forge_gallery")]


def _build_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown("## Output Gallery — browse, search, and reuse your generations")

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", variant="primary", scale=0)
            search_box = gr.Textbox(label="Search prompts / parameters / filenames",
                                    placeholder="e.g.  fu hua, cowboy_shot, seed, model name…",
                                    scale=3, elem_classes=["prompt"])
            date_dd = gr.Dropdown(label="Date", choices=["All dates"],
                                  value="All dates", scale=1)
            favs_only = gr.Checkbox(label="★ Favorites only", value=False, scale=0)

        with gr.Row():
            with gr.Column(scale=3):
                gallery = gr.Gallery(label="Images", columns=6, height=620,
                                     allow_preview=True, object_fit="contain")
                with gr.Row():
                    prev_btn = gr.Button("◀ Prev", scale=0)
                    page_label = gr.Markdown("Page 1 / 1")
                    next_btn = gr.Button("Next ▶", scale=0)
            with gr.Column(scale=2):
                sel_image = gr.Image(label="Selected", type="pil",
                                     interactive=False, height=320)
                params_box = gr.Textbox(label="Generation parameters", lines=10,
                                        interactive=True, show_copy_button=True)
                path_box = gr.Textbox(label="File path", interactive=False,
                                      show_copy_button=True)
                with gr.Row():
                    fav_btn = gr.Button("★ Toggle favorite")
                    send_t2i = gr.Button("Send to txt2img", variant="primary")
                    send_i2i = gr.Button("Send to img2img")
                    director_btn = gr.Button("→ Director")
                status = gr.Markdown("")
                with gr.Accordion("🧺 Selection set (bulk actions)", open=False):
                    set_md = gr.Markdown("*empty*")
                    with gr.Row():
                        set_add_btn = gr.Button("➕ Add current", scale=0)
                        set_clear_btn = gr.Button("Clear", scale=0)
                        set_fav_btn = gr.Button("★ Favorite set", scale=0)
                        set_trash_btn = gr.Button("🗑 Trash set",
                                                  variant="stop", scale=0)
                with gr.Accordion("⚖ A/B compare", open=False):
                    with gr.Row():
                        set_a_btn = gr.Button("Set current as A", scale=0)
                        set_b_btn = gr.Button("Set current as B", scale=0)
                    with gr.Row():
                        img_a = gr.Image(label="A", type="pil",
                                         interactive=False, height=300)
                        img_b = gr.Image(label="B", type="pil",
                                         interactive=False, height=300)

        # server-side state
        st_filtered = gr.State([])   # full filtered path list
        st_page_paths = gr.State([])  # paths shown on current page
        st_page = gr.State(1)
        st_selected = gr.State("")

        # ------------------------------------------------------------------
        def do_refresh(search, date_pick, favs_flag, progress=gr.Progress()):
            paths = _filtered_list(search, date_pick, favs_flag, progress)
            items, kept, page, label = _page_items(paths, 1)
            dates = _dates_for_dropdown()
            date_val = date_pick if date_pick in dates else "All dates"
            return (items, paths, kept, page, label,
                    gr.update(choices=dates, value=date_val))

        def do_filter(search, date_pick, favs_flag, progress=gr.Progress()):
            paths = _filtered_list(search, date_pick, favs_flag, progress)
            items, kept, page, label = _page_items(paths, 1)
            return items, paths, kept, page, label

        def do_page(paths, page, delta):
            items, kept, page, label = _page_items(paths, (page or 1) + delta)
            return items, kept, page, label

        def do_select(page_paths, evt: gr.SelectData):
            try:
                path = page_paths[evt.index]
            except Exception:
                return None, "", "", "", ""
            params = (_load_json(CACHE_FILE, {}).get(path) or {}).get("p")
            if params is None or params == "":
                params = _read_params_from_file(path)
            img = None
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                pass
            star = "★ favorited" if path in _favs() else ""
            return img, params or "(no embedded parameters found)", path, path, star

        def do_fav(path):
            if not path:
                return "Select an image first."
            return "★ added to favorites" if _toggle_fav(path) else "☆ removed from favorites"

        # ------------------------------------------------------------------
        refresh_btn.click(do_refresh,
                          inputs=[search_box, date_dd, favs_only],
                          outputs=[gallery, st_filtered, st_page_paths, st_page,
                                   page_label, date_dd])
        search_box.submit(do_filter,
                          inputs=[search_box, date_dd, favs_only],
                          outputs=[gallery, st_filtered, st_page_paths, st_page, page_label])
        date_dd.input(do_filter,
                      inputs=[search_box, date_dd, favs_only],
                      outputs=[gallery, st_filtered, st_page_paths, st_page, page_label])
        favs_only.input(do_filter,
                        inputs=[search_box, date_dd, favs_only],
                        outputs=[gallery, st_filtered, st_page_paths, st_page, page_label])
        prev_btn.click(lambda paths, page: do_page(paths, page, -1),
                       inputs=[st_filtered, st_page],
                       outputs=[gallery, st_page_paths, st_page, page_label])
        next_btn.click(lambda paths, page: do_page(paths, page, +1),
                       inputs=[st_filtered, st_page],
                       outputs=[gallery, st_page_paths, st_page, page_label])
        gallery.select(do_select,
                       inputs=[st_page_paths],
                       outputs=[sel_image, params_box, path_box, st_selected, status])
        fav_btn.click(do_fav, inputs=[st_selected], outputs=[status])
        director_btn.click(send_to_director, [st_selected], [status])

        # ---- selection set ------------------------------------------------
        st_set = gr.State([])

        def set_add(sel, cur):
            cur = list(cur or [])
            if sel and sel not in cur:
                cur.append(sel)
            names = ", ".join(os.path.basename(p) for p in cur) or "*empty*"
            return cur, f"**{len(cur)} in set:** {names[:400]}"

        def set_clear():
            return [], "*empty*"

        def set_fav(cur):
            n = 0
            with _lock:
                favs = _favs()
                for p in cur or []:
                    if p not in favs:
                        favs.add(p)
                        n += 1
                _save_json(FAV_FILE, sorted(favs))
            return f"★ {n} added to favorites."

        def set_trash(cur, search, date_pick, favs_flag):
            _, msg = trash_files(list(cur or []))
            paths = _filtered_list(search, date_pick, favs_flag)
            items, kept, page, label = _page_items(paths, 1)
            return ([], "*empty*", msg, items, paths, kept, page, label)

        set_add_btn.click(set_add, [st_selected, st_set], [st_set, set_md])
        set_clear_btn.click(set_clear, [], [st_set, set_md])
        set_fav_btn.click(set_fav, [st_set], [status])
        set_trash_btn.click(set_trash,
                            [st_set, search_box, date_dd, favs_only],
                            [st_set, set_md, status, gallery, st_filtered,
                             st_page_paths, st_page, page_label])

        # ---- A/B compare --------------------------------------------------
        def _load_ab(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None

        set_a_btn.click(_load_ab, [st_selected], [img_a])
        set_b_btn.click(_load_ab, [st_selected], [img_b])

        # Send-to buttons via Forge's paste-params API (same as NAI Converter).
        _wire_send_buttons(send_t2i, send_i2i, params_box, sel_image, status)

    return [(ui, "Gallery", "forge_gallery")]


def _wire_send_buttons(send_t2i, send_i2i, params_box, sel_image, status):
    import importlib
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
                source_text_component=params_box,
                source_image_component=sel_image,
            ))
    else:
        msg = "Paste API not found — copy the parameters manually."
        send_t2i.click(lambda: msg, None, [status])
        send_i2i.click(lambda: msg, None, [status])


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
