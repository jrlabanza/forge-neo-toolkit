"""
sd-forge-civitai-helper
=======================

"Civitai" tab: fills in the boring metadata for your LoRA collection.

For every LoRA it can fetch from Civitai (matched by SHA256 hash):
- trigger words  -> written into <lora>.json "activation text", so clicking
  the card in the Lora tab auto-inserts the right activation tags
- preview image  -> saved as <lora>.preview.png for the card thumbnail
- full metadata  -> saved as <lora>.civitai.json (model name, version,
  base model, page URL)
- update check   -> compares your file's version against the model's
  newest version on Civitai

Nothing is ever overwritten: existing previews and existing keys in
<lora>.json are kept. Hashes are cached (hashes.json) so re-runs are fast.
Needs internet at click-time; failures are logged per file and skipped.

LoRA folders searched: --lora-dirs launch arg (if set), models/Lora under
the install, plus any "extra_dirs" in civitai_settings.json here.

Author: built by Claude on 2026-07-20.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr

try:
    from modules import script_callbacks, shared
except ImportError:
    script_callbacks = None  # type: ignore
    shared = None            # type: ignore

logger = logging.getLogger(__name__)
TAG = "[civitai-helper]"

EXT_ROOT = Path(__file__).resolve().parents[1]
HASH_CACHE = EXT_ROOT / "hashes.json"
RESULT_CACHE = EXT_ROOT / "results.json"
SETTINGS_FILE = EXT_ROOT / "civitai_settings.json"

API_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/{sha}"
API_MODEL = "https://civitai.com/api/v1/models/{mid}"
LORA_EXTS = {".safetensors", ".pt", ".ckpt"}
SLEEP_BETWEEN = 0.6  # be polite to the API


def _load_json(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _save_json(path: Path, data) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
    except Exception as exc:
        logger.warning("%s could not save %s: %s", TAG, path.name, exc)


# ---------------------------------------------------------------------------
# LoRA discovery
# ---------------------------------------------------------------------------

def _data_path() -> Path:
    try:
        from modules import paths
        return Path(paths.data_path)
    except Exception:
        return EXT_ROOT.parents[1]


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
    dirs.append(_data_path() / "models" / "Lora")
    dirs.append(_data_path() / "models" / "LyCORIS")
    for x in _load_json(SETTINGS_FILE, {}).get("extra_dirs", []):
        dirs.append(Path(x))
    seen, out = set(), []
    for d in dirs:
        k = str(d).lower()
        if k not in seen and d.is_dir():
            seen.add(k)
            out.append(d)
    return out


def _scan_loras() -> List[Path]:
    found: List[Path] = []
    seen = set()
    for root in _lora_dirs():
        try:
            for p in root.rglob("*"):
                if p.suffix.lower() in LORA_EXTS and p.is_file():
                    k = str(p).lower()
                    if k not in seen:
                        seen.add(k)
                        found.append(p)
        except Exception as exc:
            logger.warning("%s scan failed in %s: %s", TAG, root, exc)
    return sorted(found, key=lambda p: p.name.lower())


# ---------------------------------------------------------------------------
# Hashing (cached)
# ---------------------------------------------------------------------------

def _sha256(path: Path, cache: Dict) -> Optional[str]:
    try:
        st = path.stat()
        key = str(path)
        ent = cache.get(key)
        if ent and ent.get("size") == st.st_size and abs(ent.get("mtime", 0) - st.st_mtime) < 2:
            return ent["sha256"]
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024 * 4), b""):
                h.update(chunk)
        digest = h.hexdigest()
        cache[key] = {"size": st.st_size, "mtime": st.st_mtime, "sha256": digest}
        return digest
    except Exception as exc:
        logger.warning("%s hashing failed for %s: %s", TAG, path.name, exc)
        return None


# ---------------------------------------------------------------------------
# Civitai API
# ---------------------------------------------------------------------------

def _api_get(url: str):
    import requests
    r = requests.get(url, timeout=25,
                     headers={"User-Agent": "sd-forge-civitai-helper/1.0"})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def _fetch_version_info(sha: str, results: Dict) -> Optional[dict]:
    if sha in results:
        return results[sha]
    data = _api_get(API_BY_HASH.format(sha=sha))
    if data is None:
        results[sha] = None
        return None
    trimmed = {
        "versionId": data.get("id"),
        "versionName": data.get("name"),
        "modelId": data.get("modelId"),
        "modelName": (data.get("model") or {}).get("name"),
        "baseModel": data.get("baseModel"),
        "trainedWords": data.get("trainedWords") or [],
        "images": [im.get("url") for im in (data.get("images") or [])
                   if im.get("url")][:3],
        "page": f"https://civitai.com/models/{data.get('modelId')}",
    }
    results[sha] = trimmed
    return trimmed


def _write_sidecars(lora: Path, info: dict, log: List[str]) -> None:
    base = lora.with_suffix("")
    # 1) full metadata
    civ_file = base.parent / (base.name + ".civitai.json")
    if not civ_file.exists():
        _save_json(civ_file, info)
    # 2) activation text into the A1111 card-metadata json (merge, don't clobber)
    words = ", ".join(info.get("trainedWords") or [])
    if words:
        card_file = base.parent / (base.name + ".json")
        card = _load_json(card_file, {})
        if not card.get("activation text"):
            card["activation text"] = words
            card.setdefault("description",
                            f"{info.get('modelName')} — {info.get('versionName')} "
                            f"({info.get('baseModel')})  {info.get('page')}")
            _save_json(card_file, card)
            log.append(f"    triggers: {words}")
    # 3) preview image
    preview = base.parent / (base.name + ".preview.png")
    if not preview.exists() and info.get("images"):
        try:
            import requests
            r = requests.get(info["images"][0], timeout=30,
                             headers={"User-Agent": "sd-forge-civitai-helper/1.0"})
            r.raise_for_status()
            preview.write_bytes(r.content)
            log.append("    preview saved")
        except Exception as exc:
            log.append(f"    preview failed: {exc}")


# ---------------------------------------------------------------------------
# Actions (generators -> live log streaming)
# ---------------------------------------------------------------------------

def _overview_rows() -> List[List[str]]:
    results = _load_json(RESULT_CACHE, {})
    hashes = _load_json(HASH_CACHE, {})
    rows = []
    for p in _scan_loras():
        base = p.with_suffix("")
        has_prev = (base.parent / (base.name + ".preview.png")).exists() or \
                   (base.parent / (base.name + ".png")).exists()
        card = _load_json(base.parent / (base.name + ".json"), {})
        sha = (hashes.get(str(p)) or {}).get("sha256")
        info = results.get(sha) if sha else None
        rows.append([
            p.name,
            f"{p.stat().st_size / 1e6:.0f} MB",
            "✓" if has_prev else "—",
            (card.get("activation text") or "")[:60] or "—",
            (info or {}).get("modelName") or ("not found" if sha and info is None else "—"),
        ])
    return rows


def do_scan():
    rows = _overview_rows()
    dirs = ", ".join(str(d) for d in _lora_dirs()) or "(none found)"
    return rows, f"{len(rows)} LoRA files across: {dirs}"


def do_fetch(only_missing: bool):
    loras = _scan_loras()
    hashes = _load_json(HASH_CACHE, {})
    results = _load_json(RESULT_CACHE, {})
    log: List[str] = [f"Fetching metadata for {len(loras)} LoRAs "
                      f"({'missing only' if only_missing else 'all'})…"]
    yield "\n".join(log), gr.update()
    done = 0
    for p in loras:
        base = p.with_suffix("")
        card = _load_json(base.parent / (base.name + ".json"), {})
        has_prev = (base.parent / (base.name + ".preview.png")).exists()
        if only_missing and card.get("activation text") and has_prev:
            continue
        log.append(f"» {p.name}")
        yield "\n".join(log[-40:]), gr.update()
        sha = _sha256(p, hashes)
        _save_json(HASH_CACHE, hashes)
        if not sha:
            log.append("    hash failed, skipped")
            continue
        try:
            info = _fetch_version_info(sha, results)
            _save_json(RESULT_CACHE, results)
        except Exception as exc:
            log.append(f"    Civitai request failed: {exc}")
            yield "\n".join(log[-40:]), gr.update()
            continue
        if info is None:
            log.append("    not on Civitai (hash unknown)")
        else:
            log.append(f"    matched: {info['modelName']} / {info['versionName']}")
            _write_sidecars(p, info, log)
            done += 1
        yield "\n".join(log[-40:]), gr.update()
        time.sleep(SLEEP_BETWEEN)
    log.append(f"Done — {done} LoRAs updated. Refresh the Lora tab to see "
               f"previews/triggers.")
    yield "\n".join(log[-40:]), _overview_rows()


def do_check_updates():
    results = _load_json(RESULT_CACHE, {})
    hashes = _load_json(HASH_CACHE, {})
    by_sha = {(hashes.get(str(p)) or {}).get("sha256"): p for p in _scan_loras()}
    log = ["Checking for newer versions on Civitai…"]
    yield "\n".join(log)
    checked_models = {}
    updates = 0
    for sha, p in by_sha.items():
        info = results.get(sha) if sha else None
        if not info:
            continue
        mid = info.get("modelId")
        if not mid:
            continue
        try:
            if mid not in checked_models:
                data = _api_get(API_MODEL.format(mid=mid))
                versions = (data or {}).get("modelVersions") or []
                checked_models[mid] = versions[0] if versions else None
                time.sleep(SLEEP_BETWEEN)
            latest = checked_models[mid]
            if latest and latest.get("id") != info.get("versionId"):
                updates += 1
                log.append(f"⬆ {p.name}: you have '{info.get('versionName')}', "
                           f"newest is '{latest.get('name')}' → {info.get('page')}")
                yield "\n".join(log[-40:])
        except Exception as exc:
            log.append(f"  {p.name}: check failed ({exc})")
            yield "\n".join(log[-40:])
    log.append(f"Done — {updates} update(s) available."
               if updates else "Done — everything is current.")
    yield "\n".join(log[-40:])


# ---------------------------------------------------------------------------
# In-app model downloader
# ---------------------------------------------------------------------------

API_SEARCH = "https://civitai.com/api/v1/models"
TYPE_DIRS = {"LORA": ("models", "Lora"), "Checkpoint": ("models", "Stable-diffusion")}


def _api_key() -> str:
    return str(_load_json(SETTINGS_FILE, {}).get("api_key") or "")


def parse_search_results(data: dict) -> List[dict]:
    """Civitai /models response -> flat result entries (pure, unit tested)."""
    out = []
    for item in (data or {}).get("items", []):
        versions = item.get("modelVersions") or []
        if not versions:
            continue
        v = versions[0]
        files = v.get("files") or []
        f = next((x for x in files if x.get("primary")), files[0] if files else None)
        if not f or not f.get("downloadUrl"):
            continue
        out.append({
            "name": item.get("name", "?"),
            "type": item.get("type", "?"),
            "version": v.get("name", "?"),
            "filename": f.get("name") or "model.safetensors",
            "size_mb": round(float(f.get("sizeKB") or 0) / 1024, 1),
            "downloads": item.get("stats", {}).get("downloadCount", 0),
            "url": f["downloadUrl"],
            "trainedWords": v.get("trainedWords") or [],
            "modelId": item.get("id"),
            "versionId": v.get("id"),
            "baseModel": v.get("baseModel"),
            "images": [im.get("url") for im in (v.get("images") or [])
                       if im.get("url")][:3],
        })
    return out


def results_rows(entries: List[dict]) -> List[List[str]]:
    return [[str(i + 1), e["name"][:40], e["type"], e["version"][:20],
             f"{e['size_mb']} MB", str(e["downloads"])]
            for i, e in enumerate(entries)]


def dl_search(query, mtype):
    if not (query or "").strip():
        return [], [], "Type a search term first."
    try:
        import requests
        params = {"query": query.strip(), "types": mtype, "limit": 10,
                  "nsfw": "true", "sort": "Most Downloaded"}
        headers = {"User-Agent": "sd-forge-civitai-helper/1.0"}
        if _api_key():
            headers["Authorization"] = f"Bearer {_api_key()}"
        r = requests.get(API_SEARCH, params=params, headers=headers, timeout=25)
        r.raise_for_status()
        entries = parse_search_results(r.json())
        return (results_rows(entries), entries,
                f"{len(entries)} result(s) — pick a # and Download.")
    except Exception as exc:
        return [], [], f"Search failed: {exc}"


def _dl_target(entry: dict) -> Path:
    from modules import paths
    base = Path(paths.data_path)
    sub = TYPE_DIRS.get(entry.get("type"), ("models", "Lora"))
    d = base.joinpath(*sub)
    d.mkdir(parents=True, exist_ok=True)
    return d / entry["filename"]


def dl_download(entries, number):
    try:
        idx = int(number) - 1
        entry = (entries or [])[idx]
    except Exception:
        yield "Pick a valid result # first."
        return
    try:
        target = _dl_target(entry)
    except Exception as exc:
        yield f"Cannot resolve target folder: {exc}"
        return
    if target.exists():
        yield f"Already exists: {target.name} — nothing to do."
        return
    import requests
    headers = {"User-Agent": "sd-forge-civitai-helper/1.0"}
    if _api_key():
        headers["Authorization"] = f"Bearer {_api_key()}"
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with requests.get(entry["url"], headers=headers, stream=True,
                          timeout=60, allow_redirects=True) as r:
            if r.status_code in (401, 403):
                yield ("⛔ Civitai requires a login for this file. Put your API "
                       "key in civitai_settings.json as {\"api_key\": \"…\"} "
                       "(create one at civitai.com/user/account) and retry.")
                return
            r.raise_for_status()
            total = int(r.headers.get("content-length") or 0)
            done = 0
            next_mark = 0
            yield f"⬇ {entry['name']} → {target.name} ({entry['size_mb']} MB)…"
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if total and done >= next_mark:
                        pct = done * 100 // total
                        yield f"⬇ {entry['name']}: {pct}% ({done/1e6:.0f} MB)"
                        next_mark = done + max(total // 20, 20_000_000)
        tmp.rename(target)
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        yield f"Download failed: {exc}"
        return
    log: List[str] = [f"✅ Saved: {target}"]
    if entry.get("type") == "LORA":
        info = {"versionId": entry.get("versionId"), "versionName": entry.get("version"),
                "modelId": entry.get("modelId"), "modelName": entry.get("name"),
                "baseModel": entry.get("baseModel"),
                "trainedWords": entry.get("trainedWords") or [],
                "images": entry.get("images") or [],
                "page": f"https://civitai.com/models/{entry.get('modelId')}"}
        _write_sidecars(target, info, log)
        log.append("Card metadata + preview written — refresh the Lora tab.")
    yield "\n".join(log)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _on_ui_tabs():
    try:
        return _build_tab()
    except Exception:
        logger.exception("%s failed to build tab", TAG)
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown("Civitai helper failed to load - see console log.")
        return [(ui, "Civitai", "forge_civitai_helper")]


def _build_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## Civitai LoRA helper — trigger words, previews, update checks\n"
            "Matches your LoRA files to Civitai by hash. Trigger words are "
            "written into each card's metadata, so clicking a LoRA card in the "
            "Lora tab inserts its activation tags automatically."
        )
        with gr.Row():
            scan_btn = gr.Button("🔍 Scan LoRA folders", variant="primary", scale=0)
            fetch_missing_btn = gr.Button("⬇ Fetch missing metadata", scale=0)
            fetch_all_btn = gr.Button("⬇ Re-fetch all", scale=0)
            updates_btn = gr.Button("⬆ Check for updates", scale=0)
        info_md = gr.Markdown("")
        table = gr.Dataframe(
            headers=["file", "size", "preview", "triggers", "Civitai model"],
            interactive=False, wrap=True)
        log_box = gr.Textbox(label="Log", lines=14, interactive=False)

        def do_fetch_missing():
            yield from do_fetch(True)

        def do_fetch_all():
            yield from do_fetch(False)

        with gr.Accordion("⬇ Get new models from Civitai", open=False):
            with gr.Row():
                dl_query = gr.Textbox(label="Search", scale=3,
                                      placeholder="e.g.  fu hua illustrious",
                                      elem_classes=["prompt"])
                dl_type = gr.Radio(["LORA", "Checkpoint"], value="LORA",
                                   label="Type")
                dl_search_btn = gr.Button("🔍 Search", scale=0)
            dl_table = gr.Dataframe(
                headers=["#", "name", "type", "version", "size", "downloads"],
                interactive=False, wrap=True)
            with gr.Row():
                dl_num = gr.Number(label="Result #", value=1, precision=0,
                                   scale=0)
                dl_btn = gr.Button("⬇ Download", variant="primary", scale=0)
                gr.Markdown("*LoRAs land in models/Lora with trigger words + "
                            "preview auto-written. Checkpoints go to "
                            "models/Stable-diffusion (multi-GB — watch the log).*")
            dl_state = gr.State([])

        scan_btn.click(do_scan, [], [table, info_md])
        fetch_missing_btn.click(do_fetch_missing, [], [log_box, table])
        fetch_all_btn.click(do_fetch_all, [], [log_box, table])
        updates_btn.click(do_check_updates, [], [log_box])
        dl_search_btn.click(dl_search, [dl_query, dl_type],
                            [dl_table, dl_state, info_md])
        dl_btn.click(dl_download, [dl_state, dl_num], [log_box])
        ui.load(do_scan, [], [table, info_md])

    return [(ui, "Civitai", "forge_civitai_helper")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
