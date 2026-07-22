"""
sd-forge-job-runner
===================

One tab, three tools, all driven through Forge's own local API
(http://127.0.0.1:<port>/sdapi/v1 — requires the --api launch flag, which
webui-user.bat now sets):

1. QUEUE — paste any infotext (from the Gallery, Model Presets, PNG Info,
   anywhere), stack up jobs, hit Run and walk away. Images are saved by
   Forge into your normal output folders with correct filenames/metadata.
   The queue survives restarts (queue.json in this extension folder).

2. BATCH RE-PROCESS — point at a folder of past outputs (or your Gallery
   favorites) and run each image through:
   - Upscale (Extras pipeline), or
   - img2img refine at low denoise reusing each image's own embedded
     prompt, optionally with an ADetailer face pass.

3. TEST CARDS — renders the same prompt+seed on every selected checkpoint
   and composes a labeled contact sheet, so you can compare model families
   at a glance. (Each checkpoint switch loads ~7 GB — expect ~1 min per
   model on the 3070.)

Author: built by Claude on 2026-07-20.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image, ImageDraw, PngImagePlugin

try:
    from modules import script_callbacks, shared
except ImportError:
    script_callbacks = None  # type: ignore
    shared = None            # type: ignore

logger = logging.getLogger(__name__)
TAG = "[job-runner]"

EXT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_FILE = EXT_ROOT / "queue.json"

DEFAULT_TEST_PROMPT = ("1girl, solo, upper body, looking at viewer, smile, "
                       "school uniform, cherry blossoms, masterpiece, best quality")
DEFAULT_TEST_NEGATIVE = "worst quality, low quality, watermark, signature"

KNOWN_SCHEDULER_SUFFIXES = (" Karras", " Exponential", " SGM Uniform")


# ---------------------------------------------------------------------------
# Local API client
# ---------------------------------------------------------------------------

def _port() -> int:
    try:
        return int(getattr(shared.cmd_opts, "port", None) or 7860)
    except Exception:
        return 7860


def _base() -> str:
    return f"http://127.0.0.1:{_port()}"


def _api_get(path: str, timeout: int = 10):
    import requests
    r = requests.get(_base() + path, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _api_post(path: str, payload: dict, timeout: int = 7200):
    import requests
    r = requests.post(_base() + path, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_ok() -> bool:
    try:
        _api_get("/sdapi/v1/samplers", timeout=5)
        return True
    except Exception:
        return False


API_HINT = ("⚠ Forge API not reachable. Make sure you launched with the "
            "--api flag (webui-user.bat has it now) and generation isn't "
            "blocked by another queue.")


# ---------------------------------------------------------------------------
# Infotext -> API payload
# ---------------------------------------------------------------------------

def _parse_infotext(text: str) -> Dict[str, str]:
    """Prefer Forge's own parser; fall back to a minimal one."""
    try:
        from modules import infotext_utils
        res = infotext_utils.parse_generation_parameters(text)
        return {str(k): v for k, v in res.items()}
    except Exception:
        pass
    # minimal fallback: prompt / negative / key: value pairs on the last line
    res: Dict[str, str] = {}
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return res
    param_line = ""
    if re.search(r"\bSteps:\s*\d+", lines[-1]):
        param_line = lines.pop()
    neg_idx = next((i for i, ln in enumerate(lines)
                    if ln.startswith("Negative prompt:")), None)
    if neg_idx is not None:
        res["Prompt"] = "\n".join(lines[:neg_idx]).strip()
        res["Negative prompt"] = "\n".join(
            [lines[neg_idx][len("Negative prompt:"):].strip()] + lines[neg_idx + 1:])
    else:
        res["Prompt"] = "\n".join(lines).strip()
    for m in re.finditer(r'\s*([\w ]+):\s*("(?:\\.|[^"])*"|[^,]*)(?:,|$)', param_line):
        res[m.group(1).strip()] = m.group(2).strip().strip('"')
    m = re.match(r"(\d+)x(\d+)", res.get("Size", ""))
    if m:
        res["Size-1"], res["Size-2"] = m.group(1), m.group(2)
    return res


def _split_sampler(name: str) -> Tuple[str, Optional[str]]:
    for suffix in KNOWN_SCHEDULER_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix.strip().lower().replace(" ", "_")
    return name, None


def payload_from_infotext(text: str, n_iter: int = 1) -> dict:
    d = _parse_infotext(text)
    payload: dict = {
        "prompt": d.get("Prompt", "") or "",
        "negative_prompt": d.get("Negative prompt", "") or "",
        "steps": int(float(d.get("Steps", 28) or 28)),
        "cfg_scale": float(d.get("CFG scale", 5.5) or 5.5),
        "seed": int(float(d.get("Seed", -1) or -1)),
        "width": int(float(d.get("Size-1", 832) or 832)),
        "height": int(float(d.get("Size-2", 1216) or 1216)),
        "n_iter": max(1, int(n_iter or 1)),
        "batch_size": 1,
        "save_images": True,
        "send_images": False,
    }
    sampler = str(d.get("Sampler", "Euler a") or "Euler a")
    sampler, scheduler = _split_sampler(sampler)
    payload["sampler_name"] = sampler
    sched = str(d.get("Schedule type", "") or "")
    if sched and sched.lower() != "automatic":
        payload["scheduler"] = sched.lower().replace(" ", "_")
    elif scheduler:
        payload["scheduler"] = scheduler
    override: dict = {}
    if d.get("Model"):
        override["sd_model_checkpoint"] = str(d["Model"])
    if d.get("Clip skip"):
        try:
            override["CLIP_stop_at_last_layers"] = int(float(d["Clip skip"]))
        except Exception:
            pass
    if override:
        payload["override_settings"] = override
        payload["override_settings_restore_afterwards"] = True
    return payload


# ---------------------------------------------------------------------------
# Queue persistence
# ---------------------------------------------------------------------------

def _load_queue() -> List[dict]:
    try:
        with open(QUEUE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def _save_queue(q: List[dict]) -> None:
    try:
        with open(QUEUE_FILE, "w", encoding="utf-8") as fh:
            json.dump(q, fh, ensure_ascii=False, indent=1)
    except Exception as exc:
        logger.warning("%s could not save queue: %s", TAG, exc)


def _queue_rows(q: List[dict]) -> List[List[str]]:
    rows = []
    for i, job in enumerate(q):
        d = _parse_infotext(job.get("infotext", ""))
        rows.append([
            str(i + 1),
            job.get("status", "pending"),
            str(d.get("Model", "(current)"))[:40],
            f"{d.get('Steps', '?')}st / {d.get('Sampler', '?')}",
            f"{d.get('Size-1', '?')}x{d.get('Size-2', '?')} ×{job.get('n_iter', 1)}",
            (d.get("Prompt", "") or "")[:80],
        ])
    return rows


# ---------------------------------------------------------------------------
# Shared image helpers
# ---------------------------------------------------------------------------

def _b64_of(img_path: Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode()


def _img_from_b64(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s)))


def _read_infotext(path: Path) -> str:
    try:
        with Image.open(path) as im:
            txt = getattr(im, "text", None)
            if isinstance(txt, dict) and txt.get("parameters"):
                return str(txt["parameters"])
    except Exception:
        pass
    return ""


def _out_dir(kind: str) -> Path:
    base = None
    try:
        from modules import paths
        base = Path(paths.data_path)
    except Exception:
        base = EXT_ROOT.parents[1]
    d = base / "output" / kind / datetime.now().strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Tab 1: Queue
# ---------------------------------------------------------------------------

def q_add(infotext: str, n_iter):
    q = _load_queue()
    if not (infotext or "").strip():
        return _queue_rows(q), "Paste an infotext first."
    q.append({"infotext": infotext.strip(), "n_iter": int(n_iter or 1),
              "status": "pending"})
    _save_queue(q)
    return _queue_rows(q), f"Job {len(q)} added."


def q_clear_done():
    q = [j for j in _load_queue() if j.get("status") == "pending"]
    _save_queue(q)
    return _queue_rows(q), "Cleared finished/errored jobs."


def q_clear_all():
    _save_queue([])
    return [], "Queue emptied."


def q_run():
    q = _load_queue()
    pending = [j for j in q if j.get("status") == "pending"]
    if not pending:
        yield _queue_rows(q), "Queue is empty — add jobs first."
        return
    if not api_ok():
        yield _queue_rows(q), API_HINT
        return
    log = [f"Running {len(pending)} job(s)…"]
    for job in q:
        if job.get("status") != "pending":
            continue
        idx = q.index(job) + 1
        job["status"] = "running"
        _save_queue(q)
        yield _queue_rows(q), "\n".join(log + [f"→ job {idx} running…"])
        t0 = time.monotonic()
        try:
            payload = payload_from_infotext(job["infotext"], job.get("n_iter", 1))
            _api_post("/sdapi/v1/txt2img", payload)
            dur = int(time.monotonic() - t0)
            job["status"] = "done"
            log.append(f"✓ job {idx} done in {dur}s")
        except Exception as exc:
            job["status"] = "error"
            log.append(f"✗ job {idx} failed: {exc}")
        _save_queue(q)
        yield _queue_rows(q), "\n".join(log)
    log.append("Queue finished. Images are in your normal output folders "
               "(see the Gallery tab).")
    yield _queue_rows(q), "\n".join(log)


def q_interrupt():
    try:
        _api_post("/sdapi/v1/interrupt", {}, timeout=10)
        return "Interrupt sent."
    except Exception as exc:
        return f"Interrupt failed: {exc}"


# ---------------------------------------------------------------------------
# Tab 2: Batch re-process
# ---------------------------------------------------------------------------

def _batch_sources(source: str, folder: str) -> List[Path]:
    if source == "Gallery favorites":
        fav_file = EXT_ROOT.parent / "sd-forge-gallery" / "favorites.json"
        try:
            with open(fav_file, "r", encoding="utf-8") as fh:
                return [Path(p) for p in json.load(fh) if os.path.isfile(p)]
        except Exception:
            return []
    p = Path((folder or "").strip().strip('"'))
    if not p.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])


ADETAILER_ARGS = {
    "ADetailer": {
        "args": [
            True,   # enable
            False,  # skip img2img pre-pass
            {"ad_model": "face_yolov8n.pt", "ad_denoising_strength": 0.4,
             "ad_inpaint_only_masked": True},
        ]
    }
}


def b_run(source: str, folder: str, mode: str, upscaler: str, denoise: float,
          limit):
    files = _batch_sources(source, folder)
    if limit:
        files = files[: int(limit)]
    if not files:
        yield "No images found for that source."
        return
    if not api_ok():
        yield API_HINT
        return
    out = _out_dir("batch-reprocessed")
    log = [f"{len(files)} image(s) → {mode} → {out}"]
    yield "\n".join(log)
    ok = err = 0
    for f in files:
        try:
            if mode.startswith("Upscale"):
                res = _api_post("/sdapi/v1/extra-single-image", {
                    "image": _b64_of(f),
                    "upscaler_1": upscaler or "R-ESRGAN 4x+ Anime6B",
                    "upscaling_resize": 2,
                }, timeout=1800)
                img = _img_from_b64(res["image"])
                info = _read_infotext(f)
                pnginfo = PngImagePlugin.PngInfo()
                if info:
                    pnginfo.add_text("parameters", info)
                dest = out / f"{f.stem}_up2x.png"
                img.save(dest, pnginfo=pnginfo)
            else:
                info = _read_infotext(f)
                base_payload = payload_from_infotext(info) if info else {}
                with Image.open(f) as im:
                    w, h = im.size
                payload = {
                    "init_images": [_b64_of(f)],
                    "prompt": base_payload.get("prompt", "") or
                              "masterpiece, best quality",
                    "negative_prompt": base_payload.get("negative_prompt", ""),
                    "denoising_strength": float(denoise or 0.3),
                    "steps": 24,
                    "sampler_name": base_payload.get("sampler_name", "Euler a"),
                    "cfg_scale": base_payload.get("cfg_scale", 5.5),
                    "width": w, "height": h,
                    "seed": -1,
                    "save_images": True,
                    "send_images": False,
                }
                if "ADetailer" in mode:
                    payload["alwayson_scripts"] = ADETAILER_ARGS
                _api_post("/sdapi/v1/img2img", payload)
            ok += 1
            log.append(f"✓ {f.name}")
        except Exception as exc:
            err += 1
            log.append(f"✗ {f.name}: {exc}")
        yield "\n".join(log[-30:])
    where = out if mode.startswith("Upscale") else "the img2img output folder"
    log.append(f"Done: {ok} ok, {err} failed. Results in {where}.")
    yield "\n".join(log[-30:])


def b_upscalers() -> List[str]:
    try:
        return [u["name"] for u in _api_get("/sdapi/v1/upscalers")
                if u.get("name") and u["name"] != "None"]
    except Exception:
        return ["R-ESRGAN 4x+ Anime6B", "R-ESRGAN 4x+", "Lanczos"]


# ---------------------------------------------------------------------------
# Tab 3: Test cards
# ---------------------------------------------------------------------------

def t_checkpoints() -> List[str]:
    try:
        return [m["title"] for m in _api_get("/sdapi/v1/sd-models")]
    except Exception:
        return []


def _contact_sheet(cells: List[Tuple[str, Image.Image]], cell_px=448) -> Image.Image:
    cols = min(3, max(1, len(cells)))
    rows = (len(cells) + cols - 1) // cols
    label_h = 28
    sheet = Image.new("RGB", (cols * cell_px, rows * (cell_px + label_h)),
                      (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(cells):
        im = im.copy()
        im.thumbnail((cell_px, cell_px))
        x = (i % cols) * cell_px + (cell_px - im.width) // 2
        y = (i // cols) * (cell_px + label_h)
        sheet.paste(im, (x, y))
        draw.text(((i % cols) * cell_px + 8, y + cell_px + 6),
                  name[:52], fill=(230, 230, 230))
    return sheet


def t_run(checkpoints: List[str], prompt: str, negative: str, seed, steps):
    if not checkpoints:
        yield None, "Pick at least one checkpoint."
        return
    if not api_ok():
        yield None, API_HINT
        return
    log = [f"Rendering {len(checkpoints)} checkpoint(s) — roughly a minute "
           f"per model (each switch loads ~7 GB)…"]
    yield None, "\n".join(log)
    cells: List[Tuple[str, Image.Image]] = []
    for ck in checkpoints:
        t0 = time.monotonic()
        try:
            res = _api_post("/sdapi/v1/txt2img", {
                "prompt": prompt or DEFAULT_TEST_PROMPT,
                "negative_prompt": negative or DEFAULT_TEST_NEGATIVE,
                "steps": int(steps or 20),
                "sampler_name": "Euler a",
                "cfg_scale": 5.5,
                "seed": int(seed or 123456789),
                "width": 832, "height": 1216,
                "save_images": False, "send_images": True,
                "override_settings": {"sd_model_checkpoint": ck},
                "override_settings_restore_afterwards": False,
            })
            img = _img_from_b64(res["images"][0])
            short = os.path.basename(ck).split(" [")[0]
            cells.append((short, img))
            log.append(f"✓ {short} ({int(time.monotonic() - t0)}s)")
        except Exception as exc:
            log.append(f"✗ {ck}: {exc}")
        sheet = _contact_sheet(cells) if cells else None
        yield sheet, "\n".join(log)
    if cells:
        out = _out_dir("test-cards") / f"testcards_{datetime.now().strftime('%H%M%S')}.png"
        sheet = _contact_sheet(cells)
        sheet.save(out)
        log.append(f"Saved: {out}")
        yield sheet, "\n".join(log)


# ---------------------------------------------------------------------------
# Tab 4: Sweet Spot finder — same-seed ladders into labeled contact sheets
# ---------------------------------------------------------------------------

LADDERS = {
    "LoRA weight": [0.4, 0.6, 0.8, 1.0, 1.2],
    "CFG": [3.0, 4.5, 5.5, 6.5, 8.0],
    "Steps": [16, 22, 28, 34, 40],
}


def _ss_lora_list() -> List[str]:
    names = set()
    try:
        from modules import paths
        base = Path(paths.data_path)
    except Exception:
        base = EXT_ROOT.parents[1]
    roots = [base / "models" / "Lora", base / "models" / "LyCORIS"]
    try:
        v = getattr(shared.cmd_opts, "lora_dirs", None)
        if isinstance(v, str) and v:
            roots.append(Path(v))
        elif isinstance(v, (list, tuple)):
            roots.extend(Path(x) for x in v if x)
    except Exception:
        pass
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for f in root.rglob("*"):
                if f.suffix.lower() in (".safetensors", ".pt", ".ckpt") and f.is_file():
                    names.add(f.stem)
        except Exception:
            pass
    return sorted(names, key=str.lower)


def build_ladder(mode: str, prompt: str, lora: str, seed: int, steps: int,
                 cfg: float, width: int, height: int,
                 sampler: str, negative: str) -> List[Tuple[str, dict]]:
    """-> [(label, payload)] — one same-seed generation per ladder rung."""
    out: List[Tuple[str, dict]] = []
    base = {
        "negative_prompt": negative, "seed": int(seed),
        "width": int(width), "height": int(height),
        "sampler_name": sampler or "Euler a",
        "steps": int(steps), "cfg_scale": float(cfg),
        "n_iter": 1, "batch_size": 1,
        "send_images": True, "save_images": False,
    }
    for v in LADDERS.get(mode, []):
        p = dict(base)
        if mode == "LoRA weight":
            p["prompt"] = f"{prompt}, <lora:{lora}:{v:g}>"
            label = f"{lora[:24]} @ {v:g}"
        elif mode == "CFG":
            p["prompt"] = prompt
            p["cfg_scale"] = float(v)
            label = f"CFG {v:g}"
        else:  # Steps
            p["prompt"] = prompt
            p["steps"] = int(v)
            label = f"{int(v)} steps"
        out.append((label, p))
    return out


def ss_run(mode, prompt, lora, seed, steps, cfg, width, height, sampler,
           negative):
    prompt = (prompt or "").strip().rstrip(",")
    if not prompt:
        yield None, "Write a base prompt first."
        return
    if mode == "LoRA weight" and not lora:
        yield None, "Pick a LoRA for the weight ladder."
        return
    if not api_ok():
        yield None, API_HINT
        return
    import random
    seed = int(seed) if int(seed or -1) > 0 else random.randint(1, 2**31 - 1)
    ladder = build_ladder(mode, prompt, lora or "", seed, steps, cfg,
                          width, height, sampler, negative or "")
    log = [f"🎯 {mode} ladder — fixed seed {seed}, {len(ladder)} rungs…"]
    yield None, "\n".join(log)
    cells: List[Tuple[str, Image.Image]] = []
    for label, payload in ladder:
        t0 = time.monotonic()
        try:
            res = _api_post("/sdapi/v1/txt2img", payload)
            img = _img_from_b64(res["images"][0])
            cells.append((label, img))
            log.append(f"✓ {label} ({int(time.monotonic()-t0)}s)")
        except Exception as exc:
            log.append(f"✗ {label}: {exc}")
        sheet = _contact_sheet(cells) if cells else None
        yield sheet, "\n".join(log)
    if cells:
        out = _out_dir("sweet-spot") / (
            f"{mode.replace(' ', '_').lower()}_{datetime.now().strftime('%H%M%S')}.png")
        _contact_sheet(cells).save(out)
        log.append(f"Saved: {out}")
        log.append("Same seed everywhere — differences you see come only "
                   "from the ladder value.")
        yield _contact_sheet(cells), "\n".join(log)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _on_ui_tabs():
    try:
        return _build_tab()
    except Exception:
        logger.exception("%s failed to build tab", TAG)
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown("Job Runner failed to load - see console log.")
        return [(ui, "Job Runner", "forge_job_runner")]


def _build_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown("## Job Runner — queue, batch re-process, test cards  "
                    "*(needs the --api launch flag; webui-user.bat sets it)*")

        with gr.Tab("Queue"):
            infotext_box = gr.Textbox(
                label="Job infotext (paste from Gallery / Model Presets / PNG Info)",
                lines=6,
                placeholder="prompt…\nNegative prompt: …\nSteps: 28, Sampler: Euler a, "
                            "CFG scale: 5.5, Seed: -1, Size: 832x1216, Model: …")
            with gr.Row():
                n_iter_num = gr.Number(label="Batch count (images per job)",
                                       value=1, precision=0, scale=0)
                add_btn = gr.Button("➕ Add to queue", variant="primary", scale=0)
                run_btn = gr.Button("▶ Run queue", scale=0)
                stop_btn = gr.Button("⏹ Interrupt current", scale=0)
                cleardone_btn = gr.Button("Clear finished", scale=0)
                clearall_btn = gr.Button("Empty queue", variant="stop", scale=0)
            q_table = gr.Dataframe(
                headers=["#", "status", "model", "steps/sampler", "size×n", "prompt"],
                interactive=False, wrap=True, value=_queue_rows(_load_queue()))
            q_log = gr.Textbox(label="Run log", lines=8, interactive=False)

            add_btn.click(q_add, [infotext_box, n_iter_num], [q_table, q_log])
            run_btn.click(q_run, [], [q_table, q_log])
            stop_btn.click(q_interrupt, [], [q_log])
            cleardone_btn.click(q_clear_done, [], [q_table, q_log])
            clearall_btn.click(q_clear_all, [], [q_table, q_log])

        with gr.Tab("Batch re-process"):
            with gr.Row():
                source_radio = gr.Radio(["Folder path", "Gallery favorites"],
                                        value="Folder path", label="Source")
                folder_box = gr.Textbox(label="Folder (when source = Folder path)",
                                        placeholder=r"F:\...\output\txt2img-images\2026-07-19")
            with gr.Row():
                mode_radio = gr.Radio(
                    ["Upscale 2x (Extras)",
                     "img2img refine (reuse image's own prompt)",
                     "img2img refine + ADetailer face pass"],
                    value="Upscale 2x (Extras)", label="Mode")
            with gr.Row():
                upscaler_dd = gr.Dropdown(label="Upscaler", choices=b_upscalers(),
                                          value="R-ESRGAN 4x+ Anime6B",
                                          allow_custom_value=True)
                denoise_sl = gr.Slider(0.1, 0.6, value=0.3, step=0.05,
                                       label="Refine denoise")
                limit_num = gr.Number(label="Max images (0 = all)", value=0,
                                      precision=0)
            with gr.Row():
                b_run_btn = gr.Button("▶ Run batch", variant="primary", scale=0)
                b_upsc_btn = gr.Button("🔄 Reload upscaler list", scale=0)
            b_log = gr.Textbox(label="Batch log", lines=12, interactive=False)

            b_run_btn.click(
                b_run, [source_radio, folder_box, mode_radio, upscaler_dd,
                        denoise_sl, limit_num], [b_log])
            b_upsc_btn.click(lambda: gr.update(choices=b_upscalers()), [],
                             [upscaler_dd])

        with gr.Tab("Test cards"):
            with gr.Row():
                ckpt_group = gr.CheckboxGroup(label="Checkpoints", choices=[])
                load_ck_btn = gr.Button("🔄 Load checkpoint list", scale=0)
            t_prompt = gr.Textbox(label="Test prompt", value=DEFAULT_TEST_PROMPT,
                                  lines=2, elem_classes=["prompt"])
            t_negative = gr.Textbox(label="Negative", value=DEFAULT_TEST_NEGATIVE)
            with gr.Row():
                t_seed = gr.Number(label="Seed", value=123456789, precision=0)
                t_steps = gr.Number(label="Steps", value=20, precision=0)
                t_run_btn = gr.Button("▶ Render test cards", variant="primary",
                                      scale=0)
            t_sheet = gr.Image(label="Contact sheet", type="pil",
                               interactive=False)
            t_log = gr.Textbox(label="Log", lines=8, interactive=False)

            load_ck_btn.click(lambda: gr.update(choices=t_checkpoints()), [],
                              [ckpt_group])
            t_run_btn.click(t_run, [ckpt_group, t_prompt, t_negative, t_seed,
                                    t_steps], [t_sheet, t_log])

        with gr.Tab("Sweet Spot"):
            gr.Markdown("**Find the best value with your own eyes** — renders "
                        "a same-seed ladder so the only variable is the one "
                        "you're testing.")
            with gr.Row():
                ss_mode = gr.Radio(list(LADDERS.keys()), value="LoRA weight",
                                   label="Ladder")
                ss_lora = gr.Dropdown(label="LoRA (for weight ladder)",
                                      choices=_ss_lora_list(),
                                      allow_custom_value=True)
                ss_lora_refresh = gr.Button("🔄", scale=0)
            ss_prompt = gr.Textbox(label="Base prompt (without the LoRA tag)",
                                   lines=2, elem_classes=["prompt"],
                                   placeholder="1girl, fu hua, upper body, "
                                               "classroom, sunset")
            ss_negative = gr.Textbox(label="Negative",
                                     value=DEFAULT_TEST_NEGATIVE)
            with gr.Row():
                ss_seed = gr.Number(label="Seed (-1 = pick once, reuse)",
                                    value=-1, precision=0)
                ss_steps = gr.Number(label="Steps", value=28, precision=0)
                ss_cfg = gr.Number(label="CFG", value=5.5)
                ss_w = gr.Number(label="Width", value=832, precision=0)
                ss_h = gr.Number(label="Height", value=1216, precision=0)
                ss_sampler = gr.Textbox(label="Sampler", value="Euler a")
            ss_btn = gr.Button("🎯 Render ladder", variant="primary")
            ss_sheet = gr.Image(label="Ladder (labeled)", type="pil",
                                interactive=False)
            ss_log = gr.Textbox(label="Log", lines=8, interactive=False)

            ss_btn.click(ss_run,
                         [ss_mode, ss_prompt, ss_lora, ss_seed, ss_steps,
                          ss_cfg, ss_w, ss_h, ss_sampler, ss_negative],
                         [ss_sheet, ss_log])
            ss_lora_refresh.click(lambda: gr.update(choices=_ss_lora_list()),
                                  [], [ss_lora])

    return [(ui, "Job Runner", "forge_job_runner")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
