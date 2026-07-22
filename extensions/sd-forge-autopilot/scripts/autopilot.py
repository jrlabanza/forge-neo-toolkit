"""
sd-forge-autopilot
==================

"Auto Pilot" tab: type what you want, set how many keepers you need, press
one button. The extension then:

1. generates a batch through Forge's own local API (same pipeline as the
   Generate button — ADetailer, LoRAs, everything applies),
2. JUDGES every image locally:
     - black/empty frame detection      (NaN-collapse catcher)
     - sharpness check                  (Laplacian variance, catches blur)
     - face check (optional)            (YOLO face_yolov8n on CPU: a clear,
                                         sharp face must be present)
     - anime aesthetic score            (cafeai/cafe_aesthetic ViT, CPU)
3. keeps only images that pass every gate AND beat your quality threshold,
4. regenerates with fresh seeds until it has the number you asked for
   (or hits the safety cap),
5. saves keepers to output/autopilot/<date>/ with full metadata + score,
   and reports the best seeds so you can reuse them.

All judging runs on CPU — zero VRAM taken from generation. The aesthetic
model (~350 MB) downloads once on first run and is cached.

Honest note: gates catch objective failures (blank, blurry, faceless,
low-aesthetic). Taste remains yours — raise the threshold for stricter cuts.

Requires the --api launch flag (webui-user.bat sets it).

Author: built by Claude on 2026-07-21.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, PngImagePlugin

try:
    from modules import script_callbacks, shared
except ImportError:
    script_callbacks = None  # type: ignore
    shared = None            # type: ignore

logger = logging.getLogger(__name__)
TAG = "[autopilot]"

EXT_ROOT = Path(__file__).resolve().parents[1]

AESTHETIC_MODEL = "cafeai/cafe_aesthetic"

DEFAULT_NEGATIVE = ("bad quality, worst quality, worst detail, sketch, censor, "
                    "signature, watermark, username, jpeg artifacts, "
                    "bad anatomy, bad hands, extra digits, missing fingers")
QUALITY_TAGS = "masterpiece, best quality, amazing quality, very aesthetic, absurdres"

_STATE: Dict = {"aesthetic": None, "yolo": None}


# ---------------------------------------------------------------------------
# Local API client (same pattern as sd-forge-job-runner)
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
        r = requests.get(f"http://127.0.0.1:{_port()}/sdapi/v1/samplers",
                         timeout=5)
        return r.ok
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------

def _judge_black(im: Image.Image) -> Optional[str]:
    a = np.asarray(im.convert("L"), dtype=np.float32)
    if a.mean() < 8 or a.std() < 4:
        return "blank/black frame"
    return None


def _judge_blur(im: Image.Image, min_var: float) -> Tuple[float, Optional[str]]:
    try:
        import cv2
        g = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
        v = float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        # cv2 unavailable — approximate with numpy gradient variance
        g = np.asarray(im.convert("L"), dtype=np.float32)
        v = float(np.gradient(g)[0].var() + np.gradient(g)[1].var())
    if v < min_var:
        return v, f"too soft/blurry (sharpness {v:.0f} < {min_var:.0f})"
    return v, None


ADETAILER_REPO = "Bingsu/adetailer"  # same source ADetailer itself uses


def _yolo(name: str):
    """Load a YOLO detector: local models/adetailer first, else HF hub cache."""
    cache = _STATE.setdefault("yolo_models", {})
    if name in cache:
        return cache[name]
    try:
        from ultralytics import YOLO
        path = None
        try:
            from modules import paths
            local = Path(paths.data_path) / "models" / "adetailer" / name
            if local.is_file():
                path = str(local)
        except Exception:
            pass
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(ADETAILER_REPO, name)
        cache[name] = YOLO(path)
    except Exception as exc:
        logger.warning("%s detector %s unavailable: %s", TAG, name, exc)
        cache[name] = False
    return cache[name]


def _boxes(im: Image.Image, model_name: str, conf: float = 0.4):
    """-> list of [x1,y1,x2,y2] boxes, [] if none, None if model unavailable."""
    model = _yolo(model_name)
    if model is False:
        return None
    try:
        res = model.predict(np.asarray(im.convert("RGB")), device="cpu",
                            verbose=False, conf=conf)
        b = res[0].boxes
        if b is None or len(b) == 0:
            return []
        return [list(map(float, row)) for row in b.xyxy.cpu().numpy()]
    except Exception as exc:
        logger.warning("%s %s detect failed: %s", TAG, model_name, exc)
        return None


def _judge_face(im: Image.Image, min_var: float) -> Optional[str]:
    boxes = _boxes(im, "face_yolov8n.pt", 0.4)
    if boxes is None:
        return None  # face checking unavailable — skip silently
    if not boxes:
        return "no clear face detected"
    # sharpest-face rule: the largest face must itself be sharp
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    x1, y1, x2, y2 = [int(v) for v in boxes[areas.index(max(areas))]]
    face = im.crop((max(0, x1), max(0, y1), x2, y2))
    if face.width >= 48 and face.height >= 48:
        _, blur_reason = _judge_blur(face, min_var * 0.6)
        if blur_reason:
            return "face is blurry/mushy"
    return None


# ---- anatomy gates --------------------------------------------------------

_SOLO_RE = re.compile(r"\b(solo|1girl|1boy)\b")
_MULTI_RE = re.compile(r"\b(2girls|2boys|3girls|3boys|multiple_girls|"
                       r"multiple_boys|multiple girls|multiple boys|couple)\b")


def _expects_solo(prompt: str) -> bool:
    p = (prompt or "").lower()
    return bool(_SOLO_RE.search(p)) and not _MULTI_RE.search(p)


def _proportion_reason(face_h: float, person_h: float,
                       img_h: float) -> Optional[str]:
    """Head/figure ratio sanity — only judged on mostly-full figures."""
    if face_h <= 0 or person_h <= 0 or img_h <= 0:
        return None
    if person_h / img_h < 0.55:
        return None  # small/cropped figure — ratio unreliable, skip
    r = face_h / person_h
    if r > 0.45:
        return f"proportions off — head is {r:.0%} of the figure"
    if r < 0.06:
        return f"proportions off — head is only {r:.0%} of the figure"
    return None


def _judge_anatomy(im: Image.Image, prompt: str) -> Optional[str]:
    hands = _boxes(im, "hand_yolov8n.pt", 0.5)
    if hands is not None and len(hands) > 2:
        return f"extra hands ({len(hands)} detected)"
    persons = _boxes(im, "person_yolov8n-seg.pt", 0.5)
    if persons is not None and persons:
        if len(persons) >= 2 and _expects_solo(prompt):
            return f"{len(persons)} people in a solo prompt"
        faces = _boxes(im, "face_yolov8n.pt", 0.4) or []
        if faces:
            p = max(persons, key=lambda b: (b[3] - b[1]) * (b[2] - b[0]))
            f = max(faces, key=lambda b: (b[3] - b[1]) * (b[2] - b[0]))
            reason = _proportion_reason(f[3] - f[1], p[3] - p[1], im.height)
            if reason:
                return reason
    return None


# ---- deep tag scan (reuses the Prompt Enhancer's WD14 tagger) -------------

BAD_ANATOMY_TAGS = {
    "bad hands", "bad_hands", "bad anatomy", "bad_anatomy", "extra digits",
    "extra_digits", "extra arms", "extra_arms", "extra legs", "extra_legs",
    "extra hands", "extra_hands", "missing finger", "missing_finger",
    "deformed", "mutated hands", "mutated_hands", "malformed limbs",
    "malformed_limbs", "bad feet", "bad_feet", "bad proportions",
    "bad_proportions", "6+girls", "6+boys",
}


def _wd14_module():
    if "wd14" in _STATE:
        return _STATE["wd14"]
    import sys
    mod = None
    for m in list(sys.modules.values()):
        f = (getattr(m, "__file__", "") or "").replace("\\", "/")
        if f.endswith("sd-forge-prompt-enhancer/scripts/prompt_enhancer.py"):
            mod = m
            break
    _STATE["wd14"] = mod if (mod and hasattr(mod, "analyze_image_wd14")) else False
    return _STATE["wd14"]


def _harvest_strings(obj, out: set, depth: int = 0):
    if depth > 4:
        return
    if isinstance(obj, str):
        out.add(obj.strip().lower())
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _harvest_strings(k, out, depth + 1)
            _harvest_strings(v, out, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _harvest_strings(v, out, depth + 1)


def _judge_deep_tags(im: Image.Image, prompt: str) -> Optional[str]:
    mod = _wd14_module()
    if mod is False:
        return None  # tagger not loaded — skip silently
    try:
        result = mod.analyze_image_wd14(im.convert("RGB"))
    except Exception as exc:
        logger.warning("%s deep scan failed: %s", TAG, exc)
        return None
    tags: set = set()
    _harvest_strings(result, tags)
    bad = tags & BAD_ANATOMY_TAGS
    if bad:
        return f"tagger flagged: {', '.join(sorted(bad)[:3])}"
    if _expects_solo(prompt) and tags & {"2girls", "2boys", "multiple girls",
                                          "multiple_girls", "multiple boys",
                                          "multiple_boys", "3girls", "3boys"}:
        return "tagger sees multiple people in a solo prompt"
    return None


def _aesthetic_pipe():
    if _STATE["aesthetic"] is not None:
        return _STATE["aesthetic"]
    try:
        from transformers import pipeline
        _STATE["aesthetic"] = pipeline("image-classification",
                                       model=AESTHETIC_MODEL, device=-1)
    except Exception as exc:
        logger.warning("%s aesthetic model unavailable: %s", TAG, exc)
        _STATE["aesthetic"] = False
    return _STATE["aesthetic"]


def _judge_aesthetic(im: Image.Image) -> Optional[float]:
    pipe = _aesthetic_pipe()
    if pipe is False:
        return None
    try:
        out = pipe(im.convert("RGB"))
        for entry in out:
            if entry.get("label") == "aesthetic":
                return float(entry.get("score", 0.0))
        return None
    except Exception as exc:
        logger.warning("%s aesthetic judge failed: %s", TAG, exc)
        return None


def judge_image(im: Image.Image, threshold: float, min_sharp: float,
                want_face: bool, want_anatomy: bool = False,
                deep_scan: bool = False, prompt: str = "") -> Tuple[bool, float, str]:
    """-> (passed, score, verdict)"""
    reason = _judge_black(im)
    if reason:
        return False, 0.0, reason
    sharp, reason = _judge_blur(im, min_sharp)
    if reason:
        return False, 0.0, reason
    if want_face:
        reason = _judge_face(im, min_sharp)
        if reason:
            return False, 0.0, reason
    if want_anatomy:
        reason = _judge_anatomy(im, prompt)
        if reason:
            return False, 0.0, reason
    if deep_scan:
        reason = _judge_deep_tags(im, prompt)
        if reason:
            return False, 0.0, reason
    score = _judge_aesthetic(im)
    if score is None:
        # aesthetic model unavailable: heuristics-only mode, pass with note
        return True, -1.0, f"pass (heuristics only, sharpness {sharp:.0f})"
    if score < threshold:
        return False, score, f"aesthetic {score:.2f} below threshold {threshold:.2f}"
    return True, score, f"PASS  aesthetic {score:.2f}, sharpness {sharp:.0f}"


def unload_judges() -> str:
    _STATE["aesthetic"] = None
    _STATE["yolo"] = None
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    return "Judges unloaded, memory freed."


# ---------------------------------------------------------------------------
# LoRA selector support
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
    try:
        from modules import paths
        base = Path(paths.data_path)
    except Exception:
        base = EXT_ROOT.parents[1]
    dirs += [base / "models" / "Lora", base / "models" / "LyCORIS"]
    seen, out = set(), []
    for d in dirs:
        k = str(d).lower()
        if k not in seen and d.is_dir():
            seen.add(k)
            out.append(d)
    return out


def list_loras() -> List[str]:
    """Available LoRA names (file stems), as used in <lora:name:w> syntax."""
    names = set()
    for root in _lora_dirs():
        try:
            for p in root.rglob("*"):
                if p.suffix.lower() in (".safetensors", ".pt", ".ckpt") and p.is_file():
                    names.add(p.stem)
        except Exception:
            pass
    return sorted(names, key=str.lower)


def _lora_activation(stem: str) -> str:
    """Trigger words from the card json (written by the Civitai helper)."""
    for root in _lora_dirs():
        try:
            for p in root.rglob(stem + ".json"):
                try:
                    card = json.loads(p.read_text(encoding="utf-8"))
                    return str(card.get("activation text") or "").strip()
                except Exception:
                    continue
        except Exception:
            pass
    return ""


def _lora_suffix(names: List[str], weight: float, add_triggers: bool) -> str:
    parts: List[str] = []
    for name in names or []:
        if not name:
            continue
        if add_triggers:
            trig = _lora_activation(name)
            if trig:
                parts.append(trig)
        parts.append(f"<lora:{name}:{weight:g}>")
    return (", " + ", ".join(parts)) if parts else ""


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _out_dirs() -> Tuple[Path, Path]:
    try:
        from modules import paths
        base = Path(paths.data_path)
    except Exception:
        base = EXT_ROOT.parents[1]
    day = datetime.now().strftime("%Y-%m-%d")
    keep = base / "output" / "autopilot" / day
    rej = keep / "rejected"
    keep.mkdir(parents=True, exist_ok=True)
    return keep, rej


def _save(im: Image.Image, infotext: str, dest: Path, stem: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    if infotext:
        info.add_text("parameters", infotext)
    path = dest / f"{stem}.png"
    im.save(path, pnginfo=info)
    return path


def run_autopilot(prompt, negative, add_quality, keepers_target, batch_size,
                  max_batches, threshold, min_sharp, want_face, want_anatomy,
                  deep_scan, keep_rejects,
                  lora_names, lora_weight, lora_triggers,
                  steps, cfg, width, height, sampler):
    prompt = (prompt or "").strip().rstrip(",")
    if not prompt:
        yield [], "Write a prompt first (or build one in the Describe tab and paste it)."
        return
    if not api_ok():
        yield [], ("⚠ Forge API not reachable — launch with webui-user.bat "
                   "(it sets --api).")
        return
    lora_part = _lora_suffix(list(lora_names or []), float(lora_weight or 0.8),
                             bool(lora_triggers))
    if lora_part:
        prompt = f"{prompt}{lora_part}"
    if add_quality and "masterpiece" not in prompt:
        prompt = f"{prompt}, {QUALITY_TAGS}"

    keep_dir, rej_dir = _out_dirs()
    keepers: List[Tuple[Image.Image, str]] = []
    best: List[Tuple[float, int]] = []  # (score, seed)
    log: List[str] = [f"Target: {int(keepers_target)} keepers · "
                      f"threshold {threshold:.2f} · batches of {int(batch_size)} "
                      f"(max {int(max_batches)})"]
    if _STATE["aesthetic"] is None:
        log.append("Loading the quality judge (first ever run downloads ~350 MB)…")
    yield [], "\n".join(log)
    _aesthetic_pipe()  # warm up before the GPU starts working

    total = kept = 0
    for batch_no in range(1, int(max_batches) + 1):
        if kept >= keepers_target:
            break
        log.append(f"— batch {batch_no}: generating {int(batch_size)} image(s)…")
        yield [k[0] for k in keepers], "\n".join(log[-25:])
        t0 = time.monotonic()
        try:
            res = _api_post("/sdapi/v1/txt2img", {
                "prompt": prompt,
                "negative_prompt": negative or DEFAULT_NEGATIVE,
                "steps": int(steps), "cfg_scale": float(cfg),
                "width": int(width), "height": int(height),
                "sampler_name": sampler or "Euler a",
                "seed": -1, "n_iter": int(batch_size), "batch_size": 1,
                "send_images": True, "save_images": False,
            })
        except Exception as exc:
            log.append(f"  ✗ generation failed: {exc}")
            yield [k[0] for k in keepers], "\n".join(log[-25:])
            break
        gen_s = time.monotonic() - t0
        try:
            info = json.loads(res.get("info") or "{}")
            seeds = info.get("all_seeds") or []
            infotexts = info.get("infotexts") or []
        except Exception:
            seeds, infotexts = [], []

        images = res.get("images") or []
        log.append(f"  {len(images)} generated in {gen_s:.0f}s — judging…")
        yield [k[0] for k in keepers], "\n".join(log[-25:])

        for i, b64 in enumerate(images):
            total += 1
            try:
                im = Image.open(io.BytesIO(base64.b64decode(b64)))
            except Exception:
                continue
            seed = seeds[i] if i < len(seeds) else -1
            infotext = infotexts[i] if i < len(infotexts) else ""
            passed, score, verdict = judge_image(
                im, float(threshold), float(min_sharp), bool(want_face),
                bool(want_anatomy), bool(deep_scan), prompt)
            stem = f"{datetime.now().strftime('%H%M%S')}_{seed}_{score:.2f}"
            if passed:
                kept += 1
                _save(im, infotext, keep_dir, f"keep_{stem}")
                keepers.append((im, f"seed {seed} · {score:.2f}"))
                best.append((score, seed))
                log.append(f"  ✔ image {i+1}: {verdict}  (seed {seed})")
            else:
                if keep_rejects:
                    _save(im, infotext, rej_dir, f"rej_{stem}")
                log.append(f"  ✘ image {i+1}: {verdict}")
            yield [k[0] for k in keepers], "\n".join(log[-25:])
            if kept >= keepers_target:
                break

    best.sort(reverse=True)
    rate = f"{kept}/{total}" if total else "0/0"
    log.append("")
    if kept >= keepers_target:
        log.append(f"✅ Done — {kept} keeper(s) from {total} generated ({rate} pass rate).")
    else:
        log.append(f"⚠ Stopped at the safety cap with {kept}/{int(keepers_target)} "
                   f"keepers ({total} generated). Lower the threshold, refine the "
                   f"prompt, or run again.")
    if best:
        top = ", ".join(f"{s} ({sc:.2f})" for sc, s in best[:3])
        log.append(f"Best seeds: {top} — reuse them for variations/upscales.")
    log.append(f"Saved to: {keep_dir}")
    yield [k[0] for k in keepers], "\n".join(log[-30:])


# ---------------------------------------------------------------------------
# Character Passports (read from <data>/passports)
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


def insert_passport(name: str, current: str) -> str:
    if not name:
        return current
    try:
        p = json.loads((_passports_dir() / f"{name}.json").read_text("utf-8"))
    except Exception:
        return current
    parts = []
    for key in ("trigger", "tags"):
        v = (p.get(key) or "").strip().rstrip(",")
        if v:
            parts.append(v)
    lora = (p.get("lora") or "").strip()
    if lora:
        parts.append(f"<lora:{lora}:{float(p.get('lora_weight') or 0.8):g}>")
    frag = ", ".join(parts)
    if not frag:
        return current
    cur = (current or "").strip().rstrip(",")
    return f"{cur}, {frag}" if cur else frag


# ---------------------------------------------------------------------------
# Scene Queue — run the Auto Pilot loop once per scene line
# ---------------------------------------------------------------------------

def run_scene_queue(scenes_text, prompt, negative, add_quality, keepers_target,
                    batch_size, max_batches, threshold, min_sharp, want_face,
                    want_anatomy, deep_scan, keep_rejects,
                    lora_names, lora_weight, lora_triggers,
                    steps, cfg, width, height, sampler):
    scenes = [ln.strip().rstrip(",") for ln in (scenes_text or "").splitlines()
              if ln.strip()]
    if not scenes:
        yield [], "Add scenes first — one per line (e.g. 'classroom, sunset')."
        return
    base = (prompt or "").strip().rstrip(",")
    total_keep = 0
    last_gallery: list = []
    for i, scene in enumerate(scenes, 1):
        scene_prompt = f"{base}, {scene}" if base else scene
        header = f"🌙 Scene {i}/{len(scenes)}: {scene[:60]}"
        for gallery, log in run_autopilot(
                scene_prompt, negative, add_quality, keepers_target,
                batch_size, max_batches, threshold, min_sharp, want_face,
                want_anatomy, deep_scan, keep_rejects,
                lora_names, lora_weight, lora_triggers,
                steps, cfg, width, height, sampler):
            last_gallery = gallery
            yield gallery, f"{header}\n{log}"
        total_keep += len(last_gallery)
    yield last_gallery, (f"🌙 Scene queue finished — {len(scenes)} scene(s), "
                         f"~{total_keep} keeper(s) total. "
                         f"Everything is in output/autopilot/ — good morning!")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _on_ui_tabs():
    try:
        return _build_tab()
    except Exception:
        logger.exception("%s failed to build tab", TAG)
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown("Auto Pilot failed to load - see console log.")
        return [(ui, "Auto Pilot", "forge_autopilot")]


def _build_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## 🚀 Auto Pilot — generate, judge, retry until it's right\n"
            "Every image is quality-checked locally (blank/blur/face/aesthetic). "
            "Only passing images are kept; new seeds are rolled until you have "
            "the number you asked for. Build your prompt here or in the "
            "**Describe** tab and paste it."
        )
        prompt_box = gr.Textbox(label="Prompt", lines=3,
                                elem_classes=["prompt"],
                                placeholder="1girl, solo, silver hair, school uniform, "
                                            "classroom, sunset, looking at viewer…")
        with gr.Row():
            pass_dd = gr.Dropdown(label="🪪 Passport", choices=list_passports(),
                                  value="", scale=2)
            pass_ins = gr.Button("Insert into prompt", scale=0)
            pass_refresh = gr.Button("🔄", scale=0)
        negative_box = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE,
                                  lines=2, elem_classes=["prompt"])
        with gr.Row():
            add_quality = gr.Checkbox(label="Add quality tags", value=True)
            want_face = gr.Checkbox(label="Require a clear, sharp face", value=True)
            want_anatomy = gr.Checkbox(
                label="Anatomy gates (extra hands/people, proportions)", value=True)
            deep_scan = gr.Checkbox(
                label="Deep tag scan (bad_hands etc — slower)", value=False)
            keep_rejects = gr.Checkbox(label="Keep rejects (in /rejected)", value=False)
        with gr.Row():
            keepers_num = gr.Slider(1, 12, value=4, step=1, label="Keepers wanted")
            batch_num = gr.Slider(1, 8, value=4, step=1, label="Images per batch")
            maxbatch_num = gr.Slider(1, 12, value=5, step=1,
                                     label="Max batches (safety cap)")
        with gr.Row():
            threshold_sl = gr.Slider(0.5, 0.95, value=0.72, step=0.01,
                                     label="Quality threshold (higher = stricter)")
            sharp_sl = gr.Slider(20, 300, value=60, step=5,
                                 label="Min sharpness")
        with gr.Accordion("LoRA", open=False):
            with gr.Row():
                lora_dd = gr.Dropdown(label="LoRAs to apply", multiselect=True,
                                      choices=list_loras(), value=[], scale=3)
                lora_refresh = gr.Button("🔄", scale=0)
            with gr.Row():
                lora_weight_sl = gr.Slider(0.1, 1.5, value=0.8, step=0.05,
                                           label="LoRA weight")
                lora_triggers_chk = gr.Checkbox(
                    label="Auto-add trigger words (from Civitai helper data)",
                    value=True)

        with gr.Accordion("Generation settings", open=False):
            with gr.Row():
                steps_num = gr.Number(label="Steps", value=28, precision=0)
                cfg_num = gr.Number(label="CFG", value=5.5)
                width_num = gr.Number(label="Width", value=832, precision=0)
                height_num = gr.Number(label="Height", value=1216, precision=0)
                sampler_box = gr.Textbox(label="Sampler", value="Euler a")
        with gr.Row():
            go_btn = gr.Button("🚀 Generate until it's right", variant="primary")
            unload_btn = gr.Button("Unload judges (free memory)", scale=0)
        gallery = gr.Gallery(label="Keepers", columns=4, height=420,
                             object_fit="contain")
        log_box = gr.Textbox(label="Flight log", lines=14, interactive=False)

        go_btn.click(run_autopilot,
                     [prompt_box, negative_box, add_quality, keepers_num,
                      batch_num, maxbatch_num, threshold_sl, sharp_sl,
                      want_face, want_anatomy, deep_scan, keep_rejects,
                      lora_dd, lora_weight_sl, lora_triggers_chk,
                      steps_num, cfg_num,
                      width_num, height_num, sampler_box],
                     [gallery, log_box])
        unload_btn.click(lambda: unload_judges(), [], [log_box])
        lora_refresh.click(lambda: gr.update(choices=list_loras()), [], [lora_dd])
        pass_ins.click(insert_passport, [pass_dd, prompt_box], [prompt_box])
        pass_refresh.click(lambda: gr.update(choices=list_passports()), [],
                           [pass_dd])

        with gr.Accordion("🌙 Scene Queue — overnight multi-scene runs",
                          open=False):
            gr.Markdown(
                "One scene per line. Each scene runs the full Auto Pilot loop "
                "(same judges, keepers count and settings as above) with the "
                "prompt above as the base subject. Start it before bed."
            )
            scenes_box = gr.Textbox(
                label="Scenes (one per line)", lines=6,
                elem_classes=["prompt"],
                placeholder="classroom, morning light\n"
                            "rooftop, sunset, wind\n"
                            "street at night, neon lights, rain\n"
                            "beach, blue sky, sun sparkle")
            scenes_btn = gr.Button("🌙 Run scene queue", variant="primary")
        scenes_btn.click(run_scene_queue,
                         [scenes_box, prompt_box, negative_box, add_quality,
                          keepers_num, batch_num, maxbatch_num, threshold_sl,
                          sharp_sl, want_face, want_anatomy, deep_scan,
                          keep_rejects, lora_dd, lora_weight_sl,
                          lora_triggers_chk, steps_num, cfg_num, width_num,
                          height_num, sampler_box],
                         [gallery, log_box])

    return [(ui, "Auto Pilot", "forge_autopilot")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
