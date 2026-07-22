"""
sd-forge-guide — "Guide" tab: the manual for this studio.
Static reference for every custom tool, plus the full optimization notes.
Author: built by Claude on 2026-07-22.
"""
from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

try:
    from modules import script_callbacks
except ImportError:
    script_callbacks = None  # type: ignore

logger = logging.getLogger(__name__)
EXT_ROOT = Path(__file__).resolve().parents[1]

GUIDE_MD = """
# 🎓 Studio Guide

**The pipeline at a glance**

> 🪪 Passports → ✍️ Describe → 🚀 Auto Pilot / 👥 Characters → 🖼 Gallery → 🎬 Director → 🎯 Sweet Spot

---

## Creating

| Tab | What it's for | The move |
|---|---|---|
| **Describe** | Plain English → perfect Illustrious tags | Write a paragraph → *Translate* → tick Scene ideas → 🤖 AI-complete → *Send to txt2img* |
| **Auto Pilot** | Hands-off generation with quality gates | Prompt (or 🪪 insert) → set keepers → 🚀. Rejects blur/blank/extra-hands/off-aesthetic automatically |
| └ 🌙 Scene Queue | Overnight batches | One scene per line → run before bed → keepers per scene in the morning |
| **Characters** | Two or three characters, each their own prompt | 🪪 per slot → shared tags → background line → *Generate scene*. Each holds their column |
| **Model Presets** | Right settings per checkpoint family | *Detect* → *Load into txt2img*. Pony needs `score_9…`, Illustrious doesn't — this remembers so you don't |
| **txt2img 📸 accordion** | NAI-style reference | Up to 3 stacked refs (Vibe/Precise each) + 💾 vibe library |

## Curating

| Tab | What it's for | The move |
|---|---|---|
| **Gallery** | Find & reuse anything you ever made | Search by tag/seed/model → ★ favorites → *Send to txt2img* / *→ Director*. Selection set for bulk trash/favorite. A/B compare for finals |
| **Director** | One-click fixes on any image | ✨Enhance · 🎲Variations · 😊Emotion (face-only) · 🎨Recolor · 🪄Remove BG · 🖌Fix Region (paint + describe) · ✒Line art · ✏Sketch · 🖍Colorize · 🏁 Finalize macros |
| **Sweet Spot** (Job Runner) | Find the best value by eye | Same-seed ladder over LoRA weight / CFG / steps → labeled sheet |

## Managing

| Tab | What it's for |
|---|---|
| **Passports** | One saved identity per character (tags+LoRA+trigger+reference). Build once, insert everywhere |
| **Styles** | Search/edit your styles.csv library with auto-backup |
| **Civitai** | Trigger words + previews for your LoRAs, update checks, and in-app model downloads |
| **Job Runner** | Queue infotext jobs · batch re-process folders · checkpoint test cards |
| **NAI Converter** | Import NovelAI PNGs → A1111 format |
| **LoRA Trainer** | Drop 5–15 images + trigger word → train |

## Everywhere

- **Tag autocomplete works in every prompt field of every tab** — type, pick, `Tab`
- Every tool saves with full metadata → everything is searchable in the Gallery

## Maintenance (double-click in the install folder)

| File | When |
|---|---|
| `webui-user.bat` / desktop “Forge Neo” | Normal launch (also installs new deps) |
| `webui-user-faststart.bat` | Daily quick launch (skip env checks) |
| `update_forge.bat` | Pull latest Forge Neo (backup branch auto-created) |
| `sync_and_push.bat` (in forge-neo-toolkit) | Commit + push all custom work to GitHub |
| `fix_and_launch.bat` | If a launch dies weirdly (kills stale python, relaunches) |
| `check_autopilot.bat` | Syntax-check all custom extensions |
| `apply_ui_layout.bat` | Re-apply quicksettings/tab order (Forge closed) |

**Rollbacks:** theme → delete `user.css` · layout → restore `config.json.pre-layout.bak` · Forge update → `git reset --hard backup/pre-update-20260720` · configs → `_attic/config-autobackups/`
"""


def _on_ui_tabs():
    try:
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown(GUIDE_MD)
            with gr.Accordion("📜 Full optimization notes & history", open=False):
                try:
                    notes = (EXT_ROOT.parents[1] /
                             "FORGE_NEO_OPTIMIZATION_NOTES.md").read_text("utf-8")
                except Exception:
                    notes = "*Notes file not found.*"
                gr.Markdown(notes)
        return [(ui, "Guide", "forge_guide")]
    except Exception:
        logger.exception("[guide] failed to build tab")
        with gr.Blocks(analytics_enabled=False) as ui:
            gr.Markdown("Guide failed to load.")
        return [(ui, "Guide", "forge_guide")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
