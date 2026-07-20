"""
sd-forge-notify
===============

Fires a notification when a generation takes longer than a threshold —
useful for big batches, hires-fix runs and ControlNet stacks you walk away
from.

Two channels, both optional:
- Windows toast (no dependencies; uses the built-in WinRT toast API
  through PowerShell)
- Discord webhook (paste a webhook URL)

Configured from the "Notify" accordion added at the bottom of txt2img and
img2img. Settings persist to notify_settings.json in this extension folder.

Implementation: a Script (AlwaysVisible). process() stamps the start time,
postprocess() runs once after the whole job (all batch images) finishes and
fires the notification if the job exceeded the threshold.

Author: built by Claude on 2026-07-20.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import gradio as gr

try:
    from modules import scripts
except ImportError:
    scripts = None  # type: ignore

logger = logging.getLogger(__name__)
TAG = "[notify]"

EXT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = EXT_ROOT / "notify_settings.json"

DEFAULTS = {
    "enabled": False,
    "min_seconds": 60,
    "toast": True,
    "webhook_url": "",
}


def _load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {**DEFAULTS, **data}
    except Exception:
        return dict(DEFAULTS)


def _save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception as exc:
        logger.warning("%s could not save settings: %s", TAG, exc)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

_PS_TOAST = (
    "$null = [Windows.UI.Notifications.ToastNotificationManager, "
    "Windows.UI.Notifications, ContentType=WindowsRuntime];"
    "$t=[Windows.UI.Notifications.ToastTemplateType]::ToastText02;"
    "$x=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($t);"
    "$s=$x.GetElementsByTagName('text');"
    "$null=$s.Item(0).AppendChild($x.CreateTextNode('Forge Neo'));"
    "$null=$s.Item(1).AppendChild($x.CreateTextNode('{msg}'));"
    "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
    "'Forge Neo').Show([Windows.UI.Notifications.ToastNotification]::new($x))"
)


def _fire_toast(message: str) -> None:
    if os.name != "nt":
        return
    try:
        safe = message.replace("'", " ").replace('"', " ").replace("\n", " ")
        cmd = _PS_TOAST.replace("{msg}", safe)
        subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        logger.warning("%s toast failed: %s", TAG, exc)


def _fire_webhook(url: str, message: str) -> None:
    if not url.startswith("https://"):
        return
    try:
        import requests
        requests.post(url, json={"content": message[:1900]}, timeout=5)
    except Exception as exc:
        logger.warning("%s webhook failed: %s", TAG, exc)


def _notify(message: str, settings: dict) -> None:
    def work():
        if settings.get("toast"):
            _fire_toast(message)
        url = (settings.get("webhook_url") or "").strip()
        if url:
            _fire_webhook(url, message)
    threading.Thread(target=work, daemon=True, name="forge-notify").start()


# ---------------------------------------------------------------------------
# Script hook
# ---------------------------------------------------------------------------

if scripts is not None:

    class NotifyScript(scripts.Script):
        def title(self):
            return "Notify when done"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            s = _load_settings()
            suffix = "img2img" if is_img2img else "txt2img"
            try:
                with gr.Accordion("🔔 Notify when done", open=False,
                                  elem_id=f"notify_accordion_{suffix}"):
                    enabled = gr.Checkbox(label="Enable notifications",
                                          value=s["enabled"])
                    min_secs = gr.Number(
                        label="Only notify if the job took at least (seconds)",
                        value=s["min_seconds"], precision=0)
                    toast = gr.Checkbox(label="Windows toast", value=s["toast"])
                    webhook = gr.Textbox(
                        label="Discord webhook URL (optional)",
                        value=s["webhook_url"], type="password",
                        placeholder="https://discord.com/api/webhooks/…")
                    with gr.Row():
                        save_btn = gr.Button("Save settings", scale=0)
                        test_btn = gr.Button("Send test notification", scale=0)
                    info = gr.Markdown("")

                def do_save(en, ms, to, wh):
                    _save_settings({
                        "enabled": bool(en),
                        "min_seconds": int(ms or 0),
                        "toast": bool(to),
                        "webhook_url": (wh or "").strip(),
                    })
                    return "Saved."

                def do_test(en, ms, to, wh):
                    _notify("Test notification — channel works.",
                            {"toast": bool(to), "webhook_url": (wh or "").strip()})
                    return "Test sent (check your notifications)."

                save_btn.click(do_save, [enabled, min_secs, toast, webhook], [info])
                test_btn.click(do_test, [enabled, min_secs, toast, webhook], [info])
            except Exception:
                logger.exception("%s ui build failed", TAG)
                enabled = gr.Checkbox(value=False, visible=False)
                min_secs = gr.Number(value=60, visible=False)
                toast = gr.Checkbox(value=True, visible=False)
                webhook = gr.Textbox(value="", visible=False)
            return [enabled, min_secs, toast, webhook]

        def process(self, p, *args):
            try:
                p._notify_t0 = time.monotonic()
            except Exception:
                pass

        def postprocess(self, p, processed, *args):
            try:
                # live UI values take precedence; fall back to saved settings
                if len(args) >= 4:
                    enabled, min_secs, toast, webhook = args[0], args[1], args[2], args[3]
                    settings = {"toast": bool(toast),
                                "webhook_url": (webhook or "").strip()}
                else:
                    s = _load_settings()
                    enabled, min_secs = s["enabled"], s["min_seconds"]
                    settings = s
                if not enabled:
                    return
                t0 = getattr(p, "_notify_t0", None)
                if t0 is None:
                    return
                dur = time.monotonic() - t0
                if dur < float(min_secs or 0):
                    return
                n = len(getattr(processed, "images", []) or [])
                mins, secs = divmod(int(dur), 60)
                _notify(f"Generation finished: {n} image(s) in {mins}m {secs}s.",
                        settings)
            except Exception as exc:
                logger.warning("%s postprocess failed: %s", TAG, exc)
