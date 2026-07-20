"""
sd-forge-lora-trainer / install.py
==================================

Intentionally minimal. Forge runs every extension's `install.py` at WebUI
startup, so anything heavy here would slow boot. The real bootstrap
(cloning kohya-ss/sd-scripts and provisioning a sandboxed venv) is run
lazily the first time the user clicks "Start training" in the new tab.

We only verify a couple of pure-Python deps that the UI itself needs to
even render. Anything else is deferred.
"""
from __future__ import annotations

import importlib
import subprocess
import sys


def _have(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


# tomli_w is used to write the kohya training TOML. tomllib (read-only) is
# in the stdlib from Python 3.11 onwards, which Forge requires, so we don't
# need to install that one. PyYAML is sometimes useful for kohya configs but
# diffusers already requires it, so it's certainly present.
for pkg, pip_name in [("tomli_w", "tomli_w")]:
    if not _have(pkg):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        except Exception as e:
            # Non-fatal: the trainer tab will surface a clear error on first use.
            print(f"[lora-trainer] install.py: optional dep {pip_name} not installed ({e})")
