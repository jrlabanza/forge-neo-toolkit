"""
sd-forge-lora-trainer / setup_bootstrap.py
==========================================

Standalone install script. Run via setup_venv.bat which invokes us with a
suitable Python interpreter (the .bat picks Python 3.10 if available -
kohya's own README recommends 3.10 and pins torch 2.6.0 + cu124 for it).

Does the heavy one-time setup the LoRA Trainer tab needs:
  1. Clone kohya-ss/sd-scripts into ./sd-scripts/ (if not already cloned)
  2. Create a sandboxed venv at ./sd-scripts-venv/ using THIS interpreter
     (so kohya's torch 2.6.x does not clobber Forge's torch 2.10.0+cu130).
  3. pip-install torch 2.6.0 + cu124 (per kohya's official README).
  4. pip-install kohya's own requirements.txt (most reliable version pins).
  5. pip-install onnxruntime-gpu (for WD14 tagger; not in kohya's reqs).

Idempotent: re-running it is safe and skips steps already done.

This file deliberately has NO imports of gradio / modules / backend so it
can be invoked outside the WebUI.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SD_SCRIPTS_DIR = ROOT / "sd-scripts"
SD_SCRIPTS_VENV = ROOT / "sd-scripts-venv"

SD_SCRIPTS_REPO = "https://github.com/kohya-ss/sd-scripts.git"
SD_SCRIPTS_BRANCH = "main"

TORCH_INDEX_URL = os.environ.get(
    "LORA_TRAINER_TORCH_INDEX",
    "https://download.pytorch.org/whl/cu124",
)
TORCH_PACKAGES = ["torch==2.6.0", "torchvision==0.21.0"]

EXTRA_PACKAGES = ["onnxruntime-gpu", "numpy<2.0"]


def venv_python() -> Path:
    if os.name == "nt":
        return SD_SCRIPTS_VENV / "Scripts" / "python.exe"
    return SD_SCRIPTS_VENV / "bin" / "python"


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if rc != 0:
        print(f"\n!! command failed with exit code {rc}", flush=True)
        sys.exit(rc)


def step_1_clone():
    train_script = SD_SCRIPTS_DIR / "sdxl_train_network.py"
    if train_script.exists():
        print(f"[1/5] sd-scripts already cloned at {SD_SCRIPTS_DIR}, skipping.", flush=True)
        return
    print(f"[1/5] cloning {SD_SCRIPTS_REPO} ({SD_SCRIPTS_BRANCH} branch) ...", flush=True)
    if SD_SCRIPTS_DIR.exists():
        import shutil
        shutil.rmtree(SD_SCRIPTS_DIR, ignore_errors=True)
    run(["git", "clone", "--depth=1", "--branch", SD_SCRIPTS_BRANCH,
         SD_SCRIPTS_REPO, str(SD_SCRIPTS_DIR)])


def step_2_venv():
    vp = venv_python()
    if vp.exists():
        print(f"[2/5] venv already exists at {SD_SCRIPTS_VENV}, skipping.", flush=True)
        return
    print(f"[2/5] creating sandboxed venv at {SD_SCRIPTS_VENV} ...", flush=True)
    print(f"      using interpreter: {sys.executable}", flush=True)
    print(f"      python version   : {sys.version.split()[0]}", flush=True)

    # Try the stdlib venv module first. Some stripped/uv-distributed Python
    # installs (notably F:\Data\Assets\Python310) have NO stdlib venv module
    # but DO have a working virtualenv package on the path - fall back to that.
    cmd_venv = [sys.executable, "-m", "venv", str(SD_SCRIPTS_VENV)]
    cmd_vex  = [sys.executable, "-m", "virtualenv", str(SD_SCRIPTS_VENV)]

    venv_cmd_str = " ".join(str(c) for c in cmd_venv)
    print("\n$ " + venv_cmd_str, flush=True)
    rc = subprocess.call([str(c) for c in cmd_venv])
    if rc != 0 or not vp.exists():
        print("   stdlib venv failed (rc={}); trying virtualenv ...".format(rc), flush=True)
        vex_cmd_str = " ".join(str(c) for c in cmd_vex)
        print("\n$ " + vex_cmd_str, flush=True)
        rc = subprocess.call([str(c) for c in cmd_vex])
        if rc != 0:
            print(f"\n!! virtualenv also failed (rc={rc})", flush=True)
            sys.exit(rc)
    if not vp.exists():
        print(f"!! venv python missing at {vp}", flush=True)
        sys.exit(1)


def step_3_pip_base():
    print(f"[3/5] upgrading pip / setuptools / wheel in sandboxed venv ...", flush=True)
    run([str(venv_python()), "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel"])


def step_4_pip_torch_and_requirements():
    print(f"[4a/5] installing torch from {TORCH_INDEX_URL} ...", flush=True)
    run([str(venv_python()), "-m", "pip", "install",
         "--index-url", TORCH_INDEX_URL, *TORCH_PACKAGES])

    reqs = SD_SCRIPTS_DIR / "requirements.txt"
    if not reqs.exists():
        print(f"!! kohya requirements.txt missing at {reqs}", flush=True)
        sys.exit(1)

    print(f"[4b/5] installing kohya's requirements.txt ...", flush=True)
    run([str(venv_python()), "-m", "pip", "install",
         "--use-pep517", "-r", str(reqs)],
        cwd=SD_SCRIPTS_DIR)


def step_5_extras():
    print(f"[5/5] installing extras (WD14 tagger deps) ...", flush=True)
    run([str(venv_python()), "-m", "pip", "install", *EXTRA_PACKAGES])


def main():
    print("=" * 70, flush=True)
    print("sd-forge-lora-trainer / bootstrap", flush=True)
    print("=" * 70, flush=True)
    print(f"  install root  : {ROOT}", flush=True)
    print(f"  sd-scripts    : {SD_SCRIPTS_DIR}", flush=True)
    print(f"  sandboxed venv: {SD_SCRIPTS_VENV}", flush=True)
    print(f"  torch index   : {TORCH_INDEX_URL}", flush=True)
    print(f"  invoker python: {sys.executable}", flush=True)
    print(f"  invoker version: {sys.version.split()[0]}", flush=True)
    if sys.version_info[:2] not in [(3, 10), (3, 11)]:
        v = "{}.{}".format(sys.version_info[0], sys.version_info[1])
        print(f"  WARNING: kohya recommends Python 3.10; running on {v}", flush=True)
        print(f"  Some deps may lack wheels for newer Python.", flush=True)
    print(f"  expected runtime: 5-15 minutes (depends on bandwidth)", flush=True)
    print("=" * 70, flush=True)

    step_1_clone()
    step_2_venv()
    step_3_pip_base()
    step_4_pip_torch_and_requirements()
    step_5_extras()

    print()
    print("=" * 70, flush=True)
    print("  DONE.", flush=True)
    print(f"  The LoRA Trainer is ready. Open Forge and look for the", flush=True)
    print(f"  'LoRA Trainer' tab next to txt2img / img2img.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n!! interrupted by user", flush=True)
        sys.exit(130)
