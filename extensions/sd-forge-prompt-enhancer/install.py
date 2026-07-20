"""
sd-forge-prompt-enhancer / install.py
=====================================

Runs on every Forge startup. Two jobs:
  1. Ensure deps (onnxruntime, Pillow) for image-to-prompt.
  2. Re-apply jrlabanza's persistent patches so extension updates don't wipe them.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys


def _have(pkg):
    return importlib.util.find_spec(pkg) is not None


def _pip_install(pkg):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", pkg]
        )
        return True
    except Exception as e:
        print("[sd-forge-prompt-enhancer] pip install " + pkg + " failed: " + str(e))
        return False


if not _have("onnxruntime"):
    print("[sd-forge-prompt-enhancer] installing onnxruntime (for image-to-prompt)")
    if not _pip_install("onnxruntime-gpu"):
        _pip_install("onnxruntime")

if not _have("PIL"):
    _pip_install("Pillow")


PERSISTENT_PATCHES = [
    {
        "rel_path": "adetailer/lib_adetailer/ui.py",
        "find":     'value=False, label="ADetailer", elem_id=eid("ad_main_accordion")',
        "replace":  'value=True, label="ADetailer", elem_id=eid("ad_main_accordion")',
        "desc":     "ADetailer auto-enable accordion",
    },
    {
        "rel_path": "adetailer/lib_adetailer/detection/common.py",
        "find":     '"face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",\n    }',
        "replace":  '"face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",\n        "mediapipe_face_mesh_eyes_only.task": "ALIAS:face_landmarker.task",\n    }',
        "desc":     "ADetailer-Neo: register mediapipe_face_mesh_eyes_only.task alias",
    },
    {
        "rel_path": "adetailer/lib_adetailer/detection/mediapipe.py",
        "find":     'if "landmarker" in model_path:\n        return _mediapipe_face_mesh(model_path, image, confidence)\n    elif "face" in model_path:',
        "replace":  '_mp_str = str(model_path).replace("\\\\", "/")\n    basename = _mp_str.rsplit("/", 1)[-1].lower()\n    if "eyes_only" in basename:\n        return _mediapipe_face_mesh_eyes_only(model_path, image, confidence)\n    if "landmarker" in basename or basename.endswith(".task"):\n        return _mediapipe_face_mesh(model_path, image, confidence)\n    if "face" in basename:',
        "desc":     "ADetailer-Neo: hardened dispatcher (eyes_only first, .task=>landmarker)",
    },
]


def _apply_persistent_patches():
    here = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.dirname(here)
    for patch in PERSISTENT_PATCHES:
        target = os.path.join(extensions_dir, patch["rel_path"].replace("/", os.sep))
        if not os.path.exists(target):
            print("[sd-forge-prompt-enhancer] patch SKIPPED (file missing): " + patch["desc"])
            continue
        try:
            with open(target, "r", encoding="utf-8") as f:
                src = f.read()
        except Exception as e:
            print("[sd-forge-prompt-enhancer] patch READ FAIL: " + str(e))
            continue
        if patch["replace"] in src:
            continue
        if patch["find"] not in src:
            print("[sd-forge-prompt-enhancer] patch NEEDS REVIEW: " + patch["desc"])
            continue
        try:
            new_src = src.replace(patch["find"], patch["replace"], 1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(new_src)
            print("[sd-forge-prompt-enhancer] RE-APPLIED patch: " + patch["desc"])
        except Exception as e:
            print("[sd-forge-prompt-enhancer] patch WRITE FAIL: " + str(e))


_apply_persistent_patches()
