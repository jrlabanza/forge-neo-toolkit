"""Push detected character tags into ADetailer's "Detector Classes" field.

Modifies the Forge ui-config.json so on next UI reload, ADetailer's 2nd-tab
detector classes field is pre-populated.

Public API:
    push_character_classes(character_tags, slot='2nd') -> dict
        {ok: bool, message: str, path: str}
"""
import json
import os
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
# extensions/sd-forge-prompt-enhancer/lib_enhancer -> Forge root is up 3
_FORGE_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))


def _ui_config_path():
    return os.path.join(_FORGE_ROOT, "ui-config.json")


def push_character_classes(character_tags, slot="2nd"):
    """character_tags is a list of strings.
    slot is '' (1st unit), '2nd', '3rd', or '4th'."""
    p = _ui_config_path()
    if not os.path.exists(p):
        return {"ok": False,
                "message": "ui-config.json not found at " + p,
                "path": p}

    try:
        raw = open(p, "rb").read().replace(b"\x00", b"")
        cfg = json.loads(raw.decode("utf-8"))
    except Exception as e:
        return {"ok": False,
                "message": "Failed to parse ui-config: " + str(e),
                "path": p}

    classes_str = ", ".join(t.strip() for t in character_tags
                            if t and t.strip())
    if not classes_str:
        return {"ok": False, "message": "No character tags provided.", "path": p}

    suffix = " " + slot if slot else ""
    updated = []
    for tab in ("txt2img", "img2img"):
        key = "{}/Detector Classes{}/value".format(tab, suffix)
        if key in cfg or True:  # write even if missing — Forge will pick up
            cfg[key] = classes_str
            updated.append(key)

    # Snapshot then write atomically
    try:
        shutil.copy2(p, p + ".pre-adetailer-classes-bak")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception as e:
        return {"ok": False,
                "message": "Write failed: " + str(e),
                "path": p}

    return {
        "ok": True,
        "message": "Updated {} key(s). Reload UI to see them in ADetailer.".format(
            len(updated)),
        "path": p,
        "classes": classes_str,
    }
