"""Generic named-preset storage for the Prompt Enhancer.

A "preset" is just a dict. Each kind of preset (prompt_builder, cn_recipe,
etc.) lives in its own JSON file under user_data/.

Public API:
    save(kind, name, payload) -> bool
    load(kind, name) -> dict | None
    list_names(kind) -> list[str]
    delete(kind, name) -> bool
"""
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_USER_DATA = os.path.normpath(os.path.join(_HERE, "..", "user_data"))


def _ensure_user_data():
    try:
        os.makedirs(_USER_DATA, exist_ok=True)
    except Exception:
        pass


def _store_path(kind):
    _ensure_user_data()
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", kind)
    return os.path.join(_USER_DATA, "presets_{}.json".format(safe))


def _read_store(kind):
    p = _store_path(kind)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read().replace("\x00", "")
        return json.loads(raw) or {}
    except Exception:
        return {}


def _write_store(kind, store):
    p = _store_path(kind)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def save(kind, name, payload):
    if not name or not name.strip():
        return False
    store = _read_store(kind)
    store[name.strip()] = payload
    return _write_store(kind, store)


def load(kind, name):
    return _read_store(kind).get(name)


def list_names(kind):
    return sorted(_read_store(kind).keys(), key=lambda s: s.lower())


def delete(kind, name):
    store = _read_store(kind)
    if name in store:
        store.pop(name)
        return _write_store(kind, store)
    return False
