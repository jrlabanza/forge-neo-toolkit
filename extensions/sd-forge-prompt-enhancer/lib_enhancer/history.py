"""Per-session generation history. Persisted to user_data/history.json.

Records every Build/Replicate action so the user can browse the last N
configurations and re-load any of them. Capped to 100 entries (LRU).
"""
import json
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.normpath(os.path.join(_HERE, "..", "user_data", "history.json"))
_MAX_ENTRIES = 100


def _load():
    if not os.path.exists(_PATH):
        return []
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            raw = f.read().replace("\x00", "")
        return json.loads(raw) or []
    except Exception:
        return []


def _save(entries):
    try:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def record(kind, label, payload):
    """Append one entry. payload should be a dict of inputs.
    Returns the new history list."""
    entries = _load()
    entries.insert(0, {
        "ts":      time.time(),
        "kind":    kind,
        "label":   label,
        "payload": payload,
    })
    entries = entries[:_MAX_ENTRIES]
    _save(entries)
    return entries


def list_entries(limit=50):
    return _load()[:limit]


def get(index):
    entries = _load()
    if 0 <= index < len(entries):
        return entries[index]
    return None


def clear():
    return _save([])


def format_label(entry):
    """Pretty one-line label for a history entry."""
    import datetime as _dt
    ts = _dt.datetime.fromtimestamp(entry["ts"]).strftime("%m-%d %H:%M")
    return "{} · {} · {}".format(ts, entry["kind"], entry.get("label", ""))
