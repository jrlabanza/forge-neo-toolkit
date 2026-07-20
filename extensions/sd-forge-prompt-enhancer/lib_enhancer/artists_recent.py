"""Track recently-used artists so the user can quick-pick from a short list
above the 44K-entry full dropdown.

Cap at 12 entries (LRU)."""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.normpath(os.path.join(_HERE, "..", "user_data", "recent_artists.json"))
_MAX = 12


def _load():
    if not os.path.exists(_PATH):
        return []
    try:
        return json.loads(open(_PATH, encoding="utf-8").read().replace("\x00", "")) or []
    except Exception:
        return []


def _save(lst):
    try:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump(lst, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def record(artist_label):
    if not artist_label or not artist_label.strip():
        return
    lst = _load()
    if artist_label in lst:
        lst.remove(artist_label)
    lst.insert(0, artist_label)
    _save(lst[:_MAX])


def list_recent():
    return _load()
