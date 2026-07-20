"""
sd-forge-config-backup
======================

Zero-UI insurance extension. On every app start it snapshots the three files
that got mangled in the June 2026 crash saga:

    config.json, ui-config.json, styles.csv

into  _attic/config-autobackups/<timestamp>/  and keeps the newest 10 sets.

Restoring = copy the files back from the newest good snapshot while Forge is
closed. Costs ~1 MB and a few milliseconds per launch.

Author: built by Claude on 2026-07-20.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

try:
    from modules import script_callbacks
except ImportError:
    script_callbacks = None  # type: ignore

logger = logging.getLogger(__name__)
TAG = "[config-backup]"

KEEP = 10
FILES = ("config.json", "ui-config.json", "styles.csv")


def _data_path() -> Path:
    try:
        from modules import paths
        return Path(paths.data_path)
    except Exception:
        return Path(__file__).resolve().parents[3]


def _run_backup(*_args, **_kwargs) -> None:
    try:
        base = _data_path()
        dest_root = base / "_attic" / "config-autobackups"
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dest = dest_root / stamp
        copied = 0
        for name in FILES:
            src = base / name
            if src.is_file():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / name)
                copied += 1
        if not copied:
            return
        # prune oldest beyond KEEP
        snaps = sorted((d for d in dest_root.iterdir() if d.is_dir()),
                       key=lambda d: d.name, reverse=True)
        for old in snaps[KEEP:]:
            shutil.rmtree(old, ignore_errors=True)
        logger.info("%s snapshot %s (%d files, keeping %d sets)",
                    TAG, stamp, copied, min(len(snaps), KEEP))
        print(f"{TAG} config snapshot saved: _attic/config-autobackups/{stamp}")
    except Exception as exc:
        # never break startup over a backup
        logger.warning("%s backup failed: %s", TAG, exc)


if script_callbacks is not None:
    script_callbacks.on_app_started(_run_backup)
