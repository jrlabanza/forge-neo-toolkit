"""Run WD14 over every image in a folder, write CSV/JSON output.

Public API:
    batch_run(folder, analyzer_fn, output_path=None,
              recursive=False, progress_cb=None) -> dict
"""
import csv
import json
import os

_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _walk_images(folder, recursive):
    if not folder or not os.path.isdir(folder):
        return []
    out = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith(_EXTS):
                    out.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(folder)):
            full = os.path.join(folder, f)
            if os.path.isfile(full) and f.lower().endswith(_EXTS):
                out.append(full)
    return out


def batch_run(folder, analyzer_fn, output_path=None,
              recursive=False, progress_cb=None):
    """analyzer_fn(PIL.Image) -> dict with whatever fields you want logged."""
    from PIL import Image
    paths = _walk_images(folder, recursive)
    rows = []
    errors = []
    for i, p in enumerate(paths):
        if progress_cb:
            progress_cb("[{}/{}] {}".format(i + 1, len(paths), os.path.basename(p)))
        try:
            with Image.open(p) as img:
                img.load()
                result = analyzer_fn(img)
            row = {"file": p}
            row.update(result or {})
            rows.append(row)
        except Exception as e:
            errors.append({"file": p, "error": str(e)})

    written = None
    if output_path and rows:
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            if output_path.lower().endswith(".json"):
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"rows": rows, "errors": errors}, f,
                              indent=2, ensure_ascii=False)
                written = output_path
            else:
                # CSV: derive columns from union of row keys
                cols = sorted({k for r in rows for k in r.keys()})
                with open(output_path, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writeheader()
                    for r in rows:
                        w.writerow({k: r.get(k, "") for k in cols})
                written = output_path
        except Exception as e:
            errors.append({"file": "<output>", "error": str(e)})

    return {
        "rows":    rows,
        "errors":  errors,
        "written": written,
        "count":   len(rows),
    }
