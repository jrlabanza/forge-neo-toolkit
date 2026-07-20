# sd-forge-gallery

Output gallery browser tab for Forge Neo. Read-only — never touches your image files.

## Features
- Browses all configured output folders (txt2img/img2img/extras/saved + date subfolders), newest first
- Full-text search across embedded generation parameters (prompt, seed, model name, …) and filenames
- Date filter + favorites (★)
- Click an image → full parameters + file path → **Send to txt2img / img2img** with one click

## Usage
Open the **Gallery** tab → **Refresh**. First search indexes parameter text of all images (cached afterwards, only new files get read on later refreshes).

## Extra folders
If some outputs live outside the standard folders, add them in `gallery_settings.json` in this extension folder:

```json
{ "extra_roots": ["F:/some/other/images"] }
```

## Files written (all inside this extension folder)
- `param_cache.json` — parameter-text index cache
- `favorites.json` — your starred image paths

Delete either at any time; they rebuild.
