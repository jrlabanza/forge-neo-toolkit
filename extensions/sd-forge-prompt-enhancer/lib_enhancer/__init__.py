"""
sd-forge-prompt-enhancer / lib_enhancer
========================================

Helper modules for the Prompt Enhancer "Pro Tools" tab. Each module is
self-contained pure-logic so it can be tested without Gradio / Forge.

Submodules:
  - wildcards      : {a|b|c} expansion + nested
  - tokens         : CLIP token counting + chunk indicator
  - linter         : prompt sanity checks (contradictions, syntax, weights)
  - neg_presets    : pre-built negative-prompt blocks
  - presets        : save/load named JSON presets to disk
  - lora_suggest   : scan models/Lora and match against detected tags
  - cn_suggest     : heuristic-pick ControlNet units from analyzed image
  - batch_analyze  : run WD14 over a folder of images, dump CSV/JSON
  - history        : per-session generation history log
  - artists_recent : recently-used artist tracker
  - send_adetailer : write detector classes back to ui-config.json
  - card_export    : recipe-card PNG composition
  - image_compare  : side-by-side prompt+settings diff between 2 images
"""
