# sd-forge-job-runner
"Job Runner" tab, driven by Forge's local API (127.0.0.1 only; needs the --api launch flag — webui-user.bat sets it).

- **Queue**: paste infotexts (from Gallery / Model Presets / PNG Info), stack jobs, Run. Forge saves results into the normal output folders. Queue survives restarts (queue.json).
- **Batch re-process**: run a folder (or your Gallery favorites) through Extras upscale, or img2img refine at low denoise reusing each image's own embedded prompt, optionally + ADetailer face pass.
- **Test cards**: same prompt+seed on every selected checkpoint -> labeled contact sheet saved to output/test-cards/. Roughly a minute per model (each switch loads ~7 GB).
