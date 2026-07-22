@echo off
setlocal
REM ============================================================================
REM apply_ui_layout.bat — one-time UI layout upgrade (run while Forge is CLOSED)
REM  - VAE + Clip Skip pinned to the top bar (quicksettings)
REM  - Tabs reordered: creation tools first, utilities after
REM Config is backed up first. Rerun-safe.
REM ============================================================================
cd /D "%~dp0"

REM refuse to run while Forge is up (it would overwrite the config on exit)
powershell -NoProfile -Command "exit [int](Test-NetConnection 127.0.0.1 -Port 7860 -InformationLevel Quiet -WarningAction SilentlyContinue)" >nul 2>&1
if errorlevel 1 (
    echo Forge appears to be RUNNING on port 7860.
    echo Close it first ^(its settings save on exit and would undo this^), then rerun me.
    pause & exit /b 1
)

copy /Y config.json "config.json.pre-layout.bak" >nul
"%~dp0venv\Scripts\python.exe" -c "import json; p='config.json'; c=json.load(open(p,encoding='utf-8')); c['quicksettings_list']=['sd_vae','CLIP_stop_at_last_layers']; c['ui_tab_order']=['txt2img','img2img','Describe','Auto Pilot','Gallery','Model Presets','Styles','Prompt Enhancer','Civitai','Job Runner','NAI Converter','LoRA Trainer','Extras','PNG Info','Settings','Extensions']; json.dump(c,open(p,'w',encoding='utf-8'),indent=4); print('Layout applied: quicksettings + tab order set.')"
if errorlevel 1 (
    echo Failed - config restored from config.json.pre-layout.bak
    copy /Y "config.json.pre-layout.bak" config.json >nul
    pause & exit /b 1
)
echo.
echo Done. Launch Forge (desktop shortcut or webui-user.bat) to see:
echo   - the new Midnight Violet theme (user.css)
echo   - VAE + Clip Skip in the top bar
echo   - creation tabs first: txt2img, img2img, Describe, Auto Pilot, Gallery...
echo Revert theme: delete user.css.  Revert layout: restore config.json.pre-layout.bak
pause
