@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM sync_and_push.bat — refresh this repo from the live install, commit, push.
REM Double-click me. First push may pop a GitHub browser-login window (approve it).
REM ============================================================================
cd /D "%~dp0"
set "SRC=%~dp0.."

echo Syncing live extension code into the repo...
for %%E in (sd-forge-gallery sd-forge-model-presets sd-forge-styles-manager sd-forge-civitai-helper sd-forge-job-runner sd-forge-notify sd-forge-config-backup sd-forge-tag-translator sd-forge-prompt-enhancer sd-forge-lora-trainer sd-forge-reference-image sd-forge-ipadapter-compat sd-forge-naitosd) do (
    robocopy "%SRC%\extensions\%%E" "extensions\%%E" /MIR /NFL /NDL /NJH /NJS ^
        /XD .git .vscode __pycache__ sd-scripts sd-scripts-venv downloads projects wd14_model user_data ^
        /XF *.pyc param_cache.json favorites.json gallery_settings.json hashes.json results.json civitai_settings.json queue.json notify_settings.json >nul
)
copy /Y "%SRC%\webui-user.bat" "launchers\" >nul
copy /Y "%SRC%\webui-user-faststart.bat" "launchers\" >nul
copy /Y "%SRC%\FORGE_NEO_OPTIMIZATION_NOTES.md" "docs\" >nul

echo Committing...
git add -A
git diff --cached --quiet && echo Nothing new to commit. || git commit -m "sync %date% %time:~0,8%"

echo Pushing to GitHub...
git push -u origin main
if errorlevel 1 (
    echo.
    echo Push failed. If this is the first push, make sure the GitHub repo exists
    echo and approve the browser login window if one appeared.
) else (
    echo.
    echo Done — https://github.com/jrlabanza/forge-neo-toolkit
)
pause
