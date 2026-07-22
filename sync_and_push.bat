@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM sync_and_push.bat — refresh this repo from the live install, commit, push.
REM Double-click me. First push may pop a GitHub browser-login window (approve it).
REM ============================================================================
cd /D "%~dp0"
set "SRC=%~dp0.."

REM ---- locate git (Forge folder's PortableGit, system PATH, or Stability Matrix) ----
set "GIT="
if exist "%~dp0..\PortableGit\cmd\git.exe" set "GIT=%~dp0..\PortableGit\cmd\git.exe"
if not defined GIT where git >nul 2>&1 && set "GIT=git"
if not defined GIT if exist "%~dp0..\..\..\Assets\PortableGit\cmd\git.exe" set "GIT=%~dp0..\..\..\Assets\PortableGit\cmd\git.exe"
if not defined GIT if exist "%~dp0..\..\..\Assets\PortableGit\bin\git.exe" set "GIT=%~dp0..\..\..\Assets\PortableGit\bin\git.exe"
if not defined GIT if exist "%LOCALAPPDATA%\StabilityMatrix\Assets\PortableGit\cmd\git.exe" set "GIT=%LOCALAPPDATA%\StabilityMatrix\Assets\PortableGit\cmd\git.exe"
if not defined GIT if exist "%APPDATA%\StabilityMatrix\Assets\PortableGit\cmd\git.exe" set "GIT=%APPDATA%\StabilityMatrix\Assets\PortableGit\cmd\git.exe"
if not defined GIT (
    echo Could not find git - neither on PATH nor in Stability Matrix's assets.
    echo Install Git for Windows with:   winget install --id Git.Git
    echo then rerun this script.
    pause & exit /b 1
)
echo Using git: %GIT%
"%GIT%" config --global --add safe.directory "*" >nul 2>&1

REM Clear ALL stale locks + temp objects left by the sandboxed setup session
del /f /q ".git\*.lock" 2>nul
if exist ".git\refs\heads\main.lock" del /f /q ".git\refs\heads\main.lock"
for /d %%D in (".git\objects\*") do del /f /q "%%D\tmp_obj_*" 2>nul

echo Syncing live extension code into the repo...
for %%E in (sd-forge-autopilot sd-forge-characters sd-forge-director sd-forge-gallery sd-forge-guide sd-forge-passport sd-forge-model-presets sd-forge-styles-manager sd-forge-civitai-helper sd-forge-job-runner sd-forge-notify sd-forge-config-backup sd-forge-tag-translator sd-forge-prompt-enhancer sd-forge-lora-trainer sd-forge-reference-image sd-forge-ipadapter-compat sd-forge-naitosd) do (
    robocopy "%SRC%\extensions\%%E" "extensions\%%E" /MIR /NFL /NDL /NJH /NJS ^
        /XD .git .vscode __pycache__ sd-scripts sd-scripts-venv downloads projects wd14_model user_data ^
        /XF *.pyc param_cache.json favorites.json gallery_settings.json hashes.json results.json civitai_settings.json queue.json notify_settings.json >nul
)
copy /Y "%SRC%\webui-user.bat" "launchers\" >nul
copy /Y "%SRC%\webui-user-faststart.bat" "launchers\" >nul
copy /Y "%SRC%\update_forge.bat" "launchers\" >nul 2>nul
copy /Y "%SRC%\fix_and_launch.bat" "launchers\" >nul 2>nul
copy /Y "%SRC%\make_shortcuts.bat" "launchers\" >nul 2>nul
copy /Y "%SRC%\apply_ui_layout.bat" "launchers\" >nul 2>nul
if not exist "ui" mkdir ui
copy /Y "%SRC%\user.css" "ui\" >nul 2>nul
copy /Y "%SRC%\FORGE_NEO_OPTIMIZATION_NOTES.md" "docs\" >nul

echo Committing...
"%GIT%" add -A
"%GIT%" diff --cached --quiet && echo Nothing new to commit. || "%GIT%" commit -m "sync %date% %time:~0,8%"

REM ---------------------------------------------------------------------------
REM Make sure the GitHub repo exists (first run only)
REM ---------------------------------------------------------------------------
"%GIT%" ls-remote origin >nul 2>&1
if errorlevel 1 (
    echo.
    echo The GitHub repo does not exist yet. Trying GitHub CLI...
    where gh >nul 2>&1
    if not errorlevel 1 (
        gh repo create forge-neo-toolkit --private --source . --remote origin 2>nul
    ) else (
        echo GitHub CLI not installed - opening the browser instead.
        echo.
        echo   In the page that opens:  name is prefilled, pick PRIVATE, click
        echo   "Create repository". I'll detect it automatically - no key press.
        echo.
        start "" "https://github.com/new?name=forge-neo-toolkit&description=Custom+Forge+Neo+extensions+and+tooling"
        for /L %%i in (1,1,30) do (
            timeout /t 10 /nobreak >nul
            "%GIT%" ls-remote origin >nul 2>&1 && goto :repo_ready
            echo   waiting for repo creation... %%i/30
        )
        echo Gave up waiting. Create the repo, then rerun me.
        pause & exit /b 1
    )
)
:repo_ready

echo Pushing to GitHub...
ipconfig /flushdns >nul 2>&1
"%GIT%" push -4 -u origin main
if errorlevel 1 (
    echo First try failed - retrying once...
    timeout /t 5 /nobreak >nul
    ipconfig /flushdns >nul 2>&1
    "%GIT%" push -4 -u origin main
)
if errorlevel 1 (
    echo.
    echo Push failed. Usual causes:
    echo   - repo not created yet on github.com  ^(create it, then rerun me^)
    echo   - a browser sign-in window appeared   ^(approve it, then rerun me^)
) else (
    echo.
    echo Done - https://github.com/jrlabanza/forge-neo-toolkit
)
pause
