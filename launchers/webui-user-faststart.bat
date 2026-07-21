@echo off

:: ============================================================================
:: FAST-START launcher - daily driver
:: ============================================================================
:: Identical GPU tuning to webui-user.bat, plus flags that skip the
:: environment preparation Forge repeats on every launch (git checks,
:: pip dependency scans, extension install.py runs). Saves roughly
:: 15-40 seconds per start on this install.
::
:: WHEN NOT TO USE THIS:
::   - right after installing/updating an extension  -> run webui-user.bat
::     once so its dependencies get installed, then come back to this one.
::   - after a Forge/torch update                    -> same thing.
::
:: If the UI ever fails to start from here, just launch webui-user.bat -
:: it repairs the environment automatically.
:: ============================================================================

cd /D "%~dp0"
set PYTHON="%~dp0venv\Scripts\python.exe"
if exist "%~dp0PortableGit\cmd\git.exe" set "GIT=%~dp0PortableGit\cmd\git.exe"

set COMMANDLINE_ARGS=--reserve-vram 2 --pin-shared-memory --cuda-malloc --cuda-stream --skip-python-version-check --disable-gpu-warning --api --skip-prepare-environment --skip-torch-cuda-test --skip-version-check

call webui.bat
