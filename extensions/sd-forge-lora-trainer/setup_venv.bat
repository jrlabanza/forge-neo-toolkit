@echo off
REM ===================================================================
REM sd-forge-lora-trainer / setup_venv.bat
REM
REM One-time bootstrap. Double-click this file to install kohya-ss/sd-scripts
REM and provision its sandboxed venv.
REM
REM Python selection (in this order of preference):
REM   1. F:\Data\Assets\Python310\python.exe        (kohya recommends 3.10)
REM   2. py -3.10                                    (Windows launcher)
REM   3. py -3.11                                    (also widely compatible)
REM   4. Forge's venv python.exe                     (fallback)
REM
REM Safe to re-run: each step is idempotent.
REM ===================================================================

setlocal enabledelayedexpansion
cd /D "%~dp0"

set "BOOTSTRAP_PYTHON="

if exist "F:\Data\Assets\Python310\python.exe" (
    set "BOOTSTRAP_PYTHON=F:\Data\Assets\Python310\python.exe"
    echo Using Python 3.10 at F:\Data\Assets\Python310\python.exe
    goto :have_python
)

py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set "BOOTSTRAP_PYTHON=py -3.10"
    echo Using py -3.10 ^(Windows launcher^)
    goto :have_python
)

py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set "BOOTSTRAP_PYTHON=py -3.11"
    echo Using py -3.11 ^(Windows launcher^)
    goto :have_python
)

set "FORGE_PYTHON=..\..\venv\Scripts\python.exe"
if exist "%FORGE_PYTHON%" (
    set "BOOTSTRAP_PYTHON=%FORGE_PYTHON%"
    echo WARNING: falling back to Forge's python at %FORGE_PYTHON%.
    echo          kohya recommends Python 3.10; newer Python may have missing wheels.
    goto :have_python
)

echo.
echo ERROR: no suitable Python interpreter found.
echo.
pause
exit /b 1

:have_python
echo.
echo Running setup_bootstrap.py with: %BOOTSTRAP_PYTHON%
echo (Expect 5-15 minutes. The window stays open until you press a key at the end.)
echo.

%BOOTSTRAP_PYTHON% setup_bootstrap.py
set RC=%errorlevel%

echo.
if %RC%==0 (
    echo === Setup finished successfully. ===
) else (
    echo === Setup FAILED with exit code %RC%. Scroll up to see what went wrong. ===
)
echo.
pause
