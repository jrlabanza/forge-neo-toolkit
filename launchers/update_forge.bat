@echo off
setlocal
REM ============================================================================
REM update_forge.bat — pull the latest Forge Neo from upstream (Haoming02)
REM Verified 2026-07-20: 75 commits behind, 0 ahead -> clean fast-forward;
REM extension APIs and launch flags unaffected; requirements = minor bumps.
REM ============================================================================
cd /D "%~dp0"

REM ---- bootstrap: extract the downloaded PortableGit on first run ----
if not exist "%~dp0PortableGit\cmd\git.exe" if exist "%~dp0PortableGit-64-bit.7z.exe" (
    echo First run: extracting PortableGit ^(one-time, ~20s^)...
    start /wait "" "%~dp0PortableGit-64-bit.7z.exe" -y -o"%~dp0PortableGit"
)

REM ---- locate git (local PortableGit, system PATH, or Stability Matrix) ----
set "GIT="
if exist "%~dp0PortableGit\cmd\git.exe" set "GIT=%~dp0PortableGit\cmd\git.exe"
if not defined GIT where git >nul 2>&1 && set "GIT=git"
if not defined GIT if exist "%~dp0..\..\Assets\PortableGit\cmd\git.exe" set "GIT=%~dp0..\..\Assets\PortableGit\cmd\git.exe"
if not defined GIT if exist "%~dp0..\..\Assets\PortableGit\bin\git.exe" set "GIT=%~dp0..\..\Assets\PortableGit\bin\git.exe"
if not defined GIT if exist "%LOCALAPPDATA%\StabilityMatrix\Assets\PortableGit\cmd\git.exe" set "GIT=%LOCALAPPDATA%\StabilityMatrix\Assets\PortableGit\cmd\git.exe"
if not defined GIT if exist "%APPDATA%\StabilityMatrix\Assets\PortableGit\cmd\git.exe" set "GIT=%APPDATA%\StabilityMatrix\Assets\PortableGit\cmd\git.exe"
if not defined GIT (
    echo Could not find git - neither on PATH nor in Stability Matrix's assets.
    echo Install Git for Windows with:   winget install --id Git.Git
    echo then rerun this script.
    pause & exit /b 1
)
echo Using git: %GIT%

REM Trust repos on this drive (folder owner SID predates this Windows user)
"%GIT%" config --global --add safe.directory "*" >nul 2>&1

echo.
echo Cleaning stale git locks and sandbox leftovers...
if exist ".git\index.lock" del /f /q ".git\index.lock"
if exist ".git\objects\maintenance.lock" del /f /q ".git\objects\maintenance.lock"
for /d %%D in (".git\objects\*") do del /f /q "%%D\tmp_obj_*" 2>nul
if exist "_unlink_test.txt" del /f /q "_unlink_test.txt"
if exist "_rmdir_test" rmdir "_rmdir_test"

echo.
echo Creating safety backup branch (rollback point)...
"%GIT%" branch backup/pre-update-20260720 2>nul && echo   created: backup/pre-update-20260720 || echo   backup branch already exists, keeping it.

echo.
echo Fetching upstream...
"%GIT%" fetch upstream neo
if errorlevel 1 (
    echo Fetch failed - check your internet connection and rerun.
    pause & exit /b 1
)

echo.
echo Applying update (fast-forward only, cannot conflict)...
"%GIT%" merge --ff-only upstream/neo
if errorlevel 1 (
    echo.
    echo Merge was not a clean fast-forward. Nothing was changed.
    echo Run this to see what's in the way:   "%GIT%" status
    pause & exit /b 1
)

echo.
echo Updating your GitHub fork too (optional - OK if this fails)...
"%GIT%" push origin neo

echo.
echo ============================================================
echo  Update complete. IMPORTANT NEXT STEP:
echo  Launch ONCE with webui-user.bat (NOT the faststart one) so
echo  the updated Python dependencies get installed.
echo.
echo  Rollback if anything misbehaves:
echo     git reset --hard backup/pre-update-20260720
echo ============================================================
pause
