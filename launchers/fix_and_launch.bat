@echo off
REM One-time repair: a stale python process from an interrupted launch is
REM holding venv DLLs (cv2.pyd) hostage, so pip can't update opencv.
REM Kill leftover pythons, then start Forge normally.
cd /D "%~dp0"

echo Stopping stale python processes...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im pythonw.exe >nul 2>&1
timeout /t 3 /nobreak >nul

echo Starting Forge (dependency install resumes automatically)...
call webui-user.bat
