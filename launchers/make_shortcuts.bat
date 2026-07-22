@echo off
REM Creates desktop shortcuts for the Forge launchers (rerun any time).
set "HERE=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws=New-Object -ComObject WScript.Shell; $d=[Environment]::GetFolderPath('Desktop'); $t=$env:HERE; foreach($p in @(@('Forge Neo','webui-user.bat'),@('Forge Neo (fast start)','webui-user-faststart.bat'))){ $s=$ws.CreateShortcut((Join-Path $d ($p[0]+'.lnk'))); $s.TargetPath=(Join-Path $t $p[1]); $s.WorkingDirectory=$t; $s.IconLocation=((Join-Path $t 'venv\Scripts\python.exe')+',0'); $s.Description='Stable Diffusion WebUI Forge Neo'; $s.Save() }; Write-Host 'Desktop shortcuts created.'"
echo Done - check your desktop.
timeout /t 4 /nobreak >nul
