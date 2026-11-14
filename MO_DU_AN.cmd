@echo off
cd /d "%~dp0"
where code >nul 2>nul
if errorlevel 1 (
  start "" explorer.exe "%~dp0."
) else (
  call code --new-window "%~dp0."
)
