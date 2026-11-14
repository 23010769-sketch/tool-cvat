@echo off
setlocal
set "CVAT_PACKAGE_ROOT=%~dp0"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NoExit -Command "Set-Location -LiteralPath $env:CVAT_PACKAGE_ROOT; & '.\.venv\Scripts\python.exe' -X utf8 '.\annotate_job908.py'"
endlocal
