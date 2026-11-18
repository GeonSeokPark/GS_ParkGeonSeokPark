@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1" %*
set "DEEPPROTECT_EXIT=%ERRORLEVEL%"
pause
exit /b %DEEPPROTECT_EXIT%
