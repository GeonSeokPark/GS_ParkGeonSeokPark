@echo off
setlocal
cd /d "%~dp0.."
title DeepProtect Local Server
set "DEEPPROTECT_PYTHON=.venv\Scripts\python.exe"
if not exist "%DEEPPROTECT_PYTHON%" set "DEEPPROTECT_PYTHON=..\..\work\testenv\Scripts\python.exe"
if not exist "%DEEPPROTECT_PYTHON%" (
    echo Run setup-windows.cmd first. See README-LAPTOP.ko.md.
    pause
    exit /b 1
)
if exist ".venv\Scripts\python.exe" set "TORCH_HOME=%CD%\.runtime\torch"
"%DEEPPROTECT_PYTHON%" -u -m web_app.server %*
set "DEEPPROTECT_EXIT=%ERRORLEVEL%"
if not "%DEEPPROTECT_EXIT%"=="0" pause
exit /b %DEEPPROTECT_EXIT%
