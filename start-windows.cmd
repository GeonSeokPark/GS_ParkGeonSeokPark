@echo off
call "%~dp0web_app\start-windows.cmd" %*
exit /b %ERRORLEVEL%
