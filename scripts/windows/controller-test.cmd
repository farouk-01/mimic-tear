@echo off
setlocal

for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI\"
set "PYTHONPATH=%PROJECT_ROOT%src"
set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Project virtual environment not found: %PYTHON_EXE% 1>&2
    exit /b 1
)

"%PYTHON_EXE%" -m ai_player.cli.controller_probe %*
exit /b %ERRORLEVEL%
