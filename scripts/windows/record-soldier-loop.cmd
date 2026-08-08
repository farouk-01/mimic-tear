@echo off
setlocal

for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI\"
set "MIMIC_TEAR_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
if not exist "%MIMIC_TEAR_PYTHON%" (
  echo Python environment not found: %MIMIC_TEAR_PYTHON%
  echo Create .venv and install requirements first.
  exit /b 1
)

set "PYTHONPATH=%PROJECT_ROOT%"
"%MIMIC_TEAR_PYTHON%" -m mimic_tear.cli.record ^
  --theme gameplay/boss/soldier-of-godrick ^
  --tag soldier-of-godrick ^
  --split train ^
  --boss-loop ^
  --no-preview ^
  %*

