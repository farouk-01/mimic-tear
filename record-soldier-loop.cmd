@echo off
setlocal

set "AI_PLAYER_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%AI_PLAYER_PYTHON%" (
  echo Python environment not found: %AI_PLAYER_PYTHON%
  echo Create .venv and install requirements first.
  exit /b 1
)

"%AI_PLAYER_PYTHON%" "%~dp0recorder\record.py" ^
  --theme combat/tutorial ^
  --tag soldier-of-godrick ^
  --split train ^
  --boss-loop ^
  --no-preview ^
  %*

