@echo off
setlocal

for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI\"
set "AI_PLAYER_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
if not exist "%AI_PLAYER_PYTHON%" (
  echo Python environment not found: %AI_PLAYER_PYTHON%
  echo Create .venv and install requirements first.
  exit /b 1
)

set "PYTHONPATH=%PROJECT_ROOT%src"
"%AI_PLAYER_PYTHON%" -m ai_player.cli.record ^
  --theme gameplay/boss/soldier-of-godrick ^
  --tag soldier-of-godrick ^
  --split train ^
  --boss-loop ^
  --no-preview ^
  %*

