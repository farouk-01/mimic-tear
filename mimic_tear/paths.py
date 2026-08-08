"""Stable paths to project-owned runtime resources."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_ROOT = PROJECT_ROOT / "configs"
DEFAULT_GAME_STATE_PROFILE = CONFIG_ROOT / "game-state" / "elden-ring.json"

