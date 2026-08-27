from __future__ import annotations

import copy
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("configs/config.toml")
DEFAULT_OVERRIDE_PATH = Path("configs/config.override.toml")

DEFAULT_GAME_VERSION = "v1.16.2"
DEFAULT_GAME_STATE_PROFILE_PATH = Path(
    f"configs/game_state/raw/versions/{DEFAULT_GAME_VERSION}.json"
)

EXPECTED_GAME_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RawConfig:
    settings: dict[str, Any]
    game_state: dict[str, Any]


def load_raw_config(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    override_path: str | Path | None = DEFAULT_OVERRIDE_PATH,
    game_state_path: str | Path = DEFAULT_GAME_STATE_PROFILE_PATH,
) -> RawConfig:
    settings = _load_toml(config_path)

    if override_path is not None:
        override_path = Path(override_path)

        if override_path.exists():
            override = _load_toml(override_path)
            _merge(settings, override)

    game_state = _load_json(game_state_path)

    return RawConfig(
        settings=settings,
        game_state=game_state,
    )


def load_expected_game_state_schema(
    version: int = EXPECTED_GAME_STATE_SCHEMA_VERSION,
) -> dict[str, str]:
    path = Path(f"configs/game_state/expected/versions/v{version}.json")

    if not path.exists():
        raise FileNotFoundError(
            f"Expected game-state schema for version {version} not found at {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_toml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("rb") as file:
        return tomllib.load(file)


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
