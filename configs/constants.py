from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
import tomllib
import copy
from typing import Any

_config: Path = Path("configs/settings/config.toml")
_override: Path = Path("configs/settings/config.override.toml")
_recordings_directory: Path = Path("recordings")
_artifacts_directory: Path = Path("artifacts")

class ConfigKey(StrEnum):
    LOGGING = "logging"
    HYPERPARAMETERS = "hyperparameters"
    RECORDING = "recording"
    MODEL = "model"
    CAPTURE = "capture"
    DATA = "data"
    DATALOADER = "data_loader"

@dataclass(frozen=True, slots=True)
class RawConfig:
    config: dict = field(default_factory=lambda: _load())
    game_state: str = field(default_factory=lambda: _dump_json("configs/settings/game_state/elden-ring.json"))
    recordings_directory: Path = _recordings_directory
    artifacts_directory: Path = _artifacts_directory

    @property
    def regular_logging(self) -> dict[str, Any]:
        return self.config[ConfigKey.LOGGING]["regular"]

    @property
    def perf_logging(self) -> dict[str, Any]:
        return self.config[ConfigKey.LOGGING]["perf_logger"]

    @property
    def profiling(self) -> dict[str, Any]:
        return self.config[ConfigKey.LOGGING]["profiling"]

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return self.config[ConfigKey.HYPERPARAMETERS]

    @property
    def controller_weights(self) -> dict[str, float]:
        return self.config[ConfigKey.HYPERPARAMETERS]["controller_weights"]

    @property
    def vision(self) -> dict[str, Any]:
        return self.config[ConfigKey.MODEL]["vision"]

    @property
    def temporal(self) -> dict[str, Any]:
        return self.config[ConfigKey.MODEL]["temporal"]

    @property
    def model_game_state(self) -> dict[str, Any]:
        return self.config[ConfigKey.MODEL]["game_state"]

    @property
    def fusion(self) -> dict[str, Any]:
        return self.config[ConfigKey.MODEL]["fusion"]

    @property
    def controller(self) -> dict[str, Any]:
        return self.config[ConfigKey.MODEL]["controller"]

    @property
    def data_loader(self) -> dict[str, Any]:
        return self.config[ConfigKey.DATALOADER]

    @property
    def transform_frames(self) -> dict[str, Any]:
        return self.config[ConfigKey.DATA]["transforms"]["frames"]

    @property
    def stores_frames(self) -> dict[str, Any]:
        return self.config[ConfigKey.DATA]["stores"]["frames"]

    @property
    def recording_files(self) -> dict[str, Any]:
        return self.config[ConfigKey.RECORDING]["files"]

    @property
    def recording_video(self) -> dict[str, Any]:
        return self.config[ConfigKey.RECORDING]["video"]

    @property
    def recording_controller(self) -> dict[str, Any]:
        return self.config[ConfigKey.RECORDING]["controller"]

    @property
    def capture_screen(self) -> dict[str, Any]:
        return self.config[ConfigKey.CAPTURE]["screen"]

    @property
    def capture_gamepad(self) -> dict[str, Any]:
        return self.config[ConfigKey.CAPTURE]["gamepad"]


def _load() -> dict:
    def merge(base: dict, override: dict) -> None:
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                merge(base[k], v)
            else:
                base[k] = copy.deepcopy(v)

    with open(_config, "rb") as f:
        config = tomllib.load(f)

    if _override.exists():
        with open(_override, "rb") as f:
            override = tomllib.load(f)

        merge(config, override)

    return config


def _dump_json(path: str | Path) -> str:
    import json

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.dumps(json.load(f))

    return data