from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True, slots=True)
class EncodingPaths:
    root: Path = Path("configs/encodings")
    game_state: Path = root / "game_state"


class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    training_recordings: Path
    validation_recordings: Path
    artifacts: Path

    encodings: EncodingPaths = EncodingPaths()

    @classmethod
    def load(cls, paths_cfg: dict) -> PathsConfig:
        for key, path_str in paths_cfg.items():
            paths_cfg[key] = Path(path_str)

        return cls.model_validate(paths_cfg)
