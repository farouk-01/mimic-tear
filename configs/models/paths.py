from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    recordings: Path = Path("recordings")
    artifacts: Path = Path("artifacts")

    @classmethod
    def load(cls, raw_paths: dict) -> PathsConfig:
        for key, path_str in raw_paths.items():
            raw_paths[key] = Path(path_str)
        return cls.model_validate(raw_paths)
