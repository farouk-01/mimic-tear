from __future__ import annotations

from pathlib import Path
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ConfigDomain(StrEnum):
    GSTATE = "game_state"


class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    recordings: Path
    artifacts: Path

    schemas: Path = Path("configs/schemas")
    memory_schemas: Path = schemas / "memory"
    processed_schemas: Path = schemas / "processed"

    encodings: Path = Path("configs/encodings")

    @classmethod
    def load(cls, paths_cfg: dict) -> PathsConfig:
        for key, path_str in paths_cfg.items():
            paths_cfg[key] = Path(path_str)

        return cls.model_validate(paths_cfg)

    def encodings_for(self, domain: ConfigDomain) -> Path:
        return self.encodings / domain

    def memory_schemas_for(self, domain: ConfigDomain) -> Path:
        return self.memory_schemas / domain

    def processed_schemas_for(self, domain: ConfigDomain) -> Path:
        return self.processed_schemas / domain

    def memory_schema(self, domain: ConfigDomain, version: str) -> Path:
        return self.memory_schemas_for(domain) / f"v{version}" / "schema.json"
    
    def processed_schema(self, domain: ConfigDomain, version: str) -> Path:
        return self.processed_schemas_for(domain) / f"v{version}" / "schema.json"