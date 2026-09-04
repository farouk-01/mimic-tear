from typing import Self

from pydantic import BaseModel, Field, ConfigDict


class VersionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    gstate_tensor_schema: str
    controller_tensor_schema: str
    
    gstate_memory_schema: str


    @classmethod
    def load(cls, version_cfg: dict) -> Self:
        return cls.model_validate(version_cfg)