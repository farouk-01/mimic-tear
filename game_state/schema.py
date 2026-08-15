from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

GameStateValue = int | float | bool

class GameStateField(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    
    name: str
    type: str
    required: bool = False
    scope: str | None = None


@dataclass(frozen=True, slots=True)
class GameStateSchema:
    fields: tuple[GameStateField, ...]

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(
            field.name
            for field in self.fields
        )

    @property
    def feature_count(self) -> int:
        return len(self.fields)

    def has_feature(self, name: str) -> bool:
        return any(
            field.name == name
            for field in self.fields
        )