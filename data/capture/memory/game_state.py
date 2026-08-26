from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict

GameStateValue = int | float | bool | str | None


class GameStateReader(ABC):
    @property
    @abstractmethod
    def schema(self) -> GameStateSchema: ...

    @abstractmethod
    def read(self) -> GameStateSnapshot: ...


class GameStateField(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    type: str
    required: bool = False
    scope: str | None = None


@dataclass(frozen=True, slots=True)
class GameStateSchema:
    fields: tuple[GameStateField, ...]

    def index(self, name: str) -> int:
        for index, field in enumerate(self.fields):
            if field.name == name:
                return index

        raise KeyError(f"Unknown game-state field: {name}")

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def feature_count(self) -> int:
        return len(self.fields)

    def has_feature(self, name: str) -> bool:
        return any(field.name == name for field in self.fields)


@dataclass(frozen=True, slots=True)
class GameStateSnapshot:
    values: dict[str, GameStateValue]  # GameStateValue is the type of the value

    def get(self, name: str) -> GameStateValue:
        return self.values[name]

    def ordered_values(self, schema: GameStateSchema) -> tuple[GameStateValue, ...]:
        missing = [feature for feature in schema.features if feature not in self.values]

        if missing:
            raise ValueError(f"Snapshot is missing game-state features: {missing}")

        return tuple(self.values[feature] for feature in schema.features)
