from typing import Literal
from dataclasses import dataclass

from .base import GameStateField, GameStateSchema, GameStateSnapshot, GameStateValue


type MemoryGStatePythonType = int | float | bool | str | None


type MemoryGStateNumpyType = Literal[
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
    "str",
]


class MemoryGameStateField(GameStateField[MemoryGStateNumpyType]):
    name: str
    dtype: MemoryGStateNumpyType


class MemoryGameStateSchema(GameStateSchema[MemoryGameStateField]):
    fields: tuple[MemoryGameStateField, ...]


@dataclass(frozen=True, slots=True)
class MemoryGameStateValue(GameStateValue[MemoryGStatePythonType]):
    pass


@dataclass(frozen=True, slots=True)
class MemoryGameStateSnapshot(GameStateSnapshot[MemoryGStatePythonType]):
    pass
