from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, cast

from data.models.game_state import (
    GameStateField,
    GameStateSchema,
    GameStateSnapshot,
    GameStateValue,
)

RawGameStatePythonType = int | float | bool | str | None

MemoryGameStateType = Literal[
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
    "utf8",
    "utf16",
    "utf8_string",
    "utf16le_string",
]

RawGameStateNumpyType = Literal[
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

MEMORY_DTYPE_OVERRIDES: dict[
    MemoryGameStateType,
    RawGameStateNumpyType,
] = {
    "utf8": "str",
    "utf16": "str",
    "utf8_string": "str",
    "utf16le_string": "str",
}


class GameStateReader(ABC):
    @property
    @abstractmethod
    def schema(self) -> GameStateSchema: ...

    @abstractmethod
    def read(self) -> RawGameStateSnapshot: ...


class RawGameStateField(GameStateField[RawGameStateNumpyType]):
    name: str
    dtype: RawGameStateNumpyType


class RawGameStateSchema(GameStateSchema[RawGameStateField]):
    fields: tuple[RawGameStateField, ...]


@dataclass(frozen=True, slots=True)
class RawGameStateValue(GameStateValue[RawGameStatePythonType]):
    pass


@dataclass(frozen=True, slots=True)
class RawGameStateSnapshot(GameStateSnapshot[RawGameStatePythonType]):
    pass


def to_raw_numpy_dtype(memory_dtype: MemoryGameStateType) -> RawGameStateNumpyType:
    if memory_dtype not in MEMORY_DTYPE_OVERRIDES:
        return cast(RawGameStateNumpyType, memory_dtype)
    
    return MEMORY_DTYPE_OVERRIDES[memory_dtype]