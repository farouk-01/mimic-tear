from abc import ABC, abstractmethod
from typing import Literal, cast

from data.models.game_state.memory import (
    MemoryGStateNumpyType,
    MemoryGameStateSchema,
    MemoryGameStateSnapshot,
)


type MemoryGStateType = Literal[
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


MEMORY_DTYPE_OVERRIDES: dict[
    MemoryGStateType,
    MemoryGStateNumpyType,
] = {
    "utf8": "str",
    "utf16": "str",
    "utf8_string": "str",
    "utf16le_string": "str",
}


class GameStateReader(ABC):
    @property
    @abstractmethod
    def schema(self) -> MemoryGameStateSchema: ...

    @abstractmethod
    def read(self) -> MemoryGameStateSnapshot: ...


def to_raw_numpy_dtype(memory_dtype: MemoryGStateType) -> MemoryGStateNumpyType:
    if memory_dtype not in MEMORY_DTYPE_OVERRIDES:
        return cast(MemoryGStateNumpyType, memory_dtype)

    return MEMORY_DTYPE_OVERRIDES[memory_dtype]
