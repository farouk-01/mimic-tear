from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Self

from pydantic import model_validator

from data.models.schema import Field, Schema

type MemoryType = Literal[
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

type NumpyType = Literal[
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

NUMPY_OVERRIDES: dict[MemoryType, NumpyType] = {
    "utf8": "str",
    "utf16": "str",
    "utf8_string": "str",
    "utf16le_string": "str",
}

type PythonType = int | float | bool | str | None


class MemoryField(Field[NumpyType]):
    pass

class MemorySchema(Schema[MemoryField]):
    pass


def to_numpy_type(memory_dtype: MemoryType) -> NumpyType:
    if memory_dtype not in NUMPY_OVERRIDES:
        return memory_dtype  # type: ignore
    
    return NUMPY_OVERRIDES[memory_dtype]