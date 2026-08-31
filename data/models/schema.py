from dataclasses import dataclass
from typing import Generic, Self, TYPE_CHECKING
from functools import cached_property
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, model_validator

if TYPE_CHECKING:
    import pyarrow as pa


class Field[T](BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    dtype: T


class Schema[F: Field](BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    fields: tuple[F, ...]

    @model_validator(mode="after")
    def validate_unique_names(self) -> Self:
        names = [field.name for field in self.fields]

        if len(names) != len(set(names)):
            raise ValueError("Schema field names must be unique")

        return self

    @cached_property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @cached_property
    def fields_by_name(self) -> dict[str, F]:
        return {field.name: field for field in self.fields}

    @property
    def feature_count(self) -> int:
        return len(self.fields)

    def index(self, name: str) -> int:
        try:
            return self.feature_names.index(name)
        except ValueError:
            raise KeyError(f"Unknown game-state field: {name}")

    def get_field(self, name: str) -> F:
        try:
            return self.fields_by_name[name]
        except KeyError:
            raise ValueError(f"Field '{name}' does not exist in schema")

    def has_feature(self, name: str) -> bool:
        return name in self.feature_names

    def to_pyarrow_schema(self) -> pa.Schema:
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("index", pa.int64()),
                pa.field("timestamp_ns", pa.int64()),
                *(
                    pa.field(field.name, pa.from_numpy_dtype(field.dtype))
                    for field in self.fields
                ),
            ]
        )


