from dataclasses import dataclass
from typing import Generic, Self, TYPE_CHECKING, overload, Literal
from functools import cached_property
from collections.abc import Iterable, Sequence
from pathlib import Path
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, model_validator

from data.constants import DEFAULT_COLUMNS

if TYPE_CHECKING:
    import pyarrow as pa


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
    def width(self) -> int:
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

        cols = []
        cols.append(pa.field(DEFAULT_COLUMNS.frame_index, pa.int64()))
        cols.append(pa.field(DEFAULT_COLUMNS.capture_timestamp_ns, pa.int64()))

        for field in self.fields:
            cols.append(pa.field(field.name, pa.from_numpy_dtype(field.dtype)))

        return pa.schema(cols)

    @classmethod
    def from_json(cls, source: str | Path | dict) -> Self:
        if isinstance(source, dict):
            schema_dict = source
        elif isinstance(source, (str, Path)):
            from utils.files import load_json

            schema_path = Path(source)

            if not schema_path.is_file():
                raise FileNotFoundError(f"Schema file does not exist: {schema_path}")

            schema_dict = load_json(schema_path)
        else:
            raise TypeError(
                f"Invalid schema source type: {type(source)}. "
                "Expected dict, str, or Path."
            )

        return cls.model_validate(
            {
                **schema_dict,
                "fields": tuple(schema_dict["fields"]),
            }
        )


class Snapshot[F: Field]:
    def __init__(
        self,
        values: Iterable[object],
        schema: Schema[F],
        timestamp_ns: int,
    ) -> None:
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns cannot be negative")
        
        self.values = tuple(values)
        self.schema = schema
        self.timestamp_ns = timestamp_ns

    def __getitem__(self, name: str) -> object:
        if name not in self.columns:
            raise KeyError(f"Snapshot does not contain field: {name}")

        index = self.schema.index(name)
        return self.values[index]

    @property
    def columns(self) -> ...:
        return self.schema.feature_names

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, object],
        schema: Schema[F],
        timestamp_ns: int,
        nullable: bool = False,
    ) -> Self:
        ordered_data = []
        for col in schema.fields:
            if col.name not in values and nullable:
                ordered_data.append(None)
            else:
                ordered_data.append(values[col.name])

        if len(ordered_data) != schema.width:
            raise ValueError(f"Expected {schema.width} values, got {len(ordered_data)}")

        return cls(tuple(ordered_data), schema, timestamp_ns)

    def to_dict(self) -> dict[str, object]:
        py_data = {}

        for i, col in enumerate(self.schema.fields):
            py_data[col.name] = self.values[i]

        return py_data
