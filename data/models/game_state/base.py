from dataclasses import dataclass
from typing import TypeVar, Generic, Self, TYPE_CHECKING
from functools import cached_property
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import pyarrow as pa

FieldT = TypeVar("FieldT", bound="GameStateField")
DataTypeT = TypeVar("DataTypeT")


class GameStateField(BaseModel, Generic[DataTypeT]):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    dtype: DataTypeT


class GameStateSchema(BaseModel, Generic[FieldT]):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    fields: tuple[FieldT, ...]

    @cached_property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def feature_count(self) -> int:
        return len(self.fields)

    def index(self, name: str) -> int:
        try:
            return self.feature_names.index(name)
        except ValueError:
            raise KeyError(f"Unknown game-state field: {name}")

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


@dataclass(frozen=True, slots=True)
class GameStateValue(Generic[DataTypeT]):
    name: str
    value: DataTypeT


@dataclass(frozen=True, slots=True)
class GameStateSnapshot(Generic[DataTypeT]):
    values: tuple[GameStateValue[DataTypeT], ...]

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(value.name for value in self.values)

    def get(self, index: int) -> DataTypeT:
        return self.values[index].value

    def get_by_name(self, name: str) -> DataTypeT:
        try:
            index = self.names.index(name)
            return self.values[index].value
        except ValueError:
            raise KeyError(f"Unknown game-state field: {name}")

    def to_dict(self) -> dict[str, DataTypeT]:
        return {
            item.name: item.value
            for item in self.values
        }

    @classmethod
    def from_schema(
        cls,
        schema: GameStateSchema[FieldT],
        values: Iterable[GameStateValue[DataTypeT]],
    ) -> Self:
        values = tuple(values)
        values_by_name = {value.name: value for value in values}

        if len(values_by_name) != len(values):
            raise ValueError("Snapshot contains duplicate game-state fields")

        missing = [name for name in schema.feature_names if name not in values_by_name]

        if missing:
            raise ValueError(f"Snapshot is missing game-state features: {missing}")

        return cls(values=tuple(values_by_name[name] for name in schema.feature_names))
