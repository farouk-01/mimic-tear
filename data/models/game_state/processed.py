from typing import Literal, Self

from pydantic import Field, model_validator
import torch
from .base import GameStateField, GameStateSchema

type FieldKind = Literal[
    "binary",
    "categorical",
    "continuous",
    "discrete",
    "constant",
    "ordinal",
    "nominal",
]

type DataType = Literal[
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "bool",
    "str",
]

TORCH_DTYPES: dict[DataType, torch.dtype] = {
    "bool": torch.bool,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

UNSUPPORTED_TORCH_DTYPES = {"str"}

GameStateNullValue = int | float | bool


# a non nullable field results in the current frame being dropped
# while a nullable field will use fill_value, nullable fields
# have a pattern to let the model know that the value is missing
#
# e.g lock_on_active = False, then enemy_health = unknown
# so model can learn from the pattern that
# when lock_on_active is False, enemy_health is fill_value
class ProcessedGameStateField(GameStateField):
    name: str
    nullable: bool = Field(default=False)
    fill_value: GameStateNullValue | None = Field(default=None)
    kind: FieldKind
    dtype: DataType
    derived: bool = Field(default=False)
    required: bool = Field(default=True)
    is_metadata: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_dtype(self) -> Self:
        if not self.is_metadata and self.dtype in UNSUPPORTED_TORCH_DTYPES:
            raise ValueError(f"Unsupported dtype for non-metadata field: {self.dtype}")

        return self

    @model_validator(mode="after")
    def validate_nullable(self) -> Self:
        if self.nullable and self.fill_value is None:
            raise ValueError("Nullable field must have a fill_value specified")

        return self


class ProcessedGameStateSchema(GameStateSchema):
    fields: tuple[ProcessedGameStateField, ...]

    def get_required_fields_names(
        self, include_derived: bool = True
    ) -> tuple[str, ...]:
        required: tuple[str, ...] = ()
        for field in self.fields:
            if include_derived:
                if not field.is_metadata and field.required:
                    required += (field.name,)
            else:
                if not (field.is_metadata or field.derived) and field.required:
                    required += (field.name,)

        if len(required) == 0:
            raise ValueError("No required fields found in schema")

        return required

    def get_required_fields(
        self, include_derived: bool = True
    ) -> tuple[ProcessedGameStateField, ...]:
        required: tuple[ProcessedGameStateField, ...] = ()
        for field in self.fields:
            if include_derived:
                if not field.is_metadata and field.required:
                    required += (field,)
            else:
                if not (field.is_metadata or field.derived) and field.required:
                    required += (field,)

        if len(required) == 0:
            raise ValueError("No required fields found in schema")

        return required

    def get_required_derived_fields(self) -> tuple[ProcessedGameStateField, ...]:
        required: tuple[ProcessedGameStateField, ...] = ()
        for field in self.fields:
            if not field.is_metadata and field.derived and field.required:
                required += (field,)

        if len(required) == 0:
            raise ValueError("No required derived fields found in schema")

        return required

    @property
    def required_feature_count(self) -> int:
        count = 0
        for field in self.fields:
            if not field.is_metadata and field.required:
                count += 1

        if count == 0:
            raise ValueError("No required raw fields found in schema")

        return count
