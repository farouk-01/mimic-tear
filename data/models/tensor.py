from typing import Literal, Self

from pydantic import model_validator
import torch

from data.models.schema import Field, Schema

type FieldKind = Literal[
    "binary",
    "categorical",
    "continuous",
    "discrete",
    "constant",
    "ordinal",
    "nominal",
]

type TensorType = Literal[
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
]

TORCH_DTYPES: dict[TensorType, torch.dtype] = {
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

FillValueType = int | float | bool

class TensorField(Field[TensorType]):
    nullable: bool = False
    fill_value: FillValueType | None = None
    kind: FieldKind
    encoding: str | None = None
    
    is_derived: bool = False
    is_model_input: bool = True
    is_metadata: bool = False

    @model_validator(mode="after")
    def validate_nullable(self) -> Self:
        if self.nullable and self.fill_value is None:
            raise ValueError("Nullable field must have a fill_value specified")

        return self

    @model_validator(mode="after")
    def validate_nominal_encoding(self) -> Self:
        if self.kind == "nominal" and self.encoding is None:
            raise ValueError("Nominal fields must specify an encoding")

        elif self.kind != "nominal" and self.encoding is not None:
            raise ValueError("Only nominal fields can specify an encoding")

        return self


class TensorSchema(Schema[TensorField]):
    pass