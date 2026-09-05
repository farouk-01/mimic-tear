from typing import Self

from pydantic import BaseModel, ConfigDict

from data.models.tensor import TensorSchema
from data.process.transforms.tensor import TensorTransform
from configs.transforms.controller import GAMEPAD_TRANSFORMS


class ControllerConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )

    tensor_controller_schema: TensorSchema
    transforms: tuple[TensorTransform, ...] = ()

    @classmethod
    def load(cls, *, schema: TensorSchema) -> Self:
        transforms = GAMEPAD_TRANSFORMS
        return cls(tensor_controller_schema=schema, transforms=transforms)
