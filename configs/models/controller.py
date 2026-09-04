from typing import Self

from pydantic import BaseModel, ConfigDict

from data.models.tensor import TensorSchema
from data.process.transforms.types.tensor import TransformNode
from graph.base import Graph, Plan, Value

from configs.transforms.controller import GAMEPAD_TRANSFORMS


class ControllerConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )

    tensor_controller_schema: TensorSchema
    plan: Plan

    @classmethod
    def load(cls, *, schema: TensorSchema) -> Self:
        graph = Graph()

        for transform in GAMEPAD_TRANSFORMS:
            graph.add(TransformNode(transform=transform))

        outputs = tuple(
            graph.value(field.name) for field in schema.fields if field.is_model_input
        )

        plan = graph.resolve(outputs)

        return cls(tensor_controller_schema=schema, plan=plan)
