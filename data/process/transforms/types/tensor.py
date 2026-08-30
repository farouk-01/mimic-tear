from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from graph.base import Value
from graph.types.tensor import TensorNode
from data.process.transforms.base import Transform


class TensorTransform(Transform[Tensor]):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformNode(TensorNode):
    transform: TensorTransform

    @property
    def inputs(self) -> tuple[Value, ...]:
        return tuple(Value(name) for name in self.transform.inputs)

    @property
    def outputs(self) -> tuple[Value, ...]:
        return (Value(self.transform.output),)

    def execute(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        return (self.transform(*inputs),)


class Ratio(TensorTransform):
    name: ClassVar[str] = "ratio"

    numerator: str
    denominator: str

    @property
    def inputs(self) -> tuple[str, str]:
        return self.numerator, self.denominator

    def __call__(self, numerator: Tensor, denominator: Tensor) -> Tensor:
        safe_denominator = torch.where(
            denominator != 0,
            denominator,
            torch.ones_like(denominator),
        )

        return torch.where(
            denominator != 0,
            numerator / safe_denominator,
            torch.zeros_like(numerator),
        )
