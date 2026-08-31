from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor
from torchvision.transforms import v2

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


class Clamp(TensorTransform):
    name: ClassVar[str] = "clamp"

    input: str
    min: float
    max: float

    @property
    def inputs(self) -> tuple[str]:
        return (self.input,)

    def __call__(self, input: Tensor) -> Tensor:
        return torch.clamp(input, min=self.min, max=self.max)


class Resize(TensorTransform):
    name: ClassVar[str] = "resize"

    input: str
    width: int
    height: int
    antialias: bool = True

    @property
    def inputs(self) -> tuple[str]:
        return (self.input,)

    def __call__(self, input: Tensor) -> Tensor:
        return v2.Resize(
            size=(self.height, self.width),
            antialias=self.antialias,
        )(input)


class Normalize(TensorTransform):
    name: ClassVar[str] = "normalize"

    input: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @property
    def inputs(self) -> tuple[str]:
        return (self.input,)

    def __call__(self, input: Tensor) -> Tensor:
        return v2.Normalize(mean=self.mean, std=self.std)(input)


class ToDtype(TensorTransform):
    name: ClassVar[str] = "to_dtype"

    input: str
    dtype: torch.dtype
    scale: bool = False

    @property
    def inputs(self) -> tuple[str]:
        return (self.input,)

    def __call__(self, input: Tensor) -> Tensor:
        return v2.ToDtype(
            dtype=self.dtype,
            scale=self.scale,
        )(input)