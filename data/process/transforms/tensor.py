from collections.abc import Mapping
from typing import ClassVar, Literal

import torch
from torch import Tensor
from torchvision.transforms import v2
from pydantic import ConfigDict

from data.process.transforms.base import Transform


class TensorTransform(Transform[Tensor]):
    pass


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
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
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


class Contiguous(TensorTransform):
    name: ClassVar[str] = "contiguous"

    input: str

    @property
    def inputs(self) -> tuple[str]:
        return (self.input,)

    def __call__(self, input: Tensor) -> Tensor:
        return input.contiguous()


class Delta(TensorTransform):
    name: ClassVar[str] = "delta"

    lhs: str
    rhs: str

    @property
    def inputs(self) -> tuple[str, str]:
        return self.lhs, self.rhs

    def __call__(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs - rhs
    

class Lag(TensorTransform):
    name: ClassVar[str] = "lag"

    input: str
    periods: int = 1
    fill: Literal["first", "zero"] = "first"

    @property
    def inputs(self) -> tuple[str]:
        return (self.input,)

    def __call__(self, input: Tensor) -> Tensor:
        if self.periods <= 0:
            raise ValueError("periods must be greater than zero")

        if input.ndim == 0:
            raise ValueError("Lag requires a tensor with a time dimension")

        result = torch.empty_like(input)

        if self.fill == "first":
            result[: self.periods] = input[0]
        else:
            result[: self.periods] = 0

        if self.periods < input.shape[0]:
            result[self.periods :] = input[: -self.periods]

        return result
