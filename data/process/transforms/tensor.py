from abc import abstractmethod
from collections.abc import Mapping
from typing import ClassVar, Literal

import torch
from torch import Tensor
from torchvision.transforms import v2
from pydantic import ConfigDict, PositiveInt, model_validator

from data.process.transforms.base import Transform, TemporalTransform, Lookback

type TransformInput = str | TensorTransform


class TensorTransform(Transform[Tensor]):
    pass


class TemporalTensorTransform(TemporalTransform[Tensor]):
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


class Add(TensorTransform):
    name: ClassVar[str] = "add"

    lhs: str
    rhs: str

    @property
    def inputs(self) -> tuple[str, str]:
        return self.lhs, self.rhs

    def __call__(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs + rhs


class Lag(TemporalTensorTransform):
    name: ClassVar[str] = "lag"

    input: str
    fill: Literal["first", "zero"] = "first"

    @property
    def inputs(self) -> tuple[str]:
        return (self.input,)

    def __call__(self, input: Tensor) -> Tensor:
        if input.ndim == 0:
            raise ValueError("Lag requires a tensor with a time dimension")

        result = torch.empty_like(input)

        missing = self.periods - len(self.lookback)

        if missing > 0:
            if self.fill == "first":
                first_value = self.lookback[0] if self.lookback else input[0]
                self.lookback.extendleft([first_value] * missing)

            elif self.fill == "zero":
                value = torch.zeros_like(input[0])
                self.lookback.extendleft([value] * missing)

        required_fill_size = min(self.periods, input.size(0))
        fill = list(self.lookback)[:required_fill_size]

        result[:required_fill_size] = torch.stack(fill)

        if self.periods < input.size(0):
            result[self.periods :] = input[: -self.periods]

        self.lookback.extend(input)

        return result


class ConsecutiveCount(TemporalTensorTransform):
    name: ClassVar[str] = "consecutive_count"

    input: str
    max_count: PositiveInt | None = None
    periods: Literal[1] = 1 # type: ignore[reportIncompatibleVariableOverride]
    shift: Literal[None, 1] = None # maybe later make flexible but need modify periods too

    @property
    def inputs(self) -> tuple[str]:
        return (self.input,)

    def __call__(self, input: Tensor) -> Tensor:
        if input.ndim == 0:
            raise ValueError("ConsecutiveCount requires a tensor with a time dimension")

        if self.lookback:
            carry = self.lookback[-1]
        else:
            carry = torch.zeros(1, device=input.device, dtype=torch.int64)

        pressed = input.bool()

        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        cumsum = torch.cumsum(pressed, dim=0)

        # keep only the cumsum value of zeros
        # [1, 2, 2, 3, 3] -> [0, 0, 2, 0, 3]
        zeros = cumsum * pressed.logical_not()

        # propagate the zeros cumsum values until the next zero value
        # [0, 0, 2, 0, 3] -> [0, 0, 2, 2, 3]
        offsets = torch.cummax(zeros, dim=0).values

        # [1, 2, 2, 3, 3] - [0, 0, 2, 2, 3] -> [1, 2, 0, 1, 0]
        consecutive_count = cumsum - offsets

        if carry > 0:
            first_zero_index = torch.where(consecutive_count.logical_not())[0]

            if len(first_zero_index) > 0:
                first_zero_index = first_zero_index[0]
                consecutive_count[:first_zero_index] += carry
            else:
                consecutive_count += carry

        self.lookback.push(consecutive_count[-1])

        if self.shift is not None:
            held_count = torch.empty_like(consecutive_count)
            held_count[0] = carry
            held_count[1:] = consecutive_count[:-1]

            consecutive_count = held_count
            
        if self.max_count is not None:
            consecutive_count = torch.clamp(consecutive_count, max=self.max_count)

        return consecutive_count