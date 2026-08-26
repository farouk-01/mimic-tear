from __future__ import annotations
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor


class GameStateTransformConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Transform(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def inputs(self) -> tuple[str, ...]: ...

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tensor: ...


class Ratio(Transform):
    def __init__(self, numerator: str, denominator: str) -> None:
        self.numerator = numerator
        self.denominator = denominator

    @property
    def name(self) -> str:
        return "ratio"

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


class GameStateTransform:
    def __init__(
        self,
        transforms: tuple[Transform, ...],
        names_to_indice: dict[str, int],
    ) -> None:
        self.transforms = transforms
        self.name_to_indices = names_to_indice

    def __call__(self, states: Tensor) -> Tensor:
        for transform_def in self.transforms:
            inputs = transform_def.inputs
            indices = [self.name_to_indices[name] for name in inputs]

            input_tensors = [states[..., index] for index in indices]
            outputs = transform_def(*input_tensors)

            if outputs.ndim == states.ndim - 1:
                outputs = outputs.unsqueeze(-1)

            states = torch.cat((states, outputs), dim=-1)

        return states
