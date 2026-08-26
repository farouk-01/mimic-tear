from abc import ABC, abstractmethod

import torch
from torch import Tensor

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