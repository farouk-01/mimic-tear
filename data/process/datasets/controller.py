from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.models.gamepad import GamepadState


@dataclass(frozen=True, slots=True)
class ControllerSample:
    analog: Tensor
    buttons: Tensor


class ControllerStore(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(self, index: int) -> GamepadState: ...

    @abstractmethod
    def get_range(self, start: int, end: int) -> ControllerSample: ...

    @property
    @abstractmethod
    def indices(self) -> Tensor: ...

    @property
    @abstractmethod
    def timestamps_ns(self) -> Tensor: ...


class ControllerDataset(Dataset[ControllerSample]):
    def __init__(
        self,
        *,
        store: ControllerStore,
        transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        if len(store) <= 0:
            raise ValueError("Controller store cannot be empty")

        self.store = store
        self.transform = transform

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int) -> ControllerSample:
        state = self.store.get(index)
        state.validate()

        analog = torch.tensor(state.analog.values(), dtype=torch.float32)
        buttons = torch.tensor(state.buttons.values(), dtype=torch.float32)

        if self.transform is not None:
            analog, buttons = self.transform(analog, buttons)

        return ControllerSample(analog=analog, buttons=buttons)

    def get_range(self, start: int, end: int) -> ControllerSample:
        sample = self.store.get_range(start, end)

        analog = sample.analog
        buttons = sample.buttons

        if self.transform is not None:
            analog, buttons = self.transform(analog, buttons)

        return ControllerSample(
            analog=analog,
            buttons=buttons,
        )