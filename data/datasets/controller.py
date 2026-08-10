from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from controller import ControllerState


@dataclass(frozen=True, slots=True)
class ControllerSample:
    analog: Tensor
    buttons: Tensor


class ControllerStore(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(self, index: int) -> ControllerState: ...


class ControllerDataset(Dataset[ControllerSample]):
    def __init__(
        self,
        *,
        store: ControllerStore,
    ) -> None:
        if len(store) <= 0:
            raise ValueError("Controller store cannot be empty")

        self.store = store

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(
        self,
        index: int,
    ) -> ControllerSample:
        state = self.store.get(index)
        state.validate()

        analog = torch.tensor(
            state.analog.values(),
            dtype=torch.float32,
        )

        buttons = torch.tensor(
            state.buttons.values(),
            dtype=torch.float32,
        )

        return ControllerSample(
            analog=analog,
            buttons=buttons,
        )
