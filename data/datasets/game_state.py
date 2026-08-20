from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

GameStateValue = int | float | bool


class GameStateStore(ABC):
    @property
    @abstractmethod
    def features(self) -> tuple[str, ...]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(self, index: int) -> Mapping[str, GameStateValue]: ...

    @abstractmethod
    def get_range(self, start: int, end: int) -> Tensor: ...


class GameStateDataset(Dataset[Tensor]):
    def __init__(
        self,
        *,
        store: GameStateStore,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        if len(store) <= 0:
            raise ValueError("Game-state store cannot be empty")

        if not store.features:
            raise ValueError("Game-state store must expose at least one feature")

        self.store = store
        self.features = store.features
        self.transform = transform

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int) -> Tensor:
        state = self.store.get(index)

        missing = [feature for feature in self.features if feature not in state]

        if missing:
            raise ValueError(f"Game-state sample is missing features: {missing}")

        state_tensor = torch.tensor(
            [float(state[feature]) for feature in self.features],
            dtype=torch.float32,
        )

        if self.transform is not None:
            state_tensor = self.transform(state_tensor)

        return state_tensor

    def get_range(self, start: int, end: int) -> Tensor:
        states = self.store.get_range(start, end)

        if self.transform is not None:
            states = self.transform(states)

        return states