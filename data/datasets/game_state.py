from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

GameStateValue = int | float | bool


class GameStateStore(ABC):
    @property
    @abstractmethod
    def features(self) -> tuple[str, ...]:
        """
        Ordered game-state features exposed by this store.
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(
        self,
        index: int,
    ) -> Mapping[str, GameStateValue]:
        """
        Return one frame-aligned game-state snapshot.
        """
        ...


class GameStateDataset(Dataset[Tensor]):
    def __init__(
        self,
        *,
        store: GameStateStore,
    ) -> None:
        if len(store) <= 0:
            raise ValueError("Game-state store cannot be empty")

        if not store.features:
            raise ValueError("Game-state store must expose at least one feature")

        self.store = store
        self.features = store.features

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(
        self,
        index: int,
    ) -> Tensor:
        state = self.store.get(index)

        missing = [feature for feature in self.features if feature not in state]

        if missing:
            raise ValueError(f"Game-state sample is missing features: {missing}")

        return torch.tensor(
            [float(state[feature]) for feature in self.features],
            dtype=torch.float32,
        )
