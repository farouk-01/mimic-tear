from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import GameStateSchema
from .snapshot import GameStateSnapshot


class GameStateReader(ABC):
    @property
    @abstractmethod
    def schema(self) -> GameStateSchema:
        ...

    @abstractmethod
    def read(self) -> GameStateSnapshot:
        ...