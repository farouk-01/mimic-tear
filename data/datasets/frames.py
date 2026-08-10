from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import Dataset


class FrameStore(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(self, index: int) -> Tensor:
        """
        Returns:
            Frame tensor with shape [3, H, W].
        """
        ...


class FramesDataset(Dataset[Tensor]):
    def __init__(
        self,
        *,
        store: FrameStore,
    ) -> None:
        if len(store) <= 0:
            raise ValueError("Frame store cannot be empty")

        self.store = store

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(
        self,
        index: int,
    ) -> Tensor:
        frame = self.store.get(index)

        if frame.ndim != 3:
            raise ValueError(
                "Expected frame with shape [3, H, W], " f"received {tuple(frame.shape)}"
            )

        if frame.shape[0] != 3:
            raise ValueError(f"Expected 3 RGB channels, received {frame.shape[0]}")

        return frame
