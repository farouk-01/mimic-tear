from __future__ import annotations

from abc import ABC, abstractmethod

import torch
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

        if frame.dtype == torch.uint8:
            frame = frame.to(torch.float32) / 255.0
        elif not frame.is_floating_point():
            frame = frame.to(torch.float32)

        if frame.dtype != torch.float32:
            frame = frame.to(torch.float32)

        if torch.any(frame < 0.0) or torch.any(frame > 1.0):
            raise ValueError("Expected frame values in [0, 1]")

        return frame
