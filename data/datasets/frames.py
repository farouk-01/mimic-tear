from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils import profile


class FrameStore(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(
        self,
        index: int,
    ) -> Tensor:
        """Return a uint8 RGB frame shaped [3, H, W]."""
        ...

    @abstractmethod
    def get_range(
        self,
        start: int,
        end: int,
    ) -> Tensor:
        """Return uint8 RGB frames shaped [T, 3, H, W]."""
        ...


class FramesDataset(Dataset[Tensor]):
    def __init__(
        self,
        *,
        store: FrameStore,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        if len(store) <= 0:
            raise ValueError("Frame store cannot be empty")

        self.store = store
        self.transform = transform

    def __len__(self) -> int:
        return len(self.store)

    def _prepare_frame(
        self,
        frame: Tensor,
    ) -> Tensor:
        if frame.ndim != 3:
            raise ValueError(
                "Expected frame with shape [3, H, W], " f"received {tuple(frame.shape)}"
            )

        if frame.shape[0] != 3:
            raise ValueError(f"Expected 3 RGB channels, received {frame.shape[0]}")

        original_dtype = frame.dtype

        if self.transform is not None:
            frame = self.transform(frame)
        else:
            frame = frame.to(torch.float32)

            if original_dtype == torch.uint8:
                frame = frame / 255.0

        if frame.dtype != torch.float32:
            frame = frame.to(torch.float32)

        return frame

    @profile
    def __getitem__(self, index: int) -> Tensor:
        return self._prepare_frame(self.store.get(index))

    @profile
    def get_range(self, start: int, end: int) -> Tensor:
        frames = self.store.get_range(start, end)
        return torch.stack([self._prepare_frame(frame) for frame in frames])
