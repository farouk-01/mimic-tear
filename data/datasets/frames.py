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

    def _prepare(self, frames: Tensor) -> Tensor:
        if frames.ndim not in (3, 4):
            raise ValueError(
                "Expected frame(s) with shape [3, H, W] or [T, 3, H, W], "
                f"received {tuple(frames.shape)}"
            )

        channel_dim = 0 if frames.ndim == 3 else 1

        if frames.shape[channel_dim] != 3:
            raise ValueError(
                f"Expected 3 RGB channels, received {frames.shape[channel_dim]}"
            )

        original_dtype = frames.dtype

        if self.transform is not None:
            frames = self.transform(frames)
        else:
            frames = frames.to(torch.float32)

            if original_dtype == torch.uint8:
                frames = frames / 255.0

        if frames.dtype != torch.float32:
            frames = frames.to(torch.float32)

        return frames

    @profile
    def __getitem__(self, index: int) -> Tensor:
        return self._prepare(self.store.get(index))

    @profile
    def get_range(self, start: int, end: int) -> Tensor:
        frames = self.store.get_range(start, end)
        return self._prepare(frames).contiguous()
