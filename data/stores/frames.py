from __future__ import annotations

from torch import Tensor

from data.datasets.frames import FrameStore


class TensorFrameStore(FrameStore):
    def __init__(
        self,
        *,
        frames: Tensor,
    ) -> None:
        if frames.ndim != 4:
            raise ValueError(
                "Expected frames with shape [N, 3, H, W], "
                f"received {tuple(frames.shape)}"
            )

        if frames.shape[0] <= 0:
            raise ValueError("Frame store cannot be empty")

        if frames.shape[1] != 3:
            raise ValueError(
                f"Expected 3 RGB channels, received {frames.shape[1]}"
            )

        self.frames = frames

    def __len__(self) -> int:
        return self.frames.shape[0]

    def get(
        self,
        index: int,
    ) -> Tensor:
        return self.frames[index]