from __future__ import annotations
from pathlib import Path

import cv2
from torch import Tensor
import torch

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

    @classmethod
    def from_mp4(cls, path: str | Path) -> TensorFrameStore:
        path = Path(path)

        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {path}")

        frames: list[Tensor] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(
                    torch.from_numpy(frame).permute(2, 0, 1)
                )
        finally:
            capture.release()

        if not frames:
            raise RuntimeError(f"Video file contains no decodable frames: {path}")

        stack = torch.stack(frames)

        return TensorFrameStore(frames=stack)