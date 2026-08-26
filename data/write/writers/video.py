from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, BaseModel, PositiveInt

class VideoConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    width: PositiveInt = 1280
    height: PositiveInt = 720
    fps: PositiveInt = 30


class VideoFrameWriter:
    def __init__(
        self,
        *,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        if width <= 0:
            raise ValueError("width must be greater than zero")

        if height <= 0:
            raise ValueError("height must be greater than zero")

        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")

        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            fps,
            (width, height),
        )

        if not self._writer.isOpened():
            raise RuntimeError(
                f"Failed to open video writer: {self.path}"
            )

        self._closed = False

    def write(
        self,
        frame: NDArray[np.uint8],
    ) -> None:
        if self._closed:
            raise RuntimeError("Video writer is closed")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected RGB frame with shape [H, W, 3], "
                f"received {frame.shape}"
            )

        if frame.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 frame, received {frame.dtype}"
            )

        output_frame = frame

        if frame.shape[:2] != (self.height, self.width):
            output_frame = cast(
                NDArray[np.uint8],
                cv2.resize(
                    frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_AREA,
                ),
            )

        bgr = cast(
            NDArray[np.uint8],
            cv2.cvtColor(
                output_frame,
                cv2.COLOR_RGB2BGR,
            ),
        )

        self._writer.write(bgr)

    def close(self) -> None:
        if self._closed:
            return

        self._writer.release()
        self._closed = True

    def __enter__(self) -> VideoFrameWriter:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()