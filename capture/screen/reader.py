from __future__ import annotations

from time import perf_counter_ns
from dataclasses import dataclass
from typing import TypeAlias
from pydantic import ConfigDict, BaseModel, NonNegativeInt
import numpy as np
from numpy.typing import NDArray

import dxcam

CaptureRegion: TypeAlias = tuple[int, int, int, int]


class ScreenCaptureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    gpu_index: NonNegativeInt = 0
    monitor_index: NonNegativeInt = 0
    region: CaptureRegion | None = None


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    image: NDArray[np.uint8]
    timestamp_ns: int

    def validate(self) -> None:
        if self.image.ndim != 3:
            raise ValueError(
                "Expected frame shape [H, W, 3], "
                f"received {self.image.shape}"
            )

        if self.image.shape[2] != 3:
            raise ValueError(
                "Expected three RGB channels, "
                f"received {self.image.shape[2]}"
            )

        if self.image.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 frame, received {self.image.dtype}"
            )

        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns cannot be negative")


class ScreenReader:
    def __init__(
        self,
        *,
        gpu_index: int = 0,
        monitor_index: int = 0,
        region: CaptureRegion | None = None,
    ) -> None:
        self._validate_region(region)

        self._camera = dxcam.create(
            device_idx=gpu_index,
            output_idx=monitor_index,
            region=region,
            output_color="RGB",
        )
        self._closed = False

    def read(self) -> CapturedFrame:
        if self._closed:
            raise RuntimeError("Screen reader is closed")

        image = self._camera.grab(
            copy=True,
            new_frame_only=False,
        )

        timestamp_ns = perf_counter_ns()

        if image is None:
            raise RuntimeError("DXCam did not return a frame")

        frame = CapturedFrame(
            image=image,
            timestamp_ns=timestamp_ns,
        )
        frame.validate()

        return frame

    def close(self) -> None:
        if self._closed:
            return

        self._camera.release()
        self._closed = True

    def __enter__(self) -> ScreenReader:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    @staticmethod
    def _validate_region(
        region: CaptureRegion | None,
    ) -> None:
        if region is None:
            return

        left, top, right, bottom = region

        if left < 0 or top < 0:
            raise ValueError("Capture-region coordinates cannot be negative")

        if right <= left:
            raise ValueError("Capture-region right must be greater than left")

        if bottom <= top:
            raise ValueError("Capture-region bottom must be greater than top")
