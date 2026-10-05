from time import perf_counter_ns
from dataclasses import dataclass

import dxcam
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, NonNegativeInt

from data.constants import FRAME_FIELD
from data.capture.sources.base import Reader
from data.models.schema import Schema, Field, NumpyType, Snapshot


class ArrayField(Field[NumpyType]):
    shape: tuple[int, ...]


type CaptureRegion = tuple[int, int, int, int]  # (left, top, right, bottom)


class ScreenCaptureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    gpu_index: NonNegativeInt = 0
    monitor_index: NonNegativeInt = 0
    region: CaptureRegion | None = None


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    image: NDArray[np.uint8]
    timestamp_ns: int


class ScreenReader(Reader[CapturedFrame]):
    def __init__(
        self,
        *,
        gpu_index: int = 0,
        monitor_index: int = 0,
        region: CaptureRegion | None = None,
    ) -> None:
        super().__init__()
        self._camera = dxcam.create(
            device_idx=gpu_index,
            output_idx=monitor_index,
            region=region,
            output_color="RGB",
        )
        self._closed = False

        left, top, right, bottom = self._camera.region

        h = bottom - top
        w = right - left
        c = 3  # RGB

        self._shape = (h, w, c)

        self._schema = Schema[ArrayField](
            fields=(
                ArrayField(
                    name=FRAME_FIELD,
                    dtype="uint8",
                    shape=self._shape,
                ),
            )
        )

    @property
    def schema(self) -> Schema[ArrayField]:
        return self._schema

    def read(self) -> CapturedFrame:
        self._ensure_open()
        image = self._camera.grab(copy=True, new_frame_only=False)
        timestamp_ns = perf_counter_ns()

        if image is None:
            raise RuntimeError("DXCam did not return a frame")

        if image.shape != self._shape or image.dtype != np.uint8:
            raise ValueError(
                f"Expected {self._shape} uint8, got {image.shape} {image.dtype}"
            )

        return CapturedFrame(image, timestamp_ns)

    def close(self) -> None:
        if self._closed:
            return

        self._camera.release()
        super().close()
