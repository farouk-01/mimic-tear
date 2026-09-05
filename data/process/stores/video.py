from pathlib import Path
from typing import Literal
from collections.abc import Sequence

import torch
from torchcodec.decoders import (
    VideoDecoder,  # pyright: ignore[reportPrivateImportUsage]
)
from torch import Tensor
from pydantic import BaseModel, ConfigDict

from data.process.stores.base import (
    STORE_ADAPTERS,
    Store,
    StoreAdapter,
    TensorColumn,
    TensorTable,
)
from utils import profile


class VideoStoreConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    device: str
    dimension_order: Literal["NCHW", "NHWC"] = "NCHW"
    seek_mode: Literal["exact", "approximate"] = "exact"
    num_ffmpeg_threads: int = 1


class VideoStore(Store[Tensor]):
    def __init__(
        self,
        *,
        path: str | Path,
        device: str | torch.device,
        dimension_order: Literal["NCHW", "NHWC"] = "NCHW",
        seek_mode: Literal["exact", "approximate"] = "exact",
        num_ffmpeg_threads: int = 1,
    ) -> None:
        self.frames = VideoDecoder(
            source=path,
            dimension_order=dimension_order,
            seek_mode=seek_mode,
            device=device,
            num_ffmpeg_threads=num_ffmpeg_threads,
        )

    def __len__(self) -> int:
        return len(self.frames)

    @profile
    def get(self, index: int) -> Tensor:
        return self.frames.get_frame_at(index).data

    @profile
    def get_range(self, start: int, end: int) -> Tensor:
        return self.frames.get_frames_in_range(start=start, stop=end).data

    @property
    def frame_indices(self) -> Sequence[int]:
        return range(len(self.frames))

    @property
    def capture_timestamp_ns(self) -> Sequence[int]:
        raise NotImplementedError("Capture timestamps are not yet supported for ParquetStore")

@STORE_ADAPTERS.register(VideoStore)
class VideoStoreAdapter(StoreAdapter[Tensor]):
    def get(self, data: Tensor) -> TensorTable:
        return {"frames": TensorColumn(values=data, validity=None)}
