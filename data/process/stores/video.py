from pathlib import Path
from typing import Literal
from collections.abc import Sequence, Collection

import torch
from torchcodec.decoders import (
    VideoDecoder,  # pyright: ignore[reportPrivateImportUsage]
)
from torch import Tensor
from pydantic import BaseModel, ConfigDict

from data.process.stores.base import (
    STORE_ADAPTERS,
    Store,
    StoreConfig,
    StoreAdapter,
    TensorColumn,
    TensorTable,
    FILE_STORES,
)
from utils import profile


class VideoStoreConfig(StoreConfig):
    device: str
    dimension_order: Literal["NCHW", "NHWC"] = "NCHW"
    seek_mode: Literal["exact", "approximate"] = "exact"
    num_ffmpeg_threads: int = 1


@FILE_STORES.register(".mp4")
class VideoStore(Store[Tensor]):
    def __init__(
        self,
        source: str | Path,
        *,
        device: str | torch.device,
        dimension_order: Literal["NCHW", "NHWC"] = "NCHW",
        seek_mode: Literal["exact", "approximate"] = "exact",
        num_ffmpeg_threads: int = 1,
    ) -> None:
        super().__init__(source=source)
        self.frames = VideoDecoder(
            source=source,
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
    def feature_names(self) -> Collection[str]:
        return ("frames",)

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
