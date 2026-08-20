from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torchcodec.decoders import (
    VideoDecoder,  # pyright: ignore[reportPrivateImportUsage]
)
from pydantic import BaseModel, ConfigDict

from data.datasets.frames import FrameStore


class VideoDecoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    device: str
    dimension_order: Literal["NCHW", "NHWC"] = "NCHW"
    seek_mode: Literal["exact", "approximate"] = "exact"
    num_ffmpeg_threads: int = 1


class TensorFrameStore(FrameStore):
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

    def get(self, index: int) -> Tensor:
        return self.frames.get_frame_at(index).data

    def get_range(self, start: int, end: int) -> Tensor:
        return self.frames.get_frames_in_range(start=start, stop=end).data
