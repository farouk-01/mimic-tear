from __future__ import annotations

from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict, PositiveInt, PositiveFloat
from torch import Tensor
from torchvision.transforms import v2


class FrameTransformConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    width: PositiveInt
    height: PositiveInt
    mean: tuple[float, float, float] | None = None
    std: tuple[PositiveFloat, PositiveFloat, PositiveFloat] | None = None
    scale: bool = True


class FrameTransform:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        scale: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")

        transforms = []
        transforms.append(v2.ToImage())
        transforms.append(v2.Resize(size=(height, width), antialias=True))
        transforms.append(v2.ToDtype(torch.float32, scale=scale))

        if std is not None and mean is not None:
            if any(value <= 0 for value in std):
                raise ValueError("Normalization standard deviations must be positive")

            transforms.append(v2.Normalize(mean=mean, std=std))

        self.transform = v2.Compose(transforms=transforms)

    def __call__(
        self,
        frames: Tensor,
    ) -> Tensor:
        transformed = self.transform(frames)

        if transformed.dtype != torch.float32:
            raise RuntimeError("Frame transform did not produce float32")

        return transformed
