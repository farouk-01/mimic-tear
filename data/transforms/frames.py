from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.transforms import v2

@dataclass(frozen=True, slots=True)
class FrameTransformConfig:
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    scale: bool = True


class FrameTransform:
    def __init__(
        self,
        *,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        scale: bool = True,
    ) -> None:
        if (mean is None) != (std is None):
            raise ValueError(
                "mean and std must either both be provided or both be None"
            )

        transforms: list[torch.nn.Module] = [
            v2.ToImage(),
            v2.ToDtype(
                torch.float32,
                scale=scale,
            ),
        ]

        if mean is not None and std is not None:
            transforms.append(
                v2.Normalize(
                    mean=mean,
                    std=std,
                )
            )

        self.transform = v2.Compose(transforms)

    def __call__(self, frame: Tensor) -> Tensor:
        return self.transform(frame)