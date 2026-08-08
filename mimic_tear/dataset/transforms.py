from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import Tensor


Frame = np.ndarray
FrameTransform = Callable[[Frame], Tensor]


class Compose:
    """Apply frame transformations sequentially."""

    def __init__(self, transforms: Sequence[Callable]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)

        return value


@dataclass(frozen=True, slots=True)
class Resize:
    """Resize an OpenCV frame to (width, height)."""

    width: int = 320
    height: int = 180

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be greater than zero")

        if self.height <= 0:
            raise ValueError("height must be greater than zero")

    def __call__(self, frame: Frame) -> Frame:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected an OpenCV frame with shape (height, width, 3), "
                f"received {frame.shape}"
            )

        if frame.shape[1] == self.width and frame.shape[0] == self.height:
            return frame

        return cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )


class BGRToRGB:
    """Convert OpenCV's BGR channel order to RGB."""

    def __call__(self, frame: Frame) -> Frame:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected a BGR frame with shape (height, width, 3), "
                f"received {frame.shape}"
            )

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class ToFloatTensor:
    """
    Convert an HWC uint8 image to a CHW float32 tensor in [0, 1].
    """

    def __call__(self, frame: Frame) -> Tensor:
        if frame.ndim != 3:
            raise ValueError(
                f"Expected a 3-dimensional frame, received {frame.shape}"
            )

        # np.ascontiguousarray avoids negative/non-contiguous strides that
        # torch.from_numpy cannot safely consume.
        contiguous = np.ascontiguousarray(frame)

        tensor = torch.from_numpy(contiguous)
        tensor = tensor.permute(2, 0, 1).contiguous()
        tensor = tensor.to(dtype=torch.float32)

        return tensor.div_(255.0)


@dataclass(frozen=True, slots=True)
class Normalize:
    """Normalize each RGB channel using channel-wise mean and std."""

    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def __post_init__(self) -> None:
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must each contain three values")

        if any(value <= 0.0 for value in self.std):
            raise ValueError("All standard-deviation values must be positive")

    def __call__(self, image: Tensor) -> Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                "Expected an RGB tensor with shape (3, height, width), "
                f"received {tuple(image.shape)}"
            )

        mean = image.new_tensor(self.mean).view(3, 1, 1)
        std = image.new_tensor(self.std).view(3, 1, 1)

        return (image - mean) / std


@dataclass(frozen=True, slots=True)
class RandomBrightnessContrast:
    """
    Apply conservative brightness and contrast augmentation.

    The input must already be a float tensor in [0, 1].
    """

    brightness: float = 0.08
    contrast: float = 0.08
    probability: float = 0.5

    def __post_init__(self) -> None:
        if self.brightness < 0.0:
            raise ValueError("brightness cannot be negative")

        if self.contrast < 0.0:
            raise ValueError("contrast cannot be negative")

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")

    def __call__(self, image: Tensor) -> Tensor:
        if torch.rand(()) >= self.probability:
            return image

        brightness_factor = 1.0 + float(
            torch.empty(()).uniform_(-self.brightness, self.brightness)
        )
        contrast_factor = 1.0 + float(
            torch.empty(()).uniform_(-self.contrast, self.contrast)
        )

        image = image * brightness_factor

        # Preserve the general brightness while changing contrast.
        channel_mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (image - channel_mean) * contrast_factor + channel_mean

        return image.clamp_(0.0, 1.0)


def build_train_transform(
    *,
    width: int = 320,
    height: int = 180,
    augment: bool = True,
) -> FrameTransform:
    transforms: list[Callable] = [
        Resize(width=width, height=height),
        BGRToRGB(),
        ToFloatTensor(),
    ]

    if augment:
        transforms.append(RandomBrightnessContrast())

    transforms.append(Normalize())

    return Compose(transforms)


def build_eval_transform(
    *,
    width: int = 320,
    height: int = 180,
) -> FrameTransform:
    return Compose(
        [
            Resize(width=width, height=height),
            BGRToRGB(),
            ToFloatTensor(),
            Normalize(),
        ]
    )
