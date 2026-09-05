from torch import float32

from data.process.transforms.types.tensor import (
    Contiguous,
    TensorTransform,
    Resize,
    ToDtype,
    Normalize,
)


def get_frame_transforms(
    *,
    width: int,
    height: int,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
    scale: bool = True,
) -> tuple[TensorTransform, ...]:
    transforms: list[TensorTransform] = [
        Resize(
            input="frames",
            output="frames",
            width=width,
            height=height,
        ),
        ToDtype(
            input="frames",
            output="frames",
            dtype=float32,
            scale=scale,
        ),
    ]

    if mean is not None and std is not None:
        transforms.append(
            Normalize(
                input="frames",
                output="frames",
                mean=mean,
                std=std,
            )
        )

    transforms.append(
        Contiguous(
            input="frames",
            output="frames",
        )
    )

    return tuple(transforms)