from typing import Self

from pydantic import BaseModel, ConfigDict
from torchvision.models import ResNet18_Weights

from data.models.tensor import TensorSchema
from data.process.transforms.tensor import TensorTransform
from data.process.stores.video import VideoStoreConfig

from configs.transforms.frame import get_frame_transforms


class FrameConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )

    video_store_cfg: VideoStoreConfig
    tensor_frame_schema: TensorSchema
    transforms: tuple[TensorTransform, ...] = ()

    @classmethod
    def load(
        cls,
        *,
        schema: TensorSchema,
        video_store_cfg: dict,
        transform_cfg: dict,
        weights_name: str | None,
    ) -> Self:
        mean: tuple[float, float, float] | None = None
        std: tuple[float, float, float] | None = None

        if weights_name is not None:
            presets = ResNet18_Weights[weights_name].transforms()
            mean = tuple(presets.mean)
            std = tuple(presets.std)

        transforms = get_frame_transforms(**transform_cfg, mean=mean, std=std)

        video_cfg = VideoStoreConfig(**video_store_cfg)

        return cls(
            video_store_cfg=video_cfg,
            tensor_frame_schema=schema,
            transforms=transforms,
        )
