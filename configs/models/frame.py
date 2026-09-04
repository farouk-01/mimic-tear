from typing import Self

from pydantic import BaseModel, ConfigDict
from torchvision.models import ResNet18_Weights

from data.models.tensor import TensorSchema
from data.process.transforms.types.tensor import TransformNode
from data.process.stores.video import VideoStoreConfig
from graph.base import Graph, Plan, Value

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
    plan: Plan

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

        graph = Graph()

        transforms = get_frame_transforms(**transform_cfg, mean=mean, std=std)

        for transform in transforms:
            graph.add(TransformNode(transform=transform))

        plan = graph.resolve((graph.value("frames"),))

        video_cfg = VideoStoreConfig(**video_store_cfg)

        return cls(video_store_cfg=video_cfg, tensor_frame_schema=schema, plan=plan)
