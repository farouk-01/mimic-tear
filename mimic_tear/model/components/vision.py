from abc import ABC, abstractmethod

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from utils import profile


class Vision(nn.Module, ABC):
    def __init__(self, output_size: int) -> None:
        super().__init__()

        self.output_size = output_size

    @abstractmethod
    def forward(self, images: Tensor) -> Tensor: ...


class ResNet18(Vision):
    def __init__(
        self,
        output_size: int,
        weights_name: str | None = "DEFAULT",
    ) -> None:
        super().__init__(output_size=output_size)

        if output_size <= 0:
            raise ValueError("output_size must be greater than zero")

        weights = None if weights_name is None else ResNet18_Weights[weights_name]

        self.backbone = resnet18(weights=weights)
        self.backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # type: ignore

        self.output_size = output_size

        if output_size == self.backbone_features:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_features, output_size),
                nn.ReLU(inplace=True),
            )

    @profile
    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape [B, 3, H, W], "
                f"received {tuple(images.shape)}"
            )

        if images.shape[1] != 3:
            raise ValueError(f"Expected 3 RGB channels, received {images.shape[1]}")

        features = self.backbone(images)
        return self.projection(features)
