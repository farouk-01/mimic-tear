from __future__ import annotations
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from utils import profile


class VisionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    
    output_features: int
    weights_name: str | None

# note : current most expensive component ~31/~35
class Vision(nn.Module):
    def __init__(
        self,
        *,
        output_features: int,
        weights_name: str | None = "DEFAULT",
    ) -> None:
        super().__init__()

        if output_features <= 0:
            raise ValueError("output_features must be greater than zero")

        weights = None if weights_name is None else ResNet18_Weights[weights_name]

        self.backbone = resnet18(weights=weights)

        self.backbone_features = self.backbone.fc.in_features

        # ImageNet classes are not useful for our purposes,
        # we need the visual information before classification.
        self.backbone.fc = nn.Identity()  # type: ignore

        self.output_features = output_features

        # Allow the visual representation size to be changed without
        # coupling the rest of the model to ResNet18's 512 features.
        if output_features == self.backbone_features:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_features, output_features),
                nn.ReLU(inplace=True),
            )

    @profile
    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            features: [B, output_features]
        """
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape [B, 3, H, W], "
                f"received {tuple(images.shape)}"
            )

        if images.shape[1] != 3:
            raise ValueError(f"Expected 3 RGB channels, received {images.shape[1]}")

        features = self.backbone(images)
        return self.projection(features)