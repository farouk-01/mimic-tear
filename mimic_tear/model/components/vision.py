from __future__ import annotations
from dataclasses import dataclass

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from ..config import Config

@dataclass(frozen=True, slots=True)
class VisionConfig(Config):
    output_features: int
    pretrained: bool = True


class Vision(nn.Module):
    def __init__(
        self,
        *,
        output_features: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if output_features <= 0:
            raise ValueError("output_features must be greater than zero")

        weights = ResNet18_Weights.DEFAULT if pretrained else None

        self.backbone = resnet18(weights=weights)
        
        self.backbone_features = self.backbone.fc.in_features

        # ImageNet classes are not useful for our purposes,
        # we need the visual information before classification.
        self.backbone.fc = nn.Identity() # type: ignore

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
            raise ValueError(
                f"Expected 3 RGB channels, received {images.shape[1]}"
            )

        features = self.backbone(images)
        return self.projection(features)