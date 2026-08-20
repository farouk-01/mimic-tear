from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor, nn

from utils import profile

@dataclass(frozen=True, slots=True)
class FusionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    input_features: tuple[int, ...]
    output_features: int


class Fusion(nn.Module):
    def __init__(
        self,
        *,
        input_features: tuple[int, ...],
        output_features: int,
    ) -> None:
        super().__init__()

        if not input_features:
            raise ValueError("input_features cannot be empty")

        if any(features <= 0 for features in input_features):
            raise ValueError("All input feature sizes must be greater than zero")

        if output_features <= 0:
            raise ValueError("output_features must be greater than zero")

        self.input_features = input_features
        self.output_features = output_features
        self.total_input_features = sum(input_features)

        if len(input_features) == 1 and input_features[0] == output_features:
            self.fusion = nn.Identity()
        else:
            self.fusion = nn.Sequential(
                nn.Linear(
                    self.total_input_features,
                    output_features,
                ),
                nn.ReLU(inplace=True),
            )

    @profile
    def forward(self, *features: Tensor) -> Tensor:
        if len(features) != len(self.input_features):
            raise ValueError(
                f"Expected {len(self.input_features)} feature tensors, "
                f"received {len(features)}"
            )

        reference_shape = features[0].shape[:-1]

        for tensor, expected_features in zip(
            features,
            self.input_features,
            strict=True,
        ):
            if tensor.ndim < 2:
                raise ValueError(
                    "Expected feature tensors with shape [..., F], "
                    f"received {tuple(tensor.shape)}"
                )

            if tensor.shape[:-1] != reference_shape:
                raise ValueError(
                    "All feature tensors must have matching " "leading dimensions"
                )

            if tensor.shape[-1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, "
                    f"received {tensor.shape[-1]}"
                )

        if len(features) == 1:
            combined = features[0]
        else:
            combined = torch.cat(
                features,
                dim=-1,
            )

        return self.fusion(combined)
