from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
import torch
from torch import Tensor, nn

from utils import profile


class FusionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    d_model: int = Field(gt=0)


class Fusion(nn.Module):
    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    @profile
    def forward(self, *features: Tensor) -> Tensor:
        if not features:
            raise ValueError("Expected at least one feature tensor")

        reference_shape = features[0].shape[:-2]

        for tensor in features:
            if tensor.ndim < 3:
                raise ValueError(
                    "Expected token tensors with shape [..., N, D], "
                    f"received {tuple(tensor.shape)}"
                )

            if tensor.shape[:-2] != reference_shape:
                raise ValueError(
                    "All feature tensors must have matching leading dimensions"
                )

            if tensor.shape[-1] != self.d_model:
                raise ValueError(
                    f"Expected token dimension {self.d_model}, "
                    f"received {tensor.shape[-1]}"
                )

        return torch.cat(features, dim=-2)