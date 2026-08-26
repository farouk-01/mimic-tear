from __future__ import annotations

from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor

from .generic import Transform


class GameStateTransformConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class GameStateTransform:
    def __init__(
        self,
        *,
        generic_transforms: dict[str, Transform],
        names_to_indice: dict[str, int],
    ) -> None:
        self.transforms = generic_transforms
        self.names_to_indices = names_to_indice.copy()

    def __call__(self, states: Tensor) -> Tensor:
        for output_name, transform in self.transforms.items():
            indices = [self.names_to_indices[name] for name in transform.inputs]
            input_tensors = [states[..., index] for index in indices]

            outputs = transform(*input_tensors)

            if outputs.ndim == states.ndim - 1:
                outputs = outputs.unsqueeze(-1)

            self.names_to_indices[output_name] = states.shape[-1]
            states = torch.cat((states, outputs), dim=-1)

        return states
