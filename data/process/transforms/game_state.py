from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from torch import Tensor

from .generic import Transform

type GameStateTensors = dict[str, Tensor]


class GameStateTransformConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class GameStateTransform:
    def __init__(
        self,
        *,
        generic_transforms: dict[str, Transform],
    ) -> None:
        self.transforms = generic_transforms

    def __call__(self, states: GameStateTensors) -> GameStateTensors:
        for output_name, transform in self.transforms.items():
            input_tensors = [states[name] for name in transform.inputs]
            states[output_name] = transform(*input_tensors)

        return states
