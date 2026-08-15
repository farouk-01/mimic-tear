from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# from mimic_tear.model.config import ComponentConfig


# @dataclass(frozen=True, slots=True)
# class ControllerTransformConfig(ComponentConfig):
#     clamp_sticks: bool = True
#     clamp_triggers: bool = True


class ControllerTransform:
    def __init__(
        self,
        *,
        clamp_sticks: bool = True,
        clamp_triggers: bool = True,
    ) -> None:
        self.clamp_sticks = clamp_sticks
        self.clamp_triggers = clamp_triggers

    def __call__(
        self,
        analog: Tensor,
        buttons: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if analog.shape[-1] != 6:
            raise ValueError(
                f"Expected 6 analog features, received {analog.shape[-1]}"
            )

        analog = analog.to(torch.float32)
        buttons = buttons.to(torch.float32)

        sticks = analog[..., :4]
        triggers = analog[..., 4:6]

        if self.clamp_sticks:
            sticks = sticks.clamp(-1.0, 1.0)

        if self.clamp_triggers:
            triggers = triggers.clamp(0.0, 1.0)

        analog = torch.cat(
            (sticks, triggers),
            dim=-1,
        )

        return analog, buttons