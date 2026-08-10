from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mimic_tear.model.components import ControllerOutput


@dataclass(frozen=True, slots=True)
class PolicyLossOutput:
    total: Tensor
    analog: Tensor
    buttons: Tensor


class PolicyLoss(nn.Module):
    def __init__(
        self,
        *,
        analog_weight: float = 1.0,
        button_weight: float = 1.0,
        button_positive_weights: Tensor | None = None,
    ) -> None:
        super().__init__()

        if analog_weight < 0.0:
            raise ValueError("analog_weight cannot be negative")

        if button_weight < 0.0:
            raise ValueError("button_weight cannot be negative")

        self.analog_weight = analog_weight
        self.button_weight = button_weight

        self.analog_loss = nn.SmoothL1Loss()

        self.button_loss = nn.BCEWithLogitsLoss(
            pos_weight=button_positive_weights,
        )

    def forward(
        self,
        output: ControllerOutput,
        *,
        analog_target: Tensor,
        button_target: Tensor,
    ) -> PolicyLossOutput:
        analog_loss = self.analog_loss(
            output.analog,
            analog_target,
        )

        button_loss = self.button_loss(
            output.button_logits,
            button_target,
        )

        total = (
            self.analog_weight * analog_loss
            + self.button_weight * button_loss
        )

        return PolicyLossOutput(
            total=total,
            analog=analog_loss,
            buttons=button_loss,
        )