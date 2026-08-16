from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from pydantic import BaseModel

from mimic_tear.model.components import ControllerOutput


@dataclass(frozen=True, slots=True)
class PolicyLossOutput:
    total: Tensor
    analog: Tensor
    buttons: Tensor


class PolicyLoss(nn.Module):
    analog_weight: Tensor
    button_weight: Tensor

    def __init__(
        self,
        *,
        button_weight: Tensor,
        analog_weight: Tensor,
    ) -> None:
        super().__init__()

        self.register_buffer("analog_weight", analog_weight)
        self.register_buffer("button_weight", button_weight)

        self.analog_criterion = nn.SmoothL1Loss(reduction="none")
        self.button_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        output: ControllerOutput,
        *,
        analog_target: Tensor,
        button_target: Tensor,
    ) -> PolicyLossOutput:
        analog_loss = self.analog_criterion(
            output.analog,
            analog_target,
        )

        button_loss = self.button_criterion(
            output.button_logits,
            button_target,
        )

        analog_loss = (analog_loss * self.analog_weight).sum(
            dim=-1
        ) / self.analog_weight.sum()

        button_loss = (button_loss * self.button_weight).sum(
            dim=-1
        ) / self.button_weight.sum()

        analog_loss = analog_loss.mean()
        button_loss = button_loss.mean()

        total = analog_loss + button_loss

        return PolicyLossOutput(
            total=total,
            analog=analog_loss,
            buttons=button_loss,
        )
