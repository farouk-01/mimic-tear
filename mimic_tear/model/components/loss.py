from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from mimic_tear.model.components.controller import ControllerOutput
from utils import profile


@dataclass(frozen=True, slots=True)
class LossOutput:
    total: Tensor


class Loss(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, output: ControllerOutput, *args: Any, **kwargs: Any
    ) -> LossOutput: ...


@dataclass(frozen=True, slots=True)
class GamepadLossOutput(LossOutput):
    analog: Tensor
    buttons: Tensor


class GamepadLoss(Loss):
    analog_weight: Tensor
    button_weight: Tensor
    button_transition_weight: Tensor

    def __init__(
        self,
        *,
        button_weight: Tensor,
        analog_weight: Tensor,
        button_transition_weight: Tensor,
    ) -> None:
        super().__init__()

        self.register_buffer("analog_weight", analog_weight)
        self.register_buffer("button_weight", button_weight)
        self.register_buffer("button_transition_weight", button_transition_weight)

        self.analog_criterion = nn.SmoothL1Loss(reduction="none")
        self.button_criterion = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=button_weight,
        )

    @profile
    def forward(
        self,
        output: ControllerOutput,
        *,
        analog_target: Tensor,
        button_target: Tensor,
    ) -> GamepadLossOutput:
        analog_loss = self.analog_criterion(output.analog, analog_target)
        button_loss = self.button_criterion(output.button_logits, button_target)

        analog_loss = (analog_loss * self.analog_weight).sum(
            dim=-1
        ) / self.analog_weight.sum()

        analog_loss = analog_loss.mean()
        transition_weights = self._transition_weights(button_target)
        button_loss = (button_loss * transition_weights).sum() / transition_weights.sum()

        total = analog_loss + button_loss

        return GamepadLossOutput(total=total, analog=analog_loss, buttons=button_loss)

    def _transition_weights(self, button_target: Tensor) -> Tensor:
        first_frame = button_target[:, :1]
        previous = torch.cat([first_frame, button_target[:, :-1]], dim=1)

        is_transitioning = button_target != previous

        return torch.where(is_transitioning, self.button_transition_weight, 1.0)
