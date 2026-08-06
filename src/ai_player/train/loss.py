from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ai_player.model.elden_ring import PolicyOutput


@dataclass(frozen=True, slots=True)
class PolicyLossResult:
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
        prediction: PolicyOutput,
        analog_target: Tensor,
        button_target: Tensor,
    ) -> PolicyLossResult:
        self._validate_shapes(
            prediction=prediction,
            analog_target=analog_target,
            button_target=button_target,
        )

        analog_loss = self.analog_loss(
            prediction.analog,
            analog_target,
        )

        button_loss = self.button_loss(
            prediction.button_logits,
            button_target,
        )

        total_loss = (
            self.analog_weight * analog_loss
            + self.button_weight * button_loss
        )

        return PolicyLossResult(
            total=total_loss,
            analog=analog_loss.detach(),
            buttons=button_loss.detach(),
        )

    @staticmethod
    def _validate_shapes(
        *,
        prediction: PolicyOutput,
        analog_target: Tensor,
        button_target: Tensor,
    ) -> None:
        if prediction.analog.shape != analog_target.shape:
            raise ValueError(
                "Analog prediction and target shapes do not match: "
                f"{tuple(prediction.analog.shape)} vs "
                f"{tuple(analog_target.shape)}"
            )

        if prediction.button_logits.shape != button_target.shape:
            raise ValueError(
                "Button prediction and target shapes do not match: "
                f"{tuple(prediction.button_logits.shape)} vs "
                f"{tuple(button_target.shape)}"
            )

        if analog_target.dtype != torch.float32:
            raise TypeError(
                "analog_target must be float32, "
                f"received {analog_target.dtype}"
            )

        if button_target.dtype != torch.float32:
            raise TypeError(
                "button_target must be float32, "
                f"received {button_target.dtype}"
            )