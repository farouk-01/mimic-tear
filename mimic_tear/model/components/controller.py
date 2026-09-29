from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn
import torch


@dataclass(frozen=True, slots=True)
class ControllerOutput:
    analog: Tensor
    button_logits: Tensor


class Controller(nn.Module, ABC):
    def __init__(self, input_size: int) -> None:
        super().__init__()

        self.input_size = input_size

    @abstractmethod
    def forward(self, features: Tensor) -> ControllerOutput: ...


class Gamepad(Controller):
    def __init__(
        self,
        input_size: int,
        num_buttons: int,
    ) -> None:
        super().__init__(input_size)

        self.num_buttons = num_buttons

        self.left_stick = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

        self.right_stick = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

        self.triggers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        self.buttons = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_buttons),
        )

    def forward(self, features: Tensor) -> ControllerOutput:
        if features.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"received {features.shape[-1]}"
            )

        left_stick = torch.tanh(self.left_stick(features))
        right_stick = torch.tanh(self.right_stick(features))
        triggers = torch.sigmoid(self.triggers(features))

        analog = torch.cat([left_stick, right_stick, triggers], dim=-1)
        btn_logits = self.buttons(features)

        return ControllerOutput(analog, btn_logits)
