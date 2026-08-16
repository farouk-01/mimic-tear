from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor, nn

class ControllerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    input_features: int
    button_outputs: int


@dataclass(frozen=True, slots=True)
class ControllerOutput:
    analog: Tensor
    button_logits: Tensor


class Controller(nn.Module):
    def __init__(
        self,
        *,
        input_features: int,
        button_outputs: int,
    ) -> None:
        super().__init__()

        if input_features <= 0:
            raise ValueError("input_features must be greater than zero")

        if button_outputs <= 0:
            raise ValueError("button_outputs must be greater than zero")

        self.input_features = input_features
        self.button_outputs = button_outputs

        # Left stick:
        # x, y in [-1, 1]
        self.left_stick = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

        # Right stick:
        # x, y in [-1, 1]
        self.right_stick = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

        # Bumpers:
        # left, right in [0, 1]
        self.bumpers = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        # No sigmoid here because BCEWithLogitsLoss
        # expects raw logits during training.
        self.buttons = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, button_outputs),
        )

    def forward(self, features: Tensor) -> ControllerOutput:
        """
        Args:
            features:
                [B, T, input_features]

        Returns:
            ControllerOutput:
                analog:
                    [B, T, 6]

                button_logits:
                    [B, T, button_outputs]
        """
        if features.ndim != 3:
            raise ValueError(
                "Expected features with shape [B, T, F], "
                f"received {tuple(features.shape)}"
            )

        if features.shape[-1] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} features, "
                f"received {features.shape[-1]}"
            )

        left_stick = torch.tanh(
            self.left_stick(features)
        )

        right_stick = torch.tanh(
            self.right_stick(features)
        )

        bumpers = torch.sigmoid(
            self.bumpers(features)
        )

        analog = torch.cat(
            (
                left_stick,
                right_stick,
                bumpers,
            ),
            dim=-1,
        )

        button_logits = self.buttons(features)

        return ControllerOutput(
            analog=analog,
            button_logits=button_logits,
        )