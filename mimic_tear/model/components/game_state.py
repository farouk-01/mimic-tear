from __future__ import annotations
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from utils import profile

class GameStateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    input_features: int = Field(gt=0)
    hidden_features: int = Field(gt=0)
    output_features: int = Field(gt=0)


class GameState(nn.Module):
    def __init__(
        self,
        *,
        input_features: int,
        hidden_features: int,
        output_features: int,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, output_features),
            nn.ReLU(inplace=True),
        )

    @profile
    def forward(self, state: Tensor) -> Tensor:
        """
        Args:
            state: [B, T, input_features]

        Returns:
            features: [B, T, output_features]
        """
        if state.ndim != 3:
            raise ValueError(
                "Expected game state with shape [B, T, F], "
                f"received {tuple(state.shape)}"
            )

        if state.shape[2] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} game-state features, "
                f"received {state.shape[2]}"
            )

        return self.encoder(state)