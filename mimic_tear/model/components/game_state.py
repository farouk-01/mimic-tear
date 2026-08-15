from __future__ import annotations
from dataclasses import dataclass

from torch import Tensor, nn

from ..config import ComponentConfig


@dataclass(frozen=True, slots=True)
class GameStateConfig(ComponentConfig):
    input_features: int
    hidden_features: int
    output_features: int


class GameState(nn.Module):
    def __init__(
        self,
        *,
        input_features: int,
        hidden_features: int,
        output_features: int,
    ) -> None:
        super().__init__()

        if input_features <= 0:
            raise ValueError("input_features must be greater than zero")

        if hidden_features <= 0:
            raise ValueError("hidden_features must be greater than zero")

        if output_features <= 0:
            raise ValueError("output_features must be greater than zero")

        self.input_features = input_features
        self.output_features = output_features

        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, output_features),
            nn.ReLU(inplace=True),
        )

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