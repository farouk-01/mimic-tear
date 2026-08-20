from __future__ import annotations
from dataclasses import dataclass
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from utils import profile

LSTMState: TypeAlias = tuple[Tensor, Tensor]

class TemporalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    input_features: int
    hidden_features: int
    num_layers: int
    dropout: float

class Temporal(nn.Module):
    def __init__(
        self,
        *,
        input_features: int,
        hidden_features: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if input_features <= 0:
            raise ValueError("input_features must be greater than zero")

        if hidden_features <= 0:
            raise ValueError("hidden_features must be greater than zero")

        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")

        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")

        self.input_features = input_features
        self.output_features = hidden_features
        self.num_layers = num_layers
        self.hidden_features = hidden_features

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_features,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    @profile
    def forward(
        self,
        features: Tensor,
        state: LSTMState | None = None,
    ) -> tuple[Tensor, LSTMState]:
        """
        Args:
            features:
                [B, T, input_features]

            state:
                (
                    hidden_state,
                    cell_state,
                )

                Each has shape:
                [num_layers, B, hidden_features]

        Returns:
            temporal_features:
                [B, T, hidden_features]

            next_state:
                (
                    hidden_state,
                    cell_state,
                )

                Each has shape:
                [num_layers, B, hidden_features]
        """
        if features.ndim != 3:
            raise ValueError(
                "Expected features with shape [B, T, F], "
                f"received {tuple(features.shape)}"
            )

        if features.shape[2] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} input features, "
                f"received {features.shape[2]}"
            )

        if state is not None:
            hidden_state, cell_state = state

            expected_shape = (
                self.num_layers,
                features.shape[0],
                self.hidden_features,
            )

            if tuple(hidden_state.shape) != expected_shape:
                raise ValueError(
                    "Expected hidden state with shape "
                    f"{expected_shape}, received "
                    f"{tuple(hidden_state.shape)}"
                )

            if tuple(cell_state.shape) != expected_shape:
                raise ValueError(
                    "Expected cell state with shape "
                    f"{expected_shape}, received "
                    f"{tuple(cell_state.shape)}"
                )

        temporal_features, next_state = self.lstm(
            features,
            state,
        )

        return temporal_features, next_state

    @staticmethod
    def detach_state(state: LSTMState) -> LSTMState:
        hidden_state, cell_state = state

        return (
            hidden_state.detach(),
            cell_state.detach(),
        )