from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
import yaml

from mimic_tear.model.components import (
    Controller,
    ControllerConfig,
    ControllerOutput,
    Fusion,
    FusionConfig,
    GameState,
    GameStateConfig,
    LSTMState,
    Temporal,
    TemporalConfig,
    Vision,
    VisionConfig,
)


class PolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    vision: VisionConfig
    temporal: TemporalConfig
    game_state: GameStateConfig | None
    fusion: FusionConfig | None
    controller: ControllerConfig


class EldenRingPolicy(nn.Module):
    def __init__(
        self,
        *,
        config: PolicyConfig,
    ) -> None:
        super().__init__()

        self.vision = Vision(**config.vision.model_dump())

        # important that input_feature of temporal is
        # equal to output_feature of vision
        self.temporal = Temporal(**config.temporal.model_dump())

        self.game_state: GameState | None = None
        if config.game_state is not None:
            self.game_state = GameState(**config.game_state.model_dump())

        self.fusion: Fusion | None = None
        if config.fusion is not None:
            self.fusion = Fusion(**config.fusion.model_dump())

        self.controller = Controller(**config.controller.model_dump())

    def forward(
        self,
        images: Tensor,
        game_state: Tensor | None = None,
        state: LSTMState | None = None,
    ) -> tuple[ControllerOutput, LSTMState]:
        if images.ndim != 5:
            raise ValueError(
                "Expected images with shape [B, T, 3, H, W], "
                f"received {tuple(images.shape)}"
            )

        batch_size, sequence_length, channels, height, width = images.shape

        if channels != 3:
            raise ValueError(f"Expected 3 RGB channels, received {channels}")

        if sequence_length <= 0:
            raise ValueError("Sequence length must be greater than zero")

        # combine dim 0 through 1 into one dimension
        # [B, T, 3, H, W] -> [B*T, 3, H, W]
        frames = images.flatten(0, 1)

        # [B*T, 3, H, W] -> [B*T, 3*H*W]
        visual_features = self.vision(frames)

        # [B*T, F] -> [B, T, F]
        visual_features = visual_features.reshape(
            batch_size,
            sequence_length,
            self.vision.output_features,
        )

        # [B, T, F] --LSTM--> [B, T, temporal_features]
        temporal_features, next_state = self.temporal(
            visual_features,
            state,
        )

        features = temporal_features

        if self.game_state is not None:
            if game_state is None:
                raise ValueError("This policy requires game-state input")

            if game_state.ndim != 3:
                raise ValueError(
                    "Expected game state with shape [B, T, F], "
                    f"received {tuple(game_state.shape)}"
                )

            if game_state.shape[:2] != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "Image and game-state batch/sequence dimensions " "must match"
                )

            # [B, T, raw_state_features] -> [B, T, state_features]
            state_features = self.game_state(game_state)

            assert self.fusion is not None

            features = self.fusion(
                temporal_features,
                state_features,
            )

        elif game_state is not None:
            raise ValueError(
                "Game-state input was provided, but this policy "
                "was created without game-state support"
            )

        output = self.controller(features)

        return output, next_state

    @staticmethod
    def detach_state(
        state: LSTMState,
    ) -> LSTMState:
        return Temporal.detach_state(state)
