from __future__ import annotations
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor, nn
import torch

from mimic_tear.model.components import *

from utils import profile


class LSTMPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    vision: VisionConfig
    temporal: TemporalConfig
    game_state: StructuredDataConfig
    fusion: VectorFusionConfig
    controller: ControllerConfig

    @model_validator(mode="after")
    def validate_fusion(self) -> Self:
        if self.game_state is not None and self.fusion is None:
            raise ValueError(
                "Fusion config must be provided when game-state config is present"
            )

        if self.game_state is None and self.fusion is not None:
            raise ValueError(
                "Game-state config must be provided when fusion config is present"
            )

        return self


class LSTMPolicy(nn.Module):
    def __init__(
        self,
        *,
        config: LSTMPolicyConfig,
    ) -> None:
        super().__init__()
        self.config = config

        self.vision = Vision(**config.vision.model_dump())
        self.temporal = Temporal(**config.temporal.model_dump())
        self.game_state = StructuredData(
            fields=config.game_state.fields,
            d_model=config.game_state.d_model,
        )
        self.fusion = VectorFusion(**config.fusion.model_dump())
        self.controller = Controller(**config.controller.model_dump())

    @profile
    def forward(
        self,
        images: Tensor,
        structured_data: dict[str, Tensor],
        presence_mask: dict[str, Tensor],
        state: LSTMState,
    ) -> tuple[ControllerOutput, LSTMState]:
        if images.ndim != 5:
            raise ValueError(
                "Expected images with shape [B, T, 3, H, W], "
                f"received {tuple(images.shape)}"
            )

        batch_size, sequence_length, channels, _, _ = images.shape

        if channels != 3:
            raise ValueError(f"Expected 3 RGB channels, received {channels}")

        if sequence_length <= 0:
            raise ValueError("Sequence length must be greater than zero")

        # [B, T, 3, H, W] -> [B*T, 3, H, W]
        frames = images.flatten(0, 1)

        # [B*T, 3, H, W] -> [B*T, F]
        visual_features = self.vision(frames)

        # [B*T, F] -> [B, T, F]
        visual_features = visual_features.reshape(
            batch_size,
            sequence_length,
            self.vision.output_features,
        )

        # [B, T, F] -> [B, T, H]
        temporal_features, next_state = self.temporal(visual_features, state)

        # [B, T, N, D]
        state_tokens = self.game_state(structured_data, presence_mask)
        # [B, T, N*D]
        # TODO : use TokenFusion instead
        state_tokens = state_tokens.mean(dim=-2)

        features = self.fusion(temporal_features, state_tokens)

        output = self.controller(features)

        return output, next_state

    @staticmethod
    def detach_state(
        state: LSTMState,
    ) -> LSTMState:
        return Temporal.detach_state(state)
