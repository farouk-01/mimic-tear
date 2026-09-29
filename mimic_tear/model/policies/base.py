from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from tensordict import TensorDict
import torch
from torch import Tensor, nn

from data.process import PresenceMasks
from mimic_tear.model.components import (
    Aggregator,
    ControllerOutput,
    Controller,
    Encoder,
    Fusion,
    Temporal,
    Vision,
)

from utils import profile

@dataclass(frozen=True, slots=True)
class PolicyInputs:
    images: Tensor
    game_state: TensorDict | None
    presence_mask: PresenceMasks | None

    @classmethod
    def from_batch(
        cls,
        batch: TensorDict,
        presence: PresenceMasks | None,
    ) -> Self:
        images = batch["video", "frames"]
        gstate = batch.get("game_state")

        return cls(images=images, game_state=gstate, presence_mask=presence)


class BasePolicy[State](nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        inputs: PolicyInputs,
        state: State | None = None,
    ) -> tuple[ControllerOutput, State | None]: ...

    @abstractmethod
    def detach_state(self, state: State | None) -> State | None: ...


class Policy[State](BasePolicy[State]):
    def __init__(
        self,
        *,
        vision: Vision,
        controller: Controller,
        temporal: Temporal[State] | None = None,
        encoders: dict[str, Encoder] | None = None,
        aggregator: Aggregator | None = None,
        fusion: Fusion | None = None,
    ) -> None:
        super().__init__()

        self.vision = vision
        self.controller = controller

        self.temporal = temporal
        self.encoders = nn.ModuleDict(encoders) if encoders else None

        self.aggregator = aggregator
        self.fusion = fusion

        self.missing = (
            nn.ParameterDict(
                {
                    name: nn.Parameter(torch.zeros(encoder.output_size))
                    for name, encoder in encoders.items()
                }
            )
            if encoders
            else None
        )

    @profile
    def forward(
        self,
        inputs: PolicyInputs,
        state: State | None = None,
    ) -> tuple[ControllerOutput, State | None]:
        features = []

        images = inputs.images
        b, t = inputs.images.shape[:2]

        game_state = inputs.game_state

        vision_features = self._apply_vision(images, b, t)
        features.append(vision_features)

        masks = (inputs.presence_mask or {}).get("game_state")

        if self.encoders is not None:
            features.append(self._encode_game_state(game_state, masks, b, t))

        x = (
            self.fusion(*features)
            if self.fusion is not None
            else torch.cat(features, dim=-1)
        )

        if self.temporal is not None:
            x, state = self.temporal(x, state)

        predictions = self.controller(x)

        return predictions, state

    def _apply_vision(self, images: Tensor, b: int, t: int) -> Tensor:
        # [B, T, 3, H, W] -> [B*T, 3, H, W]
        frames = images.flatten(0, 1)

        # [B*T, F] -> [B, T, F]
        vision_features = self.vision(frames).unflatten(0, (b, t))

        return vision_features

    def _encode_game_state(
        self,
        game_state: TensorDict | None,
        masks: dict[str, Tensor] | None,
        b: int,
        t: int,
    ) -> Tensor:
        tokens = []

        if self.encoders is None:
            raise ValueError("No encoders defined for game state.")

        if self.missing is None:
            raise ValueError("Missing encoders not defined.")

        for name, encoder in self.encoders.items():
            if (
                game_state is not None
                and masks is not None
                and bool(masks.get(name, False))
            ):
                tokens.append(encoder(game_state[name]))
            else:
                tokens.append(self.missing[name].expand(b, t, -1))

        return (
            self.aggregator(torch.stack(tokens, dim=-2))
            if self.aggregator
            else torch.stack(tokens, dim=-2)
        )

    def detach_state(self, state: State | None) -> State | None:
        if self.temporal is None or state is None:
            return None

        return self.temporal.detach_state(state)
