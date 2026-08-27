from typing import Self

from pydantic import BaseModel, ConfigDict

from mimic_tear.model import PolicyConfig
from mimic_tear.model.components.controller import ControllerConfig
from mimic_tear.model.components.fusion import FusionConfig
from mimic_tear.model.components.game_state import GameStateConfig
from mimic_tear.model.components.temporal import TemporalConfig
from mimic_tear.model.components.vision import VisionConfig


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    vision: VisionConfig
    temporal: TemporalConfig
    game_state: GameStateConfig
    fusion: FusionConfig
    controller: ControllerConfig

    @property
    def policy(self) -> PolicyConfig:
        return PolicyConfig(
            vision=self.vision,
            temporal=self.temporal,
            game_state=self.game_state,
            fusion=self.fusion,
            controller=self.controller,
        )

    @classmethod
    def load(cls, raw_model: dict, *, game_state_features: int) -> Self:
        vision = VisionConfig.model_validate(raw_model["vision"])

        temporal = TemporalConfig.model_validate(
            {**raw_model["temporal"], "input_features": vision.output_features}
        )

        game_state = GameStateConfig.model_validate(
            {**raw_model["game_state"], "input_features": game_state_features}
        )

        fusion = FusionConfig.model_validate(
            {
                **raw_model["fusion"],
                "input_features": (
                    temporal.hidden_features,
                    game_state.output_features,
                ),
            }
        )

        controller = ControllerConfig.model_validate(
            {**raw_model["controller"], "input_features": fusion.output_features}
        )

        return cls(
            vision=vision,
            temporal=temporal,
            game_state=game_state,
            fusion=fusion,
            controller=controller,
        )
