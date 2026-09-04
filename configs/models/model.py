from typing import Self
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from mimic_tear.model import LSTMPolicyConfig
from mimic_tear.model.components.controller import ControllerConfig
from mimic_tear.model.components.fusions import VectorFusionConfig
from mimic_tear.model.components.game_state import (
    GameStateConfig,
    GameStateFieldConfig,
    GameStateFieldKind,
)
from mimic_tear.model.components.temporal import TemporalConfig
from mimic_tear.model.components.vision import VisionConfig
from data.models.tensor import TensorSchema, FieldKind

MODEL_FIELD_KINDS: dict[FieldKind, GameStateFieldKind] = {
    "binary": "numeric",
    "continuous": "numeric",
    "discrete": "numeric",
    "constant": "numeric",
    "ordinal": "numeric",
    "categorical": "categorical",
    "nominal": "categorical",
}


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    d_model: int = Field(gt=0)

    vision: VisionConfig
    temporal: TemporalConfig
    game_state: GameStateConfig
    fusion: VectorFusionConfig
    controller: ControllerConfig

    @property
    def policy(self) -> LSTMPolicyConfig:
        return LSTMPolicyConfig(
            vision=self.vision,
            temporal=self.temporal,
            game_state=self.game_state,
            fusion=self.fusion,
            controller=self.controller,
        )

    @classmethod
    def load(
        cls,
        raw_model: dict,
        *,
        gstate_schema: TensorSchema,
        encoding_cardinalities: Mapping[str, int],
    ) -> Self:
        vision = VisionConfig.model_validate(raw_model["vision"])

        temporal = TemporalConfig.model_validate(
            {**raw_model["temporal"], "input_features": vision.output_features}
        )

        fields: list[GameStateFieldConfig] = []
        for field in gstate_schema.fields:
            if not field.is_model_input:
                continue

            kind = MODEL_FIELD_KINDS[field.kind]

            # later this might change to keyed by kind
            cardinality = encoding_cardinalities.get(field.name, None)
            if kind == "categorical" and cardinality is None:
                raise ValueError(
                    f"Missing encoding cardinality for categorical field {field.name}"
                )

            fields.append(
                GameStateFieldConfig(
                    name=field.name,
                    kind=kind,
                    cardinality=cardinality,
                )
            )

        d_model = raw_model["d_model"]
        game_state = GameStateConfig(fields=tuple(fields), d_model=d_model)

        fusion = VectorFusionConfig.model_validate(
            {
                "input_features": (temporal.hidden_features, d_model),
                **raw_model["fusion"],
            }
        )

        controller = ControllerConfig.model_validate(
            {**raw_model["controller"], "input_features": fusion.output_features}
        )

        return cls(
            d_model=d_model,
            vision=vision,
            temporal=temporal,
            game_state=game_state,
            fusion=fusion,
            controller=controller,
        )
