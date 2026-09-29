from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from data.models.tensor import TensorSchema
from mimic_tear.model.components import (
    LSTM,
    Aggregator,
    Concat,
    ContinuousEncoder,
    Embedding,
    Encoder,
    Gamepad,
    Mean,
    ResNet18,
)
from mimic_tear.model.components.aggregator import Flatten

from .base import Policy

if TYPE_CHECKING:
    from configs.models.policy import AggregatorConfig, GameStateConfig, PolicyConfig

CATEGORICAL_KINDS = frozenset({"categorical", "nominal"})


def resnet_lstm(
    config: PolicyConfig,
    *,
    gstate_schema: TensorSchema,
    cardinalities: Mapping[str, int],
) -> Policy:
    if config.temporal is None:
        raise ValueError("resnet_lstm requires a temporal config")

    vision = ResNet18(config.vision.output_size, config.vision.weights_name)
    streams = [vision.output_size]

    encoders = None
    aggregator = None

    if config.game_state is not None:
        encoders = _build_encoders(config.game_state, gstate_schema, cardinalities)
        aggregator = _build_aggregator(
            config.game_state.aggregator,
            d_model=config.game_state.d_model,
            num_tokens=len(encoders),
        )
        streams.append(aggregator.output_size)

    fusion = None
    if config.fusion is not None:
        fusion = Concat(tuple(streams), config.fusion.output_size)
        size = fusion.output_size
    else:
        size = sum(streams)

    temporal = LSTM(
        input_size=size,
        hidden_size=config.temporal.hidden_size,
        num_layers=config.temporal.num_layers,
        dropout=config.temporal.dropout,
    )

    controller = Gamepad(temporal.output_size, config.controller.num_buttons)

    return Policy(
        vision=vision,
        controller=controller,
        temporal=temporal,
        encoders=encoders,
        aggregator=aggregator,
        fusion=fusion,
    )


def _build_encoders(
    config: GameStateConfig,
    schema: TensorSchema,
    cardinalities: Mapping[str, int],
) -> dict[str, Encoder]:
    encoders: dict[str, Encoder] = {}

    for field in schema.fields:
        if not field.is_model_input:
            continue

        if field.kind in CATEGORICAL_KINDS:
            if field.name not in cardinalities:
                raise ValueError(f"Missing cardinality for categorical field {field.name}")

            encoders[field.name] = Embedding(cardinalities[field.name], config.d_model)
        else:
            encoders[field.name] = ContinuousEncoder(1, config.d_model)

    return encoders


def _build_aggregator(
    config: AggregatorConfig,
    *,
    d_model: int,
    num_tokens: int,
) -> Aggregator:
    match config.type:
        case "mean":
            return Mean(d_model)
        case "flatten":
            return Flatten(d_model, num_tokens)


PRESETS = {
    "resnet_lstm": resnet_lstm,
}
