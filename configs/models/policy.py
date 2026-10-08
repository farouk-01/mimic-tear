from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _SlotConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


# ---------------------------------------------------------------- vision


class ResNet18Config(_SlotConfig):
    type: Literal["resnet18"]
    output_size: int = Field(gt=0)
    weights_name: str | None = "DEFAULT"


type VisionConfig = Annotated[ResNet18Config, Field(discriminator="type")]


# ---------------------------------------------------------------- game state


class MeanConfig(_SlotConfig):
    type: Literal["mean"]


class FlattenConfig(_SlotConfig):
    type: Literal["flatten"]


type AggregatorConfig = Annotated[
    MeanConfig | FlattenConfig, Field(discriminator="type")
]


class GameStateConfig(_SlotConfig):
    d_model: int = Field(gt=0)
    aggregator: AggregatorConfig


# ---------------------------------------------------------------- fusion


class ConcatConfig(_SlotConfig):
    type: Literal["concat"]
    output_size: int = Field(gt=0)


type FusionConfig = Annotated[ConcatConfig, Field(discriminator="type")]


# ---------------------------------------------------------------- temporal


class LSTMConfig(_SlotConfig):
    type: Literal["lstm"]
    hidden_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0)


type TemporalConfig = Annotated[LSTMConfig, Field(discriminator="type")]


# ---------------------------------------------------------------- controller


class GamepadConfig(_SlotConfig):
    type: Literal["gamepad"]
    num_buttons: int = Field(gt=0)


type ControllerConfig = Annotated[GamepadConfig, Field(discriminator="type")]


# ---------------------------------------------------------------- policy


class PolicyConfig(_SlotConfig):
    preset: Literal["resnet_lstm"]

    vision: VisionConfig
    controller: ControllerConfig

    game_state: GameStateConfig | None = None
    fusion: FusionConfig | None = None
    temporal: TemporalConfig | None = None
