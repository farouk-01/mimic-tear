from __future__ import annotations

from pydantic import BaseModel, Field, model_validator, ConfigDict
from pathlib import Path
import torch
import yaml


class ControllerInputsWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    # buttons
    south: float = 1.0
    east: float = 1.0
    west: float = 1.0
    north: float = 1.0
    left_bumper: float = 1.0
    right_bumper: float = 1.0
    left_stick: float = 1.0
    right_stick: float = 1.0
    dpad_up: float = 1.0
    dpad_down: float = 1.0
    dpad_left: float = 1.0
    dpad_right: float = 1.0
    start: float = 1.0
    back: float = 1.0

    # analogs
    left_stick_x: float = 1.0
    left_stick_y: float = 1.0
    right_stick_x: float = 1.0
    right_stick_y: float = 1.0
    left_trigger: float = 1.0
    right_trigger: float = 1.0

    @model_validator(mode="before")
    def validate_number_of_fields(self) -> ControllerInputsWeights:
        from controller.inputs import BUTTON_INPUTS, ANALOG_INPUTS

        expected_fields = set(BUTTON_INPUTS) | set(ANALOG_INPUTS)
        received_fields = ControllerInputsWeights.model_fields.keys()

        if expected_fields != received_fields:
            missing = expected_fields - received_fields
            extra = received_fields - expected_fields

            raise ValueError(
                f"Expected fields: {expected_fields}, "
                f"received: {received_fields}. "
                f"Missing: {missing}, Extra: {extra}"
            )

        return self

    @classmethod
    def global_weights(
        cls, btn_weight: float = 1.0, analog_weight: float = 1.0
    ) -> ControllerInputsWeights:
        from controller.inputs import BUTTON_INPUTS, ANALOG_INPUTS

        weights = {name: btn_weight for name in BUTTON_INPUTS}
        weights.update({name: analog_weight for name in ANALOG_INPUTS})

        return cls.model_validate(weights)

    @property
    def button_weights(self) -> torch.Tensor:
        from controller.inputs import BUTTON_INPUTS

        return torch.tensor([getattr(self, name) for name in BUTTON_INPUTS], dtype=torch.float32)

    @property
    def analog_weights(self) -> torch.Tensor:
        from controller.inputs import ANALOG_INPUTS

        return torch.tensor([getattr(self, name) for name in ANALOG_INPUTS], dtype=torch.float32)



class Hyperparameters(BaseModel):
    epochs: int = Field(default=20)
    learning_rate: float = Field(default=1e-4)
    weight_decay: float = Field(default=1e-4)

    use_amp: bool = Field(default=True)
    gradient_clip_norm: float | None = Field(default=1.0)

    controller_weights: ControllerInputsWeights = Field(default_factory=ControllerInputsWeights)

    @classmethod
    def load(cls, path: str | Path) -> Hyperparameters:
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        return cls.model_validate(cfg)
