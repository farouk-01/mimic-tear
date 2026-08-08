from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from mimic_tear.recording.schema import ANALOG_COLUMNS, BUTTON_COLUMNS


@dataclass(slots=True)
class ControllerState:
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    south: bool = False
    east: bool = False
    west: bool = False
    north: bool = False
    left_bumper: bool = False
    right_bumper: bool = False
    left_stick: bool = False
    right_stick: bool = False
    dpad_up: bool = False
    dpad_down: bool = False
    dpad_left: bool = False
    dpad_right: bool = False
    start: bool = False
    back: bool = False

    @classmethod
    def from_predictions(
        cls,
        analog: Sequence[float],
        buttons: Sequence[bool],
    ) -> ControllerState:
        if len(analog) != len(ANALOG_COLUMNS):
            raise ValueError(
                f"Expected {len(ANALOG_COLUMNS)} analog values, "
                f"received {len(analog)}."
            )
        if len(buttons) != len(BUTTON_COLUMNS):
            raise ValueError(
                f"Expected {len(BUTTON_COLUMNS)} button values, "
                f"received {len(buttons)}."
            )

        values: dict[str, float | bool] = {
            name: float(value)
            for name, value in zip(ANALOG_COLUMNS, analog, strict=True)
        }
        values.update(
            {
                name: bool(value)
                for name, value in zip(BUTTON_COLUMNS, buttons, strict=True)
            }
        )
        state = cls(**values)
        state.validate()
        return state

    def validate(self) -> None:
        _validate_range("left_x", self.left_x, -1.0, 1.0)
        _validate_range("left_y", self.left_y, -1.0, 1.0)
        _validate_range("right_x", self.right_x, -1.0, 1.0)
        _validate_range("right_y", self.right_y, -1.0, 1.0)
        _validate_range("left_trigger", self.left_trigger, 0.0, 1.0)
        _validate_range("right_trigger", self.right_trigger, 0.0, 1.0)


def _validate_range(name: str, value: float, low: float, high: float) -> None:
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
