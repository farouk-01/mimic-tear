from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .inputs import ANALOG_INPUTS, BUTTON_INPUTS


@dataclass(frozen=True, slots=True)
class AnalogState:
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    left_trigger: float = 0.0
    right_trigger: float = 0.0

    def validate(self) -> None:
        _validate_range("left_x", self.left_x, -1.0, 1.0)
        _validate_range("left_y", self.left_y, -1.0, 1.0)
        _validate_range("right_x", self.right_x, -1.0, 1.0)
        _validate_range("right_y", self.right_y, -1.0, 1.0)
        _validate_range("left_trigger", self.left_trigger, 0.0, 1.0)
        _validate_range("right_trigger", self.right_trigger, 0.0, 1.0)

    def values(self) -> tuple[float, ...]:
        return tuple(
            float(getattr(self, name))
            for name in ANALOG_INPUTS
        )

    @classmethod
    def from_values(
        cls,
        values: Sequence[float],
    ) -> AnalogState:
        if len(values) != len(ANALOG_INPUTS):
            raise ValueError(
                f"Expected {len(ANALOG_INPUTS)} analog values, "
                f"received {len(values)}"
            )

        state = cls(
            left_x=float(values[0]),
            left_y=float(values[1]),
            right_x=float(values[2]),
            right_y=float(values[3]),
            left_trigger=float(values[4]),
            right_trigger=float(values[5]),
        )

        state.validate()
        return state


@dataclass(frozen=True, slots=True)
class ButtonState:
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

    def values(self) -> tuple[bool, ...]:
        return tuple(
            bool(getattr(self, name))
            for name in BUTTON_INPUTS
        )

    @classmethod
    def from_values(
        cls,
        values: Sequence[bool],
    ) -> ButtonState:
        if len(values) != len(BUTTON_INPUTS):
            raise ValueError(
                f"Expected {len(BUTTON_INPUTS)} button values, "
                f"received {len(values)}"
            )

        return cls(
            south=bool(values[0]),
            east=bool(values[1]),
            west=bool(values[2]),
            north=bool(values[3]),
            left_bumper=bool(values[4]),
            right_bumper=bool(values[5]),
            left_stick=bool(values[6]),
            right_stick=bool(values[7]),
            dpad_up=bool(values[8]),
            dpad_down=bool(values[9]),
            dpad_left=bool(values[10]),
            dpad_right=bool(values[11]),
            start=bool(values[12]),
            back=bool(values[13]),
        )


@dataclass(frozen=True, slots=True)
class ControllerState:
    analog: AnalogState = AnalogState()
    buttons: ButtonState = ButtonState()

    def validate(self) -> None:
        self.analog.validate()

    @classmethod
    def from_values(
        cls,
        *,
        analog: Sequence[float],
        buttons: Sequence[bool],
    ) -> ControllerState:
        return cls(
            analog=AnalogState.from_values(analog),
            buttons=ButtonState.from_values(buttons),
        )


def _validate_range(
    name: str,
    value: float,
    low: float,
    high: float,
) -> None:
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(
            f"{name} must be finite and in [{low}, {high}]"
        )