from __future__ import annotations

from importlib import import_module
from typing import Any
from pydantic import ConfigDict, BaseModel, NonNegativeFloat

from data.models.gamepad import (
    AnalogState,
    ButtonState,
    GamepadState,
)


class GamepadReaderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    stick_deadzone: NonNegativeFloat


class GamepadReader:
    def __init__(self, stick_deadzone: float) -> None:
        native = import_module("ai_controller")
        self._gamepad: Any = native.Controller(stick_deadzone)

    @property
    def name(self) -> str:
        return self._gamepad.name

    @property
    def connected(self) -> bool:
        return self._gamepad.connected

    def read(self) -> GamepadState:
        if not self.connected:
            raise RuntimeError("Gamepad disconnected")

        native = self._gamepad.poll()

        state = GamepadState(
            analog=AnalogState(
                left_x=native.left_x,
                left_y=native.left_y,
                right_x=native.right_x,
                right_y=native.right_y,
                left_trigger=native.left_trigger,
                right_trigger=native.right_trigger,
            ),
            buttons=ButtonState(
                south=native.south,
                east=native.east,
                west=native.west,
                north=native.north,
                left_bumper=native.left_bumper,
                right_bumper=native.right_bumper,
                left_stick=native.left_stick,
                right_stick=native.right_stick,
                dpad_up=native.dpad_up,
                dpad_down=native.dpad_down,
                dpad_left=native.dpad_left,
                dpad_right=native.dpad_right,
                start=native.start,
                back=native.back,
            ),
        )

        state.validate()
        return state
