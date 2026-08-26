from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO

from data.models.gamepad import AnalogState, ButtonState, GamepadState


class GamepadWriter:
    def __init__(self, pipe_name: str = "mimic-tear-controller") -> None:
        self.path = Path(rf"\\.\pipe\{pipe_name}")
        self._pipe: BinaryIO | None = None

    def connect(self) -> None:
        if self._pipe is not None:
            return

        self._pipe = open(self.path, "r+b", buffering=0)

        ready = self._pipe.readline()

        if not ready:
            self.close()
            raise RuntimeError("Controller bridge disconnected before ready")

        message = json.loads(ready.decode("utf-8"))

        if message.get("type") != "ready":
            self.close()
            raise RuntimeError(f"Unexpected controller bridge response: {message}")

    def write(self, state: GamepadState) -> None:
        state.validate()

        payload = {
            "type": "state",
            "left_x": state.analog.left_x,
            "left_y": state.analog.left_y,
            "right_x": state.analog.right_x,
            "right_y": state.analog.right_y,
            "left_trigger": state.analog.left_trigger,
            "right_trigger": state.analog.right_trigger,
            "south": state.buttons.south,
            "east": state.buttons.east,
            "west": state.buttons.west,
            "north": state.buttons.north,
            "left_bumper": state.buttons.left_bumper,
            "right_bumper": state.buttons.right_bumper,
            "back": state.buttons.back,
            "start": state.buttons.start,
            "left_stick": state.buttons.left_stick,
            "right_stick": state.buttons.right_stick,
            "dpad_up": state.buttons.dpad_up,
            "dpad_down": state.buttons.dpad_down,
            "dpad_left": state.buttons.dpad_left,
            "dpad_right": state.buttons.dpad_right,
        }

        self._write(payload)

    def _ensure_connected(self) -> BinaryIO:
        if self._pipe is None:
            raise RuntimeError("Controller bridge is not connected")

        return self._pipe

    def reset(self) -> None:
        if self._pipe is not None:
            self._write({"type": "reset"})

    def close(self) -> None:
        if self._pipe is None:
            return

        try:
            self.reset()
            self._write({"type": "disconnect"})
        finally:
            self._pipe.close()
            self._pipe = None

    def _write(self, payload: dict[str, object]) -> None:
        pipe = self._ensure_connected()

        message = json.dumps(payload, separators=(",", ":")) + "\n"
        pipe.write(message.encode("utf-8"))

    def __enter__(self) -> GamepadWriter:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()
