from importlib import import_module
from time import perf_counter_ns

from pydantic import BaseModel, ConfigDict, NonNegativeFloat

from data.capture.sources.base import Reader
from data.models.schema import Schema, Field, NumpyType, Snapshot
from data.models.gamepad import ANALOG_INPUTS, BUTTON_INPUTS


class GamepadField(Field[NumpyType]):
    pass

GAMEPAD_SCHEMA = Schema[GamepadField](
    fields=(
        *(GamepadField(name=name, dtype="float32") for name in ANALOG_INPUTS),
        *(GamepadField(name=name, dtype="bool") for name in BUTTON_INPUTS),
    )
)


class GamepadReaderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    stick_deadzone: NonNegativeFloat


class GamepadReader(Reader):
    def __init__(self, stick_deadzone: float) -> None:
        super().__init__()

        native = import_module("ai_controller")
        self._gamepad = native.Controller(stick_deadzone)

    @property
    def schema(self) -> Schema[GamepadField]:
        return GAMEPAD_SCHEMA

    def read(self) -> Snapshot:
        self._ensure_open()

        if not self._gamepad.connected:
            raise RuntimeError("Gamepad disconnected")

        native = self._gamepad.poll()
        timestamp_ns = perf_counter_ns()
        values = (getattr(native, name) for name in self.schema.feature_names)

        return Snapshot(values, self.schema, timestamp_ns)