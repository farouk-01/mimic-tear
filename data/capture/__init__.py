from pydantic import BaseModel, ConfigDict
from functools import cached_property
from collections.abc import Iterator
from threading import Event
from typing import Self

from .synchronizer import CaptureSynchronizer, CaptureSample
from .gamepad import GamepadReader, GamepadReaderConfig
from .screen import CaptureRegion, ScreenReader, CapturedFrame, ScreenCaptureConfig


from controller import ControllerState


class CaptureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    gamepad: GamepadReaderConfig
    screen: ScreenCaptureConfig

    fps: float = 30.0


class Capture:
    def __init__(self, *, config: CaptureConfig) -> None:
        self.config = config
        self.fps = config.fps

    @cached_property
    def _gamepad(self) -> GamepadReader:
        return GamepadReader(**self.config.gamepad.model_dump())

    @cached_property
    def _screen(self) -> ScreenReader:
        return ScreenReader(**self.config.screen.model_dump())

    def capture_one_screen(self) -> CapturedFrame:
        return self._screen.read()

    def capture_one_gamepad(self) -> ControllerState:
        return self._gamepad.read()

    @cached_property
    def _synchronizer(self) -> CaptureSynchronizer:
        return CaptureSynchronizer(
            screen=self._screen,
            gamepad=self._gamepad,
            fps=self.fps,
        )

    def capture_one(self) -> CaptureSample:
        return CaptureSynchronizer(
            screen=self._screen,
            gamepad=self._gamepad,
            fps=self.fps,
        ).capture()

    def capture_stream(
        self, stop_event: Event | None = None
    ) -> Iterator[CaptureSample]:
        return self._synchronizer.run(stop_event=stop_event)

    def close(self) -> None:
        screen = self.__dict__.get("_screen")
        if screen is not None:
            screen.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


__all__ = [
    "CaptureSample",
    "CaptureSynchronizer",
    "CaptureConfig",
    "Capture",
]
