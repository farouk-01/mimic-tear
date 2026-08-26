from pydantic import BaseModel, ConfigDict
from contextlib import ExitStack
from functools import cached_property
from collections.abc import Iterator
from threading import Event
from typing import Self

from .synchronizer import CaptureSynchronizer, CaptureSample
from .gamepad import GamepadReader, GamepadReaderConfig
from .screen import ScreenReader, CapturedFrame, ScreenCaptureConfig
from .memory import EldenRingReader, EldenRingMemoryProfile, GameStateSnapshot


from data.models.gamepad import GamepadState

__all__ = [
    "CaptureSample",
    "CaptureSynchronizer",
    "CaptureConfig",
    "Capture",
]


class CaptureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    gamepad: GamepadReaderConfig
    screen: ScreenCaptureConfig
    game_state_profile: EldenRingMemoryProfile

    fps: float = 30.0


class Capture:
    def __init__(self, *, config: CaptureConfig) -> None:
        self.config = config
        self.fps = config.fps
        self._stack = ExitStack()
        self._closed = False

    @cached_property
    def _gamepad(self) -> GamepadReader:
        return GamepadReader(**self.config.gamepad.model_dump())

    @cached_property
    def _screen(self) -> ScreenReader:
        self._ensure_open()
        return self._stack.enter_context(
            ScreenReader(**self.config.screen.model_dump())
        )

    @cached_property
    def _game_state(self) -> EldenRingReader:
        self._ensure_open()
        return self._stack.enter_context(
            EldenRingReader.open(self.config.game_state_profile)
        )

    def capture_one_screen(self) -> CapturedFrame:
        self._ensure_open()
        return self._screen.read()

    def capture_one_gamepad(self) -> GamepadState:
        self._ensure_open()
        return self._gamepad.read()

    def capture_one_game_state(self) -> GameStateSnapshot:
        self._ensure_open()
        return self._game_state.read()

    @cached_property
    def _synchronizer(self) -> CaptureSynchronizer:
        return CaptureSynchronizer(
            screen=self._screen,
            gamepad=self._gamepad,
            game_state=self._game_state,
            fps=self.fps,
        )

    def capture_one(self) -> CaptureSample:
        self._ensure_open()
        return CaptureSynchronizer(
            screen=self._screen,
            gamepad=self._gamepad,
            game_state=self._game_state,
            fps=self.fps,
        ).capture()

    def capture_stream(
        self, stop_event: Event | None = None
    ) -> Iterator[CaptureSample]:
        self._ensure_open()
        return self._synchronizer.run(stop_event=stop_event)

    def close(self) -> None:
        if self._closed:
            return

        try:
            self._stack.close()
        finally:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Capture is closed")

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
