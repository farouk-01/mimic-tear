from collections.abc import Iterator
from contextlib import ExitStack
from functools import cached_property
from threading import Event
from types import TracebackType
from typing import Self

from pydantic import BaseModel, ConfigDict

from data.models.schema import Snapshot

from .synchronizer import CaptureSynchronizer, CaptureSample
from .sources.controller import GamepadReader, GamepadReaderConfig
from .sources.screen import CapturedFrame, ScreenReader, ScreenCaptureConfig
from .sources.memory import EldenRingReader, EldenRingMemoryProfile

__all__ = [
    "CaptureSample",
    "CaptureSynchronizer",
    "CaptureConfig",
    "Capture",
    "EldenRingMemoryProfile",
    "GamepadReaderConfig",
    "ScreenCaptureConfig",
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
        self._ensure_open()
        return self._stack.enter_context(
            GamepadReader(**self.config.gamepad.model_dump())
        )

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

    @cached_property
    def _synchronizer(self) -> CaptureSynchronizer:
        return CaptureSynchronizer(
            screen=self._screen,
            gamepad=self._gamepad,
            game_state=self._game_state,
            fps=self.fps,
        )

    def capture_one_screen(self) -> CapturedFrame:
        self._ensure_open()
        return self._screen.read()

    def capture_one_gamepad(self) -> Snapshot:
        self._ensure_open()
        return self._gamepad.read()

    def capture_one_game_state(self) -> Snapshot:
        self._ensure_open()
        return self._game_state.read()

    def capture_one(self) -> CaptureSample:
        self._ensure_open()
        return self._synchronizer.capture()

    def capture_stream(
        self,
        *,
        stop_event: Event | None = None,
        include_gamepad: bool = True,
    ) -> Iterator[CaptureSample]:
        self._ensure_open()
        synchronizer = CaptureSynchronizer(
            screen=self._screen,
            gamepad=self._gamepad if include_gamepad else None,
            game_state=self._game_state,
            fps=self.fps,
        )

        return synchronizer.run(stop_event=stop_event)

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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
