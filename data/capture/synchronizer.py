from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from threading import Event
from time import perf_counter_ns, sleep

from .gamepad.reader import GamepadReader
from .screen import ScreenReader, CapturedFrame
from .memory.game_state import GameStateReader, RawGameStateSnapshot

from data.models.gamepad import GamepadState


@dataclass(frozen=True, slots=True)
class CaptureSample:
    index: int
    frame: CapturedFrame
    controller: GamepadState
    game_state: RawGameStateSnapshot | None
    completed_ns: int

    @property
    def timestamp_ns(self) -> int:
        return self.frame.timestamp_ns

    @property
    def capture_duration_ns(self) -> int:
        return self.completed_ns - self.frame.timestamp_ns


class CaptureSynchronizer:
    def __init__(
        self,
        *,
        screen: ScreenReader,
        gamepad: GamepadReader,
        game_state: GameStateReader | None = None,
        fps: float = 30.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        self.screen = screen
        self.gamepad = gamepad
        self.game_state = game_state
        self.fps = fps

        self._period_ns = round(1_000_000_000 / fps)
        self._next_index = 0

    def capture(self) -> CaptureSample:
        frame = self.screen.read()
        controller = self.gamepad.read()

        game_state = (
            self.game_state.read()
            if self.game_state is not None
            else None
        )

        sample = CaptureSample(
            index=self._next_index,
            frame=frame,
            controller=controller,
            game_state=game_state,
            completed_ns=perf_counter_ns(),
        )

        self._next_index += 1
        return sample

    def run(
        self,
        *,
        stop_event: Event | None = None,
    ) -> Iterator[CaptureSample]:
        next_tick_ns = perf_counter_ns()

        while stop_event is None or not stop_event.is_set():
            yield self.capture()

            next_tick_ns += self._period_ns
            now_ns = perf_counter_ns()
            remaining_ns = next_tick_ns - now_ns

            if remaining_ns > 0:
                sleep(remaining_ns / 1_000_000_000)
                continue

            # If capture took too long,
            # skip expired deadlines instead of accumulating drift.
            missed_ticks = (
                (now_ns - next_tick_ns) // self._period_ns
            ) + 1

            next_tick_ns += missed_ticks * self._period_ns