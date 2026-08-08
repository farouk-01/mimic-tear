from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mimic_tear.game_state.reader import EldenRingStateReader, GameStateSnapshot


@dataclass(frozen=True, slots=True)
class GameStateSample:
    timestamp_ns: int
    snapshot: GameStateSnapshot


@dataclass(frozen=True, slots=True)
class GameStateSamplerStats:
    sample_count: int
    valid_sample_count: int
    invalid_sample_count: int
    measured_hz: float
    missed_polls: int
    maximum_poll_duration_ms: float
    last_read_errors: tuple[str, ...]


def nearest_game_state_sample(
    samples: Sequence[GameStateSample],
    timestamp_ns: int,
) -> GameStateSample:
    if not samples:
        raise ValueError("At least one game-state sample is required")
    return min(samples, key=lambda sample: abs(sample.timestamp_ns - timestamp_ns))


class GameStateSampler:
    """Poll process memory independently and align snapshots to frame clocks."""

    def __init__(
        self,
        reader_factory: Callable[[], EldenRingStateReader],
        *,
        polling_hz: int,
    ) -> None:
        if polling_hz <= 0:
            raise ValueError("polling_hz must be greater than zero")
        self._reader_factory = reader_factory
        self._polling_hz = polling_hz
        self._period_ns = round(1_000_000_000 / polling_hz)
        self._samples: deque[GameStateSample] = deque(maxlen=polling_hz * 4)
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._sample_count = 0
        self._valid_sample_count = 0
        self._missed_polls = 0
        self._maximum_poll_duration_ns = 0
        self._first_sample_ns: int | None = None
        self._last_sample_ns: int | None = None
        self._last_read_errors: tuple[str, ...] = ()

    def start(self, *, timeout_seconds: float = 3.0) -> None:
        if self._thread is not None:
            raise RuntimeError("Game-state sampler has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="GameStateSampler",
            daemon=True,
        )
        self._thread.start()
        deadline = time.perf_counter() + timeout_seconds
        with self._condition:
            while not self._samples and self._error is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for the first game-state sample")
                self._condition.wait(timeout=remaining)
            self._raise_if_failed_locked()
            if not self._samples[0].snapshot.valid:
                details = "; ".join(self._samples[0].snapshot.read_errors)
                raise RuntimeError(
                    "The first game-state sample was invalid"
                    + (f": {details}" if details else "")
                )

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def closest(
        self,
        timestamp_ns: int,
        *,
        timeout_seconds: float,
    ) -> GameStateSample:
        deadline = time.perf_counter() + timeout_seconds
        with self._condition:
            while (
                (not self._samples or self._samples[-1].timestamp_ns < timestamp_ns)
                and self._error is None
            ):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            self._raise_if_failed_locked()
            if not self._samples:
                raise RuntimeError("No game-state samples are available")
            return nearest_game_state_sample(self._samples, timestamp_ns)

    def stats(self) -> GameStateSamplerStats:
        with self._condition:
            measured_hz = 0.0
            if (
                self._sample_count > 1
                and self._first_sample_ns is not None
                and self._last_sample_ns is not None
                and self._last_sample_ns > self._first_sample_ns
            ):
                measured_hz = (self._sample_count - 1) / (
                    (self._last_sample_ns - self._first_sample_ns) / 1e9
                )
            return GameStateSamplerStats(
                sample_count=self._sample_count,
                valid_sample_count=self._valid_sample_count,
                invalid_sample_count=self._sample_count - self._valid_sample_count,
                measured_hz=measured_hz,
                missed_polls=self._missed_polls,
                maximum_poll_duration_ms=self._maximum_poll_duration_ns / 1e6,
                last_read_errors=self._last_read_errors,
            )

    def _run(self) -> None:
        next_poll_ns = time.perf_counter_ns()
        reader: EldenRingStateReader | None = None
        try:
            reader = self._reader_factory()
            while not self._stop_event.is_set():
                now_ns = time.perf_counter_ns()
                if now_ns < next_poll_ns:
                    self._stop_event.wait((next_poll_ns - now_ns) / 1e9)
                    continue
                poll_started_ns = time.perf_counter_ns()
                snapshot = reader.read()
                poll_ended_ns = time.perf_counter_ns()
                sample = GameStateSample(poll_ended_ns, snapshot)
                poll_duration_ns = poll_ended_ns - poll_started_ns
                with self._condition:
                    self._samples.append(sample)
                    self._sample_count += 1
                    self._valid_sample_count += int(snapshot.valid)
                    self._last_read_errors = snapshot.read_errors
                    self._maximum_poll_duration_ns = max(
                        self._maximum_poll_duration_ns,
                        poll_duration_ns,
                    )
                    if self._first_sample_ns is None:
                        self._first_sample_ns = poll_ended_ns
                    self._last_sample_ns = poll_ended_ns
                    self._condition.notify_all()
                next_poll_ns += self._period_ns
                if poll_ended_ns > next_poll_ns:
                    skipped = (poll_ended_ns - next_poll_ns) // self._period_ns + 1
                    self._missed_polls += int(skipped)
                    next_poll_ns += skipped * self._period_ns
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
        finally:
            if reader is not None:
                reader.close()

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("Game-state sampler failed") from self._error
