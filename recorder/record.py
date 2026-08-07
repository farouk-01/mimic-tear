from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import dxcam
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
NATIVE_MODULE_DIRECTORY = PROJECT_ROOT / "native" / "build" / "Release"

sys.path.insert(0, str(SOURCE_DIRECTORY))
sys.path.insert(0, str(NATIVE_MODULE_DIRECTORY))
sys.path.insert(0, str(PROJECT_ROOT))

import ai_controller  # noqa: E402
from ai_player.recording_schema import (  # noqa: E402
    ANALOG_COLUMNS,
    BUTTON_COLUMNS,
    INPUT_COLUMNS,
    PARQUET_SCHEMA,
)
from ai_player.game_state import (  # noqa: E402
    EldenRingStateReader,
    load_memory_profile,
)
from recorder.game_state_capture import (  # noqa: E402
    GameStateSampler,
    GameStateWriter,
)
from recorder.hotkey import GlobalHotkey, parse_hotkey  # noqa: E402


PREVIEW_TITLE = "AI Player Recorder"
RECORDING_LABEL_PATTERN = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
DATASET_SPLITS = ("train", "validation")


def validate_recording_label(value: str, *, field_name: str) -> None:
    if not RECORDING_LABEL_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must use lowercase letters, numbers, hyphens, or "
            f"underscores and cannot begin or end with punctuation"
        )


def validate_recording_theme(value: str) -> None:
    segments = value.split("/")
    if not segments:
        raise ValueError("theme must contain at least one path segment")
    for segment in segments:
        validate_recording_label(segment, field_name="each theme path segment")


def parse_recording_label(value: str) -> str:
    try:
        validate_recording_label(value, field_name="value")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def parse_recording_theme(value: str) -> str:
    try:
        validate_recording_theme(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


@dataclass(frozen=True, slots=True)
class RecorderConfig:
    theme: str
    tag: str
    split: str
    labels: tuple[str, ...] = ()
    output_directory: Path = PROJECT_ROOT / "recordings"
    fps: int = 30
    width: int = 640
    height: int = 360
    device_index: int = 0
    monitor_index: int = 0
    region: tuple[int, int, int, int] | None = None
    codec: str = "mp4v"
    countdown_seconds: int = 3
    preview: bool = True
    stick_deadzone: float = 0.12
    maximum_duration_seconds: float | None = None
    minimum_free_gb: float = 2.0
    input_hz: int = 250
    maximum_sync_offset_ms: float = 15.0
    game_state_profile: Path | None = None
    game_state_hz: int = 60
    maximum_game_state_sync_offset_ms: float = 25.0
    write_csv: bool = False
    stop_hotkey: str = "F8"
    cancel_hotkey: str = "CTRL+F9"
    open_replay_after_recording: bool = True
    boss_loop: bool = False
    boss_reset_hotkey: str = "F10"
    boss_reset_profile: Path = PROJECT_ROOT / "recorder" / "profiles" / "elden_ring.json"
    boss_reset_timeout_seconds: float = 45.0
    boss_title_settle_seconds: float = 2.0
    boss_snapshot_delay_seconds: float = 1.0
    boss_gameplay_settle_seconds: float = 2.0
    boss_episodes: int | None = None

    def __post_init__(self) -> None:
        validate_recording_theme(self.theme)
        validate_recording_label(self.tag, field_name="tag")
        if self.split not in DATASET_SPLITS:
            raise ValueError(f"split must be one of: {', '.join(DATASET_SPLITS)}")
        if self.boss_reset_timeout_seconds <= 0:
            raise ValueError("boss reset timeout must be greater than zero")
        if min(
            self.boss_title_settle_seconds,
            self.boss_snapshot_delay_seconds,
            self.boss_gameplay_settle_seconds,
        ) < 0:
            raise ValueError("boss reset delays cannot be negative")
        if self.boss_episodes is not None and self.boss_episodes <= 0:
            raise ValueError("boss episodes must be greater than zero")
        for label in self.labels:
            validate_recording_label(label, field_name="label")
        if self.fps <= 0:
            raise ValueError("fps must be greater than zero")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be greater than zero")
        if len(self.codec) != 4:
            raise ValueError("codec must contain exactly four characters")
        if self.countdown_seconds < 0:
            raise ValueError("countdown cannot be negative")
        if not 0.0 <= self.stick_deadzone < 1.0:
            raise ValueError("stick deadzone must be in [0, 1)")
        if (
            self.maximum_duration_seconds is not None
            and self.maximum_duration_seconds <= 0
        ):
            raise ValueError("maximum duration must be greater than zero")
        if self.minimum_free_gb < 0:
            raise ValueError("minimum free space cannot be negative")
        if self.input_hz < self.fps:
            raise ValueError("input polling rate must be at least the video FPS")
        if self.maximum_sync_offset_ms <= 0:
            raise ValueError("maximum sync offset must be greater than zero")
        if self.game_state_profile is not None and self.game_state_hz < self.fps:
            raise ValueError("game-state polling rate must be at least the video FPS")
        if self.maximum_game_state_sync_offset_ms <= 0:
            raise ValueError("maximum game-state sync offset must be greater than zero")
        try:
            parse_hotkey(self.stop_hotkey)
            parse_hotkey(self.cancel_hotkey)
        except ValueError as error:
            raise ValueError(str(error)) from error


@dataclass(frozen=True, slots=True)
class InputRow:
    frame_index: int
    timestamp_ns: int
    source_timestamp_ns: int
    frame_timestamp_ns: int
    input_timestamp_ns: int
    input_offset_ns: int

    left_x: float
    left_y: float
    right_x: float
    right_y: float
    left_trigger: float
    right_trigger: float

    south: bool
    east: bool
    west: bool
    north: bool
    left_bumper: bool
    right_bumper: bool
    left_stick: bool
    right_stick: bool
    dpad_up: bool
    dpad_down: bool
    dpad_left: bool
    dpad_right: bool
    start: bool
    back: bool

    @classmethod
    def from_state(
        cls,
        *,
        frame_index: int,
        timestamp_ns: int,
        source_timestamp_ns: int,
        frame_timestamp_ns: int,
        input_timestamp_ns: int,
        state: ai_controller.ControllerState,
    ) -> "InputRow":
        return cls(
            frame_index=frame_index,
            timestamp_ns=timestamp_ns,
            source_timestamp_ns=source_timestamp_ns,
            frame_timestamp_ns=frame_timestamp_ns,
            input_timestamp_ns=input_timestamp_ns,
            input_offset_ns=input_timestamp_ns - frame_timestamp_ns,
            left_x=state.left_x,
            left_y=state.left_y,
            right_x=state.right_x,
            right_y=state.right_y,
            left_trigger=state.left_trigger,
            right_trigger=state.right_trigger,
            south=state.south,
            east=state.east,
            west=state.west,
            north=state.north,
            left_bumper=state.left_bumper,
            right_bumper=state.right_bumper,
            left_stick=state.left_stick,
            right_stick=state.right_stick,
            dpad_up=state.dpad_up,
            dpad_down=state.dpad_down,
            dpad_left=state.dpad_left,
            dpad_right=state.dpad_right,
            start=state.start,
            back=state.back,
        )

    def as_csv_row(self) -> list[int | float]:
        values: list[int | float] = []
        for field in fields(self):
            value = getattr(self, field.name)
            values.append(int(value) if isinstance(value, bool) else value)
        return values

    def as_record(self) -> dict[str, int | float | bool]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


class InputWriters:
    """Write canonical Parquet inputs and an optional CSV mirror."""

    def __init__(
        self,
        parquet_path: Path,
        *,
        csv_path: Path | None = None,
        row_group_size: int = 300,
    ) -> None:
        if row_group_size <= 0:
            raise ValueError("row_group_size must be greater than zero")
        self._parquet_writer = pq.ParquetWriter(
            parquet_path,
            PARQUET_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        self._row_group_size = row_group_size
        self._buffer: list[dict[str, int | float | bool]] = []
        self._csv_file = None
        self._csv_writer = None
        self._closed = False
        try:
            if csv_path is not None:
                self._csv_file = csv_path.open("w", newline="", encoding="utf-8")
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(INPUT_COLUMNS)
        except Exception:
            self._parquet_writer.close()
            raise

    def write(self, row: InputRow) -> None:
        self._buffer.append(row.as_record())
        if self._csv_writer is not None:
            self._csv_writer.writerow(row.as_csv_row())
        if len(self._buffer) >= self._row_group_size:
            self._flush_parquet()

    def flush_csv(self) -> None:
        if self._csv_file is not None:
            self._csv_file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._flush_parquet()
        finally:
            try:
                self._parquet_writer.close()
            finally:
                if self._csv_file is not None:
                    self._csv_file.close()

    def _flush_parquet(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=PARQUET_SCHEMA)
        self._parquet_writer.write_table(table)
        self._buffer.clear()

    def __enter__(self) -> "InputWriters":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class RecordingStats:
    frame_count: int
    elapsed_seconds: float
    measured_fps: float
    dropped_frames_estimate: int
    synchronization_drops: int
    capture_failures: int
    largest_frame_gap_ms: float
    stop_reason: str
    input_statistics: dict[str, Any]
    input_sample_count: int
    input_sample_rate_hz: float
    missed_input_polls: int
    mean_abs_input_offset_ms: float
    maximum_abs_input_offset_ms: float


@dataclass(frozen=True, slots=True)
class RecordingResult:
    statistics: RecordingStats
    source_width: int
    source_height: int
    source_channels: int
    source_dtype: str


@dataclass(slots=True)
class InputStatisticsAccumulator:
    frame_count: int = 0
    active_frames: dict[str, int] = field(
        default_factory=lambda: {
            column: 0 for column in (*ANALOG_COLUMNS, *BUTTON_COLUMNS)
        }
    )
    button_press_counts: dict[str, int] = field(
        default_factory=lambda: {column: 0 for column in BUTTON_COLUMNS}
    )
    previous_buttons: dict[str, bool] = field(
        default_factory=lambda: {column: False for column in BUTTON_COLUMNS}
    )
    analog_sums: dict[str, float] = field(
        default_factory=lambda: {column: 0.0 for column in ANALOG_COLUMNS}
    )
    analog_squared_sums: dict[str, float] = field(
        default_factory=lambda: {column: 0.0 for column in ANALOG_COLUMNS}
    )
    analog_minimums: dict[str, float] = field(
        default_factory=lambda: {column: math.inf for column in ANALOG_COLUMNS}
    )
    analog_maximums: dict[str, float] = field(
        default_factory=lambda: {column: -math.inf for column in ANALOG_COLUMNS}
    )

    def add(self, state: Any) -> None:
        for column in ANALOG_COLUMNS:
            value = float(getattr(state, column))
            if abs(value) > 1e-4:
                self.active_frames[column] += 1
            self.analog_sums[column] += value
            self.analog_squared_sums[column] += value * value
            self.analog_minimums[column] = min(self.analog_minimums[column], value)
            self.analog_maximums[column] = max(self.analog_maximums[column], value)

        for column in BUTTON_COLUMNS:
            pressed = bool(getattr(state, column))
            if pressed:
                self.active_frames[column] += 1
            if pressed and not self.previous_buttons[column]:
                self.button_press_counts[column] += 1
            self.previous_buttons[column] = pressed

        self.frame_count += 1

    def summary(self, *, duration_seconds: float | None = None) -> dict[str, Any]:
        presses_per_minute = {
            column: (
                count * 60.0 / duration_seconds
                if duration_seconds is not None and duration_seconds > 0.0
                else 0.0
            )
            for column, count in self.button_press_counts.items()
        }
        if self.frame_count == 0:
            return {
                "active_frames": dict(self.active_frames),
                "active_ratios": {column: 0.0 for column in self.active_frames},
                "button_press_counts": dict(self.button_press_counts),
                "button_presses_per_minute": presses_per_minute,
                "analog": {
                    column: {"minimum": 0.0, "maximum": 0.0, "mean": 0.0, "std": 0.0}
                    for column in ANALOG_COLUMNS
                },
            }

        analog_statistics: dict[str, dict[str, float]] = {}
        for column in ANALOG_COLUMNS:
            mean = self.analog_sums[column] / self.frame_count
            variance = max(
                0.0,
                self.analog_squared_sums[column] / self.frame_count - mean * mean,
            )
            analog_statistics[column] = {
                "minimum": self.analog_minimums[column],
                "maximum": self.analog_maximums[column],
                "mean": mean,
                "std": math.sqrt(variance),
            }

        return {
            "active_frames": dict(self.active_frames),
            "active_ratios": {
                column: count / self.frame_count
                for column, count in self.active_frames.items()
            },
            "button_press_counts": dict(self.button_press_counts),
            "button_presses_per_minute": presses_per_minute,
            "analog": analog_statistics,
        }


@dataclass(frozen=True, slots=True)
class InputSample:
    timestamp_ns: int
    state: Any


@dataclass(frozen=True, slots=True)
class InputSamplerStats:
    sample_count: int
    measured_hz: float
    missed_polls: int
    maximum_poll_duration_ms: float


class ControllerSampler:
    """Poll a controller independently and retain timestamped states."""

    def __init__(
        self,
        controller_factory: Callable[[], Any],
        *,
        polling_hz: int,
    ) -> None:
        if polling_hz <= 0:
            raise ValueError("polling_hz must be greater than zero")
        self._controller_factory = controller_factory
        self._polling_hz = polling_hz
        self._period_ns = round(1_000_000_000 / polling_hz)
        self._samples: deque[InputSample] = deque(maxlen=polling_hz * 4)
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._disconnected = False
        self._sample_count = 0
        self._missed_polls = 0
        self._maximum_poll_duration_ns = 0
        self._first_sample_ns: int | None = None
        self._last_sample_ns: int | None = None
        self._controller_name: str | None = None

    @property
    def connected(self) -> bool:
        with self._condition:
            return not self._disconnected and self._error is None

    @property
    def controller_name(self) -> str:
        with self._condition:
            if self._controller_name is None:
                raise RuntimeError("Controller sampler has not initialized")
            return self._controller_name

    def start(self, *, timeout_seconds: float = 2.0) -> None:
        if self._thread is not None:
            raise RuntimeError("Controller sampler has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="ControllerSampler",
            daemon=True,
        )
        self._thread.start()
        deadline = time.perf_counter() + timeout_seconds
        with self._condition:
            while not self._samples and self._error is None and not self._disconnected:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for the first controller sample")
                self._condition.wait(timeout=remaining)
            self._raise_if_failed_locked()
            if not self._samples:
                raise RuntimeError("Controller disconnected before sampling began")

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def closest(
        self,
        timestamp_ns: int,
        *,
        timeout_seconds: float,
    ) -> InputSample:
        deadline = time.perf_counter() + timeout_seconds
        with self._condition:
            while (
                (not self._samples or self._samples[-1].timestamp_ns < timestamp_ns)
                and self._error is None
                and not self._disconnected
            ):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            self._raise_if_failed_locked()
            if not self._samples:
                raise RuntimeError("No controller samples are available")
            return nearest_input_sample(self._samples, timestamp_ns)

    def latest_at_or_before(self, timestamp_ns: int) -> InputSample | None:
        with self._condition:
            self._raise_if_failed_locked()

            return latest_input_sample_at_or_before(
                        self._samples,
                        timestamp_ns,
                    )

    def stats(self) -> InputSamplerStats:
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
            return InputSamplerStats(
                sample_count=self._sample_count,
                measured_hz=measured_hz,
                missed_polls=self._missed_polls,
                maximum_poll_duration_ms=self._maximum_poll_duration_ns / 1e6,
            )

    def _run(self) -> None:
        next_poll_ns = time.perf_counter_ns()
        try:
            controller = self._controller_factory()
            with self._condition:
                self._controller_name = controller.name
                self._condition.notify_all()
            while not self._stop_event.is_set():
                now_ns = time.perf_counter_ns()
                if now_ns < next_poll_ns:
                    self._wait_until(next_poll_ns)
                    continue
                if not controller.connected:
                    with self._condition:
                        self._disconnected = True
                        self._condition.notify_all()
                    return

                poll_started_ns = time.perf_counter_ns()
                state = controller.poll()
                poll_ended_ns = time.perf_counter_ns()
                sample = InputSample(
                    timestamp_ns=(poll_started_ns + poll_ended_ns) // 2,
                    state=state,
                )
                with self._condition:
                    self._samples.append(sample)
                    self._sample_count += 1
                    self._maximum_poll_duration_ns = max(
                        self._maximum_poll_duration_ns,
                        poll_ended_ns - poll_started_ns,
                    )
                    if self._first_sample_ns is None:
                        self._first_sample_ns = sample.timestamp_ns
                    self._last_sample_ns = sample.timestamp_ns
                    self._condition.notify_all()

                next_poll_ns += self._period_ns
                current_ns = time.perf_counter_ns()
                if current_ns - next_poll_ns > self._period_ns:
                    skipped = (current_ns - next_poll_ns) // self._period_ns
                    with self._condition:
                        self._missed_polls += int(skipped)
                    next_poll_ns = current_ns + self._period_ns
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def _wait_until(self, deadline_ns: int) -> None:
        """Wait for a polling deadline without coarse Windows oversleep."""

        while not self._stop_event.is_set():
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns <= 0:
                return

            # Leave the final millisecond to a yield loop. Windows timer waits
            # can oversleep a 4 ms controller polling period by a full timer
            # quantum, which creates stale input/frame pairings.
            if remaining_ns > 1_000_000:
                self._stop_event.wait((remaining_ns - 1_000_000) / 1e9)
            else:
                time.sleep(0)

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("Controller input sampler failed") from self._error


def nearest_input_sample(
    samples: deque[InputSample] | list[InputSample] | tuple[InputSample, ...],
    timestamp_ns: int,
) -> InputSample:
    if not samples:
        raise ValueError("At least one input sample is required")
    return min(samples, key=lambda sample: abs(sample.timestamp_ns - timestamp_ns))

def latest_input_sample_at_or_before(
    samples: deque[InputSample],
    timestamp_ns: int,
) -> InputSample | None:
    for sample in reversed(samples):
        if sample.timestamp_ns <= timestamp_ns:
            return sample
    return None


class FrameClockSynchronizer:
    """Translate DXGI presentation ticks into perf_counter's clock domain."""

    def __init__(self) -> None:
        self._minimum_clock_offset_ns: int | None = None
        self._previous_source_ns: int | None = None
        self._previous_frame_ns: int | None = None

    def align(
        self,
        source_timestamp_seconds: float,
        received_timestamp_ns: int,
    ) -> tuple[int, int, bool]:
        source_timestamp_ns = round(source_timestamp_seconds * 1e9)
        clock_offset_ns = received_timestamp_ns - source_timestamp_ns
        if (
            self._minimum_clock_offset_ns is None
            or clock_offset_ns < self._minimum_clock_offset_ns
        ):
            self._minimum_clock_offset_ns = clock_offset_ns

        duplicate = source_timestamp_ns == self._previous_source_ns
        if duplicate:
            frame_timestamp_ns = received_timestamp_ns
        else:
            frame_timestamp_ns = source_timestamp_ns + self._minimum_clock_offset_ns
            frame_timestamp_ns = min(frame_timestamp_ns, received_timestamp_ns)

        if self._previous_frame_ns is not None:
            frame_timestamp_ns = max(frame_timestamp_ns, self._previous_frame_ns + 1)
        self._previous_source_ns = source_timestamp_ns
        self._previous_frame_ns = frame_timestamp_ns
        return source_timestamp_ns, frame_timestamp_ns, duplicate


class RecordingCancelled(RuntimeError):
    pass


def create_session_directories(
    output_directory: Path,
    *,
    split: str,
    theme: str,
    tag: str,
) -> tuple[Path, Path]:
    if split not in DATASET_SPLITS:
        raise ValueError(f"split must be one of: {', '.join(DATASET_SPLITS)}")
    validate_recording_theme(theme)
    validate_recording_label(tag, field_name="tag")
    theme_directory = output_directory / split / theme
    theme_directory.mkdir(parents=True, exist_ok=True)
    final_directory = theme_directory / tag
    if final_directory.exists():
        raise FileExistsError(
            f"Recording destination already exists: {final_directory}. "
            f"Choose a different --tag."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    staging_directory = theme_directory / f".{tag}.{timestamp}.inprogress"
    staging_directory.mkdir()
    return staging_directory, final_directory


def write_metadata(session_directory: Path, metadata: dict[str, Any]) -> None:
    path = session_directory / "metadata.json"
    temporary_path = session_directory / ".metadata.json.tmp"
    temporary_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def configuration_metadata(config: RecorderConfig) -> dict[str, Any]:
    data = asdict(config)
    data["output_directory"] = str(config.output_directory.resolve())
    data["game_state_profile"] = (
        str(config.game_state_profile.resolve())
        if config.game_state_profile is not None
        else None
    )
    data["boss_reset_profile"] = str(config.boss_reset_profile.resolve())
    data["region"] = list(config.region) if config.region is not None else None
    return data


def open_video_writer(path: Path, config: RecorderConfig) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*config.codec),
        config.fps,
        (config.width, config.height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def resize_frame(frame: Any, config: RecorderConfig) -> Any:
    return cv2.resize(
        frame,
        (config.width, config.height),
        interpolation=cv2.INTER_AREA,
    )


def warm_up_capture(
    camera: Any,
    frame_clock: FrameClockSynchronizer,
    *,
    sample_count: int,
) -> None:
    for _ in range(sample_count):
        result = camera.get_latest_frame(with_timestamp=True)
        if result is None:
            raise RuntimeError("Desktop capture stopped during warm-up")
        _, source_timestamp_seconds = result
        frame_clock.align(source_timestamp_seconds, time.perf_counter_ns())


def wait_for_countdown(
    camera: Any,
    config: RecorderConfig,
    frame_clock: FrameClockSynchronizer,
    input_sampler: ControllerSampler,
    stop_event: threading.Event,
    cancel_event: threading.Event,
) -> bool:
    if cancel_event.is_set():
        raise RecordingCancelled("Cancelled by cancel hotkey")
    if config.countdown_seconds == 0:
        return True

    deadline = time.perf_counter() + config.countdown_seconds
    while True:
        if cancel_event.is_set():
            raise RecordingCancelled("Cancelled by cancel hotkey")
        if stop_event.is_set():
            return False
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return True

        result = camera.get_latest_frame(with_timestamp=True)
        if result is None:
            raise RuntimeError("Desktop capture stopped during countdown")
        frame, source_timestamp_seconds = result
        frame_clock.align(source_timestamp_seconds, time.perf_counter_ns())
        if not input_sampler.connected:
            raise RuntimeError("Controller disconnected during countdown")

        if config.preview:
            preview = resize_frame(frame, config)
            draw_text_lines(
                preview,
                [
                    f"Recording starts in {math.ceil(remaining)}",
                    f"Q / Esc: cancel  Stop: {config.stop_hotkey}",
                    f"Cancel: {config.cancel_hotkey}",
                ],
                color=(0, 220, 255),
            )
            cv2.imshow(PREVIEW_TITLE, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return False


def record_frames(
    *,
    camera: Any,
    input_sampler: ControllerSampler,
    frame_clock: FrameClockSynchronizer,
    video_writer: cv2.VideoWriter,
    input_writers: InputWriters,
    game_state_sampler: GameStateSampler | None,
    game_state_writer: GameStateWriter | None,
    config: RecorderConfig,
    stop_event: threading.Event,
    cancel_event: threading.Event,
) -> RecordingResult:
    if (game_state_sampler is None) != (game_state_writer is None):
        raise ValueError("Game-state sampler and writer must be configured together")
    frame_index = 0
    capture_failures = 0
    dropped_frames = 0
    synchronization_drops = 0
    largest_gap_ns = 0
    previous_frame_ns: int | None = None
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    recent_timestamps: deque[int] = deque(maxlen=max(2, config.fps * 2))
    frame_period_ns = round(1_000_000_000 / config.fps)
    started_ns = time.perf_counter_ns()
    next_disk_check_ns = started_ns
    stop_reason = "global_hotkey" if stop_event.is_set() else "controller_disconnected"
    input_statistics = InputStatisticsAccumulator()
    absolute_input_offsets_ns: list[int] = []
    source_width = 0
    source_height = 0
    source_channels = 0
    source_dtype = ""

    try:
        while input_sampler.connected and not stop_event.is_set():
            if cancel_event.is_set():
                raise RecordingCancelled("Cancelled by cancel hotkey")
            elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1e9
            if (
                config.maximum_duration_seconds is not None
                and elapsed_seconds >= config.maximum_duration_seconds
            ):
                stop_reason = "maximum_duration"
                break

            now_ns = time.perf_counter_ns()
            if now_ns >= next_disk_check_ns:
                free_bytes = shutil.disk_usage(config.output_directory).free
                if free_bytes < config.minimum_free_gb * 1024**3:
                    stop_reason = "low_disk_space"
                    break
                next_disk_check_ns = now_ns + 5_000_000_000

            result = camera.get_latest_frame(with_timestamp=True)
            if result is None:
                capture_failures += 1
                raise RuntimeError("Desktop capture stopped unexpectedly")

            frame, source_timestamp_seconds = result
            if frame_index == 0:
                if frame.ndim != 3:
                    raise RuntimeError(
                        f"Captured frame has unexpected shape: {frame.shape}"
                    )
                source_height, source_width, source_channels = frame.shape
                source_dtype = str(frame.dtype)
            elif frame.shape != (source_height, source_width, source_channels):
                raise RuntimeError(
                    "Captured frame dimensions changed during recording: "
                    f"expected {(source_height, source_width, source_channels)}, "
                    f"got {frame.shape}"
                )
            timestamp_ns = time.perf_counter_ns()
            source_timestamp_ns, frame_timestamp_ns, _ = frame_clock.align(
                source_timestamp_seconds,
                timestamp_ns,
            )

            if previous_frame_ns is not None:
                gap_ns = max(0, timestamp_ns - previous_frame_ns)
                largest_gap_ns = max(largest_gap_ns, gap_ns)
                dropped_frames += max(0, round(gap_ns / frame_period_ns) - 1)

            previous_frame_ns = timestamp_ns

            input_sample = input_sampler.latest_at_or_before(frame_timestamp_ns)

            if input_sample is None:
                synchronization_drops += 1
                continue

            input_timestamp_ns = input_sample.timestamp_ns
            input_offset_ns = input_timestamp_ns - frame_timestamp_ns

            max_offset_ns = round(config.maximum_sync_offset_ms * 1e6)

            if input_offset_ns > 0 or -input_offset_ns > max_offset_ns:
                synchronization_drops += 1
                continue

            game_state_sample = None

            if game_state_sampler is not None and game_state_writer is not None:
                game_state_sample = game_state_sampler.closest(
                    frame_timestamp_ns,
                    timeout_seconds=max(0.025, 4 / config.game_state_hz),
                )

                game_state_offset_ns = (
                    game_state_sample.timestamp_ns - frame_timestamp_ns
                )
                max_game_state_offset_ns = round(
                    config.maximum_game_state_sync_offset_ms * 1e6
                )

                if abs(game_state_offset_ns) > max_game_state_offset_ns:
                    synchronization_drops += 1
                    continue
                    
                game_state_writer.write(
                    frame_index=frame_index,
                    timestamp_ns=timestamp_ns,
                    frame_timestamp_ns=frame_timestamp_ns,
                    sample=game_state_sample,
                )

            state = input_sample.state
            absolute_input_offsets_ns.append(abs(input_offset_ns))
            input_statistics.add(state)

            resized_frame = resize_frame(frame, config)
            row = InputRow.from_state(
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                source_timestamp_ns=source_timestamp_ns,
                frame_timestamp_ns=frame_timestamp_ns,
                input_timestamp_ns=input_timestamp_ns,
                state=state,
            )
            
            video_writer.write(resized_frame)
            input_writers.write(row)

            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            last_timestamp_ns = timestamp_ns
            recent_timestamps.append(timestamp_ns)
            frame_index += 1

            if frame_index % config.fps == 0:
                input_writers.flush_csv()

            if config.preview:
                preview = resized_frame.copy()
                current_fps = measured_fps(recent_timestamps)
                free_gb = shutil.disk_usage(config.output_directory).free / 1024**3
                draw_text_lines(
                    preview,
                    [
                        f"REC  {elapsed_seconds:7.1f}s  frame {frame_index}",
                        f"FPS {current_fps:4.1f}/{config.fps}  drops~{dropped_frames}",
                        f"Input {config.input_hz} Hz  sync {input_offset_ns / 1e6:+.2f} ms",
                        f"Disk {free_gb:.1f} GB free",
                        summarize_state(state),
                        f"Q / Esc: stop  Stop: {config.stop_hotkey}",
                        f"Cancel: {config.cancel_hotkey}",
                    ],
                    color=(80, 255, 80),
                )
                cv2.imshow(PREVIEW_TITLE, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_reason = "user"
                    break
        if stop_event.is_set():
            stop_reason = "global_hotkey"
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"

    ended_ns = time.perf_counter_ns()
    elapsed_seconds = (ended_ns - started_ns) / 1e9
    if first_timestamp_ns is not None and last_timestamp_ns is not None:
        measured = (
            (frame_index - 1) / ((last_timestamp_ns - first_timestamp_ns) / 1e9)
            if frame_index > 1 and last_timestamp_ns > first_timestamp_ns
            else 0.0
        )
    else:
        measured = 0.0

    sampler_stats = input_sampler.stats()
    mean_abs_input_offset_ms = (
        sum(absolute_input_offsets_ns) / len(absolute_input_offsets_ns) / 1e6
        if absolute_input_offsets_ns
        else 0.0
    )
    maximum_abs_input_offset_ms = (
        max(absolute_input_offsets_ns) / 1e6
        if absolute_input_offsets_ns
        else 0.0
    )
    return RecordingResult(
            statistics=RecordingStats(
                frame_count=frame_index,
                elapsed_seconds=elapsed_seconds,
                measured_fps=measured,
                dropped_frames_estimate=dropped_frames,
                synchronization_drops=synchronization_drops,
                capture_failures=capture_failures,
            largest_frame_gap_ms=largest_gap_ns / 1e6,
            stop_reason=stop_reason,
            input_statistics=input_statistics.summary(
                duration_seconds=elapsed_seconds
            ),
            input_sample_count=sampler_stats.sample_count,
            input_sample_rate_hz=sampler_stats.measured_hz,
            missed_input_polls=sampler_stats.missed_polls,
            mean_abs_input_offset_ms=mean_abs_input_offset_ms,
            maximum_abs_input_offset_ms=maximum_abs_input_offset_ms,
        ),
        source_width=source_width,
        source_height=source_height,
        source_channels=source_channels,
        source_dtype=source_dtype,
    )


def measured_fps(timestamps: deque[int]) -> float:
    if len(timestamps) < 2:
        return 0.0
    elapsed = (timestamps[-1] - timestamps[0]) / 1e9
    return (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def summarize_state(state: Any) -> str:
    buttons = [
        name
        for name in (
            "south",
            "east",
            "west",
            "north",
            "left_bumper",
            "right_bumper",
            "left_stick",
            "right_stick",
            "dpad_up",
            "dpad_down",
            "dpad_left",
            "dpad_right",
            "start",
            "back",
        )
        if getattr(state, name)
    ]
    active = "+".join(buttons) if buttons else "none"
    return (
        f"L {state.left_x:+.2f},{state.left_y:+.2f}  "
        f"R {state.right_x:+.2f},{state.right_y:+.2f}  "
        f"LT/RT {state.left_trigger:.2f}/{state.right_trigger:.2f}  {active}"
    )


def draw_text_lines(
    frame: Any,
    lines: list[str],
    *,
    color: tuple[int, int, int],
) -> None:
    line_height = 22
    overlay_height = 12 + line_height * len(lines)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (8, 20 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def finalize_session(
    staging_directory: Path,
    final_directory: Path,
    metadata: dict[str, Any],
) -> None:
    from recorder.validate import validate_session

    metadata["status"] = "complete"
    write_metadata(staging_directory, metadata)
    report = validate_session(staging_directory)
    if report.errors:
        metadata["status"] = "invalid"
        metadata["validation_errors"] = report.errors
        write_metadata(staging_directory, metadata)
        raise RuntimeError("Recording validation failed: " + "; ".join(report.errors))

    metadata["validation_warnings"] = report.warnings
    write_metadata(staging_directory, metadata)
    staging_directory.rename(final_directory)


def open_replay(session_directory: Path) -> None:
    from recorder.replay import load_replay_session, run_replay

    session = load_replay_session(session_directory)
    print(f"Opening replay for {session_directory}")
    run_replay(session)


def run(config: RecorderConfig) -> Path:
    memory_profile = (
        load_memory_profile(config.game_state_profile)
        if config.game_state_profile is not None
        else None
    )
    staging_directory, final_directory = create_session_directories(
        config.output_directory,
        split=config.split,
        theme=config.theme,
        tag=config.tag,
    )
    video_path = staging_directory / "frames.mp4"
    input_path = staging_directory / "inputs.parquet"
    game_state_path = staging_directory / "game_state.parquet"
    csv_path = staging_directory / "inputs.csv" if config.write_csv else None
    files = {"video": "frames.mp4", "inputs": "inputs.parquet"}
    if csv_path is not None:
        files["inputs_csv"] = "inputs.csv"
    if memory_profile is not None:
        files["game_state"] = "game_state.parquet"
    metadata: dict[str, Any] = {
        "status": "initializing",
        "session_id": f"{config.split}/{config.theme}/{config.tag}",
        "created_at": datetime.now().astimezone().isoformat(),
        "config": configuration_metadata(config),
        "files": files,
    }
    if memory_profile is not None:
        metadata["game_state"] = memory_profile.metadata()
    write_metadata(staging_directory, metadata)

    camera = None
    video_writer = None
    input_sampler = None
    game_state_sampler = None
    global_hotkey = None
    cancel_hotkey = None
    stop_event = threading.Event()
    cancel_event = threading.Event()
    cancellation_error: RecordingCancelled | None = None
    try:
        free_gb = shutil.disk_usage(config.output_directory).free / 1024**3
        if free_gb < config.minimum_free_gb:
            raise RuntimeError(
                f"Only {free_gb:.2f} GB is free; "
                f"{config.minimum_free_gb:.2f} GB is required"
            )

        input_sampler = ControllerSampler(
            lambda: ai_controller.Controller(config.stick_deadzone),
            polling_hz=config.input_hz,
        )
        input_sampler.start()
        metadata["controller"] = {"name": input_sampler.controller_name}
        if memory_profile is not None:
            game_state_sampler = GameStateSampler(
                lambda: EldenRingStateReader.open(memory_profile),
                polling_hz=config.game_state_hz,
            )
            game_state_sampler.start()
        camera = dxcam.create(
            device_idx=config.device_index,
            output_idx=config.monitor_index,
            region=config.region,
            output_color="BGR",
            max_buffer_len=max(4, config.fps),
        )
        camera.start(target_fps=config.fps, video_mode=True)
        frame_clock = FrameClockSynchronizer()
        warm_up_capture(
            camera,
            frame_clock,
            sample_count=max(3, min(10, config.fps // 3)),
        )

        metadata["status"] = "countdown"
        write_metadata(staging_directory, metadata)
        print(f"Controller: {input_sampler.controller_name}")
        print(f"Preparing: {final_directory}")
        print(
            f"Press Q or Esc in the preview, {config.stop_hotkey} globally, "
            f"{config.cancel_hotkey} globally to cancel, or "
            "Ctrl+C in the terminal, to stop."
        )

        hotkey_spec = parse_hotkey(config.stop_hotkey)
        if hotkey_spec is not None:
            global_hotkey = GlobalHotkey(hotkey_spec, stop_event.set)
            global_hotkey.start()
        cancel_hotkey_spec = parse_hotkey(config.cancel_hotkey)
        if cancel_hotkey_spec is not None:
            cancel_hotkey = GlobalHotkey(
                cancel_hotkey_spec,
                cancel_event.set,
            )
            cancel_hotkey.start()

        try:
            ready = wait_for_countdown(
                camera,
                config,
                frame_clock,
                input_sampler,
                stop_event,
                cancel_event,
            )
        except KeyboardInterrupt as error:
            raise RecordingCancelled("Cancelled during countdown") from error
        if not ready:
            raise RecordingCancelled("Cancelled during countdown")

        video_writer = open_video_writer(video_path, config)
        metadata["status"] = "recording"
        metadata["started_at"] = datetime.now().astimezone().isoformat()
        write_metadata(staging_directory, metadata)

        with ExitStack() as stack:
            input_writers = stack.enter_context(
                InputWriters(input_path, csv_path=csv_path)
            )
            game_state_writer = (
                stack.enter_context(GameStateWriter(game_state_path))
                if game_state_sampler is not None
                else None
            )
            recording_result = record_frames(
                camera=camera,
                input_sampler=input_sampler,
                frame_clock=frame_clock,
                video_writer=video_writer,
                input_writers=input_writers,
                game_state_sampler=game_state_sampler,
                game_state_writer=game_state_writer,
                config=config,
                stop_event=stop_event,
                cancel_event=cancel_event,
            )
    except RecordingCancelled as error:
        cancellation_error = error
        metadata["status"] = "cancelled"
        metadata["cancelled_at"] = datetime.now().astimezone().isoformat()
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_metadata(staging_directory, metadata)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["failed_at"] = datetime.now().astimezone().isoformat()
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_metadata(staging_directory, metadata)
        print(f"Incomplete recording kept at: {staging_directory}", file=sys.stderr)
        raise
    finally:
        if global_hotkey is not None:
            global_hotkey.stop()
        if cancel_hotkey is not None:
            cancel_hotkey.stop()
        if video_writer is not None:
            video_writer.release()
        if input_sampler is not None:
            input_sampler.stop()
        if game_state_sampler is not None:
            game_state_sampler.stop()
        if camera is not None:
            try:
                if camera.is_capturing:
                    camera.stop()
            finally:
                camera.release()
        cv2.destroyAllWindows()

    if cancellation_error is not None:
        resolved_staging = staging_directory.resolve()
        expected_parent = (
            config.output_directory
            / config.split
            / config.theme
        ).resolve()
        if (
            resolved_staging.parent != expected_parent
            or not resolved_staging.name.endswith(".inprogress")
        ):
            raise RuntimeError(
                f"Refusing to discard unexpected staging directory: "
                f"{resolved_staging}"
            )
        if resolved_staging.exists():
            shutil.rmtree(resolved_staging)
        print(f"Recording cancelled; discarded {resolved_staging}")
        raise cancellation_error

    stats = recording_result.statistics
    if stats.frame_count == 0:
        metadata["status"] = "invalid"
        metadata["error"] = "Recording contains no frames"
        write_metadata(staging_directory, metadata)
        raise RuntimeError("Recording contains no frames")

    metadata["completed_at"] = datetime.now().astimezone().isoformat()
    metadata["recording"] = asdict(stats)
    metadata["video"] = {
        "codec": config.codec,
        "fps": config.fps,
        "width": config.width,
        "height": config.height,
    }
    metadata["preprocessing"] = {
        "capture": {
            "backend": "dxcam",
            "color_order": "BGR",
            "dtype": recording_result.source_dtype,
            "resolution": [
                recording_result.source_width,
                recording_result.source_height,
            ],
            "channels": recording_result.source_channels,
        },
        "recorded_frames": {
            "color_order": "BGR",
            "decoded_dtype": "uint8",
            "resolution": [config.width, config.height],
            "resize_interpolation": "cv2.INTER_AREA",
        },
    }
    if game_state_sampler is not None:
        metadata["game_state"]["sampler"] = asdict(game_state_sampler.stats())
    finalize_session(staging_directory, final_directory, metadata)

    print(f"Saved {stats.frame_count} frames to {final_directory}")
    print(f"Measured FPS: {stats.measured_fps:.2f}")
    print(f"Estimated dropped frames: {stats.dropped_frames_estimate}")
    print(f"Dropped for input sync: {stats.synchronization_drops}")
    print(
        f"Input sampling: {stats.input_sample_rate_hz:.1f} Hz, "
        f"max sync offset {stats.maximum_abs_input_offset_ms:.2f} ms"
    )
    print(f"Stop reason: {stats.stop_reason}")
    if config.open_replay_after_recording:
        open_replay(final_directory)
    return final_directory


def parse_region(value: str) -> tuple[int, int, int, int]:
    try:
        left, top, right, bottom = (int(part.strip()) for part in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "region must be LEFT,TOP,RIGHT,BOTTOM"
        ) from error
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("region coordinates are invalid")
    return left, top, right, bottom


def parse_args(argv: list[str] | None = None) -> RecorderConfig:
    parser = argparse.ArgumentParser(
        description="Record synchronized desktop frames and controller inputs."
    )
    parser.add_argument(
        "--theme",
        required=True,
        type=parse_recording_theme,
        help="recording category; use / for nested folders, e.g. movement/jump",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        required=True,
        type=parse_recording_label,
        help="recording tag/class, for example basic-movement",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=DATASET_SPLITS,
        help="assign the entire recording to training or validation",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        type=parse_recording_label,
        help="optional reusable label; repeat this flag to add more",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "recordings")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--region", type=parse_region)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--input-hz", type=int, default=250)
    parser.add_argument("--max-sync-offset-ms", type=float, default=15.0)
    parser.add_argument(
        "--game-state-profile",
        type=Path,
        help="enable read-only game-state capture using this offset profile",
    )
    parser.add_argument("--game-state-hz", type=int, default=60)
    parser.add_argument(
        "--max-game-state-sync-offset-ms",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="also write an inputs.csv mirror",
    )
    parser.add_argument(
        "--stop-hotkey",
        default="F8",
        help="global stop hotkey, e.g. F8 or CTRL+F8; use NONE to disable",
    )
    parser.add_argument(
        "--cancel-hotkey",
        default="CTRL+F9",
        help=(
            "global cancel hotkey; discards the in-progress recording; "
            "use NONE to disable"
        ),
    )
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="do not open the replay window after finalizing a recording",
    )
    parser.add_argument(
        "--boss-loop",
        action="store_true",
        help=(
            "record repeated boss attempts; the reset hotkey finalizes the "
            "episode, quitouts, reloads the practice-tool savefile, continues, "
            "and starts the next recording"
        ),
    )
    parser.add_argument(
        "--boss-reset-hotkey",
        default="F10",
        help="global finalize-and-reset key used by --boss-loop (default: F10)",
    )
    parser.add_argument(
        "--boss-reset-profile",
        type=Path,
        default=PROJECT_ROOT / "recorder" / "profiles" / "elden_ring.json",
        help="read-only Elden Ring profile used to detect title/gameplay transitions",
    )
    parser.add_argument("--boss-reset-timeout", type=float, default=45.0)
    parser.add_argument("--boss-title-settle", type=float, default=2.0)
    parser.add_argument("--boss-snapshot-delay", type=float, default=1.0)
    parser.add_argument("--boss-gameplay-settle", type=float, default=2.0)
    parser.add_argument(
        "--boss-episodes",
        type=int,
        help="stop after this many completed attempts (default: run until cancelled)",
    )
    args = parser.parse_args(argv)
    return RecorderConfig(
        theme=args.theme,
        tag=args.tag,
        split=args.split,
        labels=tuple(dict.fromkeys(args.label)),
        output_directory=args.output.expanduser().resolve(),
        fps=args.fps,
        width=args.width,
        height=args.height,
        device_index=args.device,
        monitor_index=args.monitor,
        region=args.region,
        codec=args.codec,
        countdown_seconds=args.countdown,
        preview=not args.no_preview,
        stick_deadzone=args.deadzone,
        maximum_duration_seconds=args.max_duration,
        minimum_free_gb=args.min_free_gb,
        input_hz=args.input_hz,
        maximum_sync_offset_ms=args.max_sync_offset_ms,
        game_state_profile=(
            args.game_state_profile.expanduser().resolve()
            if args.game_state_profile is not None
            else None
        ),
        game_state_hz=args.game_state_hz,
        maximum_game_state_sync_offset_ms=args.max_game_state_sync_offset_ms,
        write_csv=args.csv,
        stop_hotkey=args.stop_hotkey,
        cancel_hotkey=args.cancel_hotkey,
        open_replay_after_recording=not args.no_replay,
        boss_loop=args.boss_loop,
        boss_reset_hotkey=args.boss_reset_hotkey,
        boss_reset_profile=args.boss_reset_profile.expanduser().resolve(),
        boss_reset_timeout_seconds=args.boss_reset_timeout,
        boss_title_settle_seconds=args.boss_title_settle,
        boss_snapshot_delay_seconds=args.boss_snapshot_delay,
        boss_gameplay_settle_seconds=args.boss_gameplay_settle,
        boss_episodes=args.boss_episodes,
    )


def main() -> None:
    try:
        config = parse_args()
        if config.boss_loop:
            from recorder.boss_loop import run_boss_recording_loop

            run_boss_recording_loop(config, record_once=run)
        else:
            run(config)
    except RecordingCancelled:
        return


if __name__ == "__main__":
    main()
