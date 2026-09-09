from __future__ import annotations

from collections.abc import Mapping, Generator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from types import TracebackType
from typing import Self
import subprocess
from time import monotonic, sleep

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from data.models.game_state.memory import MemoryGameStateSchema
from data.models.gamepad import GamepadState
from data.models.record import RecordingConfig

from .metadata import RecordingMetadata
from .writers import (
    ControllerWriter,
    ControllerWriterConfig,
    GamepadWriter,
    GameStateWriter,
    GameStateWriterConfig,
    VideoConfig,
    VideoFrameWriter,
)

__all__ = [
    "RecordingMetadata",
    "WriterConfig",
    "Writer",
]


class WriterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    recording: RecordingConfig
    video: VideoConfig
    game_state: GameStateWriterConfig
    controller: ControllerWriterConfig


class Writer:
    def __init__(self, *, config: WriterConfig) -> None:
        self.config = config

    @contextmanager
    def recording(
        self,
        *,
        path: str | Path,
        schema: MemoryGameStateSchema,
    ) -> Generator[RecordingWriter]:
        with RecordingWriter(path=path, schema=schema, config=self.config) as writer:
            yield writer

    @contextmanager
    def gamepad(self) -> Generator[GamepadWriter]:
        writer = GamepadWriter()

        launched_bridge = False

        try:
            try:
                writer.connect()

            except FileNotFoundError:
                self._start_gamepad_bridge()
                launched_bridge = True

                self._wait_for_gamepad_bridge(writer)

            yield writer

        finally:
            writer.close(shutdown=launched_bridge)

    def _start_gamepad_bridge(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[2]

        script = root / "bridges" / "hidmaestro" / "start.ps1"

        if not script.exists():
            raise FileNotFoundError(
                "Controller bridge startup script " f"not found: {script}"
            )

        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
            ],
            check=True,
        )

    def _wait_for_gamepad_bridge(
        self,
        writer: GamepadWriter,
        *,
        timeout: float = 10.0,
        interval: float = 0.1,
    ) -> None:
        deadline = monotonic() + timeout

        while True:
            try:
                writer.connect()
                return

            except FileNotFoundError:
                if monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for " "controller bridge")

                sleep(interval)


class RecordingWriter:
    def __init__(
        self,
        *,
        path: str | Path,
        schema: MemoryGameStateSchema,
        config: WriterConfig,
    ) -> None:
        self.config = config
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema = schema

        self._validate_targets()

        self._stack = ExitStack()
        self._closed = False
        self._sample_count = 0

        try:
            self.video_writer = self._stack.enter_context(
                VideoFrameWriter(
                    path=self.root / config.recording.video_file,
                    width=config.video.width,
                    height=config.video.height,
                    fps=config.video.fps,
                )
            )

            self.controller_writer = self._stack.enter_context(
                ControllerWriter(
                    path=self.root / config.recording.controller_file,
                    flush_every=config.controller.flush_every,
                )
            )

            self.game_state_writer = self._stack.enter_context(
                GameStateWriter(
                    path=self.root / config.recording.game_state_file,
                    schema=schema,
                    flush_every=config.game_state.flush_every,
                    compression=config.game_state.compression,
                )
            )

        except BaseException:
            self._stack.close()
            raise

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def write_record(
        self,
        *,
        index: int,
        timestamp_ns: int,
        video_frame: NDArray[np.uint8],
        controller_state: GamepadState,
        game_state: Mapping[str, object],
    ) -> None:
        if self._closed:
            raise RuntimeError("Writer is closed")

        if index != self._sample_count:
            raise ValueError(f"Expected record {self._sample_count}, received {index}")

        self.video_writer.write(frame=video_frame)

        self.controller_writer.write(
            index=index,
            timestamp_ns=timestamp_ns,
            state=controller_state,
        )

        self.game_state_writer.write(
            index=index,
            timestamp_ns=timestamp_ns,
            values=game_state,
        )

        self._sample_count += 1

    def _validate_targets(self) -> None:
        targets = [
            self.root / self.config.recording.video_file,
            self.root / self.config.recording.controller_file,
            self.root / self.config.recording.metadata_file,
        ]

        if self.schema is not None:
            targets.append(self.root / self.config.recording.game_state_file)

        existing = [path for path in targets if path.exists()]

        if existing:
            paths = ", ".join(str(path) for path in existing)

            raise FileExistsError(f"Recording files already exist: {paths}")

    def _write_metadata(self) -> None:
        metadata = RecordingMetadata(
            format_version=self.config.recording.version,
            fps=float(self.config.video.fps),
            width=self.config.video.width,
            height=self.config.video.height,
            sample_count=self._sample_count,
        )

        path = self.root / self.config.recording.metadata_file

        path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

    def close(
        self,
        *,
        finalize: bool = True,
    ) -> None:
        if self._closed:
            return

        self._stack.close()
        self._closed = True

        if finalize:
            self._write_metadata()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(finalize=exc_type is None)
