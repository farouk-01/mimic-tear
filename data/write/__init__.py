from __future__ import annotations

from collections.abc import Generator
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

from data.constants import DEFAULT_COLUMNS
from data.models.record import RecordingConfig
from data.models.schema import Schema, Snapshot

from .metadata import RecordingMetadata
from .writers import (
    GamepadWriter,
    ParquetWriter,
    ParquetWriterConfig,
    VideoConfig,
    VideoFrameWriter,
)

__all__ = [
    "ParquetWriterConfig",
    "RecordingMetadata",
    "VideoConfig",
    "WriterConfig",
    "Writer",
]


class WriterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    recording: RecordingConfig
    video: VideoConfig
    game_state: ParquetWriterConfig
    controller: ParquetWriterConfig


class Writer:
    def __init__(self, *, config: WriterConfig) -> None:
        self.config = config

    @contextmanager
    def recording(
        self,
        *,
        path: str | Path,
        controller_schema: Schema,
        game_state_schema: Schema,
    ) -> Generator[RecordingWriter]:
        with RecordingWriter(
            path=path,
            controller_schema=controller_schema,
            game_state_schema=game_state_schema,
            config=self.config,
        ) as writer:
            yield writer

    @contextmanager
    def gamepad(self) -> Generator[GamepadWriter]:
        writer = GamepadWriter()

        try:
            writer.connect()

        except FileNotFoundError as error:
            raise RuntimeError(
                "Controller bridge is not running. Connect the virtual gamepad first."
            ) from error

        try:
            yield writer

        finally:
            writer.close()

    def start_gamepad_bridge(self) -> bool:
        writer = GamepadWriter()

        try:
            writer.connect()
            writer.close()
            return False

        except FileNotFoundError:
            pass

        self._launch_gamepad_bridge()
        self._wait_for_gamepad_bridge(writer)
        writer.close()

        return True

    def stop_gamepad_bridge(self) -> bool:
        writer = GamepadWriter()

        try:
            writer.connect()

        except FileNotFoundError:
            return False

        except OSError:
            raise RuntimeError(
                "Controller is currently used, end whatever is using it before."
            )

        writer.close(shutdown=True)

        return True

    def _launch_gamepad_bridge(
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
        controller_schema: Schema,
        game_state_schema: Schema,
        config: WriterConfig,
    ) -> None:
        self.config = config
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.controller_schema = controller_schema
        self.game_state_schema = game_state_schema

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
                ParquetWriter(
                    self.root / config.recording.controller_file,
                    controller_schema.to_pyarrow_schema(),
                    flush_every=config.controller.flush_every,
                    compression=config.controller.compression,
                )
            )

            self.game_state_writer = self._stack.enter_context(
                ParquetWriter(
                    self.root / config.recording.game_state_file,
                    game_state_schema.to_pyarrow_schema(),
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
        controller_state: Snapshot,
        game_state: Snapshot,
    ) -> None:
        if self._closed:
            raise RuntimeError("Writer is closed")

        if index != self._sample_count:
            raise ValueError(f"Expected record {self._sample_count}, received {index}")

        self.video_writer.write(frame=video_frame)

        self.controller_writer.write(
            self._snapshot_row(index, timestamp_ns, controller_state)
        )
        self.game_state_writer.write(
            self._snapshot_row(index, timestamp_ns, game_state)
        )

        self._sample_count += 1

    @staticmethod
    def _snapshot_row(
        index: int,
        timestamp_ns: int,
        snapshot: Snapshot,
    ) -> dict[str, object]:
        return {
            DEFAULT_COLUMNS.frame_index: index,
            DEFAULT_COLUMNS.capture_timestamp_ns: timestamp_ns,
            **snapshot.to_dict(),
        }

    def _validate_targets(self) -> None:
        targets = [
            self.root / self.config.recording.video_file,
            self.root / self.config.recording.controller_file,
            self.root / self.config.recording.game_state_file,
            self.root / self.config.recording.metadata_file,
        ]

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
