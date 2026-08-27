from pathlib import Path
from contextlib import ExitStack
import json
from dataclasses import asdict
from types import TracebackType
from typing import Self
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict
import numpy as np
from numpy.typing import NDArray

from data.capture.memory.game_state import RawGameStateSchema
from data.models.record import RecordingConfig
from .metadata import RecordingMetadata
from .writers import (
    ControllerWriterConfig,
    VideoConfig,
    VideoFrameWriter,
    GameStateWriter,
    GameStateWriterConfig,
    ControllerWriter,
)
from data.models.gamepad import GamepadState

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
    def __init__(
        self,
        *,
        path: str | Path,
        schema: RawGameStateSchema,
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
            index=index, timestamp_ns=timestamp_ns, state=controller_state
        )

        self.game_state_writer.write(
            index=index, timestamp_ns=timestamp_ns, values=game_state
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

            raise FileExistsError("Recording files already exist: " f"{paths}")

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

    def close(self, *, finalize: bool = True) -> None:
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
