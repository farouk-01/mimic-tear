from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import TracebackType

from capture import CaptureSample
from game_state import GameStateSchema
from .manifest import RecordingConfig
from .metadata import RecordingMetadata
from .writers.controller import (
    ControllerWriter,
    ControllerWriterConfig,
)
from .writers.game_state import GameStateWriter
from .writers.video import (
    VideoConfig,
    VideoFrameWriter,
)


class Recorder:
    def __init__(
        self,
        *,
        root: str | Path,
        files: RecordingConfig,
        video: VideoConfig,
        controller: ControllerWriterConfig,
        game_state_schema: GameStateSchema | None = None,
    ) -> None:
        self.root = Path(root)
        self.files = files
        self.video_config = video
        self.game_state_schema = game_state_schema

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._validate_targets()

        self._stack = ExitStack()
        self._closed = False
        self._sample_count = 0

        try:
            self.video = self._stack.enter_context(
                VideoFrameWriter(
                    path=self.root / files.video_file,
                    width=video.width,
                    height=video.height,
                    fps=video.fps,
                )
            )

            self.controller = self._stack.enter_context(
                ControllerWriter(
                    path=self.root / files.controller_file,
                    flush_every=controller.flush_every,
                )
            )

            self.game_state = (
                self._stack.enter_context(
                    GameStateWriter(
                        path=(self.root / files.game_state_file),
                        schema=game_state_schema,
                        flush_every=self.controller.flush_every,
                    )
                )
                if game_state_schema is not None
                else None
            )

        except BaseException:
            self._stack.close()
            raise

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def write(
        self,
        sample: CaptureSample,
    ) -> None:
        if self._closed:
            raise RuntimeError("Recorder is closed")

        if sample.index != self._sample_count:
            raise ValueError(
                "Capture samples must be sequential: "
                f"expected {self._sample_count}, "
                f"received {sample.index}"
            )

        if self.game_state is not None and sample.game_state is None:
            raise ValueError(
                "Recorder requires game-state data, "
                f"but sample {sample.index} has none"
            )

        self.video.write(sample.frame.image)

        self.controller.write(
            index=sample.index,
            timestamp_ns=sample.timestamp_ns,
            state=sample.controller,
        )

        if self.game_state is not None and sample.game_state is not None:
            self.game_state.write(
                index=sample.index,
                timestamp_ns=sample.timestamp_ns,
                snapshot=sample.game_state,
            )

        self._sample_count += 1

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

    def _write_metadata(self) -> None:
        metadata = RecordingMetadata(
            format_version=self.files.version,
            fps=float(self.video_config.fps),
            width=self.video_config.width,
            height=self.video_config.height,
            sample_count=self._sample_count,
        )

        path = self.root / self.files.metadata_file

        path.write_text(
            json.dumps(
                asdict(metadata),
                indent=2,
            ),
            encoding="utf-8",
        )

    def _validate_targets(self) -> None:
        targets = [
            self.root / self.files.video_file,
            self.root / self.files.controller_file,
            self.root / self.files.metadata_file,
        ]

        if self.game_state_schema is not None:
            targets.append(self.root / self.files.game_state_file)

        existing = [path for path in targets if path.exists()]

        if existing:
            paths = ", ".join(str(path) for path in existing)

            raise FileExistsError("Recording files already exist: " f"{paths}")

    def __enter__(self) -> Recorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(
            finalize=exc_type is None,
        )
