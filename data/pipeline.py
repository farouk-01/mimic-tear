from datetime import datetime
from functools import cached_property
from pathlib import Path

from .capture import CaptureConfig, Capture
from .process import ProcessConfig, Process, SequenceDataset
from .write import Writer, WriterConfig


class DataPipeline:
    def __init__(
        self,
        *,
        capture_config: CaptureConfig,
        process_config: ProcessConfig,
        writer_config: WriterConfig,
    ) -> None:
        self._validate_config_compatibility(
            capture_config=capture_config,
            process_config=process_config,
            writer_config=writer_config,
        )

        self.capture_config = capture_config
        self.process_config = process_config
        self.writer_config = writer_config

    @cached_property
    def processor(self) -> Process:
        return Process(config=self.process_config)

    def record_session(
        self,
        *,
        root: str | Path,
        theme: str | None = None,
        sub_theme: str | None = None,
        name: str,
        seconds: float | None = None,
    ) -> Path:
        if seconds is not None and seconds <= 0:
            raise ValueError("seconds must be greater than zero")

        path = self._recording_path(
            root=root,
            theme=theme,
            sub_theme=sub_theme,
            name=name,
        )

        fps = self.capture_config.fps
        max_frames = max(1, round(seconds * fps)) if seconds is not None else None

        print("Press Ctrl+C to stop recording.")

        if seconds is not None:
            print(f"Recording for {seconds:.1f} seconds.")

        with (
            Capture(config=self.capture_config) as capture,
            Writer(config=self.writer_config, path=path) as writer,
        ):
            try:
                for sample in capture.capture_stream():
                    game_state = (
                        sample.game_state.values
                        if sample.game_state is not None
                        else None
                    )

                    if game_state is None:
                        raise RuntimeError(
                            "Game state is None, but it is required for writing."
                        )

                    writer.write_record(
                        index=sample.index,
                        timestamp_ns=sample.timestamp_ns,
                        video_frame=sample.frame.image,
                        controller_state=sample.controller,
                        game_state=game_state,
                    )

                    elapsed_seconds = writer.sample_count / fps

                    print(
                        f"\r"
                        f"Frames: {writer.sample_count} "
                        f"Time: {elapsed_seconds:.1f}s "
                        f"Capture: "
                        f"{sample.capture_duration_ns / 1_000_000:.2f}ms",
                        end="",
                        flush=True,
                    )

                    if max_frames is not None and writer.sample_count >= max_frames:
                        break

            except KeyboardInterrupt:
                print("\nStopping recording...")

        print()
        print(f"Saved {writer.sample_count} samples " f"to {writer.root.resolve()}")

        return path

    def process_recording(
        self,
        *,
        source: str | Path,
    ) -> SequenceDataset:
        recording_path = Path(source).resolve()
        return self.processor.process_sequence(source=recording_path)

    @staticmethod
    def _validate_config_compatibility(
        *,
        capture_config: CaptureConfig,
        process_config: ProcessConfig,
        writer_config: WriterConfig,
    ) -> None:
        if capture_config.fps != writer_config.video.fps:
            raise ValueError(
                "Capture FPS must match video writer FPS: "
                f"{capture_config.fps} != {writer_config.video.fps}"
            )

        if process_config.recording != writer_config.recording:
            raise ValueError("Process and writer recording configurations must match")

    @staticmethod
    def _recording_path(
        root: str | Path,
        *,
        theme: str | None,
        sub_theme: str | None,
        name: str,
    ) -> Path:
        root_path = Path(root).resolve()
        path = root_path

        def path_component(value: str, *, label: str) -> str:
            value = value.strip()

            if not value:
                raise ValueError(f"{label} cannot be empty")

            component = Path(value)

            if (
                "/" in value
                or "\\" in value
                or value in {".", ".."}
                or component.is_absolute()
                or bool(component.drive)
                or len(component.parts) != 1
            ):
                raise ValueError(f"{label} must be a single path component: {value!r}")

            return value

        if theme is not None:
            path /= path_component(theme, label="Theme")

        if sub_theme is not None:
            path /= path_component(sub_theme, label="Sub-theme")

        path /= path_component(name, label="Recording name")
        path /= datetime.now().strftime("%Y%m%d-%H%M%S")
        path = path.resolve()

        if not path.is_relative_to(root_path):
            raise ValueError(f"Recording path escapes root directory: {path}")

        return path