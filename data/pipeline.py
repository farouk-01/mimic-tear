from collections.abc import Iterator, Mapping
from datetime import datetime
from functools import cached_property
from pathlib import Path
from threading import Event
from time import sleep

from .capture import Capture, CaptureConfig
from .process import Process, ProcessConfig, ProcessedRecording, ProcessedSample
from .write import Writer, WriterConfig

from utils.hotkeys import VK_F9, register_hotkey, unregister_hotkey


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

    @cached_property
    def writer(self) -> Writer:
        return Writer(config=self.writer_config)

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

        stop_event = Event()
        hotkey = register_hotkey(VK_F9, stop_event.set)

        print("Recording starts in 5 seconds...")

        for seconds_left in range(5, 0, -1):
            print(f"\rStarting in {seconds_left}...", end="", flush=True)
            sleep(1.0)

        print("Press F8 to stop recording.")

        if seconds is not None:
            print(f"Recording for {seconds:.1f} seconds.")

        schema = self.capture_config.game_state_profile.raw_schema

        try:
            with (
                Capture(config=self.capture_config) as capture,
                self.writer.recording(path=path, schema=schema) as writer,
            ):
                try:
                    for sample in capture.capture_stream(stop_event=stop_event):
                        game_state = (
                            sample.game_state.to_dict()
                            if sample.game_state is not None
                            else None
                        )

                        if game_state is None:
                            raise RuntimeError(
                                "Game state is None, but it is required for writing."
                            )

                        if sample.controller is None:
                            raise RuntimeError(
                                "Controller state is None, but it is required for writing."
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

        finally:
            unregister_hotkey(hotkey)

        print()
        print(f"Saved {writer.sample_count} samples " f"to {writer.root.resolve()}")

        return path

    def prepare_recordings(self, *, root: str | Path) -> Iterator[ProcessedRecording]:
        root_path = Path(root).resolve()

        if not root_path.is_dir():
            raise ValueError(f"Recording root path is not a directory: {root_path}")

        for metadata in sorted(root_path.rglob("metadata.json")):
            yield self.prepare_one_recording(source=metadata.parent)

    def prepare_one_recording(self, *, source: str | Path) -> ProcessedRecording:
        return self.processor.process_sequence(source=Path(source).resolve())

    def discover_encodings(self, *, root: str | Path) -> None:
        root_path = Path(root).resolve()

        if not root_path.is_dir():
            raise ValueError(f"Recording root path is not a directory: {root_path}")

        for metadata in sorted(root_path.rglob("metadata.json")):
            self.processor.discover_encodings(recording_root=metadata.parent)

    @property
    def encoding_cardinalities(self) -> Mapping[str, Mapping[str, int]]:
        return self.processor.encoding_cardinalities

    def process_stream(
        self,
        *,
        stop_event: Event | None = None,
    ) -> Iterator[ProcessedSample]:
        with Capture(config=self.capture_config) as capture:
            for sample in capture.capture_stream(
                stop_event=stop_event,
                include_gamepad=False,
            ):
                yield self.processor.process_sample(sample)

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
