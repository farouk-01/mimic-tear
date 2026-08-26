from datetime import datetime
from functools import cached_property
from pathlib import Path

from .capture import CaptureConfig, Capture
from .process import ProcessConfig, Process
from .write import Writer, WriterConfig


class DataPipeline:
    def __init__(
        self,
        *,
        capture_config: CaptureConfig,
        process_config: ProcessConfig,
        writer_config: WriterConfig,
    ) -> None:
        if capture_config.fps != writer_config.video.fps:
            raise ValueError("Capture and video FPS must match")
        
        self.capture_config = capture_config
        self.process_config = process_config
        self.writer_config = writer_config

    @cached_property
    def capture(self) -> Capture:
        return Capture(config=self.capture_config)

    @cached_property
    def process(self) -> Process:
        return Process(config=self.process_config)

    @cached_property
    def writer(self) -> Writer:
        return Writer(config=self.writer_config)

    def record_one_session(
        self,
        *,
        seconds: float | None = None,
    ) -> None:
        if seconds is not None and seconds <= 0:
            raise ValueError("seconds must be greater than zero")

        capture = self.capture
        writer = self.writer

        finalize = False
        fps = capture.fps
        max_frames = max(1, round(seconds * fps)) if seconds is not None else None

        print("Press Ctrl+C to stop recording.")

        if seconds is not None:
            print(f"Recording for {seconds:.1f} seconds.")

        try:
            stream = capture.capture_stream()

            for sample in stream:
                game_state = (
                    sample.game_state.values if sample.game_state is not None else None
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
            finalize = True
            print("\nStopping recording...")

        else:
            finalize = True
            print("\nRecording complete.")

        finally:
            try:
                writer.close(finalize=finalize)
            finally:
                capture.close()

        print()
        print(f"Saved {writer.sample_count} samples " f"to {writer.root.resolve()}")

    # @staticmethod
    # def _session_name(name: str | None) -> str:
    #     if name is None:
    #         return datetime.now().strftime("%Y%m%d-%H%M%S")

    #     name = name.strip()

    #     if not name:
    #         raise ValueError("Recording name cannot be empty")

    #     if "/" in name or "\\" in name:
    #         raise ValueError("Recording name cannot contain a path")

    #     if name in {".", ".."}:
    #         raise ValueError(f"Invalid recording name: {name!r}")

    #     return name
