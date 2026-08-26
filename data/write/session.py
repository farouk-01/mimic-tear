from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from capture import CaptureSynchronizer
from capture.gamepad.reader import GamepadReader
from capture.screen import ScreenReader
from game_state.elden_ring import (
    EldenRingGameStateReader,
)
from .recorder import Recorder

if TYPE_CHECKING:
    from configs.config import MimicTearConfig


class RecordingSession:
    def __init__(
        self,
        *,
        config: MimicTearConfig,
    ) -> None:
        self.config = config

    def run(
        self,
        *,
        name: str | None = None,
        seconds: float | None = None,
    ) -> Path:
        if seconds is not None and seconds <= 0:
            raise ValueError("seconds must be greater than zero")

        session_name = self._session_name(name)

        output = self.config.recordings_directory / "raw" / session_name

        fps = float(self.config.video_config.fps)

        maximum_frames = max(1, round(seconds * fps)) if seconds is not None else None

        gamepad = GamepadReader(self.config.capture_gamepad.stick_deadzone)

        print(f"Gamepad: {gamepad.name}")
        print(f"Output: {output.resolve()}")

        if seconds is None:
            print("Press Ctrl+C to stop recording.")
        else:
            print(f"Recording for {seconds:.1f} seconds.")

        with (
            ScreenReader(
                gpu_index=(
                    self.config.capture_screen.gpu_index
                ),
                monitor_index=(
                    self.config.capture_screen.monitor_index
                ),
                region=self.config.capture_screen.region,
            ) as screen,
            EldenRingGameStateReader.open(self.config.game_state) as game_state,
            Recorder(
                root=output,
                files=self.config.recording_files,
                video=self.config.video_config,
                controller=self.config.recording_controller,
                game_state_schema=game_state.schema,
            ) as recorder,
        ):
            synchronizer = CaptureSynchronizer(
                screen=screen,
                gamepad=gamepad,
                game_state=game_state,
                fps=fps,
            )

            try:
                for sample in synchronizer.run():
                    recorder.write(sample)

                    elapsed_seconds = recorder.sample_count / fps

                    print(
                        f"\r"
                        f"Frames: {recorder.sample_count} "
                        f"Time: {elapsed_seconds:.1f}s "
                        f"Capture: "
                        f"{sample.capture_duration_ns / 1_000_000:.2f}ms",
                        end="",
                        flush=True,
                    )

                    if (
                        maximum_frames is not None
                        and recorder.sample_count >= maximum_frames
                    ):
                        break

            except KeyboardInterrupt:
                print("\nStopping recording...")

        print()
        print(f"Saved {recorder.sample_count} samples " f"to {output.resolve()}")

        return output

    @staticmethod
    def _session_name(
        name: str | None,
    ) -> str:
        if name is None:
            return datetime.now().strftime("%Y%m%d-%H%M%S")

        name = name.strip()

        if not name:
            raise ValueError(
                "Recording name cannot be empty"
            )

        if "/" in name or "\\" in name:
            raise ValueError(
                "Recording name cannot contain a path"
            )

        if name in {".", ".."}:
            raise ValueError(
                f"Invalid recording name: {name!r}"
            )

        return name
