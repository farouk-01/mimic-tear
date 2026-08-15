from __future__ import annotations

from pathlib import Path
from pydantic import ConfigDict, BaseModel
from torchvision.models import (
    ResNet18_Weights,
)


from capture.screen.reader import ScreenCaptureConfig
from capture.gamepad.reader import GamepadReaderConfig
from data.stores.frames import VideoDecoderConfig
from mimic_tear.model.components.controller import ControllerConfig
from mimic_tear.model.components.fusion import FusionConfig
from mimic_tear.model.components.temporal import TemporalConfig
from mimic_tear.model.components.vision import VisionConfig
from mimic_tear.model.components.game_state import GameStateConfig
from recording.writers.controller import ControllerWriterConfig
from recording.writers.video import VideoConfig
from data.transforms.frames import FrameTransformConfig

from .constants import RawConfig
from mimic_tear.model import PolicyConfig
from mimic_tear.utils.logging import LoggingConfig
from game_state.elden_ring.config import EldenRingConfig
from recording import RecordingConfig, Recording
from mimic_tear.training.trainer import Hyperparameters
from data import SequenceDataset

_raw = RawConfig()


class MimicTearConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    regular_logging: LoggingConfig = LoggingConfig.model_validate(_raw.regular_logging)
    perf_logging: LoggingConfig = LoggingConfig.model_validate(_raw.perf_logging)

    game_state: EldenRingConfig = EldenRingConfig.model_validate_json(_raw.game_state)

    recording_files: RecordingConfig = RecordingConfig.model_validate(
        _raw.recording_files
    )
    video_config: VideoConfig = VideoConfig.model_validate(_raw.recording_video)
    capture_screen: ScreenCaptureConfig = ScreenCaptureConfig.model_validate(
        _raw.capture_screen
    )
    capture_gamepad: GamepadReaderConfig = GamepadReaderConfig.model_validate(
        _raw.capture_gamepad
    )
    recording_controller: ControllerWriterConfig = (
        ControllerWriterConfig.model_validate(_raw.recording_controller)
    )

    hyperparameters: Hyperparameters = Hyperparameters.model_validate(
        _raw.hyperparameters
    )

    recordings_directory: Path = _raw.recordings_directory
    artifacts_directory: Path = _raw.artifacts_directory

    vision: VisionConfig = VisionConfig.model_validate(_raw.vision)
    temporal: TemporalConfig = TemporalConfig.model_validate(
        {**_raw.temporal, "input_features": vision.output_features}
    )
    model_game_state: GameStateConfig = GameStateConfig.model_validate(
        {
            **_raw.model_game_state,
            "input_features": game_state.schema_.feature_count,
        }
    )
    fusion: FusionConfig = FusionConfig.model_validate(
        {
            **_raw.fusion,
            "input_features": (
                temporal.hidden_features,
                model_game_state.output_features,
            ),
        }
    )
    controller: ControllerConfig = ControllerConfig.model_validate(
        {**_raw.controller, "input_features": fusion.output_features}
    )

    policy: PolicyConfig = PolicyConfig(
        vision=vision,
        temporal=temporal,
        game_state=model_game_state,
        fusion=fusion,
        controller=controller,
    )

    _weights_name = vision.weights_name
    _mean = None
    _std = None
    if _weights_name is not None:
        _presets = ResNet18_Weights[_weights_name].transforms()
        _mean = tuple(_presets.mean)
        _std = tuple(_presets.std)

    transform_frames: FrameTransformConfig = FrameTransformConfig.model_validate(
        {**_raw.transform_frames, "mean": _mean, "std": _std}
    )
    stores_frames: VideoDecoderConfig = VideoDecoderConfig.model_validate(
        _raw.stores_frames
    )

    def load_recordings(
        self,
        train_subdir: str = "train",
        val_subdir: str = "val",
    ) -> tuple[list[SequenceDataset], list[SequenceDataset]]:
        train_dir = self.recordings_directory / train_subdir
        val_dir = self.recordings_directory / val_subdir

        def _load_dir(directory: Path) -> list[SequenceDataset]:
            recording_dir = []

            for path in directory.rglob(self.recording_files.video_file):
                recording_dir.append(path.parent)

            sorted(recording_dir)

            if not recording_dir:
                raise FileNotFoundError(
                    f"No recordings found in directory: {directory}"
                )

            datasets: list[SequenceDataset] = []
            for directory in recording_dir:
                recording = Recording.from_directory(directory, self.recording_files)

                if not recording.has_game_state:
                    raise ValueError(
                        "The configured policy requires game-state data, but the "
                        f"recording does not provide it: {directory}"
                    )

                datasets.append(
                    SequenceDataset.load(
                        video_decoder_config=self.stores_frames,
                        recording=recording,
                        game_state_schema=self.game_state.schema_,
                        sequence_length=self.hyperparameters.sequence_length,
                        drop_incomplete=False,
                    )
                )
            return datasets

        train_datasets = _load_dir(train_dir)
        val_datasets = _load_dir(val_dir)

        return train_datasets, val_datasets
