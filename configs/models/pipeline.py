from typing import Self
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict
from torchvision.models import ResNet18_Weights

from data.capture import (
    CaptureConfig,
    EldenRingMemoryProfile,
    GamepadReaderConfig,
    ScreenCaptureConfig,
)
from data.models.record import RecordingConfig
from data.process import (
    ControllerTransformConfig,
    FrameTransformConfig,
    GameStateTransformConfig,
    ParquetGameStateStoreConfig,
    ProcessConfig,
    VideoDecoderConfig,
)
from data.write import (
    ControllerWriterConfig,
    GameStateWriterConfig,
    VideoConfig,
    WriterConfig,
)

from .model import ModelConfig
from .training import TrainingConfig
from .game_state import GameStateConfig
from data.models.game_state.processed import ProcessedGameStateSchema


class DataPipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    capture: CaptureConfig
    process: ProcessConfig
    writer: WriterConfig

    @classmethod
    def load(
        cls,
        raw_pipeline: dict,
        *,
        model: ModelConfig,
        gstate: GameStateConfig,
        training: TrainingConfig,
    ) -> Self:
        recording = RecordingConfig.model_validate(raw_pipeline["recording"]["files"])

        capture = cls._load_capture(raw_pipeline, game_state=gstate.memory_profile)

        process = cls._load_process(
            raw_pipeline,
            recording=recording,
            game_state_schema=gstate.processed_schema,
            model=model,
            training=training,
        )

        writer = cls._load_writer(
            raw_pipeline,
            recording=recording,
        )

        return cls(capture=capture, process=process, writer=writer)

    @staticmethod
    def _load_capture(
        raw: dict,
        *,
        game_state: EldenRingMemoryProfile,
    ) -> CaptureConfig:
        video = VideoConfig.model_validate(raw["recording"]["video"])
        screen = ScreenCaptureConfig.model_validate(raw["capture"]["screen"])
        gamepad = GamepadReaderConfig.model_validate(raw["capture"]["gamepad"])

        return CaptureConfig(
            fps=video.fps,
            screen=screen,
            gamepad=gamepad,
            game_state_profile=game_state,
        )

    @staticmethod
    def _load_writer(
        raw: dict,
        *,
        recording: RecordingConfig,
    ) -> WriterConfig:
        video = VideoConfig.model_validate(raw["recording"]["video"])

        controller = ControllerWriterConfig.model_validate(
            raw["recording"]["controller"]
        )

        game_state = GameStateWriterConfig.model_validate(
            raw["recording"]["game_state"]
        )

        return WriterConfig(
            recording=recording,
            video=video,
            controller=controller,
            game_state=game_state,
        )

    @staticmethod
    def _load_process(
        raw: dict,
        *,
        recording: RecordingConfig,
        game_state_schema: ProcessedGameStateSchema,
        model: ModelConfig,
        training: TrainingConfig,
    ) -> ProcessConfig:
        frame_transform_raw = dict(raw["data"]["transforms"]["frames"])

        mean: tuple[float, ...] | None = None
        std: tuple[float, ...] | None = None

        if model.vision.weights_name is not None:
            presets = ResNet18_Weights[model.vision.weights_name].transforms()
            mean = tuple(presets.mean)
            std = tuple(presets.std)

        frame_transform = FrameTransformConfig.model_validate(
            {
                **frame_transform_raw,
                "mean": mean,
                "std": std,
            }
        )

        game_state_transform = GameStateTransformConfig.model_validate(
            raw["data"]["transforms"]["game_state"]
        )

        controller_transform = ControllerTransformConfig.model_validate(
            raw["data"]["transforms"]["controller"]
        )

        video_decoder = VideoDecoderConfig.model_validate(
            raw["data"]["stores"]["frames"]
        )

        game_state_store = ParquetGameStateStoreConfig(
            features=game_state_schema.get_required_fields_names(include_derived=False),
        )

        return ProcessConfig(
            recording=recording,
            video_decoder=video_decoder,
            game_state_store=game_state_store,
            controller_transform=controller_transform,
            frame_transform=frame_transform,
            game_state_transform=game_state_transform,
            game_state_schema=game_state_schema,
            sequence_length=training.hyperparameters.sequence_length,
            drop_incomplete=True,
        )
