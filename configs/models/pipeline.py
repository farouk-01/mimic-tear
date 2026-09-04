from typing import Self

from pydantic import BaseModel, ConfigDict

from graph.base import Plan
from data.capture import (
    CaptureConfig,
    EldenRingMemoryProfile,
    GamepadReaderConfig,
    ScreenCaptureConfig,
)
from data.models.record import RecordingConfig
from data.models.tensor import TensorSchema
from data.process import ProcessConfig
from data.write import (
    ControllerWriterConfig,
    GameStateWriterConfig,
    VideoConfig,
    WriterConfig,
)

from .training import TrainingConfig
from .game_state import GameStateConfig
from .frame import FrameConfig
from .frame import VideoStoreConfig


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
        gstate: GameStateConfig,
        video_store_cfg: VideoStoreConfig,
        frame_schema: TensorSchema,
        frame_plan: Plan,
        controller_schema: TensorSchema,
        controller_plan: Plan,
        training: TrainingConfig,
    ) -> Self:
        recording = RecordingConfig.model_validate(raw_pipeline["recording"]["files"])

        capture = cls._load_capture(raw_pipeline, game_state=gstate.memory_profile)

        process = cls._load_process(
            raw_pipeline,
            recording=recording,
            gstate=gstate,
            video_store_cfg=video_store_cfg,
            frame_schema=frame_schema,
            frame_plan=frame_plan,
            controller_schema=controller_schema,
            controller_plan=controller_plan,
            training=training,
        )

        writer = cls._load_writer(raw_pipeline, recording=recording)

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
        gstate: GameStateConfig,
        video_store_cfg: VideoStoreConfig,
        frame_schema: TensorSchema,
        frame_plan: Plan,
        controller_schema: TensorSchema,
        controller_plan: Plan,
        training: TrainingConfig,
    ) -> ProcessConfig:
        return ProcessConfig(
            recording=recording,
            encoding_stores=gstate.encoding_stores,
            encoders=gstate.encoders,
            video_store_cfg=video_store_cfg,
            frame_schema=frame_schema,
            frame_plan=frame_plan,
            controller_schema=controller_schema,
            controller_plan=controller_plan,
            game_state_schema=gstate.tensor_gstate_schema,
            game_state_plan=gstate.plan,
            sequence_length=training.hyperparameters.sequence_length,
            drop_incomplete=True,
        )
