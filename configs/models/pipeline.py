from typing import Self

from pydantic import BaseModel, ConfigDict

from data.capture import (
    CaptureConfig,
    EldenRingMemoryProfile,
    GamepadReaderConfig,
    ScreenCaptureConfig,
)
from data.models.record import RecordingConfig
from data.models.tensor import TensorSchema
from data.process import ProcessConfig, DatasetSourceConfig
from data.process.datasets.tensor import TensorDatasetConfig
from data.process.transforms.tensor import TensorTransform
from data.write import (
    ControllerWriterConfig,
    GameStateWriterConfig,
    VideoConfig,
    WriterConfig,
)

from .training import TrainingConfig
from .game_state import GameStateConfig
from .frame import FrameConfig
from .controller import ControllerConfig


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
        frame: FrameConfig,
        controller: ControllerConfig,
        training: TrainingConfig,
    ) -> Self:
        recording = RecordingConfig.model_validate(raw_pipeline["recording"]["files"])

        capture = cls._load_capture(raw_pipeline, game_state=gstate.memory_profile)

        dataset_cfgs: list[DatasetSourceConfig] = []

        for cfg, name in zip(
            (gstate, frame, controller), ("game_state", "video", "controller")
        ):
            dataset_cfg = TensorDatasetConfig(
                tensor_schema=cfg.tensor_schema,
                transforms=cfg.transforms,
            )

            store_cfg = cfg.store_cfg
            encoding_stores = getattr(cfg, "encoding_stores", ())
            encoders = getattr(cfg, "encoders", ())

            source_cfg = DatasetSourceConfig(
                name=name,
                dataset_cfg=dataset_cfg,
                store_cfg=store_cfg,
                encoding_stores=encoding_stores,
                encoders=encoders,
            )
            dataset_cfgs.append(source_cfg)

        process = cls._load_process(
            recording=recording,
            datasets=tuple(dataset_cfgs),
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
        *,
        recording: RecordingConfig,
        datasets: tuple[DatasetSourceConfig, ...] = (),
        training: TrainingConfig,
    ) -> ProcessConfig:
        return ProcessConfig(
            recording=recording,
            datasets=datasets,
            sequence_length=training.hyperparameters.sequence_length,
            drop_incomplete=True,
        )
