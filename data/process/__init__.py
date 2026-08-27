from pydantic import ConfigDict, BaseModel
from pathlib import Path

from configs.models.game_state import ProcessedGameStateSchema

from .stores.controller import ParquetControllerStore
from .stores.game_state import ParquetGameStateStoreConfig, ParquetGameStateStore
from .stores.frames import TensorFrameStore, VideoDecoderConfig

from .transforms import (
    ControllerTransformConfig,
    ControllerTransform,
    FrameTransformConfig,
    FrameTransform,
    GameStateTransform,
    GameStateTransformConfig,
)

from .datasets import (
    FramesDataset,
    ControllerDataset,
    GameStateDataset,
)
from .sequence import SequenceDataset, SequenceSample
from data.models.record import RecordingConfig, Recording

__all__ = [
    "ProcessConfig",
    "Process",
    "SequenceDataset",
    "SequenceSample",
]


class ProcessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    recording: RecordingConfig

    video_decoder: VideoDecoderConfig
    game_state_store: ParquetGameStateStoreConfig

    controller_transform: ControllerTransformConfig
    frame_transform: FrameTransformConfig
    game_state_transform: GameStateTransformConfig

    game_state_schema: ProcessedGameStateSchema

    sequence_length: int
    drop_incomplete: bool = True


class Process:
    def __init__(
        self,
        *,
        config: ProcessConfig,
    ) -> None:
        self.config = config
        self.sequence_length = config.sequence_length
        self.drop_incomplete = config.drop_incomplete

    def process_sequence(self, source: str | Path) -> SequenceDataset:
        recording = Recording.from_directory(root=source, config=self.config.recording)

        if recording.game_state is None:
            raise ValueError(
                f"Missing game-state data in recording at {recording.root}"
            )

        frame_dataset = self._load_frames_dataset(source=recording.video)
        controller_dataset = self._load_controller_dataset(source=recording.controller)
        game_state_dataset = self._load_game_state_dataset(source=recording.game_state)

        self._validate_recording_integrity(
            frames=frame_dataset,
            controller=controller_dataset,
            game_state=game_state_dataset,
        )

        return SequenceDataset(
            frames=frame_dataset,
            controller=controller_dataset,
            game_state=game_state_dataset,
            sequence_length=self.sequence_length,
            drop_incomplete=self.drop_incomplete,
        )

    @staticmethod
    def _validate_recording_integrity(
        *,
        frames: FramesDataset,
        controller: ControllerDataset,
        game_state: GameStateDataset,
    ) -> None:
        if len(frames) != len(controller):
            raise ValueError(
                "Frame and controller sample counts do not match: "
                f"{len(frames)} != {len(controller)}"
            )

        if len(frames) != len(game_state):
            raise ValueError(
                "Frame and game-state sample counts do not match: "
                f"{len(frames)} != {len(game_state)}"
            )

        expected_indices = list(range(len(frames)))

        if list(controller.store.indices) != expected_indices:
            raise ValueError("Controller indices are not sequential")

        if list(controller.store.indices) != list(game_state.store.indices):
            raise ValueError("Controller and game-state indices do not match")

        if list(controller.store.timestamps_ns) != list(game_state.store.timestamps_ns):
            raise ValueError("Controller and game-state timestamps do not match")

    def _load_frames_dataset(self, source: str | Path) -> FramesDataset:
        frame_store = TensorFrameStore(
            path=source, **self.config.video_decoder.model_dump()
        )
        frame_transform = FrameTransform(**self.config.frame_transform.model_dump())
        return FramesDataset(store=frame_store, transform=frame_transform)

    def _load_controller_dataset(self, source: str | Path) -> ControllerDataset:
        controller_store = ParquetControllerStore(path=source)
        controller_transform = ControllerTransform(
            **self.config.controller_transform.model_dump()
        )
        return ControllerDataset(store=controller_store, transform=controller_transform)

    def _load_game_state_dataset(
        self,
        source: str | Path,
    ) -> GameStateDataset:
        from .transforms.definitions import GAME_STATE_TRANSFORMS

        # note : only the required features are passed
        game_state_store = ParquetGameStateStore(
            path=source, features=self.config.game_state_store.features
        )

        game_state_transform = GameStateTransform(
            generic_transforms=GAME_STATE_TRANSFORMS,
            **self.config.game_state_transform.model_dump(),
        )

        return GameStateDataset(
            store=game_state_store,
            schema=self.config.game_state_schema,
            transform=game_state_transform,
        )
