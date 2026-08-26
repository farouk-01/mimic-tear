from pydantic import ConfigDict, BaseModel
from pathlib import Path

import torch

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
from .sequence import SequenceDataset
from data.models.record import RecordingConfig, Recording


class ProcessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    recording: RecordingConfig

    video_decoder: VideoDecoderConfig
    game_state_store: ParquetGameStateStoreConfig

    controller_transform: ControllerTransformConfig
    frame_transform: FrameTransformConfig
    game_state_transform: GameStateTransformConfig

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

    def _load_game_state_dataset(self, source: str | Path) -> GameStateDataset:
        game_state_store = ParquetGameStateStore(
            path=source, features=self.config.game_state_store.features
        )
        game_state_transform = GameStateTransform(
            **self.config.game_state_transform.model_dump()
        )
        return GameStateDataset(store=game_state_store, transform=game_state_transform)

    def process_sequence(self, source: str | Path) -> SequenceDataset:
        recording = Recording.from_directory(root=source, config=self.config.recording)

        if recording.game_state is None:
            raise ValueError(
                f"Missing game-state data in recording at {recording.root}"
            )

        frame_dataset = self._load_frames_dataset(source=recording.video)
        controller_dataset = self._load_controller_dataset(source=recording.controller)
        game_state_dataset = self._load_game_state_dataset(source=recording.game_state)

        if len(frame_dataset) != len(controller_dataset):
            raise ValueError(
                "Frame and controller sample counts do not match: "
                f"{len(frame_dataset)} != {len(controller_dataset)}"
            )

        if len(frame_dataset) != len(game_state_dataset):
            raise ValueError(
                "Frame and game-state sample counts do not match: "
                f"{len(frame_dataset)} != {len(game_state_dataset)}"
            )

        expected_indices = torch.arange(len(frame_dataset), dtype=torch.int64)

        if not torch.equal(controller_dataset.store.indices, expected_indices):
            raise ValueError("Controller indices are not sequential")

        if not torch.equal(
            controller_dataset.store.indices,
            game_state_dataset.store.indices,
        ):
            raise ValueError("Controller and game-state indices do not match")

        if not torch.equal(
            controller_dataset.store.timestamps_ns,
            game_state_dataset.store.timestamps_ns,
        ):
            raise ValueError("Controller and game-state timestamps do not match")


        return SequenceDataset(
            frames=frame_dataset,
            controller=controller_dataset,
            game_state=game_state_dataset,
            sequence_length=self.sequence_length,
            drop_incomplete=self.drop_incomplete,
        )


__all__ = [
    "ProcessConfig",
    "Process",
    "SequenceDataset",
]
