from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.datasets.controller import ControllerDataset
from data.datasets.frames import FramesDataset
from data.datasets.game_state import GameStateDataset
from data.stores import (
    ParquetControllerStore,
    ParquetGameStateStore,
    TensorFrameStore,
)
from game_state import GameStateSchema
from recording import Recording
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SequenceSample:
    images: Tensor
    analog: Tensor
    buttons: Tensor
    game_state: Tensor | None


class SequenceDataset(Dataset[SequenceSample]):
    def __init__(
        self,
        *,
        frames: FramesDataset,
        controller: ControllerDataset,
        game_state: GameStateDataset | None = None,
        sequence_length: int,
        drop_incomplete: bool = True,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")

        if len(frames) != len(controller):
            raise ValueError("Frames and controller datasets must have the same length")

        if game_state is not None and len(game_state) != len(frames):
            raise ValueError("Game-state and frames datasets must have the same length")

        self.frames = frames
        self.controller = controller
        self.game_state = game_state

        self.sequence_length = sequence_length
        self.drop_incomplete = drop_incomplete

    def __len__(self) -> int:
        sample_count = len(self.frames)

        if self.drop_incomplete:
            return sample_count // self.sequence_length

        return (sample_count + self.sequence_length - 1) // self.sequence_length

    def __getitem__(
        self,
        index: int,
    ) -> SequenceSample:
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(index)

        start = index * self.sequence_length
        end = min(
            start + self.sequence_length,
            len(self.frames),
        )

        images = torch.stack([self.frames[i] for i in range(start, end)])

        controller_samples = [self.controller[i] for i in range(start, end)]

        analog = torch.stack([sample.analog for sample in controller_samples])

        buttons = torch.stack([sample.buttons for sample in controller_samples])

        game_state = (
            torch.stack([self.game_state[i] for i in range(start, end)])
            if self.game_state is not None
            else None
        )

        return SequenceSample(
            images=images,
            analog=analog,
            buttons=buttons,
            game_state=game_state,
        )

    @classmethod
    def load(
        cls,
        recording: Recording,
        game_state_schema: GameStateSchema | None,
        sequence_length: int,
        drop_incomplete: bool = True,
    ) -> SequenceDataset:
        frame_store = TensorFrameStore.from_mp4(recording.video)
        frame_dataset = FramesDataset(store=frame_store)
        controller_store = ParquetControllerStore(path=recording.controller)
        controller_dataset = ControllerDataset(store=controller_store)

        if recording.game_state is not None:
            if game_state_schema is None:
                raise ValueError(
                    "game_state_schema must be provided when "
                    "the recording contains game-state data"
                )

            game_state_store = ParquetGameStateStore(
                path=recording.game_state,
                features=game_state_schema.features,
            )

            game_state_dataset: GameStateDataset | None = GameStateDataset(
                store=game_state_store,
            )
        else:
            game_state_dataset = None

        if len(frame_dataset) != len(controller_dataset):
            raise ValueError(
                "Frame and controller sample counts do not match: "
                f"{len(frame_dataset)} != {len(controller_dataset)}"
            )

        if game_state_dataset is not None and len(game_state_dataset) != len(
            frame_dataset
        ):
            raise ValueError(
                "Frame and game-state sample counts do not match: "
                f"{len(frame_dataset)} != {len(game_state_dataset)}"
            )

        return SequenceDataset(
            frames=frame_dataset,
            controller=controller_dataset,
            game_state=game_state_dataset,
            sequence_length=sequence_length,
            drop_incomplete=drop_incomplete,
        )
