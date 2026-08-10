from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.datasets.controller import (
    ControllerDataset,
    ControllerSample,
)
from data.datasets.frames import FramesDataset
from data.datasets.game_state import GameStateDataset


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

        self.sequence_count = self._sequence_count()

    def _sequence_count(self) -> int:
        sample_count = len(self.frames)

        if self.drop_incomplete:
            return sample_count // self.sequence_length

        return (sample_count + self.sequence_length - 1) // self.sequence_length

    def __len__(self) -> int:
        return self.sequence_count

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

        controller_samples: list[ControllerSample] = [
            self.controller[i] for i in range(start, end)
        ]

        analog = torch.stack([sample.analog for sample in controller_samples])

        buttons = torch.stack([sample.buttons for sample in controller_samples])

        if self.game_state is not None:
            game_state = torch.stack([self.game_state[i] for i in range(start, end)])
        else:
            game_state = None

        return SequenceSample(
            images=images,
            analog=analog,
            buttons=buttons,
            game_state=game_state,
        )
