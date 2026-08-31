from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from torch import Tensor
from torch.utils.data import Dataset

from .datasets.controller import ControllerDataset
from .datasets.frames import FramesDataset
from .datasets.game_state import GameStateDataset, GameStateTensors


@dataclass(frozen=True, slots=True)
class SequenceSample:
    images: Tensor
    analog: Tensor
    buttons: Tensor
    game_state: GameStateTensors | None


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
        end = min(start + self.sequence_length, len(self.frames))

        images = self.frames.get_range(start, end)

        controller = self.controller.get_range(start, end)

        game_state = (
            self.game_state.get_range(start, end)
            if self.game_state is not None
            else None
        )

        return SequenceSample(
            images=images,
            analog=controller.analog,
            buttons=controller.buttons,
            game_state=game_state,
        )
