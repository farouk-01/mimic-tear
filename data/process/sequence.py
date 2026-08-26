from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import Dataset

from .datasets.controller import ControllerDataset
from .datasets.frames import FramesDataset
from .datasets.game_state import GameStateDataset


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

    # @classmethod
    # def load(
    #     cls,
    #     *,
    #     recording: Recording,
    #     game_state_schema: GameStateSchema | None,
    #     sequence_length: int,
    #     video_decoder_config: VideoDecoderConfig,
    #     frame_transform_config: FrameTransformConfig,
    #     game_state_transform_config: GameStateTransformConfig | None,
    #     controller_transform_config: ControllerTransformConfig,
    #     drop_incomplete: bool = True,
    # ) -> SequenceDataset:
    #     frame_store = TensorFrameStore(
    #         path=recording.video, **video_decoder_config.model_dump()
    #     )
    #     frame_transform = FrameTransform(**frame_transform_config.model_dump())

    #     frame_dataset = FramesDataset(store=frame_store, transform=frame_transform)
    #     controller_store = ParquetControllerStore(path=recording.controller)

    #     controller_transform = ControllerTransform(
    #         **controller_transform_config.model_dump()
    #     )
    #     controller_dataset = ControllerDataset(
    #         store=controller_store, transform=controller_transform
    #     )

    #     if recording.game_state is not None:
    #         if game_state_schema is None:
    #             raise ValueError(
    #                 "game_state_schema must be provided when "
    #                 "the recording contains game-state data"
    #             )

    #         game_state_store = ParquetGameStateStore(
    #             path=recording.game_state,
    #             features=game_state_schema.features,
    #         )

    #         game_state_transform = (
    #             GameStateTransform(**game_state_transform_config.model_dump())
    #             if game_state_transform_config is not None
    #             else None
    #         )

    #         # if game_state_transform is None:
    #         #     logger.warning("Game state is provided but it's transform is None")

    #         game_state_dataset: GameStateDataset | None = GameStateDataset(
    #             store=game_state_store,
    #             transform=game_state_transform,
    #         )
    #     else:
    #         game_state_dataset = None

    #     if len(frame_dataset) != len(controller_dataset):
    #         raise ValueError(
    #             "Frame and controller sample counts do not match: "
    #             f"{len(frame_dataset)} != {len(controller_dataset)}"
    #         )

    #     if game_state_dataset is not None and len(game_state_dataset) != len(
    #         frame_dataset
    #     ):
    #         raise ValueError(
    #             "Frame and game-state sample counts do not match: "
    #             f"{len(frame_dataset)} != {len(game_state_dataset)}"
    #         )

    #     return SequenceDataset(
    #         frames=frame_dataset,
    #         controller=controller_dataset,
    #         game_state=game_state_dataset,
    #         sequence_length=sequence_length,
    #         drop_incomplete=drop_incomplete,
    #     )
