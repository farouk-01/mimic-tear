from dataclasses import dataclass

from data.datasets.controller import ControllerDataset
from data.datasets.frames import FramesDataset
from data.datasets.game_state import GameStateDataset
from data.sequence import SequenceDataset
from data.stores import (
    ParquetControllerStore,
    ParquetGameStateStore,
    TensorFrameStore,
)
from recording import Recording
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RecordingDatasets:
    frames: FramesDataset
    controller: ControllerDataset
    game_state: GameStateDataset | None
    sequence: SequenceDataset


def load_recording_datasets(
    *,
    recording: Recording,
    frames: Tensor,
    sequence_length: int,
    game_state_features: tuple[str, ...] | None = None,
    drop_incomplete: bool = True,
) -> RecordingDatasets:
    frame_store = TensorFrameStore(
        frames=frames,
    )

    frame_dataset = FramesDataset(
        store=frame_store,
    )

    controller_store = ParquetControllerStore(
        path=recording.controller,
    )

    controller_dataset = ControllerDataset(
        store=controller_store,
    )

    if recording.game_state is not None:
        if game_state_features is None:
            raise ValueError(
                "game_state_features must be provided when "
                "the recording contains game-state data"
            )

        game_state_store = ParquetGameStateStore(
            path=recording.game_state,
            features=game_state_features,
        )

        game_state_dataset: GameStateDataset | None = GameStateDataset(
            store=game_state_store,
        )

    else:
        game_state_dataset = None

    # Validate frame alignment before creating sequences.
    if len(frame_dataset) != len(controller_dataset):
        raise ValueError(
            "Frame and controller sample counts do not match: "
            f"{len(frame_dataset)} != {len(controller_dataset)}"
        )

    if game_state_dataset is not None and len(game_state_dataset) != len(frame_dataset):
        raise ValueError(
            "Frame and game-state sample counts do not match: "
            f"{len(frame_dataset)} != {len(game_state_dataset)}"
        )

    sequence_dataset = SequenceDataset(
        frames=frame_dataset,
        controller=controller_dataset,
        game_state=game_state_dataset,
        sequence_length=sequence_length,
        drop_incomplete=drop_incomplete,
    )

    return RecordingDatasets(
        frames=frame_dataset,
        controller=controller_dataset,
        game_state=game_state_dataset,
        sequence=sequence_dataset,
    )
