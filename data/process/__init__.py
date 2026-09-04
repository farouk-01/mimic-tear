from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict

from data.models.gamepad import ANALOG_INPUTS, BUTTON_INPUTS
from data.models.record import Recording, RecordingConfig
from data.models.tensor import TensorSchema
from graph.base import Plan

from .datasets.tensor import TensorDataset
from .encoders.encoder import Encoder, EncoderConfig
from .sequence import SequenceDataset
from .stores.encoding import EncodingStore, EncodingStoreConfig
from .stores.parquet import ParquetStore
from .stores.video import VideoStore, VideoStoreConfig

from utils import profile

__all__ = [
    "SequenceDataset",
    "ProcessConfig",
    "Process",
]


class ProcessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    recording: RecordingConfig

    encoding_stores: tuple[EncodingStoreConfig, ...] = ()
    encoders: tuple[EncoderConfig, ...] = ()

    video_store_cfg: VideoStoreConfig
    frame_schema: TensorSchema
    frame_plan: Plan

    controller_schema: TensorSchema
    controller_plan: Plan

    game_state_schema: TensorSchema
    game_state_plan: Plan

    sequence_length: int
    drop_incomplete: bool = True


class Process:
    def __init__(self, *, config: ProcessConfig) -> None:
        self.config = config
        self.encoders = self._build_encoders()

    @profile
    def process_sequence(self, source: str | Path) -> SequenceDataset:
        recording = Recording.from_directory(root=source, config=self.config.recording)

        # datasets: dict[str, TensorDataset] = {
        #     "frames": self._load_frames_dataset(source=recording.video),
        #     "controller": self._load_controller_dataset(source=recording.controller),
        # }

        # if recording.game_state is not None:
        #     datasets["game_state"] = self._load_game_state_dataset(
        #         source=recording.game_state,
        #     )

        if recording.game_state is None:
            raise ValueError(
                f"Missing game-state data in recording at {recording.root}"
            )

        game_state = self._load_game_state_dataset(
            source=recording.game_state,
        )

        datasets: dict[str, TensorDataset] = {
            "frames": self._load_frames_dataset(
                source=recording.video,
                cfg=self.config.video_store_cfg,
                capture_timestamps_ns=game_state.store.capture_timestamp_ns, # TODO : this is temporary
            ),
            "controller": self._load_controller_dataset(
                source=recording.controller,
            ),
            "game_state": game_state,
        }
        self._validate_recording_integrity(datasets)

        return SequenceDataset(
            datasets=datasets,
            sequence_length=self.config.sequence_length,
            drop_incomplete=self.config.drop_incomplete,
        )

    def discover_encodings(self, recording_root: str | Path) -> None:
        recording = Recording.from_directory(
            root=recording_root,
            config=self.config.recording,
        )

        if recording.game_state is None:
            return

        self._load_game_state_dataset(
            source=recording.game_state,
        ).discover_encodings()

    @property
    def encoding_cardinalities(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                field_name: encoder.cardinality
                for encoder in self.encoders
                for field_name in encoder.fields
            }
        )

    def _build_encoders(self) -> tuple[Encoder, ...]:
        stores = {
            config.encoding: EncodingStore(path=config.path)
            for config in self.config.encoding_stores
        }

        return tuple(
            Encoder(
                fields=config.fields,
                get_encodings=stores[config.encoding].load,
                append_encoding=stores[config.encoding].append,
            )
            for config in self.config.encoders
        )

    def _load_frames_dataset(
        self,
        source: str | Path,
        cfg: VideoStoreConfig,
        capture_timestamps_ns: Sequence[int],
    ) -> TensorDataset:
        # metadata = ParquetStore(path=source, columns=())

        store = VideoStore(
            path=source,
            **cfg.model_dump(),
            capture_timestamps_ns=capture_timestamps_ns, # TODO remove this, see above todo
        )

        return TensorDataset(
            store=store,
            schema=self.config.frame_schema,
            plan=self.config.frame_plan,
        )

    def _load_controller_dataset(self, source: str | Path) -> TensorDataset:
        store = ParquetStore(path=source, columns=(*ANALOG_INPUTS, *BUTTON_INPUTS))

        return TensorDataset(
            store=store,
            schema=self.config.controller_schema,
            plan=self.config.controller_plan,
        )

    def _load_game_state_dataset(self, source: str | Path) -> TensorDataset:
        features = tuple(value.name for value in self.config.game_state_plan.inputs)

        store = ParquetStore(path=source, columns=features)

        return TensorDataset(
            store=store,
            schema=self.config.game_state_schema,
            encoders=self.encoders,
            plan=self.config.game_state_plan,
        )

    @staticmethod
    def _validate_recording_integrity(datasets: Mapping[str, TensorDataset]) -> None:
        lengths = {name: len(dataset) for name, dataset in datasets.items()}

        if len(set(lengths.values())) != 1:
            raise ValueError(f"Dataset sample counts do not match: {lengths}")
