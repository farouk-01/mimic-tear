from collections.abc import Mapping, Collection
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property
from typing import overload

from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor
from tensordict import TensorDict

from data.models.record import Recording, RecordingConfig, RecordingName
from data.models.tensor import TensorSchema
from data.process.transforms import TensorTransform, Graph
from data.capture import CaptureSample

from .datasets.tensor import TensorDataset, TensorDatasetConfig
from .encoders.encoder import Encoder, EncoderConfig
from .sequence import SequenceDataset
from .stores.encoding import EncodingStore, EncodingStoreConfig
from .stores.base import FILE_STORES, StoreConfig, TensorColumn, TensorTable

from utils import profile

__all__ = [
    "SequenceDataset",
    "ProcessedRecording",
    "ProcessedSample",
    "ProcessConfig",
    "Process",
]


class DatasetSourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: RecordingName
    store_cfg: StoreConfig
    dataset_cfg: TensorDatasetConfig

    encoding_stores: tuple[EncodingStoreConfig, ...] = ()
    encoders: tuple[EncoderConfig, ...] = ()


class ProcessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    recording: RecordingConfig

    datasets: tuple[DatasetSourceConfig, ...] = ()

    sequence_length: int
    drop_incomplete: bool = True


type PresenceMask = dict[str, torch.Tensor]
type PresenceMasks = dict[RecordingName, PresenceMask]

type EncodingCardinalities = Mapping[str, int]
type EncodingCardinalitiesByDataset = Mapping[
    str,
    EncodingCardinalities,
]


@dataclass(frozen=True, slots=True)
class ProcessedRecording:
    dataset: SequenceDataset
    presence: PresenceMasks


@dataclass(frozen=True, slots=True)
class ProcessedSample:
    datasets: dict[RecordingName, TensorDict]
    presence: PresenceMasks


class Process:
    def __init__(self, *, config: ProcessConfig) -> None:
        self.config = config
        self.encoders: dict[RecordingName, tuple[Encoder, ...]] = self._build_encoders()

    @profile
    def process_sequence(self, source: str | Path) -> ProcessedRecording:
        recording = Recording.from_directory(root=source, config=self.config.recording)

        datasets, presences = self._load_datasets(recording)

        self._validate_recording_integrity(datasets)

        return ProcessedRecording(
            SequenceDataset(
                datasets=datasets,
                sequence_length=self.config.sequence_length,
                drop_incomplete=self.config.drop_incomplete,
            ),
            presences,
        )

    def discover_encodings(self, recording_root: str | Path) -> None:
        recording = Recording.from_directory(
            root=recording_root,
            config=self.config.recording,
        )

        datasets, *_ = self._load_datasets(recording)

        for dataset in datasets.values():
            dataset.discover_encodings()

    @property
    def encoding_cardinalities(self) -> EncodingCardinalitiesByDataset:
        result: dict[str, Mapping[str, int]] = {}

        for dataset_name, encoders in self.encoders.items():
            cardinalities: dict[str, int] = {}

            for encoder in encoders:
                for field_name in encoder.fields:
                    cardinalities[field_name] = encoder.cardinality

            result[dataset_name] = cardinalities

        return result

    @cached_property
    def live_datasets(self) -> dict[RecordingName, TensorDataset]:
        return self._build_live_datasets()

    def _build_live_datasets(
        self,
    ) -> dict[RecordingName, TensorDataset]:
        return {
            cfg.name: TensorDataset(
                store=None,
                tensor_schema=cfg.dataset_cfg.tensor_schema,
                encoders=self.encoders[cfg.name],
                transforms=cfg.dataset_cfg.transforms,
            )
            for cfg in self.config.datasets
        }

    @cached_property
    def dataset_configs(self) -> dict[RecordingName, DatasetSourceConfig]:
        return {cfg.name: cfg for cfg in self.config.datasets}

    @overload
    def process_sample(
        self,
        sample: CaptureSample,
    ) -> ProcessedSample: ...

    @overload
    def process_sample(
        self,
        sample: Mapping[RecordingName, TensorTable | None],
    ) -> ProcessedSample: ...

    def process_sample(
        self,
        sample: CaptureSample | Mapping[RecordingName, TensorTable | None],
    ) -> ProcessedSample:
        if isinstance(sample, CaptureSample):
            sources = self._capture_sample_sources(sample)
        else:
            sources = sample

        datasets: dict[RecordingName, TensorDict] = {}
        presences: PresenceMasks = {}

        for name, table in sources.items():
            if name not in self.dataset_configs:
                raise KeyError(f"Unknown dataset: {name}")

            cfg = self.dataset_configs[name]
            dataset = self.live_datasets[name]

            datasets[name] = dataset.process_table(
                table,
                batch_size=[1],
            )

            available_features = self._available_table_features(
                table=table,
                transforms=cfg.dataset_cfg.transforms,
            )

            presences[name] = self._make_presence_mask(
                schema=dataset.schema,
                available_features=available_features,
            )

        return ProcessedSample(datasets=datasets, presence=presences)

    def _capture_sample_sources(
        self,
        sample: CaptureSample,
    ) -> dict[RecordingName, TensorTable | None]:
        frame = torch.from_numpy(sample.frame.image)
        frame = frame.permute(2, 0, 1).unsqueeze(0)

        video_table: TensorTable = {"frames": TensorColumn(values=frame)}

        game_state_table: TensorTable | None = None

        if sample.game_state is not None:
            game_state_table = {}

            config = self.dataset_configs["game_state"]
            schema = config.dataset_cfg.tensor_schema

            for name, value in sample.game_state.to_dict().items():
                if name not in schema.fields_by_name or value is None:
                    continue

                field = schema.get_field(name)

                game_state_table[name] = TensorColumn(
                    values=torch.as_tensor(
                        [value],
                        dtype=field.torch_dtype,
                    ),
                )

        return {"video": video_table, "game_state": game_state_table}

    def _build_encoders(
        self,
    ) -> dict[RecordingName, tuple[Encoder, ...]]:
        result: dict[RecordingName, tuple[Encoder, ...]] = {}

        for dataset_cfg in self.config.datasets:
            stores = {
                cfg.encoding: EncodingStore(path=cfg.path)
                for cfg in dataset_cfg.encoding_stores
            }

            result[dataset_cfg.name] = tuple(
                Encoder(
                    fields=cfg.fields,
                    get_encodings=stores[cfg.encoding].load,
                    append_encoding=stores[cfg.encoding].append,
                )
                for cfg in dataset_cfg.encoders
            )

        return result

    def _load_one_dataset(
        self,
        source: str | Path | None,
        *,
        schema: TensorSchema,
        store_cfg: StoreConfig,
        encoders: tuple[Encoder, ...] = (),
        transforms: Graph[Tensor],
    ) -> tuple[TensorDataset, dict[str, torch.Tensor]]:
        if source is None:
            dataset = TensorDataset(
                store=None,
                tensor_schema=schema,
                encoders=encoders,
                transforms=transforms,
            )
        else:
            source = Path(source)
            suffix = source.suffix.lower()

            store_cls = FILE_STORES.resolve(suffix)
            store = store_cls(source=source, **store_cfg.kwargs())

            dataset = TensorDataset(
                store=store,
                tensor_schema=schema,
                encoders=encoders,
                transforms=transforms,
            )

        available_features = dataset.available_features
        mask = self._make_presence_mask(
            schema=dataset.schema,
            available_features=available_features,
        )

        return dataset, mask

    @staticmethod
    def _make_presence_mask(
        schema: TensorSchema,
        available_features: Collection[str],
    ) -> dict[str, torch.Tensor]:
        features = schema.fields_by_name

        presence: dict[str, torch.Tensor] = {}
        for name, field in features.items():
            if field.is_model_input:
                is_available = name in available_features

                presence[name] = torch.tensor(is_available, dtype=torch.bool)

        return presence

    def _load_datasets(
        self,
        recording: Recording,
    ) -> tuple[dict[str, TensorDataset], PresenceMasks]:
        datasets: dict[str, TensorDataset] = {}
        presences: PresenceMasks = {}

        for cfg in self.config.datasets:
            name = cfg.name
            source = getattr(recording, name)

            dataset, presence_mask = self._load_one_dataset(
                source=source,
                schema=cfg.dataset_cfg.tensor_schema,
                store_cfg=cfg.store_cfg,
                encoders=self.encoders[name],
                transforms=cfg.dataset_cfg.transforms,
            )

            datasets[name] = dataset
            presences[name] = presence_mask

        return datasets, presences

    @staticmethod
    def _available_table_features(
        *,
        table: TensorTable | None,
        transforms: Graph[Tensor],
    ) -> set[str]:
        if table is None:
            return set()

        available = set(table)

        for transform in transforms.transforms:
            if all(name in available for name in transform.inputs):
                available.add(transform.output)

        return available

    @staticmethod
    def _validate_recording_integrity(datasets: Mapping[str, TensorDataset]) -> None:
        stores = [
            (name, dataset.store)
            for name, dataset in datasets.items()
            if dataset.store is not None
        ]

        if not stores:
            return

        ref_name, ref_store = stores[0]
        ref_indices = tuple(ref_store.frame_indices)

        for name, store in stores[1:]:
            indices = tuple(store.frame_indices)

            if indices != ref_indices:
                raise ValueError(
                    f"Dataset '{name}' has different frame indices from dataset '{ref_name}'"
                )
