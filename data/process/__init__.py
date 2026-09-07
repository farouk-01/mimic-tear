from collections.abc import Mapping, Collection
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict
import torch

from data.models.record import Recording, RecordingConfig, RecordingName
from data.models.tensor import TensorSchema
from data.process.transforms import TensorTransform

from .datasets.tensor import TensorDataset, TensorDatasetConfig
from .encoders.encoder import Encoder, EncoderConfig
from .sequence import SequenceDataset
from .stores.encoding import EncodingStore, EncodingStoreConfig
from .stores.base import FILE_STORES, StoreConfig

from utils import profile

__all__ = [
    "SequenceDataset",
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


type PresenceMask = dict[str, dict[str, torch.Tensor] | None]


class Process:
    def __init__(self, *, config: ProcessConfig) -> None:
        self.config = config
        self.encoders: dict[RecordingName, tuple[Encoder, ...]] = self._build_encoders()

    @profile
    def process_sequence(
        self, source: str | Path
    ) -> tuple[SequenceDataset, PresenceMask]:
        recording = Recording.from_directory(root=source, config=self.config.recording)

        datasets, presences = self._load_datasets(recording)

        self._validate_recording_integrity(datasets)

        return (
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

        datasets, _ = self._load_datasets(recording)

        for dataset in datasets.values():
            dataset.discover_encodings()

    @property
    def encoding_cardinalities(self) -> Mapping[str, int]:
        result: dict[str, int] = {}

        for encoders in self.encoders.values():
            for encoder in encoders:
                for name in encoder.fields:
                    result[name] = encoder.cardinality

        return MappingProxyType(result)

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
        source: str | Path,
        *,
        schema: TensorSchema,
        store_cfg: StoreConfig,
        encoders: tuple[Encoder, ...] = (),
        transforms: tuple[TensorTransform, ...] = (),
    ) -> tuple[TensorDataset, dict[str, torch.Tensor] | None]:
        source = Path(source)
        suffix = source.suffix.lower()

        store_cls = FILE_STORES.resolve(suffix)
        store = store_cls(source=source, **store_cfg.model_dump())

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
    ) -> tuple[dict[str, TensorDataset], PresenceMask]:
        datasets: dict[str, TensorDataset] = {}
        presences: PresenceMask = {}

        for cfg in self.config.datasets:
            name = cfg.name
            source = getattr(recording, name)

            if source is None:
                presences[name] = None
                continue

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
    def _validate_recording_integrity(datasets: Mapping[str, TensorDataset]) -> None:
        if not datasets:
            return

        iterator = iter(datasets.items())
        ref_name, ref_dataset = next(iterator)
        ref_indices = tuple(ref_dataset.store.frame_indices)

        for name, dataset in iterator:
            indices = tuple(dataset.store.frame_indices)

            if indices != ref_indices:
                raise ValueError(
                    f"Dataset '{name}' has different frame indices from dataset '{ref_name}'"
                )
