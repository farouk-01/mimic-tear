from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor

from data.process.transforms.tensor import TensorTransform
from configs.transforms.game_state import GAME_STATE_TRANSFORMS
from data.capture.memory import EldenRingMemoryProfile
from data.models.tensor import TensorSchema
from data.process.encoders.encoder import EncoderConfig
from data.process.stores.encoding import EncodingStoreConfig
from data.process.stores.parquet import ParquetStoreConfig
from data.process.transforms import Graph


class GameStateConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )

    memory_profile: EldenRingMemoryProfile
    tensor_schema: TensorSchema
    transforms: Graph[Tensor]
    store_cfg: ParquetStoreConfig

    encoding_stores: tuple[EncodingStoreConfig, ...] = ()
    encoders: tuple[EncoderConfig, ...] = ()

    @classmethod
    def load(
        cls,
        *,
        memory_profile: EldenRingMemoryProfile,
        tensor_schema: TensorSchema,
        encodings_path: Path,
    ) -> Self:
        transforms = cls._resolve_store_columns(
            schema=tensor_schema,
            transforms=GAME_STATE_TRANSFORMS,
        )
        
        nominal_fields = [
            field
            for field in tensor_schema.fields
            if field.is_model_input and field.kind == "nominal"
        ]

        grouped: dict[str, set[str]] = {}

        for field in nominal_fields:
            if field.encoding is None:
                raise ValueError(f"Nominal field '{field.name}' has no encoding")

            grouped.setdefault(field.encoding, set()).add(field.name)

        encoding_stores = tuple(
            EncodingStoreConfig(
                encoding=encoding,
                path=encodings_path / f"{encoding}.json",
            )
            for encoding in sorted(grouped)
        )

        encoders = tuple(
            EncoderConfig(encoding=encoding, fields=tuple(sorted(grouped[encoding])))
            for encoding in sorted(grouped)
        )

        store_columns = {
            name
            for name, field in tensor_schema.fields_by_name.items()
            if field.is_model_input and not field.is_derived
        }

        store_columns.update(transforms.inputs)

        store_cfg = ParquetStoreConfig(columns=tuple(sorted(store_columns)))

        return cls(
            memory_profile=memory_profile,
            tensor_schema=tensor_schema,
            transforms=transforms,
            store_cfg=store_cfg,
            encoding_stores=encoding_stores,
            encoders=encoders,
        )

    @staticmethod
    def _resolve_store_columns(
        schema: TensorSchema,
        transforms: tuple[TensorTransform, ...],
    ) -> Graph[Tensor]:
        fields = schema.fields_by_name

        available = {name for name, field in fields.items() if not field.is_derived}
        valid_transforms: list[TensorTransform] = []

        for t in transforms:
            if t.output not in fields:
                continue

            missing_inputs = set(t.inputs) - available
            if missing_inputs:
                raise ValueError(
                    f"Transform '{t}' has missing inputs: {missing_inputs}"
                )

            valid_transforms.append(t)
            available.add(t.output)

        cols = {
            name
            for name, field in fields.items()
            if field.is_model_input and not field.is_derived
        }

        for t in valid_transforms:
            for name in t.inputs:
                if name not in fields:
                    continue

                if not fields[name].is_derived:
                    cols.add(name)

        return Graph[Tensor](valid_transforms)