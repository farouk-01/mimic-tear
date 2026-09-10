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
        transforms = cls._resolve_transforms(
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
    def _resolve_transforms(
        schema: TensorSchema,
        transforms: tuple[TensorTransform, ...],
    ) -> Graph[Tensor]:
        fields = schema.fields_by_name

        by_output = {transform.output: transform for transform in transforms}

        required_outputs = {
            name
            for name, field in fields.items()
            if field.is_model_input and field.is_derived
        }

        required_transform_ids: set[int] = set()

        def resolve(name: str) -> None:
            transform = by_output.get(name)

            if transform is None:
                if name not in fields:
                    raise ValueError(
                        f"Transform dependency '{name}' is neither "
                        "produced by a transform nor defined in the schema"
                    )
                return

            if transform.id in required_transform_ids:
                return

            required_transform_ids.add(transform.id)

            for input_name in transform.inputs:
                resolve(input_name)

        for output in required_outputs:
            resolve(output)

        valid_transforms = tuple(
            transform
            for transform in transforms
            if transform.id in required_transform_ids
        )

        return Graph[Tensor](valid_transforms)
