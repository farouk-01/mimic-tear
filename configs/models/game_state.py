from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from data.models.tensor import TensorSchema
from graph.base import Graph, Value, Plan

from data.capture.memory import EldenRingMemoryProfile
from data.process.encoders.encoder import EncoderConfig
from data.process.stores.encoding import EncodingStoreConfig
from data.process.transforms.types.tensor import TransformNode
from configs.transforms.game_state import GAME_STATE_TRANSFORMS


class GameStateConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )

    memory_profile: EldenRingMemoryProfile
    tensor_gstate_schema: TensorSchema
    plan: Plan

    encoding_stores: tuple[EncodingStoreConfig, ...] = ()
    encoders: tuple[EncoderConfig, ...] = ()

    @classmethod
    def load(
        cls,
        *,
        memory_profile: EldenRingMemoryProfile,
        tensor_gstate_schema: TensorSchema,
        encodings_path: Path,
    ) -> Self:
        graph = Graph()

        for transform in GAME_STATE_TRANSFORMS:
            graph.add(TransformNode(transform=transform))

        outputs = tuple(
            graph.value(field.name)
            for field in tensor_gstate_schema.fields
            if field.is_model_input
        )

        plan = graph.resolve(outputs)

        input_names = {value.name for value in plan.inputs}

        nominal_fields = [
            field
            for field in tensor_gstate_schema.fields
            if field.name in input_names and field.kind == "nominal"
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

        return cls(
            memory_profile=memory_profile,
            tensor_gstate_schema=tensor_gstate_schema,
            plan=plan,
            encoding_stores=encoding_stores,
            encoders=encoders,
        )
