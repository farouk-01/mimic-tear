from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from utils.files import load_json
from graph.base import Graph, Value, Plan

from data.capture.memory import EldenRingMemoryProfile
from data.models.game_state.processed import ProcessedGameStateSchema
from data.process.encoders.game_state import GameStateEncoderConfig
from data.process.stores.encoding import EncodingStoreConfig
from data.process.transforms.types.tensor import Ratio, TensorTransform, TransformNode

GAME_STATE_TRANSFORMS: tuple[TensorTransform, ...] = (
    Ratio(
        output="player_hp_ratio",
        numerator="player_health",
        denominator="player_max_health",
    ),
    Ratio(
        output="player_fp_ratio",
        numerator="player_fp",
        denominator="player_max_fp",
    ),
    Ratio(
        output="player_stamina_ratio",
        numerator="player_stamina",
        denominator="player_max_stamina",
    ),
    Ratio(
        output="enemy_hp_ratio",
        numerator="enemy_health",
        denominator="enemy_max_health",
    ),
)


class GameStateConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )

    memory_profile: EldenRingMemoryProfile
    processed_schema: ProcessedGameStateSchema
    plan: Plan

    encoding_stores: tuple[EncodingStoreConfig, ...] = ()
    encoders: tuple[GameStateEncoderConfig, ...] = ()

    @classmethod
    def load(
        cls,
        *,
        memory_schema: Path,
        processed_schema: Path,
        encodings_path: Path,
    ) -> Self:
        mem_data = load_json(memory_schema)
        proc_data = load_json(processed_schema)

        mem_schema = EldenRingMemoryProfile.model_validate(mem_data)

        processed_fields = tuple(
            {"name": name, **definition} for name, definition in proc_data.items()
        )

        schema = ProcessedGameStateSchema.model_validate({"fields": processed_fields})

        graph = Graph()
        for t in GAME_STATE_TRANSFORMS:
            graph.add(TransformNode(transform=t))

        outputs = tuple(
            Value(field.name) for field in schema.fields if field.is_model_input
        )

        plan = graph.resolve(outputs)

        input_names = {v.name for v in plan.inputs}
        nominal_fields = [
            field
            for field in schema.fields
            if field.name in input_names and field.kind == "nominal"
        ]

        grouped: dict[str, set[str]] = {}
        encoders_cfg: list[GameStateEncoderConfig] = []
        encoding_stores_cfg: list[EncodingStoreConfig] = []
        for field in nominal_fields:
            if field.encoding is None:
                raise ValueError(f"Nominal field '{field.name}' has no encoding")

            grouped.setdefault(field.encoding, set()).add(field.name)

        for encoding, fields in grouped.items():
            encoding_stores_cfg.append(
                EncodingStoreConfig(
                    encoding=encoding,
                    path=encodings_path / f"{encoding}.json",
                )
            )

            encoders_cfg.append(
                GameStateEncoderConfig(
                    encoding=encoding,
                    fields=tuple(sorted(fields)),
                )
            )

        return cls(
            memory_profile=mem_schema,
            processed_schema=schema,
            plan=plan,
            encoders=tuple(encoders_cfg),
            encoding_stores=tuple(encoding_stores_cfg),
        )
