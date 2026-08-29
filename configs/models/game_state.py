from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from utils.files import load_json
from data.capture.memory import EldenRingMemoryProfile
from data.models.game_state.processed import ProcessedGameStateSchema
from data.process.encoders.game_state import GameStateEncoderConfig
from data.process.stores.encoding import EncodingStoreConfig


class GameStateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    memory_profile: EldenRingMemoryProfile
    processed_schema: ProcessedGameStateSchema

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

        proc_schema = ProcessedGameStateSchema.model_validate(
            {"fields": processed_fields}
        )

        nominal_fields = proc_schema.get_required_fields_by_kind("nominal")

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
            processed_schema=proc_schema,
            encoders=tuple(encoders_cfg),
            encoding_stores=tuple(encoding_stores_cfg),
        )
