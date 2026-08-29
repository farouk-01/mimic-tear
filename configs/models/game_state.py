from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from utils.files import load_json
from data.capture.memory import EldenRingMemoryProfile
from data.models.game_state.processed import ProcessedGameStateSchema
from data.process.encoders.game_state import GameStateEncoderConfig
from configs.encodings.handler import EncodingHandler


class GameStateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    memory_profile: EldenRingMemoryProfile
    processed_schema: ProcessedGameStateSchema

    encoders_cfg: tuple[GameStateEncoderConfig, ...] = ()

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

        fields = tuple(
            {"name": name, **definition} for name, definition in proc_data.items()
        )

        proc_schema = ProcessedGameStateSchema.model_validate({"fields": fields})

        nominal_fields = proc_schema.get_required_fields_by_kind("nominal")

        grouped: dict[str, set[str]] = {}
        encoders_cfg: list[GameStateEncoderConfig] = []
        for field in nominal_fields:
            if field.encoding is None:
                continue

            grouped.setdefault(field.encoding, set()).add(field.name)

        for encoding, fields in grouped.items():
            handler = EncodingHandler(encodings_path / f"{encoding}.json")

            encoders_cfg.append(
                GameStateEncoderConfig(
                    encoding=encoding,
                    fields=tuple(sorted(fields)),
                    load_encodings=handler.load,
                    append_encoding=handler.append,
                )
            )

        return cls(
            memory_profile=mem_schema,
            processed_schema=proc_schema,
            encoders_cfg=tuple(encoders_cfg),
        )
