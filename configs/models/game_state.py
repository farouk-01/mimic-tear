from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from utils.files import load_json
from data.capture.memory import EldenRingMemoryProfile
from data.models.game_state.processed import ProcessedGameStateSchema


class GameStateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    memory_profile: EldenRingMemoryProfile
    processed_schema: ProcessedGameStateSchema

    @classmethod
    def load(cls, *, memory_schema: Path, processed_schema: Path) -> Self:
        mem_data = load_json(memory_schema)
        proc_data = load_json(processed_schema)

        mem_schema = EldenRingMemoryProfile.model_validate(mem_data)

        fields = tuple(
            {"name": name, **definition} for name, definition in proc_data.items()
        )

        proc_schema = ProcessedGameStateSchema.model_validate({"fields": fields})

        return cls(
            memory_profile=mem_schema,
            processed_schema=proc_schema,
        )
