
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from utils.files import load_json
from data.models.tensor import TensorSchema
from data.capture.memory import EldenRingMemoryProfile

SCHEMA_DIR = Path("configs/schemas")


@dataclass(frozen=True, slots=True)
class TensorSchemaPaths:
    root: Path = SCHEMA_DIR / "tensor"

    def controller(self, name: Literal["gamepad"] = "gamepad", version: str = "1") -> TensorSchema:
        dict = load_json(self.root / "controller" / name / f"v{version}" / "schema.json")
        return TensorSchema.from_json(dict)
        
    def game_state(self, version: str) -> TensorSchema:
        dict = load_json(self.root / "game_state" / f"v{version}" / "schema.json")
        return TensorSchema.from_json(dict)

    def frame(self) -> TensorSchema:
        dict = load_json(self.root / "frame" / "schema.json")
        return TensorSchema.from_json(dict)


@dataclass(frozen=True, slots=True)
class MemorySchemaPaths:
    root: Path = SCHEMA_DIR / "memory"

    def game_state(self, version: str = "1.16.2") -> EldenRingMemoryProfile:
        dict = load_json(self.root / "game_state" / f"v{version}" / "schema.json")
        return EldenRingMemoryProfile.model_validate(dict)


@dataclass(frozen=True, slots=True)
class Schemas:
    tensor: TensorSchemaPaths = TensorSchemaPaths()
    memory: MemorySchemaPaths = MemorySchemaPaths()