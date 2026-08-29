from pathlib import Path
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from utils.files import dump_json, load_json

class EncodingStoreConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    encoding: str
    path: Path

class EncodingStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[int, int]:
        if not self.path.exists():
            return {}

        data = load_json(self.path)

        # if later key are really strings -> abstract + dict[T, int]
        return {int(key): int(value) for key, value in data.items()}
    

    def append(self, keys: Sequence[int] | int, values: Sequence[int] | int) -> None:
        if isinstance(keys, int):
            keys = (keys,)
        if isinstance(values, int):
            values = (values,)

        if len(keys) != len(values):
            raise ValueError("keys and values must have the same length")

        data = self.load()

        for k, v in zip(keys, values):
            if k in data:
                raise ValueError(f"Key {k} already exists in encoding store")

            if v in data.values():
                raise ValueError(f"Value {v} already exists in encoding store")

            data[k] = v

        dump_json(data, path=self.path, indent=4, sort_keys=True)