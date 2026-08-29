from pathlib import Path
import json


class EncodingHandler:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[int, int]:
        if not self.path.exists():
            return {}
                    
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # if later key are really strings -> abstract + dict[T, int]
        return {int(key): int(value) for key, value in data.items()}

    def append(self, key: int, value: int) -> None:
        data = self.load()

        if key in data:
            raise ValueError(f"Key {key} already exists in encoding store")

        if value in data.values():
            raise ValueError(f"Value {value} already exists in encoding store")

        data[key] = value

        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
