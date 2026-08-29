from pathlib import Path
from utils.files import load_json, dump_json

class EncodingHandler:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[int, int]:
        if not self.path.exists():
            return {}
                    
        data = load_json(self.path)

        # if later key are really strings -> abstract + dict[T, int]
        return {int(key): int(value) for key, value in data.items()}

    def append(self, key: int, value: int) -> None:
        data = self.load()

        if key in data:
            raise ValueError(f"Key {key} already exists in encoding store")

        if value in data.values():
            raise ValueError(f"Value {value} already exists in encoding store")

        data[key] = value

        dump_json(data, path=self.path, indent=4, sort_keys=True)
