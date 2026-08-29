from pathlib import Path
from typing import Any, overload
import json


def load_toml(path: str | Path) -> dict[str, Any]:
    import tomllib

    with Path(path).open("rb") as f:
        return tomllib.load(f)


def load_json(path: str | Path) -> dict[str, Any]:

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(
    data: dict[str, Any] | dict[int, Any],
    *,
    path: str | Path,
    indent: int = 4,
    sort_keys: bool = False,
) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)
