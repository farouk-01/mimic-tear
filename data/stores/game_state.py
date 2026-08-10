from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from data.datasets.game_state import (
    GameStateStore,
    GameStateValue,
)


class ParquetGameStateStore(GameStateStore):
    def __init__(
        self,
        *,
        path: str | Path,
        features: tuple[str, ...],
    ) -> None:
        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Game-state parquet does not exist: {self.path}"
            )

        if not features:
            raise ValueError(
                "Game-state features cannot be empty"
            )

        table = pq.read_table(
            self.path,
            columns=list(features),
        )

        missing = [
            feature
            for feature in features
            if feature not in table.column_names
        ]

        if missing:
            raise ValueError(
                f"Game-state parquet is missing features: {missing}"
            )

        if table.num_rows <= 0:
            raise ValueError(
                "Game-state parquet cannot be empty"
            )

        self._features = features

        self._columns = {
            feature: table[feature].to_pylist()
            for feature in features
        }

        self._length = table.num_rows

    @property
    def features(self) -> tuple[str, ...]:
        return self._features

    def __len__(self) -> int:
        return self._length

    def get(
        self,
        index: int,
    ) -> dict[str, GameStateValue]:
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(index)

        return {
            feature: self._columns[feature][index]
            for feature in self._features
        }