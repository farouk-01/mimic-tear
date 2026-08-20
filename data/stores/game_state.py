from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch import Tensor
import numpy as np

from data.datasets.game_state import GameStateStore, GameStateValue


class ParquetGameStateStore(GameStateStore):
    def __init__(
        self,
        *,
        path: str | Path,
        features: tuple[str, ...],
    ) -> None:
        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(f"Game-state parquet does not exist: {self.path}")

        if not features:
            raise ValueError("Game-state features cannot be empty")

        table = pq.read_table(self.path, columns=list(features))

        missing = [feature for feature in features if feature not in table.column_names]

        if missing:
            raise ValueError(f"Game-state parquet is missing features: {missing}")

        if table.num_rows <= 0:
            raise ValueError("Game-state parquet cannot be empty")

        self._features = features

        self._states = torch.from_numpy(
            np.stack(
                [table[feature].to_numpy(zero_copy_only=False) for feature in features],
                axis=1,
            )
        ).to(torch.float32)

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

        state = self._states[index]

        return {feature: state[i].item() for i, feature in enumerate(self._features)}

    def get_range(self, start: int, end: int) -> Tensor:
        if start < 0:
            start += len(self)

        if end < 0:
            end += len(self)

        if not 0 <= start <= end <= len(self):
            raise IndexError(f"Invalid range [{start}:{end}]")

        return self._states[start:end]
