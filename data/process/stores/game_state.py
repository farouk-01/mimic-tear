from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor
import numpy as np

from .validations import normalize_index, normalize_range

from ..datasets.game_state import GameStateStore, GameStateValue


class ParquetGameStateStoreConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    features: tuple[str, ...]


class ParquetGameStateStore(GameStateStore):
    def __init__(
        self,
        path: str | Path,
        *,
        features: tuple[str, ...],
    ) -> None:
        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(f"Game-state parquet does not exist: {self.path}")

        if not features:
            raise ValueError("Game-state features cannot be empty")

        required = ("index", "timestamp_ns", *features)

        table = pq.read_table(self.path, columns=required)

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

        self._indices = torch.tensor(
            table["index"].to_numpy(zero_copy_only=False),
            dtype=torch.int64,
        )

        self._timestamps_ns = torch.tensor(
            table["timestamp_ns"].to_numpy(zero_copy_only=False),
            dtype=torch.int64,
        )

    @property
    def indices(self) -> torch.Tensor:
        return self._indices

    @property
    def timestamps_ns(self) -> torch.Tensor:
        return self._timestamps_ns

    @property
    def features(self) -> tuple[str, ...]:
        return self._features

    def __len__(self) -> int:
        return self._length

    def get(self, index: int) -> dict[str, GameStateValue]:
        index = normalize_index(index, len(self))

        state = self._states[index]

        return {feature: state[i].item() for i, feature in enumerate(self._features)}

    def get_range(self, start: int, end: int) -> Tensor:
        start, end = normalize_range(start, end, len(self))

        return self._states[start:end]
