from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import pyarrow.parquet as pq
import pyarrow as pa
from pydantic import BaseModel, ConfigDict
import torch

from data.models.game_state.processed import (
    ProcessedGameStateField,
    ProcessedGameStateSchema,
)

from .validations import normalize_index, normalize_range

from ..datasets.game_state import (
    GameStateStore,
    GameStateStoreAdapter,
    game_state_store_adapters,
)


class ParquetGameStateStoreConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    features: tuple[str, ...]


class ParquetGameStateStore(GameStateStore[pa.Table, pa.Table, pa.ChunkedArray]):
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

        self._indices: Sequence[int] = table["index"].to_numpy(zero_copy_only=False)
        self._timestamps_ns: Sequence[int] = table["timestamp_ns"].to_numpy(
            zero_copy_only=False
        )
        self._features = features
        self._length: int = table.num_rows
        self._table = table.select(features)

    @property
    def indices(self) -> Sequence[int]:
        return self._indices

    @property
    def timestamps_ns(self) -> Sequence[int]:
        return self._timestamps_ns

    @property
    def features(self) -> tuple[str, ...]:
        return self._features

    def __len__(self) -> int:
        return self._length

    def get(self, index: int) -> pa.Table:
        index = normalize_index(index, len(self))

        return self._table.slice(index, 1)

    def get_range(self, start: int, end: int) -> pa.Table:
        start, end = normalize_range(start, end, len(self))

        return self._table.slice(start, end - start)

    def get_feature(self, name: str) -> pa.ChunkedArray:
        if name not in self._features:
            raise ValueError(f"Feature not found in game-state store: {name}")

        return self._table[name]


@game_state_store_adapters.register(ParquetGameStateStore)
class ParquetGameStateStoreAdapter(
    GameStateStoreAdapter[pa.Table, pa.Table, pa.ChunkedArray]
):
    def get(
        self, data: pa.Table, schema: ProcessedGameStateSchema
    ) -> dict[str, torch.Tensor]:
        return self._to_tensors(data, schema)

    def get_range(
        self, data: pa.Table, schema: ProcessedGameStateSchema
    ) -> dict[str, torch.Tensor]:
        return self._to_tensors(data, schema)

    def get_feature(
        self, data: pa.ChunkedArray, field: ProcessedGameStateField
    ) -> torch.Tensor: 
        if field.fill_value is not None:
            data = data.fill_null(field.fill_value)
        elif data.null_count > 0:
            raise ValueError(
                f"Unexpected null values in non-nullable field: {field.name}"
            )

        return torch.from_numpy(data.to_numpy(zero_copy_only=False))

    def _to_tensors(
        self, data: pa.Table, schema: ProcessedGameStateSchema
    ) -> dict[str, torch.Tensor]:
        tensors = {}

        for field in schema.get_required_fields(include_derived=False):
            tensor = self.get_feature(data[field.name], field)
            
            tensors[field.name] = tensor

        return tensors
