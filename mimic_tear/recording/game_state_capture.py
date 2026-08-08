from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from mimic_tear.game_state.sampler import (
    GameStateSample,
    GameStateSampler,
    GameStateSamplerStats,
    nearest_game_state_sample,
)
from mimic_tear.game_state.schema import (
    GAME_STATE_COLUMNS,
    GAME_STATE_PARQUET_SCHEMA,
    GAME_STATE_VALUE_TYPES,
)


class GameStateWriter:
    def __init__(self, path: Path, *, row_group_size: int = 300) -> None:
        if row_group_size <= 0:
            raise ValueError("row_group_size must be greater than zero")
        self._writer = pq.ParquetWriter(
            path,
            GAME_STATE_PARQUET_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        self._row_group_size = row_group_size
        self._buffer: list[dict[str, Any]] = []
        self._closed = False

    def write(
        self,
        *,
        frame_index: int,
        timestamp_ns: int,
        frame_timestamp_ns: int,
        sample: GameStateSample,
    ) -> None:
        row: dict[str, Any] = {
            "frame_index": frame_index,
            "timestamp_ns": timestamp_ns,
            "frame_timestamp_ns": frame_timestamp_ns,
            "state_timestamp_ns": sample.timestamp_ns,
            "state_offset_ns": sample.timestamp_ns - frame_timestamp_ns,
            "state_valid": sample.snapshot.valid,
        }
        row.update(
            {
                name: sample.snapshot.values.get(name)
                for name in GAME_STATE_VALUE_TYPES
            }
        )
        self._buffer.append(row)
        if len(self._buffer) >= self._row_group_size:
            self._flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._flush()
        finally:
            self._writer.close()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=GAME_STATE_PARQUET_SCHEMA)
        self._writer.write_table(table)
        self._buffer.clear()

    def __enter__(self) -> "GameStateWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def validate_game_state_columns(columns: Sequence[str]) -> None:
    if tuple(columns) != GAME_STATE_COLUMNS:
        raise ValueError("Game-state columns do not match the canonical schema")
