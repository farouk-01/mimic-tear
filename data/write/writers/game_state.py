from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import TypedDict

from pydantic import BaseModel, ConfigDict
import pyarrow as pa
import pyarrow.parquet as pq

from data.capture.memory.game_state import RawGameStateSchema


class GameStateWriterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    flush_every: int
    compression: str = "zstd"


class GameStateWriter:
    def __init__(
        self,
        *,
        path: str | Path,
        schema: RawGameStateSchema,
        flush_every: int,
        compression: str = "zstd",
    ) -> None:
        if flush_every <= 0:
            raise ValueError("flush_every must be greater than zero")

        self.path = Path(path)
        self.fields = schema
        self.flush_every = flush_every

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.schema = schema.to_pyarrow_schema()

        self._writer = pq.ParquetWriter(self.path, self.schema, compression=compression)

        self._rows: list[dict[str, object]] = []
        self._last_index: int | None = None
        self._closed = False

    def _validate_values(self, values: Mapping[str, object]) -> None:
        missing = [name for name in self.schema.feature_names if name not in values]
        unexpected = [name for name in values if not self.schema.has_feature(name)]

        errors: list[Exception] = []
        if missing:
            errors.append(ValueError(f"Missing fields: {missing}"))
        if unexpected:
            errors.append(ValueError(f"Unexpected fields: {unexpected}"))

        if errors:
            raise ExceptionGroup("Invalid game-state values", errors)

    def write(
        self,
        *,
        index: int,
        timestamp_ns: int,
        values: Mapping[str, object],
    ) -> None:
        if self._closed:
            raise RuntimeError("Game-state writer is closed")

        if index < 0:
            raise ValueError("index cannot be negative")

        if timestamp_ns < 0:
            raise ValueError("timestamp_ns cannot be negative")

        if self._last_index is not None and index != self._last_index + 1:
            raise ValueError(
                "Game-state indices must be sequential: "
                f"expected {self._last_index + 1}, "
                f"received {index}"
            )

        self._validate_values(values)

        row: dict[str, object] = {"index": index, "timestamp_ns": timestamp_ns}

        for name in self.schema.feature_names:
            row[name] = values[name]

        self._rows.append(row)
        self._last_index = index

        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._closed:
            raise RuntimeError("Game-state writer is closed")

        if not self._rows:
            return

        table = pa.Table.from_pylist(self._rows, schema=self.schema)

        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        if self._closed:
            return

        self.flush()
        self._writer.close()
        self._closed = True

    def __enter__(self) -> GameStateWriter:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()
