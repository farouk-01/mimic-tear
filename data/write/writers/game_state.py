from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import TypedDict

from pydantic import BaseModel, ConfigDict
import pyarrow as pa
import pyarrow.parquet as pq


class GameStateWriterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    path: str | Path
    schema_: Mapping[str, str] 
    flush_every: int
    compression: str = "zstd"


class GameStateWriter:
    def __init__(
        self,
        *,
        path: str | Path,
        schema: Mapping[str, str],
        flush_every: int,
        compression: str = "zstd",
    ) -> None:
        if flush_every <= 0:
            raise ValueError("flush_every must be greater than zero")

        self.path = Path(path)
        self.fields = schema
        self.flush_every = flush_every

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._schema = pa.schema(
            [
                pa.field("index", pa.int64()),
                pa.field("timestamp_ns", pa.int64()),
                *(
                    pa.field(field, pa.from_numpy_dtype(type))
                    for field, type in self.fields.items()
                ),
            ]
        )

        self._writer = pq.ParquetWriter(
            self.path, self._schema, compression=compression
        )

        self._rows: list[dict[str, object]] = []
        self._last_index: int | None = None
        self._closed = False

    def _validate_values(self, values: Mapping[str, object]) -> None:
        missing = [field for field in self.fields.keys() if field not in values]
        unexpected = [field for field in values.keys() if field not in self.fields]

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

        for field in self.fields:
            row[field] = values[field]

        self._rows.append(row)
        self._last_index = index

        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._closed:
            raise RuntimeError("Game-state writer is closed")

        if not self._rows:
            return

        table = pa.Table.from_pylist(self._rows, schema=self._schema)

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
