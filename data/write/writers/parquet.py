from pathlib import Path
from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, PositiveInt


class ParquetWriterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    flush_every: PositiveInt
    compression: str = "zstd"


class ParquetWriter:
    def __init__(
        self,
        path: str | Path,
        schema: pa.Schema,
        *,
        flush_every: int,
        compression: str = "zstd",
    ) -> None:
        if flush_every <= 0:
            raise ValueError("flush_every must be greater than zero")

        self.path = Path(path)
        self.schema = schema
        self.flush_every = flush_every

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._writer = pq.ParquetWriter(self.path, self.schema, compression=compression)

        self._columns = frozenset(schema.names)
        self._rows: list[dict[str, object]] = []
        self._row_count = 0
        self._closed = False

    @property
    def row_count(self) -> int:
        return self._row_count

    def write(self, row: Mapping[str, object]) -> None:
        self._ensure_open()
        self._validate_values(row)

        self._rows.append({name: row[name] for name in self.schema.names})
        self._row_count += 1

        if len(self._rows) >= self.flush_every:
            self.flush()

    def write_many(self, rows: Iterable[Mapping[str, object]]) -> None:
        for row in rows:
            self.write(row)

    def flush(self) -> None:
        self._ensure_open()

        if not self._rows:
            return

        table = pa.Table.from_pylist(self._rows, schema=self.schema)

        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        if self._closed:
            return

        try:
            self.flush()
        finally:
            self._writer.close()
            self._closed = True

    def _validate_values(self, values: Mapping[str, object]) -> None:
        missing = [name for name in self.schema.names if name not in values]
        unexpected = [name for name in values if name not in self._columns]

        errors: list[Exception] = []
        if missing:
            errors.append(ValueError(f"Missing columns: {missing}"))
        if unexpected:
            errors.append(ValueError(f"Unexpected columns: {unexpected}"))

        if errors:
            raise ExceptionGroup(f"Invalid row for {self.path.name}", errors)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Parquet writer is closed: {self.path}")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
