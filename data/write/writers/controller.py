from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ConfigDict, BaseModel, PositiveInt

from controller import (
    ANALOG_INPUTS,
    BUTTON_INPUTS,
    ControllerState,
)

class ControllerWriterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    flush_every: PositiveInt

class ControllerWriter:
    def __init__(
        self,
        *,
        path: str | Path,
        flush_every: int,
    ) -> None:
        if flush_every <= 0:
            raise ValueError(
                "flush_every must be greater than zero"
            )

        self.path = Path(path)
        self.flush_every = flush_every

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._schema = pa.schema(
            [
                pa.field("index", pa.int64()),
                pa.field("timestamp_ns", pa.int64()),
                *(
                    pa.field(name, pa.float32())
                    for name in ANALOG_INPUTS
                ),
                *(
                    pa.field(name, pa.bool_())
                    for name in BUTTON_INPUTS
                ),
            ]
        )

        self._writer = pq.ParquetWriter(
            self.path,
            self._schema,
            compression="zstd",
        )

        self._rows: list[dict[str, object]] = []
        self._last_index: int | None = None
        self._closed = False

    def write(
        self,
        *,
        index: int,
        timestamp_ns: int,
        state: ControllerState,
    ) -> None:
        if self._closed:
            raise RuntimeError(
                "Controller writer is closed"
            )

        if index < 0:
            raise ValueError("index cannot be negative")

        if timestamp_ns < 0:
            raise ValueError(
                "timestamp_ns cannot be negative"
            )

        if (
            self._last_index is not None
            and index != self._last_index + 1
        ):
            raise ValueError(
                "Controller indices must be sequential: "
                f"expected {self._last_index + 1}, "
                f"received {index}"
            )

        state.validate()

        row: dict[str, object] = {
            "index": index,
            "timestamp_ns": timestamp_ns,
        }

        row.update(
            {
                name: float(
                    getattr(state.analog, name)
                )
                for name in ANALOG_INPUTS
            }
        )

        row.update(
            {
                name: bool(
                    getattr(state.buttons, name)
                )
                for name in BUTTON_INPUTS
            }
        )

        self._rows.append(row)
        self._last_index = index

        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._closed:
            raise RuntimeError(
                "Controller writer is closed"
            )

        if not self._rows:
            return

        table = pa.Table.from_pylist(
            self._rows,
            schema=self._schema,
        )

        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        if self._closed:
            return

        self.flush()
        self._writer.close()
        self._closed = True

    def __enter__(self) -> ControllerWriter:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()