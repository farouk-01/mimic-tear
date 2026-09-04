from pathlib import Path
from collections.abc import Sequence

import pyarrow.parquet as pq
import pyarrow as pa
import torch

from data.process.stores.base import (
    Store,
    SampleColumns,
    DEFAULT_SAMPLE_COLUMNS,
    StoreAdapter,
    TensorColumn,
    TensorTable,
    STORE_ADAPTERS,
)
from data.process.stores.validations import normalize_index, normalize_range
from utils import profile


class ParquetStore(Store[pa.Table]):
    def __init__(
        self,
        path: str | Path,
        *,
        columns: Sequence[str],
        sample_columns: SampleColumns = DEFAULT_SAMPLE_COLUMNS,
    ) -> None:
        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(f"Parquet file does not exist: {self.path}")

        self._columns = tuple(columns)

        frame_index = sample_columns.frame_index
        capture_timestamp_ns = sample_columns.capture_timestamp_ns

        if frame_index in self._columns or capture_timestamp_ns in self._columns:
            raise ValueError("Sample columns cannot also be payload columns")

        table = pq.read_table(
            self.path,
            columns=[frame_index, capture_timestamp_ns, *self._columns],
        )

        if table.num_rows <= 0:
            raise ValueError("Parquet file cannot be empty")

        self._frame_indices = table[frame_index].to_numpy(zero_copy_only=False)
        self._capture_timestamps_ns = table[capture_timestamp_ns].to_numpy(
            zero_copy_only=False
        )

        self._length = table.num_rows
        self._table = table.select(self._columns)

    def __len__(self) -> int:
        return self._length

    @property
    def frame_indices(self) -> Sequence[int]:
        return self._frame_indices

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    @property
    def capture_timestamp_ns(self) -> Sequence[int]:
        return self._capture_timestamps_ns

    @profile
    def get(self, index: int) -> pa.Table:
        index = normalize_index(index, len(self))

        return self._table.slice(index, 1)

    @profile
    def get_range(self, start: int, end: int) -> pa.Table:
        start, end = normalize_range(start, end, len(self))

        return self._table.slice(start, end - start)


@STORE_ADAPTERS.register(ParquetStore)
class ParquetStoreAdapter(StoreAdapter[pa.Table]):
    def get(self, data: pa.Table) -> TensorTable:
        return {name: self._to_tensor_column(data[name]) for name in data.column_names}

    def _to_tensor_column(self, data: pa.ChunkedArray) -> TensorColumn:
        validity = None

        if data.null_count > 0:
            validity = torch.from_numpy(
                data.is_valid().to_numpy(zero_copy_only=False).copy()
            )

        if pa.types.is_boolean(data.type):
            fill_value = False
        elif pa.types.is_integer(data.type):
            fill_value = 0
        elif pa.types.is_floating(data.type):
            fill_value = 0.0
        else:
            raise TypeError(f"Unsupported Parquet tensor column dtype: {data.type}")

        values = torch.from_numpy(
            data.fill_null(fill_value).to_numpy(zero_copy_only=False).copy()
        )

        return TensorColumn(values=values, validity=validity)
