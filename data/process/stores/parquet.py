from pathlib import Path
from collections.abc import Sequence, Collection

import pyarrow.parquet as pq
import pyarrow as pa
import torch

from data.process.stores.base import (
    Store,
    StoreAdapter,
    StoreConfig,
    TensorColumn,
    TensorTable,
    STORE_ADAPTERS,
    FILE_STORES,
)
from data.process.stores.validations import normalize_index, normalize_range
from data.write.metadata import DEFAULT_COLUMNS, MetaDataStoreColumns

from utils import profile


class ParquetStoreConfig(StoreConfig):
    columns: Sequence[str]


@FILE_STORES.register(".parquet")
class ParquetStore(Store[pa.Table]):
    def __init__(
        self,
        source: str | Path,
        *,
        columns: Sequence[str],
        metadata_columns: MetaDataStoreColumns = DEFAULT_COLUMNS,
    ) -> None:
        super().__init__(source=source)

        if not self.source.is_file():
            raise FileNotFoundError(f"Parquet file does not exist: {self.source}")

        available_columns = pq.read_schema(self.source).names
        self._columns = [col for col in columns if col in available_columns]

        frame_index = metadata_columns.frame_index
        capture_timestamp_ns = metadata_columns.capture_timestamp_ns

        if frame_index in self._columns or capture_timestamp_ns in self._columns:
            raise ValueError("Sample columns cannot also be payload columns")

        table = pq.read_table(
            self.source,
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
    def feature_names(self) -> Collection[str]:
        return self._columns

    @property
    def capture_timestamp_ns(self) -> Sequence[int]:
        raise NotImplementedError("Capture timestamps not yet supported")

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
