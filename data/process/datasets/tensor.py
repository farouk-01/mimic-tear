from torch import Tensor
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict

from data.models.tensor import TORCH_DTYPES, TensorSchema
from data.process.stores.base import Store, STORE_ADAPTERS, TensorColumn, TensorTable
from graph.base import Plan
from graph.types.tensor import TensorGraphExecutor
from data.process.encoders.encoder import Encoder, TensorEncoder

from utils import profile


class TensorDataset(Dataset[TensorDict]):
    def __init__(
        self,
        *,
        store: Store,
        schema: TensorSchema,
        encoders: tuple[Encoder, ...] = (),
        plan: Plan,
        executor: TensorGraphExecutor | None = None,
    ) -> None:
        if len(store) <= 0:
            raise ValueError("Store cannot be empty")

        self.store = store
        self.schema = schema
        self.plan = plan
        self.executor = executor or TensorGraphExecutor()

        self.encoders = tuple(TensorEncoder(encoder) for encoder in encoders)

        adapter_cls = STORE_ADAPTERS.resolve(type(store))
        self.adapter = adapter_cls()

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int) -> TensorDict:
        data = self.store.get(index)
        table = self.adapter.get(data)

        return self._process_table(table, batch_size=[1])

    @profile
    def get_range(self, start: int, end: int) -> TensorDict:
        data = self.store.get_range(start, end)
        table = self.adapter.get(data)

        return self._process_table(table, batch_size=[end - start])

    def discover_encodings(self) -> None:
        if not self.encoders:
            return

        data = self.store.get_range(0, len(self.store))
        table = self.adapter.get(data)

        for encoder in self.encoders:
            for field_name in encoder.fields:
                col = table[field_name]
                tensor = self._materialize_column(field_name, col)
                encoder.discover(tensor)

    def _process_table(
        self,
        table: TensorTable,
        *,
        batch_size: list[int],
    ) -> TensorDict:
        tensors = TensorDict(
            {
                name: self._materialize_column(name, column)
                for name, column in table.items()
            },
            batch_size=batch_size,
            # device="cuda",
        )

        for encoder in self.encoders:
            for field_name in encoder.fields:
                tensors[field_name] = encoder.encode(tensors[field_name])

        tensors = self.executor.execute(self.plan, tensors)

        self._validate_tensors(tensors)

        return tensors

    def _materialize_column(self, name: str, column: TensorColumn) -> Tensor:
        field = self.schema.get_field(name)
        dtype = TORCH_DTYPES[field.dtype]

        values = column.values.to(dtype)
        if column.validity is None:
            return values

        if not field.nullable:
            raise ValueError(f"Non-nullable feature '{name}' contains null values")

        return torch.where(
            column.validity,
            column.values,
            torch.as_tensor(field.fill_value, dtype=dtype, device=values.device),
        )

    def _validate_tensors(self, tensors: TensorDict) -> None:
        expected = {field.name for field in self.schema.fields}

        actual = set(tensors.keys())

        missing = expected - actual
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")

        unexpected = actual - expected
        if unexpected:
            raise ValueError(f"Unexpected features: {sorted(unexpected)}")

        for field in self.schema.fields:
            tensor = tensors[field.name]
            expected_dtype = TORCH_DTYPES[field.dtype]

            if tensor.dtype != expected_dtype:
                raise TypeError(
                    f"Feature '{field.name}' has dtype {tensor.dtype}, "
                    f"expected {expected_dtype}"
                )
