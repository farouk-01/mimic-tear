from collections.abc import Collection

from torch import Tensor
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict
from pydantic import BaseModel, ConfigDict

from data.models.tensor import TensorSchema
from data.process.stores.base import (
    Store,
    StoreAdapter,
    STORE_ADAPTERS,
    TensorColumn,
    TensorTable,
)
from data.process.transforms import TensorTransform, Graph
from data.process.encoders.encoder import Encoder, TensorEncoder

from utils import profile


class TensorDatasetConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )

    tensor_schema: TensorSchema
    transforms: Graph[Tensor]


class TensorDataset(Dataset[TensorDict]):
    def __init__(
        self,
        store: Store | None,
        *,
        tensor_schema: TensorSchema,
        encoders: tuple[Encoder, ...] = (),
        transforms: Graph[Tensor],
    ) -> None:
        if store is not None:
            if len(store) <= 0:
                raise ValueError("Store cannot be empty")

        self.store = store
        self.schema = tensor_schema
        self.transforms = transforms

        self.encoders = tuple(TensorEncoder(encoder) for encoder in encoders)

        self.adapter: StoreAdapter | None

        if store is not None:
            adapter_cls = STORE_ADAPTERS.resolve(type(store))
            self.adapter = adapter_cls()
        else:
            self.adapter = None

    def __len__(self) -> int:
        if self.store is None:
            raise TypeError("TensorDataset with no Store has no length")

        return len(self.store)

    def __getitem__(self, index: int) -> TensorDict:
        if self.store is None or self.adapter is None:
            return self.process_table(None, batch_size=[1])

        data = self.store.get(index)
        table = self.adapter.get(data)

        return self.process_table(table, batch_size=[1])

    def get_range(self, start: int, end: int) -> TensorDict:
        if self.store is None or self.adapter is None:
            return self.process_table(None, batch_size=[end - start])

        data = self.store.get_range(start, end)
        table = self.adapter.get(data)

        return self.process_table(table, batch_size=[end - start])

    def discover_encodings(self) -> None:
        if self.store is None or self.adapter is None:
            return

        if not self.encoders:
            return

        data = self.store.get_range(0, len(self.store))
        table = self.adapter.get(data)

        for encoder in self.encoders:
            for field_name in encoder.fields:
                if field_name not in table:
                    continue

                column = table[field_name]
                tensor = self._materialize_column(field_name, column)

                encoder.discover(tensor)

    @property
    def available_features(self) -> set[str]:
        if self.store is None:
            return set()

        # base fields that the store can provide
        store_available = set(self.store.feature_names)

        # only return features that are model inputs and that the store can produce
        return self.available_features_for(store_available)

    def available_features_for(self, names: Collection[str]) -> set[str]:
        # everything the graph can produce from those fields
        available = self.transforms.resolve_available(names)

        return set(available) & set(self.schema.model_input_names)

    @profile
    def process_table(
        self,
        table: TensorTable | None,
        *,
        batch_size: list[int],
    ) -> TensorDict:
        source: dict[str, Tensor] = {}

        if table is not None:
            for name, column in table.items():
                source[name] = self._materialize_column(name, column)

        tensors = TensorDict(source, batch_size=batch_size)

        for encoder in self.encoders:
            for field_name in encoder.fields:
                if field_name in tensors:
                    tensors[field_name] = encoder.encode(tensors[field_name])

        if self.transforms is not None:
            tensors = self.transforms(tensors)

            for name in tuple(tensors.keys()):
                is_model_input = name in self.schema.model_input_names
                is_model_target = name in self.schema.model_target_names

                if not is_model_input and not is_model_target:
                    del tensors[name]

        for field in self.schema.fields:
            if field.is_model_input and field.name not in tensors:
                tensors[field.name] = self._materialize_missing_column(
                    field.name,
                    batch_size=batch_size,
                )
                
        tensors = self._apply_schema(tensors)
        expected = self.available_features_for(table.keys() if table is not None else ())
        self._validate_tensors(tensors, expected)

        return TensorDict(
            {
                "inputs": tensors.select(*self.schema.model_input_names),
                "targets": tensors.select(*self.schema.model_target_names),
            },
            batch_size=batch_size,
        )

    def _materialize_column(self, name: str, column: TensorColumn) -> Tensor:
        field = self.schema.get_field(name)

        values = column.values
        if column.validity is None:
            return values

        if not field.nullable:
            raise ValueError(f"Non-nullable feature '{name}' contains null values")

        return torch.where(
            column.validity,
            values,
            torch.as_tensor(field.fill_value, dtype=values.dtype, device=values.device),
        )

    def _apply_schema(self, tensors: TensorDict) -> TensorDict:
        fields = self.schema.fields_by_name

        for name in tensors.keys():
            if not isinstance(name, str):
                continue

            field = fields[name]
            tensors[name] = tensors[name].to(dtype=field.torch_dtype)

        return tensors

    def _materialize_missing_column(
        self,
        name: str,
        *,
        batch_size: list[int],
    ) -> Tensor:
        field = self.schema.get_field(name)

        return torch.zeros(batch_size, dtype=field.torch_dtype)

    def _validate_tensors(self, tensors: TensorDict, expected: set[str]) -> None:
        actual = set(tensors.keys())

        missing = expected - actual
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")

        unexpected = actual - set(self.schema.feature_names)
        if unexpected:
            raise ValueError(f"Unexpected features: {sorted(unexpected)}")

        missing_targets = set(self.schema.model_target_names) - actual
        if missing_targets:
            raise ValueError(f"Missing targets: {sorted(missing_targets)}")

        fields = self.schema.fields_by_name

        for name in actual:
            if not isinstance(name, str):
                raise ValueError(
                    f"Expected feature name to be a string, got {type(name)}. \n"
                    f"Note: TensorDict supports tuple[str, ...] keys for nested tensors"
                    f" but this is not supported in TensorSchema."
                )

            field = fields[name]
            tensor = tensors[name]
            expected_dtype = field.torch_dtype

            if tensor.dtype != expected_dtype:
                raise TypeError(
                    f"Feature '{name}' has dtype {tensor.dtype}, "
                    f"expected {expected_dtype}"
                )