from torch import Tensor
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict
from pydantic import BaseModel, ConfigDict

from data.models.tensor import TORCH_DTYPES, TensorSchema
from data.process.stores.base import Store, STORE_ADAPTERS, TensorColumn, TensorTable
from data.process.transforms.tensor import TensorTransform
from data.process.encoders.encoder import Encoder, TensorEncoder

from utils import profile


class TensorDatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    tensor_schema: TensorSchema
    transforms: tuple[TensorTransform, ...] = ()


class TensorDataset(Dataset[TensorDict]):
    def __init__(
        self,
        store: Store,
        *,
        tensor_schema: TensorSchema,
        encoders: tuple[Encoder, ...] = (),
        transforms: tuple[TensorTransform, ...] = (),
    ) -> None:
        if len(store) <= 0:
            raise ValueError("Store cannot be empty")

        self.store = store
        self.schema = tensor_schema
        self.transforms = transforms

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

    @property
    def available_features(self) -> set[str]:
        available = set(self.store.feature_names)

        for transform in self.transforms:
            # if all inputs are available 
            if all(name in available for name in transform.inputs):
                available.add(transform.output)

        return available

    def _process_table(
        self,
        table: TensorTable,
        *,
        batch_size: list[int],
    ) -> TensorDict:
        source: dict[str, Tensor] = {}
        for name, column in table.items():
            source[name] = self._materialize_column(name, column)

        tensors = TensorDict(source, batch_size=batch_size)

        for encoder in self.encoders:
            for field_name in encoder.fields:
                tensors[field_name] = encoder.encode(tensors[field_name])

        for transform in self.transforms:
            required_inputs = transform.inputs
            is_available = (name in self.available_features for name in required_inputs)
            
            if all(is_available):
                inputs = tuple(tensors[name] for name in transform.inputs)

                output_name = transform.output
                tensors[output_name] = transform(*inputs)

        self._validate_tensors(tensors)
        return tensors

    def _materialize_column(self, name: str, column: TensorColumn) -> Tensor:
        field = self.schema.get_field(name)
        dtype = field.torch_dtype

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
        expected = self.available_features
        actual = set(tensors.keys())

        missing = expected - actual
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")

        unexpected = actual - set(self.schema.feature_names)
        if unexpected:
            raise ValueError(f"Unexpected features: {sorted(unexpected)}")

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
