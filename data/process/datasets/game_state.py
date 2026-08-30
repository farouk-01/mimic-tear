from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar, Generic

from torch import Tensor
from torch.utils.data import Dataset

from data.models.game_state.processed import (
    ProcessedGameStateSchema,
    TORCH_DTYPES,
    ProcessedGameStateField,
)
from data.process.encoders.game_state import GameStateEncoder, TensorGameStateEncoder
from graph.base import Plan
from graph.types.tensor import TensorGraphExecutor
from utils.registries import Registry

type GameStateValue = int | float | bool | str | None

RowT = TypeVar("RowT")
RangeT = TypeVar("RangeT")
ColT = TypeVar("ColT")

type GameStateTensors = dict[str, Tensor]


class GameStateStore(ABC, Generic[RowT, RangeT, ColT]):
    @property
    @abstractmethod
    def features(self) -> tuple[str, ...]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(self, index: int) -> RowT: ...

    @abstractmethod
    def get_range(self, start: int, end: int) -> RangeT: ...

    @property
    @abstractmethod
    def indices(self) -> Sequence[int]: ...

    @property
    @abstractmethod
    def timestamps_ns(self) -> Sequence[int]: ...

    @abstractmethod
    def get_feature(self, name: str) -> ColT: ...


class GameStateStoreAdapter(ABC, Generic[RowT, RangeT, ColT]):
    @abstractmethod
    def get(self, data: RowT, schema: ProcessedGameStateSchema) -> GameStateTensors: ...

    @abstractmethod
    def get_range(
        self, data: RangeT, schema: ProcessedGameStateSchema
    ) -> GameStateTensors: ...

    @abstractmethod
    def get_feature(self, data: ColT, field: ProcessedGameStateField) -> Tensor: ...


game_state_store_adapters = Registry[
    type[GameStateStore], type[GameStateStoreAdapter]
]()


class GameStateDataset(Dataset[GameStateTensors]):
    def __init__(
        self,
        *,
        store: GameStateStore,
        schema: ProcessedGameStateSchema,
        encoders: tuple[GameStateEncoder, ...] = (),
        plan: Plan,
        executor: TensorGraphExecutor,
    ) -> None:
        if len(store) <= 0:
            raise ValueError("Game-state store cannot be empty")

        if not store.features:
            raise ValueError("Game-state store must expose at least one feature")

        self.store: GameStateStore = store
        self.schema = schema
        adapter_cls = game_state_store_adapters.resolve(type(store))
        self.adapter = adapter_cls()

        expected = tuple(value.name for value in plan.inputs)

        missing = [feature for feature in expected if feature not in store.features]

        if missing:
            raise ValueError(
                f"Game-state store is missing features required by schema: {missing}"
            )

        self.features = store.features
        self.encoders: tuple[TensorGameStateEncoder, ...] = tuple(
            TensorGameStateEncoder(encoder) for encoder in encoders
        )
        self.plan = plan
        self.executor = executor

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int) -> GameStateTensors:
        state = self.store.get(index)
        tensors = self.adapter.get(state, self.schema)

        return self._process_tensors(tensors)

    def get_range(self, start: int, end: int) -> GameStateTensors:
        states = self.store.get_range(start, end)
        tensors = self.adapter.get_range(states, self.schema)

        return self._process_tensors(tensors)

    def discover_encodings(self) -> dict[str, int]:
        cardinalities: dict[str, int] = {}

        for encoder in self.encoders:
            for field_name in encoder.fields:
                field_data = self.store.get_feature(field_name)
                tensor = self.adapter.get_feature(
                    field_data,
                    self.schema.get_field(field_name),
                )
                encoder.discover(tensor)

            # a encoder can be used for multiple fields
            # so need to process all fields before
            for field_name in encoder.fields:
                cardinalities[field_name] = encoder.cardinality

        return cardinalities

    def _process_tensors(self, tensors: GameStateTensors) -> GameStateTensors:
        for encoder in self.encoders:
            for field_name in encoder.fields:
                tensors[field_name] = encoder.encode(tensors[field_name])

        tensors = self.executor.execute(self.plan, tensors)
        self._validate_transformed_tensors(tensors)

        return tensors

    def _validate_transformed_tensors(
        self,
        tensors: GameStateTensors,
    ) -> None:
        expected_names = {value.name for value in self.plan.outputs}

        errors: list[Exception] = []

        missing = [name for name in expected_names if name not in tensors]
        unexpected = [name for name in tensors if name not in expected_names]

        if missing:
            errors.append(ValueError(f"Missing model input features: {missing}"))

        if unexpected:
            errors.append(ValueError(f"Unexpected model input features: {unexpected}"))

        for name in expected_names:
            if name not in tensors:
                continue

            field = self.schema.get_field(name)
            tensor = tensors[name]

            if tensor.dtype != TORCH_DTYPES[field.dtype]:
                errors.append(
                    TypeError(
                        f"Feature '{name}' has unexpected dtype: "
                        f"expected {field.dtype}, got {tensor.dtype}"
                    )
                )

        if errors:
            raise ExceptionGroup(
                "Game-state dataset processing validation failed",
                errors,
            )
