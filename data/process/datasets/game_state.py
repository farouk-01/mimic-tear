from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TypeVar, Generic

from torch import Tensor
from torch.utils.data import Dataset

from configs.models.game_state import ProcessedGameStateSchema, TORCH_DTYPES
from utils.registries import Registry

type GameStateValue = int | float | bool | str | None

RowT = TypeVar("RowT")
RangeT = TypeVar("RangeT")

type GameStateTensors = dict[str, Tensor]


class GameStateStore(ABC, Generic[RowT, RangeT]):
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


class GameStateStoreAdapter(ABC, Generic[RowT, RangeT]):
    @abstractmethod
    def get(self, data: RowT, schema: ProcessedGameStateSchema) -> GameStateTensors: ...

    @abstractmethod
    def get_range(
        self, data: RangeT, schema: ProcessedGameStateSchema
    ) -> GameStateTensors: ...


game_state_store_adapters = Registry[
    type[GameStateStore], type[GameStateStoreAdapter]
]()


class GameStateDataset(Dataset[GameStateTensors]):
    def __init__(
        self,
        *,
        store: GameStateStore,
        schema: ProcessedGameStateSchema,
        transform: Callable[[GameStateTensors], GameStateTensors] | None = None,
    ) -> None:
        if len(store) <= 0:
            raise ValueError("Game-state store cannot be empty")

        if not store.features:
            raise ValueError("Game-state store must expose at least one feature")

        self.store = store
        self.schema = schema
        adapter_cls = game_state_store_adapters.resolve(type(store))
        self.adapter = adapter_cls()

        expected = schema.get_required_fields_names(include_derived=False)

        missing = [feature for feature in expected if feature not in store.features]

        if missing:
            raise ValueError(
                f"Game-state store is missing features required by schema: {missing}"
            )

        self.features = store.features
        self.transform = transform

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int) -> GameStateTensors:
        state = self.store.get(index)
        tensors = self.adapter.get(state, self.schema)

        if self.transform is not None:
            tensors = self.transform(tensors)
            self._validate_transformed_tensors(tensors)

        return tensors

    def get_range(self, start: int, end: int) -> GameStateTensors:
        states = self.store.get_range(start, end)
        tensors = self.adapter.get_range(states, self.schema)

        if self.transform is not None:
            tensors = self.transform(tensors)
            self._validate_transformed_tensors(tensors)

        return tensors

    def _validate_transformed_tensors(self, tensors: GameStateTensors) -> None:
        required_derived = self.schema.get_required_derived_fields()
        required_derived_names = {field.name for field in required_derived}
        non_derived_names = set(
            self.schema.get_required_fields_names(include_derived=False)
        )
        expected_names = required_derived_names | non_derived_names

        errors: list[Exception] = []

        missing = [name for name in required_derived_names if name not in tensors]
        unexpected = [name for name in tensors if name not in expected_names]

        if missing:
            errors.append(ValueError(f"Missing required derived features: {missing}"))

        if unexpected:
            errors.append(
                ValueError(f"Unexpected features in transformed tensors: {unexpected}")
            )

        for field in required_derived:
            if field.name not in tensors:
                continue

            tensor = tensors[field.name]

            if tensor.dtype != TORCH_DTYPES[field.dtype]:
                errors.append(
                    TypeError(
                        f"Feature '{field.name}' has unexpected dtype: "
                        f"expected {field.dtype}, got {tensor.dtype}"
                    )
                )

        if errors:
            raise ExceptionGroup(
                "Game-state dataset transform validation failed",
                errors,
            )
