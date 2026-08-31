from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor

from utils.registries import Registry


@dataclass(frozen=True, slots=True)
class SampleColumns:
    frame_index: str
    timestamp_ns: str


class Store[Row, Column](ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(self, index: int) -> Row: ...

    @abstractmethod
    def get_range(self, start: int, end: int) -> Row: ...

    @abstractmethod
    def get_column(self, name: str) -> Column: ...

    @property
    @abstractmethod
    def frame_indices(self) -> Sequence[int]: ...

    @property
    @abstractmethod
    def timestamps_ns(self) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class TensorColumn:
    values: Tensor
    validity: Tensor | None = None


type TensorTable = dict[str, TensorColumn]


class StoreAdapter[Row, Column](ABC):
    @abstractmethod
    def get(
        self,
        data: Row,
    ) -> TensorTable: ...

    @abstractmethod
    def get_column(
        self,
        data: Column,
    ) -> TensorColumn: ...


STORE_ADAPTERS = Registry[type[Store], type[StoreAdapter]]()
