from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch import Tensor

from utils.registries import Registry


@dataclass(frozen=True, slots=True)
class SampleColumns:
    frame_index: str
    capture_timestamp_ns: str


DEFAULT_SAMPLE_COLUMNS = SampleColumns(
    frame_index="frame_index",
    capture_timestamp_ns="frame_timestamp_ns",
)


class StoreConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Store[Row](ABC):
    source: Path

    def __init__(self, *, source: str | Path) -> None:
        self.source = Path(source)

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(self, index: int) -> Row: ...

    @abstractmethod
    def get_range(self, start: int, end: int) -> Row: ...

    @property
    @abstractmethod
    def feature_names(self) -> Collection[str]: ...

    @property
    @abstractmethod
    def frame_indices(self) -> Sequence[int]: ...

    @property
    @abstractmethod
    def capture_timestamp_ns(self) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class TensorColumn:
    values: Tensor
    validity: Tensor | None = None


type TensorTable = dict[str, TensorColumn]


class StoreAdapter[Row](ABC):
    @abstractmethod
    def get(
        self,
        data: Row,
    ) -> TensorTable: ...


FILE_STORES = Registry[str, type[Store]]()
STORE_ADAPTERS = Registry[type[Store], type[StoreAdapter]]()
