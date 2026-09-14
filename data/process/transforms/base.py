from abc import ABC, abstractmethod
from typing import ClassVar
from graphlib import TopologicalSorter
from collections.abc import Iterator, Sequence, MutableMapping, Collection
import uuid
from collections import deque

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class Transform[T](BaseModel, ABC):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: int = Field(default_factory=lambda: uuid.uuid4().int)
    name: ClassVar[str]
    output: str

    @property
    @abstractmethod
    def inputs(self) -> tuple[str, ...]: ...

    @abstractmethod
    def __call__(self, *args, **kwargs) -> T: ...


class Lookback[T]:
    def __init__(self, lookback_size: int) -> None:
        self.lookback_size = lookback_size
        self.buffer = deque(maxlen=lookback_size)

    def push(self, value: T) -> None:
        self.buffer.append(value)

    def __iter__(self) -> Iterator[T]:
        return iter(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def reset(self) -> None:
        self.buffer.clear()


class TemporalTransform[T](Transform[T]):
    periods: PositiveInt = 1

    lookback: Lookback[T] = Field(
        default_factory=lambda data: Lookback(data["periods"])
    )


class Graph[T]:
    def __init__(self, transforms: Sequence[Transform[T]]) -> None:
        self.transforms = transforms

        self._by_id = {transform.id: transform for transform in transforms}
        self._by_output = {transform.output: transform for transform in transforms}
        self._sorter = TopologicalSorter()

        latest_producer: dict[str, int] = {}

        self._lookback_cache: dict[int, Lookback[T]] = {}

        for t in transforms:
            deps: list[int] = []

            for name in t.inputs:
                _id = latest_producer.get(name)

                if _id is not None:
                    deps.append(_id)

            if isinstance(t, TemporalTransform) and t.periods > 1:
                self._lookback_cache[t.id] = Lookback(t.periods)

            self._sorter.add(t.id, *deps)
            latest_producer[t.output] = t.id

        self._order = tuple(self._sorter.static_order())

    def __call__(self, data: MutableMapping[str, T]) -> MutableMapping[str, T]:
        for name in self._order:
            transform = self._by_id[name]

            if not all(name in data for name in transform.inputs):
                continue

            data[transform.output] = transform(*[data[i] for i in transform.inputs])

        return data

    @property
    def inputs(self) -> tuple[str, ...]:
        all_inputs: set[str] = set()

        for t in self.transforms:
            for _input in t.inputs:
                if _input not in self._by_output:
                    all_inputs.add(_input)

        return tuple(all_inputs)

    @property
    def outputs(self) -> tuple[str, ...]:
        all_outputs: set[str] = set()

        for t in self.transforms:
            all_outputs.add(t.output)

        return tuple(all_outputs)

    def resolve_available(self, initial_inputs: Collection[str]) -> Collection[str]:
        available = set(initial_inputs)

        for t_id in self._order:
            transform = self._by_id[t_id]

            if all(name in available for name in transform.inputs):
                available.add(transform.output)

        return available
