from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from data.models.schema import Schema, Snapshot


class Reader[T = Snapshot](ABC):
    _closed: bool = False

    @property
    def schema(self) -> Schema: ...

    @abstractmethod
    def read(self) -> T: ...

    def close(self) -> None:
        self._closed = True

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()