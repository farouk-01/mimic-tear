from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class Transform[T](BaseModel, ABC):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: ClassVar[str]
    output: str

    @property
    @abstractmethod
    def inputs(self) -> tuple[str, ...]: ...

    @abstractmethod
    def __call__(self, *args, **kwargs) -> T: ...