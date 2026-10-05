from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from time import perf_counter_ns
from typing import Any, Literal, Self

from pydantic import BaseModel, model_validator

from data.capture.sources.base import Reader
from data.capture.sources.memory.windows import ProcessMemory
from data.models.memory import (
    MemoryField,
    MemorySchema,
    MemoryType,
    PythonType,
    to_numpy_type,
)
from data.models.schema import Snapshot

STRING_READ_TYPES: dict[MemoryType, MemoryType] = {
    "utf8_string": "utf8",
    "utf16le_string": "utf16",
}


class StaticLocator(BaseModel, ABC):
    @abstractmethod
    def resolve(self, memory: ProcessMemory) -> int: ...


class DependentLocator(BaseModel, ABC):
    locator: str

    @abstractmethod
    def resolve(self, memory: ProcessMemory, base_address: int) -> int | None: ...


class ModulePointerLocator(StaticLocator):
    type: Literal["module_pointer"]
    offset: str

    def resolve(self, memory: ProcessMemory) -> int:
        pointer_address = memory.module_base + int(self.offset, 0)
        return memory.read_pointer(pointer_address)


type Locator = StaticLocator | DependentLocator


class MemoryFieldSpec(BaseModel):
    type: MemoryType


class PointerField(MemoryFieldSpec):
    locator: str
    offsets: list[str]
    max_length: int | None = None


class MemoryProfile[
    L: Locator = Locator,
    F: MemoryFieldSpec = MemoryFieldSpec,
](BaseModel):
    game_version: str
    process_name: str
    module_name: str
    pointer_size: int
    locators: dict[str, L]
    fields: dict[str, F]

    @model_validator(mode="after")
    def validate_memory_references(self) -> Self:
        for name, spec in self.fields.items():
            if isinstance(spec, PointerField) and spec.locator not in self.locators:
                raise ValueError(
                    f"Pointer field {name!r} references unknown "
                    f"locator {spec.locator!r}"
                )

        for name, locator in self.locators.items():
            if (
                isinstance(locator, DependentLocator)
                and locator.locator not in self.locators
            ):
                raise ValueError(
                    f"Locator {name!r} references unknown "
                    f"locator {locator.locator!r}"
                )

        return self

    @cached_property
    def raw_schema(self) -> MemorySchema:
        return MemorySchema(
            fields=tuple(
                MemoryField(name=name, dtype=to_numpy_type(spec.type))
                for name, spec in self.fields.items()
            )
        )


@dataclass(slots=True)
class ReadContext:
    locators: dict[str, int | None] = field(default_factory=dict)
    cache: dict[str, object] = field(default_factory=dict)


class GameStateReader[P: MemoryProfile[Any, Any]](Reader):
    def __init__(self, profile: P, memory: ProcessMemory) -> None:
        self.profile = profile
        self._memory = memory
        self._static_locator_addresses = self._resolve_static_locators()

    @classmethod
    def open(cls, profile: P, *, anti_cheat_guard: bool = True) -> Self:
        memory = ProcessMemory.open(
            profile.process_name,
            module_name=profile.module_name,
            pointer_size=profile.pointer_size,
            anti_cheat_guard=anti_cheat_guard,
        )
        try:
            return cls(profile, memory)
        except BaseException:
            memory.close()
            raise

    @property
    def schema(self) -> MemorySchema:
        return self.profile.raw_schema

    def read(self) -> Snapshot[MemoryField]:
        self._ensure_open()

        timestamp_ns = perf_counter_ns()
        context = ReadContext()
        values = {
            name: self._read_field(name, spec, context)
            for name, spec in self.profile.fields.items()
        }

        return Snapshot.from_dict(
            values,
            schema=self.schema,
            timestamp_ns=timestamp_ns,
        )

    def close(self) -> None:
        if self.is_closed:
            return
        self._memory.close()
        super().close()

    def _read_field(
        self,
        name: str,
        spec: MemoryFieldSpec,
        context: ReadContext,
    ) -> PythonType:
        if isinstance(spec, PointerField):
            base_address = self._locator_address(spec.locator, context)
            if base_address is None:
                return None
            return self._read_pointer_field(base_address, spec)

        raise TypeError(
            f"{type(self).__name__} cannot read field {name!r} "
            f"of type {type(spec).__name__}"
        )

    def _resolve_static_locators(self) -> dict[str, int]:
        return {
            name: locator.resolve(self._memory)
            for name, locator in self.profile.locators.items()
            if isinstance(locator, StaticLocator)
        }

    def _locator_address(self, name: str, context: ReadContext) -> int | None:
        locator = self.profile.locators[name]

        if isinstance(locator, DependentLocator):
            if name not in context.locators:
                base_address = self._locator_address(locator.locator, context)
                context.locators[name] = (
                    None
                    if base_address is None
                    else locator.resolve(self._memory, base_address)
                )
            return context.locators[name]

        try:
            return self._static_locator_addresses[name]
        except KeyError as error:
            raise RuntimeError(f"Static locator was not resolved: {name!r}") from error

    def _read_pointer_field(self, base_address: int, spec: PointerField) -> PythonType:
        offsets = tuple(int(offset, 0) for offset in spec.offsets)
        address = base_address
        if offsets:
            for offset in offsets[:-1]:
                address = self._memory.read_pointer(address + offset)
            address += offsets[-1]

        value = self._memory.read_typed(
            address,
            STRING_READ_TYPES.get(spec.type, spec.type),
            length=spec.max_length,
        )
        if not isinstance(value, (bool, int, float, str)):
            raise TypeError(f"Unsupported game-state value: {value!r}")
        return value
