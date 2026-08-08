from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ai_player.game_state.discovery import (
    AddressDiscoveryError,
    resolve_memory_locator,
)
from ai_player.platform.windows.process_memory import MemoryReadError, ProcessMemory
from ai_player.game_state.profile import EldenRingMemoryProfile, MemoryField
from ai_player.game_state.schema import GAME_STATE_VALUE_KINDS, GAME_STATE_VALUE_TYPES


class MemoryAccessor(Protocol):
    def resolve(self, base_offset: int, pointer_offsets: tuple[int, ...]) -> int: ...

    def read_typed(
        self,
        address: int,
        value_type: str,
        *,
        length: int | None,
    ) -> object: ...

    def resolve_address(
        self,
        base_address: int,
        pointer_offsets: tuple[int, ...],
    ) -> int: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class GameStateSnapshot:
    values: dict[str, object | None]
    valid: bool
    read_errors: tuple[str, ...]


class EldenRingStateReader:
    """Read a configured Elden Ring state snapshot without writing memory."""

    def __init__(
        self,
        profile: EldenRingMemoryProfile,
        memory: MemoryAccessor,
    ) -> None:
        self.profile = profile
        self._memory = memory
        self._locator_cache: dict[str, int | Exception] = {}
        self._static_values: dict[str, object] = {}

    @classmethod
    def open(cls, profile: EldenRingMemoryProfile) -> "EldenRingStateReader":
        memory = ProcessMemory.open(
            profile.process_name,
            module_name=profile.module_name,
            pointer_size=profile.pointer_size,
            anti_cheat_guard=True,
        )
        return cls(profile, memory)

    def read(self) -> GameStateSnapshot:
        values: dict[str, object | None] = {
            name: None for name in GAME_STATE_VALUE_TYPES
        }
        errors: list[str] = []
        required_failed = False
        successful_reads = 0
        for logical_name, field in self.profile.fields.items():
            if field.scope == "static" and logical_name in self._static_values:
                values[logical_name] = self._static_values[logical_name]
                successful_reads += 1
                continue
            try:
                address = self._resolve_field(field)
                raw_value = self._memory.read_typed(
                    address,
                    field.value_type,
                    length=field.length,
                )
                value_kind = GAME_STATE_VALUE_KINDS[logical_name]
                converted = {
                    "bool": bool,
                    "int": int,
                    "float": float,
                    "string": str,
                }[value_kind](raw_value)
                values[logical_name] = converted
                if field.scope == "static":
                    self._static_values[logical_name] = converted
                successful_reads += 1
            except (AddressDiscoveryError, MemoryReadError, OSError, ValueError) as error:
                errors.append(f"{logical_name}: {error}")
                required_failed = required_failed or field.required

        return GameStateSnapshot(
            values=values,
            valid=successful_reads > 0 and not required_failed,
            read_errors=tuple(errors),
        )

    def _resolve_field(self, field: MemoryField) -> int:
        if field.base_locator is None:
            if field.base_offset is None:
                raise ValueError("Memory field has no base address")
            return self._memory.resolve(field.base_offset, field.pointer_offsets)

        cached = self._locator_cache.get(field.base_locator)
        if isinstance(cached, Exception):
            raise cached
        if cached is None:
            locator = self.profile.locators[field.base_locator]
            try:
                discovered = resolve_memory_locator(
                    self._memory,  # type: ignore[arg-type]
                    field.base_locator,
                    locator,
                )
            except Exception as error:
                self._locator_cache[field.base_locator] = error
                raise
            cached = discovered.address
            self._locator_cache[field.base_locator] = cached
        return self._memory.resolve_address(cached, field.pointer_offsets)

    def close(self) -> None:
        self._memory.close()

    def __enter__(self) -> "EldenRingStateReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
