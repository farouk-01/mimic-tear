from __future__ import annotations

import struct
from types import TracebackType
from typing import Self

from capture.memory.windows import MemoryReadError, ProcessMemory
from game_state.reader import GameStateReader
from game_state.schema import GameStateField, GameStateSchema, GameStateValue
from game_state.snapshot import GameStateSnapshot

from .locator import (
    CharacterHandleLocator,
    FD4SingletonLocator,
    ModulePointerLocator,
)
from .profile import EldenRingMemoryProfile
from .states import InventoryEntryField, InventoryField, PointerField


class EldenRingReader(GameStateReader):
    _ENTRY_FORMATS = {
        "int8": "<b",
        "uint8": "<B",
        "int16": "<h",
        "uint16": "<H",
        "int32": "<i",
        "uint32": "<I",
        "int64": "<q",
        "uint64": "<Q",
    }

    def __init__(
        self,
        profile: EldenRingMemoryProfile,
        memory: ProcessMemory,
    ) -> None:
        self.profile = profile
        self._memory = memory
        self._closed = False
        self._schema = GameStateSchema(
            fields=tuple(
                GameStateField(name=name, type=field.type)
                for name, field in profile.fields.items()
            )
        )
        self._static_locator_addresses = self._resolve_static_locators()

    @classmethod
    def open(
        cls,
        profile: EldenRingMemoryProfile,
    ) -> Self:
        memory = ProcessMemory.open(
            profile.process_name,
            module_name=profile.module_name,
            pointer_size=profile.pointer_size,
            anti_cheat_guard=True,
        )
        try:
            return cls(profile, memory)
        except BaseException:
            memory.close()
            raise

    @property
    def schema(self) -> GameStateSchema:
        return self._schema

    def read(self) -> GameStateSnapshot:
        if self._closed:
            raise RuntimeError("Elden Ring reader is closed")

        values: dict[str, GameStateValue] = {}
        dynamic_locators: dict[str, int | None] = {}
        inventories: dict[str, dict[int, int] | None] = {}

        for name, field in self.profile.fields.items():
            if isinstance(field, PointerField):
                base_address = self._locator_address(
                    field.locator,
                    dynamic_locators,
                )
                values[name] = (
                    None
                    if base_address is None
                    else self._read_pointer_field(base_address, field)
                )
            else:
                values[name] = self._read_inventory_field(
                    field,
                    dynamic_locators,
                    inventories,
                )

        return GameStateSnapshot(values=values)

    def close(self) -> None:
        if self._closed:
            return
        self._memory.close()
        self._closed = True

    def _resolve_static_locators(self) -> dict[str, int]:
        addresses: dict[str, int] = {}
        for name, locator in self.profile.locators.items():
            if isinstance(locator, CharacterHandleLocator):
                continue
            if isinstance(locator, (ModulePointerLocator, FD4SingletonLocator)):
                addresses[name] = locator.resolve(self._memory)
        return addresses

    def _locator_address(
        self,
        name: str,
        dynamic_locators: dict[str, int | None],
    ) -> int | None:
        try:
            locator = self.profile.locators[name]
        except KeyError as error:
            raise ValueError(f"Unknown locator: {name!r}") from error

        if isinstance(locator, CharacterHandleLocator):
            if name not in dynamic_locators:
                base_address = self._locator_address(
                    locator.locator,
                    dynamic_locators,
                )
                dynamic_locators[name] = (
                    None
                    if base_address is None
                    else locator.resolve(self._memory, base_address)
                )
            return dynamic_locators[name]

        try:
            return self._static_locator_addresses[name]
        except KeyError as error:
            raise ValueError(f"Static locator was not resolved: {name!r}") from error

    def _read_pointer_field(
        self,
        base_address: int,
        field: PointerField,
    ) -> GameStateValue:
        offsets = tuple(int(offset, 0) for offset in field.offsets)
        address = base_address
        if offsets:
            for offset in offsets[:-1]:
                address = self._memory.read_pointer(address + offset)
            address += offsets[-1]

        memory_type = {
            "utf8_string": "utf8",
            "utf16le_string": "utf16",
        }.get(field.type, field.type)
        value = self._memory.read_typed(
            address,
            memory_type,
            length=field.max_length,
        )
        if not isinstance(value, (bool, int, float, str)):
            raise TypeError(f"Unsupported game-state value: {value!r}")
        return value

    def _read_inventory_field(
        self,
        field: InventoryField,
        dynamic_locators: dict[str, int | None],
        inventories: dict[str, dict[int, int] | None],
    ) -> int | None:
        if field.structure not in inventories:
            inventories[field.structure] = self._read_inventory(
                field.structure,
                dynamic_locators,
            )
        items = inventories[field.structure]
        if items is None:
            return None

        item_type_base = int(field.item_type_base, 0)
        matches = [
            quantity
            for item_id, quantity in items.items()
            if field.item_id_min <= item_id - item_type_base <= field.item_id_max
        ]
        if not matches:
            return 0
        if len(matches) > 1:
            raise MemoryReadError(
                f"Inventory field matched more than one item in "
                f"[{field.item_id_min}, {field.item_id_max}]"
            )
        return matches[0]

    def _read_inventory(
        self,
        structure_name: str,
        dynamic_locators: dict[str, int | None],
    ) -> dict[int, int] | None:
        try:
            definition = self.profile.structures[structure_name]
        except KeyError as error:
            raise ValueError(
                f"Unknown inventory structure: {structure_name!r}"
            ) from error

        base_address = self._locator_address(
            definition.locator,
            dynamic_locators,
        )
        if base_address is None:
            return None

        player_data = self._memory.read_pointer(
            base_address + int(definition.player_data_offset, 0)
        )
        inventory_data = self._memory.read_pointer(
            player_data + int(definition.inventory_data_offset, 0)
        )
        list_address = self._memory.read_pointer(
            inventory_data + int(definition.list_offset, 0)
        )
        item_count = self._memory.read_int32(
            inventory_data + int(definition.count_offset, 0)
        )
        if not 0 <= item_count <= definition.max_index + 1:
            raise MemoryReadError(f"Invalid inventory item count: {item_count}")
        if item_count == 0:
            return {}

        entry_size = int(definition.entry_size, 0)
        raw = self._memory.read(
            list_address,
            (definition.max_index + 1) * entry_size,
        )
        try:
            handle_field = definition.entry_fields["item_handle"]
            item_id_field = definition.entry_fields["item_id"]
            quantity_field = definition.entry_fields["quantity"]
        except KeyError as error:
            raise ValueError(
                f"Inventory structure {structure_name!r} is missing {error}"
            ) from error

        items: dict[int, int] = {}
        populated = 0
        for index in range(definition.max_index + 1):
            entry_offset = index * entry_size
            handle = self._read_entry(raw, entry_offset, handle_field)
            item_id = self._read_entry(raw, entry_offset, item_id_field)
            if handle == 0 or item_id == 0xFFFFFFFF:
                continue

            populated += 1
            items[item_id] = self._read_entry(raw, entry_offset, quantity_field)
            if populated >= item_count:
                break

        return items

    def _read_entry(
        self,
        raw: bytes,
        entry_offset: int,
        field: InventoryEntryField,
    ) -> int:
        try:
            format_string = self._ENTRY_FORMATS[field.type]
        except KeyError as error:
            raise ValueError(
                f"Unsupported inventory entry type: {field.type!r}"
            ) from error
        return int(
            struct.unpack_from(format_string, raw, entry_offset + int(field.offset, 0))[
                0
            ]
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
