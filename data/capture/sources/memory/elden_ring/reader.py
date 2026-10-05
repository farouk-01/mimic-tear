from __future__ import annotations

import struct

from data.capture.sources.memory.reader import (
    GameStateReader,
    MemoryFieldSpec,
    ReadContext,
)
from data.capture.sources.memory.windows import MemoryReadError
from data.models.memory import PythonType

from .profile import EldenRingMemoryProfile
from .states import (
    InventoryEntryField,
    InventoryField,
    ENTRY_FORMATS,
)


class EldenRingReader(GameStateReader[EldenRingMemoryProfile]):
    def _read_field(
        self,
        name: str,
        spec: MemoryFieldSpec,
        context: ReadContext,
    ) -> PythonType:
        if isinstance(spec, InventoryField):
            return self._read_inventory_field(spec, context)

        return super()._read_field(name, spec, context)

    def _read_inventory_field(
        self,
        field: InventoryField,
        context: ReadContext,
    ) -> int | None:
        cache_key = f"inventory:{field.structure}"
        if cache_key not in context.cache:
            context.cache[cache_key] = self._read_inventory(field.structure, context)
        items: dict[int, int] | None = context.cache[cache_key]  # type: ignore[assignment]
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
        context: ReadContext,
    ) -> dict[int, int] | None:
        definition = self.profile.structures[structure_name]

        base_address = self._locator_address(definition.locator, context)
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

        handle_field = definition.entry_fields["item_handle"]
        item_id_field = definition.entry_fields["item_id"]
        quantity_field = definition.entry_fields["quantity"]

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
        format_string = ENTRY_FORMATS[field.type]
        return int(
            # fmt: off
            struct.unpack_from(
                format_string, raw, entry_offset + int(field.offset, 0)
            )[0]
            # fmt: on
        )
