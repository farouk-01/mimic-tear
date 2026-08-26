from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from capture.memory.windows import MemoryReadError, ProcessMemory


class ModulePointerLocator(BaseModel):
    type: Literal["module_pointer"]
    offset: str

    def resolve(self, memory: ProcessMemory) -> int:
        pointer_address = memory.module_base + int(self.offset, 0)
        return memory.read_pointer(pointer_address)


class FD4SingletonLocator(BaseModel):
    type: Literal["fd4_singleton"]
    class_name: str
    offset: str

    def resolve(self, memory: ProcessMemory) -> int:
        pointer_address = memory.module_base + int(self.offset, 0)
        return memory.read_pointer(pointer_address)


class CharacterHandleLocator(BaseModel):
    type: Literal["character_handle"]
    locator: str
    player_offset: str
    target_handle_offset: str
    character_set_offset: str
    entries_offset: str
    entry_size: str
    instance_handle_offset: str
    object_id_mask: str
    set_index_shift: int
    set_index_mask: str
    entry_index_mask: str
    invalid_handle: str

    def resolve(
        self,
        memory: ProcessMemory,
        base_address: int,
    ) -> int | None:
        player = memory.read_pointer(base_address + int(self.player_offset, 0))
        handle_address = player + int(self.target_handle_offset, 0)
        handle = int.from_bytes(memory.read(handle_address, 8), "little")
        if handle in {0, int(self.invalid_handle, 0)}:
            return None

        object_id = handle & int(self.object_id_mask, 0)
        set_index = (
            object_id >> self.set_index_shift
        ) & int(self.set_index_mask, 0)
        entry_index = object_id & int(self.entry_index_mask, 0)
        character_set = memory.read_pointer(
            base_address
            + int(self.character_set_offset, 0)
            + set_index * memory.pointer_size
        )
        entries = memory.read_pointer(
            character_set + int(self.entries_offset, 0)
        )
        instance = memory.read_pointer(
            entries + entry_index * int(self.entry_size, 0)
        )
        instance_handle = int.from_bytes(
            memory.read(instance + int(self.instance_handle_offset, 0), 8),
            "little",
        )
        if instance_handle != handle:
            raise MemoryReadError(
                f"Resolved character handle {instance_handle:#x} does not "
                f"match selected handle {handle:#x}"
            )
        return instance
