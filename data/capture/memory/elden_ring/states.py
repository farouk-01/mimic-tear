from __future__ import annotations

from pydantic import BaseModel
from typing import Literal

from ..game_state import MemoryGameStateType

type InventoryEntryType = Literal[
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
]

ENTRY_FORMATS: dict[InventoryEntryType, str] = {
    "int8": "<b",
    "uint8": "<B",
    "int16": "<h",
    "uint16": "<H",
    "int32": "<i",
    "uint32": "<I",
    "int64": "<q",
    "uint64": "<Q",
}

class InventoryEntryField(BaseModel):
    offset: str
    type: InventoryEntryType


class InventoryStructure(BaseModel):
    locator: str
    player_data_offset: str
    inventory_data_offset: str
    list_offset: str
    count_offset: str
    max_index: int
    entry_size: str
    entry_fields: dict[str, InventoryEntryField]


class PointerField(BaseModel):
    locator: str
    offsets: list[str]
    type: MemoryGameStateType
    max_length: int | None = None


class InventoryField(BaseModel):
    structure: str
    type: MemoryGameStateType
    item_type_base: str
    item_id_min: int
    item_id_max: int