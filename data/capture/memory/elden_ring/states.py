from __future__ import annotations

from pydantic import BaseModel


class InventoryEntryField(BaseModel):
    offset: str
    type: str


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
    type: str
    max_length: int | None = None


class InventoryField(BaseModel):
    structure: str
    type: str
    item_type_base: str
    item_id_min: int
    item_id_max: int