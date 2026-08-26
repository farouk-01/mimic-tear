from typing import Annotated

from pydantic import BaseModel, Field

from .locator import (
    CharacterHandleLocator,
    FD4SingletonLocator,
    ModulePointerLocator,
)
from .states import InventoryField, InventoryStructure, PointerField


Locator = Annotated[
    ModulePointerLocator
    | FD4SingletonLocator
    | CharacterHandleLocator,
    Field(discriminator="type"),
]


class EldenRingMemoryProfile(BaseModel):
    game_version: str
    steam_build_id: int
    process_name: str
    module_name: str
    pointer_size: int
    locators: dict[str, Locator]
    structures: dict[str, InventoryStructure]
    fields: dict[str, PointerField | InventoryField]
