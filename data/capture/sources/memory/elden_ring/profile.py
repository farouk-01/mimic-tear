from typing import Annotated, Self

from pydantic import Field, model_validator

from data.capture.sources.memory.reader import (
    MemoryProfile,
    ModulePointerLocator,
    PointerField,
)

from .locator import (
    CharacterHandleLocator,
    FD4SingletonLocator,
)
from .states import (
    InventoryField,
    InventoryStructure,
)

Locator = Annotated[
    ModulePointerLocator | FD4SingletonLocator | CharacterHandleLocator,
    Field(discriminator="type"),
]


class EldenRingMemoryProfile(
    MemoryProfile[Locator, PointerField | InventoryField]
):
    steam_build_id: int
    structures: dict[str, InventoryStructure]

    @model_validator(mode="after")
    def validate_inventory_references(self) -> Self:
        self._validate_field_references()
        self._validate_structure_references()
        return self

    def _validate_field_references(self) -> None:
        for name, field in self.fields.items():
            if (
                isinstance(field, InventoryField)
                and field.structure not in self.structures
            ):
                raise ValueError(
                    f"Inventory field {name!r} references unknown "
                    f"structure {field.structure!r}"
                )

    def _validate_structure_references(self) -> None:
        for name, structure in self.structures.items():
            if structure.locator not in self.locators:
                raise ValueError(
                    f"Inventory structure {name!r} references unknown "
                    f"locator {structure.locator!r}"
                )

            required = {"item_handle", "item_id", "quantity"}
            missing = required - structure.entry_fields.keys()

            if missing:
                raise ValueError(
                    f"Inventory structure {name!r} is missing "
                    f"entry fields: {sorted(missing)}"
                )
