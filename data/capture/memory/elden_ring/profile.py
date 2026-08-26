from typing import Annotated, Self

from pydantic import BaseModel, Field, model_validator

from .locator import (
    CharacterHandleLocator,
    FD4SingletonLocator,
    ModulePointerLocator,
)
from .states import (
    InventoryField,
    InventoryEntryType,
    InventoryStructure,
    PointerField,
)

Locator = Annotated[
    ModulePointerLocator | FD4SingletonLocator | CharacterHandleLocator,
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

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        self._validate_field_references()
        self._validate_structure_references()
        self._validate_locator_references()
        return self

    def _validate_field_references(self) -> None:
        for name, field in self.fields.items():
            if isinstance(field, PointerField):
                if field.locator not in self.locators:
                    raise ValueError(
                        f"Pointer field {name!r} references unknown "
                        f"locator {field.locator!r}"
                    )

            elif isinstance(field, InventoryField):
                if field.structure not in self.structures:
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

    def _validate_locator_references(self) -> None:
        for name, locator in self.locators.items():
            if (
                isinstance(locator, CharacterHandleLocator)
                and locator.locator not in self.locators
            ):
                raise ValueError(
                    f"Character locator {name!r} references unknown "
                    f"locator {locator.locator!r}"
                )
