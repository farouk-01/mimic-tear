from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, config, field_validator, model_validator

from game_state.schema import GameStateField, GameStateSchema


class EldenRingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    game_version: str
    process_name: str
    module_name: str
    pointer_size: int
    locators: tuple[LocatorConfig, ...]
    fields: tuple[FieldConfig, ...]

    @property
    def schema_(self) -> GameStateSchema:
        return GameStateSchema(
            fields=tuple(
                GameStateField(
                    name=field.name,
                    type=field.type,  # type: ignore[arg-type] pydantic validates this for enabled fields
                    required=field.required,
                    scope=field.scope,
                )
                for field in self.fields
                if field.enabled
            )
        )

    class LocatorConfig(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

        name: str
        kind: str

    @field_validator("locators", mode="before")
    @classmethod
    def load_locators(
        cls, locators: dict[str, dict[str, Any]]
    ) -> tuple[LocatorConfig, ...]:
        return tuple(
            cls.LocatorConfig.model_validate({**config, "name": name})
            for name, config in locators.items()
        )

    class FieldConfig(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

        name: str
        enabled: bool = True
        type: str | None = None
        base_locator: str | None = None
        pointer_offsets: tuple[str, ...] = ()
        required: bool = False
        scope: str | None = None

        @field_validator("pointer_offsets", mode="before")
        @classmethod
        def _convert_pointer_offsets(
            cls,
            value: Any,
        ) -> Any:
            if isinstance(value, list):
                return tuple(value)

            return value

        @model_validator(mode="after")
        def _validate_enabled_fields(self) -> Self:
            if not self.enabled:
                return self

            missing: list[str] = []

            if self.type is None:
                missing.append("type")

            if self.base_locator is None:
                missing.append("base_locator")

            if "pointer_offsets" not in self.model_fields_set:
                missing.append("pointer_offsets")

            if missing:
                raise ValueError(
                    f"Enabled field '{self.name}' is missing: {', '.join(missing)}"
                )

            return self

    @field_validator("fields", mode="before")
    @classmethod
    def load_fields(cls, fields: dict[str, dict[str, Any]]) -> tuple[FieldConfig, ...]:
        return tuple(
            cls.FieldConfig.model_validate({**config, "name": name})
            for name, config in fields.items()
        )