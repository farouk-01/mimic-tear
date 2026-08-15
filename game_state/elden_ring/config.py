from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from game_state.schema import GameStateField, GameStateSchema


class GameStateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    game_version: str
    process_name: str
    module_name: str
    pointer_size: int
    locators: dict[str, dict[str, Any]]
    fields: dict[str, dict[str, Any]]


class LocatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    kind: str

    @classmethod
    def load(cls, name: str, config: dict[str, Any]) -> LocatorConfig:
        values = dict(config)
        values["name"] = name
        return cls.model_validate(values)


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
    def _validate_enabled_fields(self) -> FieldConfig:
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

    @classmethod
    def load(cls, name: str, config: dict[str, Any]) -> FieldConfig:
        values = dict(config)
        values["name"] = name
        return cls.model_validate(values)


@dataclass(frozen=True, slots=True)
class EldenRingGameStateConfig:
    name: str
    game_version: str
    process_name: str
    module_name: str
    pointer_size: int
    locators: tuple[LocatorConfig, ...]
    fields: tuple[FieldConfig, ...]
    schema: GameStateSchema

    @classmethod
    def load(cls, path: str | Path) -> EldenRingGameStateConfig:
        path = Path(path)
        config = GameStateConfig.model_validate_json(path.read_text(encoding="utf-8"))

        locators = tuple(
            LocatorConfig.load(name, values) for name, values in config.locators.items()
        )

        fields = tuple(
            FieldConfig.load(name, values) for name, values in config.fields.items()
        )

        game_state_fields = ()
        for field in fields:
            if field.enabled:
                game_state_fields += (
                    GameStateField(
                        name=field.name,
                        type=field.type,  # type: ignore[arg-type] pydantic validates this for enabled fields
                        required=field.required,
                        scope=field.scope,
                    ),
                )

        schema = GameStateSchema(fields=game_state_fields)

        return cls(
            name=config.name,
            game_version=config.game_version,
            process_name=config.process_name,
            module_name=config.module_name,
            pointer_size=config.pointer_size,
            locators=locators,
            fields=fields,
            schema=schema,
        )
