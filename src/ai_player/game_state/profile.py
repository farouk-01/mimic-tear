from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_player.game_state.discovery import BytePattern, MemoryLocator
from ai_player.game_state.schema import GAME_STATE_VALUE_KINDS, GAME_STATE_VALUE_TYPES
from ai_player.game_state.stubs import GAME_STATE_STUB_FIELDS


SUPPORTED_MEMORY_TYPES = {
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
    "utf8",
    "utf16",
}

MEMORY_TYPES_BY_VALUE_KIND = {
    "bool": {"bool", "uint8"},
    "int": {
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    },
    "float": {"float32", "float64"},
    "string": {"utf8", "utf16"},
}


def parse_offset(
    value: Any,
    *,
    field_name: str,
    allow_negative: bool = False,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer or hexadecimal string")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value, 0)
        except ValueError as error:
            raise ValueError(
                f"{field_name} must be an integer or hexadecimal string"
            ) from error
    else:
        raise ValueError(f"{field_name} must be an integer or hexadecimal string")
    if parsed < 0 and not allow_negative:
        raise ValueError(f"{field_name} cannot be negative")
    return parsed


@dataclass(frozen=True, slots=True)
class MemoryField:
    base_offset: int | None
    pointer_offsets: tuple[int, ...]
    value_type: str
    length: int | None = None
    required: bool = False
    base_locator: str | None = None
    scope: str = "dynamic"


@dataclass(frozen=True, slots=True)
class EldenRingMemoryProfile:
    name: str
    game_version: str
    process_name: str
    module_name: str
    pointer_size: int
    fields: dict[str, MemoryField]
    source_path: Path
    locators: dict[str, MemoryLocator] = field(default_factory=dict)
    sha256: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "profile_name": self.name,
            "game_version": self.game_version,
            "process_name": self.process_name,
            "module_name": self.module_name,
            "pointer_size": self.pointer_size,
            "enabled_fields": sorted(self.fields),
            "field_scopes": {
                name: field.scope for name, field in sorted(self.fields.items())
            },
            "memory_locators": {
                name: locator.kind for name, locator in sorted(self.locators.items())
            },
            "unresolved_stubs": {
                name: {"scope": stub.scope, "reason": stub.reason}
                for name, stub in sorted(GAME_STATE_STUB_FIELDS.items())
            },
            "profile_path": str(self.source_path),
            "profile_sha256": self.sha256,
        }


def load_memory_profile(path: str | Path) -> EldenRingMemoryProfile:
    profile_path = Path(path).expanduser().resolve()
    try:
        profile_bytes = profile_path.read_bytes()
        raw = json.loads(profile_bytes.decode("utf-8"))
    except OSError as error:
        raise ValueError(f"Could not read memory profile {profile_path}: {error}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid JSON in memory profile {profile_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("Memory profile must contain a JSON object")

    name = _required_string(raw, "name")
    game_version = _required_string(raw, "game_version")
    process_name = _required_string(raw, "process_name")
    module_name = _required_string(raw, "module_name")
    pointer_size = raw.get("pointer_size", 8)
    if pointer_size not in (4, 8):
        raise ValueError("Memory profile pointer_size must be 4 or 8")

    locators = _load_locators(raw.get("locators", {}))

    raw_fields = raw.get("fields")
    if not isinstance(raw_fields, dict):
        raise ValueError("Memory profile fields must be an object")
    known_fields = set(GAME_STATE_VALUE_TYPES) | set(GAME_STATE_STUB_FIELDS)
    unexpected = sorted(set(raw_fields).difference(known_fields))
    if unexpected:
        raise ValueError(
            "Memory profile contains unknown state fields: " + ", ".join(unexpected)
        )

    fields: dict[str, MemoryField] = {}
    for logical_name, raw_field in raw_fields.items():
        if not isinstance(raw_field, dict):
            raise ValueError(f"Field {logical_name} must be an object")
        if not raw_field.get("enabled", True):
            continue
        if logical_name in GAME_STATE_STUB_FIELDS:
            stub = GAME_STATE_STUB_FIELDS[logical_name]
            raise ValueError(
                f"Field {logical_name} is an unresolved {stub.scope} stub and "
                f"cannot be enabled: {stub.reason}"
            )
        value_type = _required_string(raw_field, "type")
        if value_type not in SUPPORTED_MEMORY_TYPES:
            raise ValueError(
                f"Field {logical_name} has unsupported type {value_type!r}"
            )
        value_kind = GAME_STATE_VALUE_KINDS[logical_name]
        if value_type not in MEMORY_TYPES_BY_VALUE_KIND[value_kind]:
            raise ValueError(
                f"Field {logical_name} cannot use memory type {value_type!r}; "
                f"expected a {value_kind} type"
            )
        base_locator = raw_field.get("base_locator")
        has_base_offset = "base_offset" in raw_field
        if base_locator is not None:
            if not isinstance(base_locator, str) or not base_locator.strip():
                raise ValueError(
                    f"fields.{logical_name}.base_locator must be a non-empty string"
                )
            base_locator = base_locator.strip()
            if has_base_offset:
                raise ValueError(
                    f"Field {logical_name} cannot declare both base_locator and base_offset"
                )
            if base_locator not in locators:
                raise ValueError(
                    f"Field {logical_name} uses unknown locator {base_locator!r}"
                )
            base_offset = None
        else:
            base_offset = parse_offset(
                raw_field.get("base_offset"),
                field_name=f"fields.{logical_name}.base_offset",
            )
        raw_offsets = raw_field.get("pointer_offsets", [])
        if not isinstance(raw_offsets, list):
            raise ValueError(
                f"fields.{logical_name}.pointer_offsets must be an array"
            )
        pointer_offsets = tuple(
            parse_offset(
                offset,
                field_name=f"fields.{logical_name}.pointer_offsets",
                allow_negative=True,
            )
            for offset in raw_offsets
        )
        length = raw_field.get("length")
        if value_type in ("utf8", "utf16"):
            if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
                raise ValueError(
                    f"String field {logical_name} requires a positive length"
                )
        elif length is not None:
            raise ValueError(f"Non-string field {logical_name} cannot declare length")
        fields[logical_name] = MemoryField(
            base_offset=base_offset,
            pointer_offsets=pointer_offsets,
            value_type=value_type,
            length=length,
            required=bool(raw_field.get("required", False)),
            base_locator=base_locator,
            scope=_field_scope(raw_field, logical_name),
        )

    if not fields:
        raise ValueError(
            "Memory profile has no enabled fields; fill verified offsets before use"
        )
    return EldenRingMemoryProfile(
        name=name,
        game_version=game_version,
        process_name=process_name,
        module_name=module_name,
        pointer_size=pointer_size,
        fields=fields,
        source_path=profile_path,
        locators=locators,
        sha256=hashlib.sha256(profile_bytes).hexdigest(),
    )


def _load_locators(raw_locators: Any) -> dict[str, MemoryLocator]:
    if not isinstance(raw_locators, dict):
        raise ValueError("Memory profile locators must be an object")
    locators: dict[str, MemoryLocator] = {}
    for name, raw_locator in raw_locators.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Memory locator names must be non-empty strings")
        if not isinstance(raw_locator, dict):
            raise ValueError(f"Memory locator {name!r} must be an object")
        kind = _required_string(raw_locator, "kind")
        if kind not in ("aob", "elden_ring_world_chr_man"):
            raise ValueError(f"Memory locator {name!r} has unsupported kind {kind!r}")
        section = raw_locator.get("section", ".text")
        if not isinstance(section, str) or not section.strip():
            raise ValueError(f"Memory locator {name!r} section must be a string")
        pattern = raw_locator.get("pattern")
        if kind == "aob":
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(f"AOB locator {name!r} requires a pattern")
            BytePattern.parse(pattern)
        elif pattern is not None:
            raise ValueError(
                f"Built-in locator {name!r} cannot override its discovery pattern"
            )

        occurrence = parse_offset(
            raw_locator.get("occurrence", 0),
            field_name=f"locators.{name}.occurrence",
        )
        dereference_count = parse_offset(
            raw_locator.get("dereference_count", 0),
            field_name=f"locators.{name}.dereference_count",
        )
        rip_offset_raw = raw_locator.get("rip_displacement_offset")
        instruction_size_raw = raw_locator.get("instruction_size")
        if (rip_offset_raw is None) != (instruction_size_raw is None):
            raise ValueError(
                f"Locator {name!r} must declare both rip_displacement_offset "
                "and instruction_size"
            )
        rip_offset = (
            None
            if rip_offset_raw is None
            else parse_offset(
                rip_offset_raw,
                field_name=f"locators.{name}.rip_displacement_offset",
            )
        )
        instruction_size = (
            None
            if instruction_size_raw is None
            else parse_offset(
                instruction_size_raw,
                field_name=f"locators.{name}.instruction_size",
            )
        )
        locators[name] = MemoryLocator(
            kind=kind,
            section=section.strip(),
            pattern=pattern.strip() if isinstance(pattern, str) else None,
            occurrence=occurrence,
            match_offset=parse_offset(
                raw_locator.get("match_offset", 0),
                field_name=f"locators.{name}.match_offset",
                allow_negative=True,
            ),
            rip_displacement_offset=rip_offset,
            instruction_size=instruction_size,
            addend=parse_offset(
                raw_locator.get("addend", 0),
                field_name=f"locators.{name}.addend",
                allow_negative=True,
            ),
            dereference_count=dereference_count,
        )
    return locators


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Memory profile {key} must be a non-empty string")
    return value.strip()


def _field_scope(raw_field: dict[str, Any], logical_name: str) -> str:
    scope = raw_field.get("scope", "dynamic")
    if scope not in ("dynamic", "static"):
        raise ValueError(
            f"fields.{logical_name}.scope must be 'dynamic' or 'static'"
        )
    return scope
