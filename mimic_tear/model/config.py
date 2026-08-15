from dataclasses import dataclass

from pathlib import Path
from dataclasses import dataclass, fields
from typing import Any

# TODO rename to ComponentConfig
@dataclass(frozen=True, slots=True)
class ComponentConfig:

    def unpack(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}