from dataclasses import dataclass

GameStateValue = int | float | bool

@dataclass(frozen=True, slots=True)
class GameStateField:
    name: str
    type: str
    required: bool = False
    scope: str | None = None


@dataclass(frozen=True, slots=True)
class GameStateSchema:
    fields: tuple[GameStateField, ...]

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(
            field.name
            for field in self.fields
        )

    @property
    def feature_count(self) -> int:
        return len(self.fields)

    def has_feature(self, name: str) -> bool:
        return any(
            field.name == name
            for field in self.fields
        )