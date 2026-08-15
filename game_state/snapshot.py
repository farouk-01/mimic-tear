from __future__ import annotations

from dataclasses import dataclass

from .schema import GameStateSchema, GameStateValue


@dataclass(frozen=True, slots=True)
class GameStateSnapshot:
    values: dict[str, GameStateValue]

    def get(
        self,
        name: str,
    ) -> GameStateValue:
        return self.values[name]

    def ordered_values(
        self,
        schema: GameStateSchema,
    ) -> tuple[GameStateValue, ...]:
        missing = [
            feature
            for feature in schema.features
            if feature not in self.values
        ]

        if missing:
            raise ValueError(
                f"Snapshot is missing game-state features: {missing}"
            )

        return tuple(
            self.values[feature]
            for feature in schema.features
        )