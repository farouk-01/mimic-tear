from __future__ import annotations

from types import TracebackType

from capture.memory import ProcessMemory
from game_state import (
    GameStateReader,
    GameStateSchema,
    GameStateSnapshot,
    GameStateValue,
)

from .config import EldenRingConfig
from .locator import (
    MemoryLocator,
    resolve_memory_locator,
)


class EldenRingGameStateReader(GameStateReader):
    def __init__(
        self,
        *,
        config: EldenRingConfig,
        memory: ProcessMemory,
    ) -> None:
        self.config = config
        self._memory = memory
        self._closed = False

        self._locator_addresses = (
            self._resolve_locators()
        )

    @classmethod
    def open(
        cls,
        config: EldenRingConfig,
    ) -> EldenRingGameStateReader:
        memory = ProcessMemory.open(
            config.process_name,
            module_name=config.module_name,
            pointer_size=config.pointer_size,
            anti_cheat_guard=True,
        )

        try:
            return cls(
                config=config,
                memory=memory,
            )
        except BaseException:
            memory.close()
            raise

    @property
    def schema(self) -> GameStateSchema:
        return self.config.schema_

    def read(self) -> GameStateSnapshot:
        if self._closed:
            raise RuntimeError(
                "Game-state reader is closed"
            )

        values: dict[str, GameStateValue] = {}

        for field in self.config.fields:
            if not field.enabled:
                continue

            if field.base_locator is None:
                raise ValueError(
                    f"Field '{field.name}' has no base locator"
                )

            if field.type is None:
                raise ValueError(
                    f"Field '{field.name}' has no type"
                )

            try:
                base_address = self._locator_addresses[
                    field.base_locator
                ]
            except KeyError as error:
                raise ValueError(
                    f"Field '{field.name}' references unknown "
                    f"locator '{field.base_locator}'"
                ) from error

            pointer_offsets = tuple(
                int(offset, 0)
                for offset in field.pointer_offsets
            )

            address = self._memory.resolve_address(
                base_address,
                pointer_offsets,
            )

            raw_value = self._memory.read_typed(
                address,
                field.type,
                length=None,
            )

            if not isinstance(
                raw_value,
                (bool, int, float),
            ):
                raise TypeError(
                    f"Field '{field.name}' returned "
                    f"unsupported value {raw_value!r}"
                )

            values[field.name] = raw_value

        return GameStateSnapshot(values=values)

    def close(self) -> None:
        if self._closed:
            return

        self._memory.close()
        self._closed = True

    def _resolve_locators(
        self,
    ) -> dict[str, int]:
        addresses: dict[str, int] = {}

        for locator in self.config.locators:
            result = resolve_memory_locator(
                self._memory,
                locator.name,
                MemoryLocator(
                    kind=locator.kind,
                ),
            )

            addresses[locator.name] = result.address

        return addresses

    def __enter__(
        self,
    ) -> EldenRingGameStateReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()