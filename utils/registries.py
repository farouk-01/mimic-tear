from __future__ import annotations

from typing import Generic, TypeVar

# T = TypeVar("T")


# def subclass_registry_by_name(
#     cls: type[T],
#     name_property: str = "name",
# ) -> dict[str, type[T]]:
#     return {
#         getattr(subclass, name_property): subclass for subclass in cls.__subclasses__()
#     }


KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


class Registry(Generic[KeyT, ValueT]):
    def __init__(self) -> None:
        self._items: dict[KeyT, ValueT] = {}

    def register(self, key: KeyT):
        def decorator(value: ValueT) -> ValueT:
            if key in self._items:
                raise KeyError(f"Key already registered: {key!r}")

            self._items[key] = value
            return value

        return decorator

    def resolve(self, key: KeyT) -> ValueT:
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(f"No value registered for key: {key!r}") from None

    def has(self, key: KeyT) -> bool:
        return key in self._items
