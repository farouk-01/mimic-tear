from collections.abc import Callable, Sequence
from typing import overload

from pydantic import BaseModel, ConfigDict, validate_call


class GameStateEncoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    load_encodings: Callable[[], dict[int, int]]
    append_encoding: Callable[[int, int], None]

    @validate_call(config=ConfigDict(strict=True), validate_return=True)
    def get_encodings(self) -> dict[int, int]:
        data = self.load_encodings()
        return data


class GameStateEncoder:
    def __init__(
        self,
        get_encodings: Callable[[], dict[int, int]],
        append_encoding: Callable[[int, int], None],
    ) -> None:
        self.encodings = get_encodings()
        self.append_encoding = append_encoding

    @overload
    def encode(self, values: int) -> int: ...

    @overload
    def encode(self, values: Sequence[int]) -> list[int]: ...

    def encode(self, values: int | Sequence[int]) -> int | list[int]:
        data = self.encodings

        is_scalar = isinstance(values, int)
        if is_scalar:
            values = (values,)

        for value in values:
            if value not in data:
                index = max(data.values(), default=0) + 1
                self.append_encoding(value, index)
                data[value] = index

        if is_scalar:
            return data[values[0]]

        return [data[value] for value in values]
