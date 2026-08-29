from collections.abc import Callable, Sequence
from pathlib import Path
from typing import overload

from pydantic import BaseModel, ConfigDict
from torch import Tensor
import torch

class GameStateEncoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    encoding: str
    fields: tuple[str, ...]

class GameStateEncoder:
    def __init__(
        self,
        fields: tuple[str, ...],
        *,
        get_encodings: Callable[[], dict[int, int]],
        append_encoding: Callable[[int, int], None],
        allow_new: bool = True,
    ) -> None:
        self.fields = fields
        self.encodings_data = get_encodings()
        self.append_encoding = append_encoding
        self.allow_new = allow_new

    @overload
    def encode(self, values: int) -> int: ...

    @overload
    def encode(self, values: Sequence[int]) -> list[int]: ...

    def encode(self, values: int | Sequence[int]) -> int | list[int]:
        data = self.encodings_data

        is_scalar = isinstance(values, int)
        if is_scalar:
            values = (values,)

        index = max(data.values(), default=0) + 1
        for value in values:
            if value not in data:
                if not self.allow_new:
                    return 0
                
                self.append_encoding(value, index)
                data[value] = index
                index += 1

        if is_scalar:
            return data[values[0]]

        return [data[value] for value in values]


class TensorGameStateEncoder:
    def __init__(self, encoder: GameStateEncoder) -> None:
        self.encoder = encoder

    # TODO : Tensor -> list -> Tensor (not ideal)
    def encode(self, values: Tensor) -> Tensor:
        encoded = self.encoder.encode(values.tolist())

        return torch.tensor(
            encoded,
            dtype=values.dtype,
            device=values.device,
        )

    @property
    def fields(self) -> tuple[str, ...]:
        return self.encoder.fields