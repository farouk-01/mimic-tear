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
        append_encoding: Callable[[Sequence[int] | int, Sequence[int] | int], None],
    ) -> None:
        self.fields = fields
        self.encodings_data = get_encodings()
        self.append_encoding = append_encoding

    @property
    def cardinality(self) -> int:
        return max(self.encodings_data.values(), default=0) + 1

    @overload
    def encode(self, values: int) -> int: ...

    @overload
    def encode(self, values: Sequence[int]) -> list[int]: ...

    def encode(self, values: int | Sequence[int]) -> int | list[int]:
        data = self.encodings_data

        if isinstance(values, int):
            return data.get(values, 0)

        return [data.get(value, 0) for value in values]

    def discover(self, values: Sequence[int] | int) -> None:
        data = self.encodings_data

        if isinstance(values, int):
            values = (values,)

        unseen = list(dict.fromkeys(value for value in values if value not in data))

        if not unseen:
            return

        start = self.cardinality
        indices = list(range(start, start + len(unseen)))

        self.append_encoding(unseen, indices)
        data.update(zip(unseen, indices))


class TensorGameStateEncoder:
    def __init__(self, encoder: GameStateEncoder) -> None:
        self.encoder = encoder

    @property
    def fields(self) -> tuple[str, ...]:
        return self.encoder.fields

    @property
    def cardinality(self) -> int:
        return self.encoder.cardinality

    # TODO : Tensor -> list -> Tensor (not ideal)
    def encode(self, values: Tensor) -> Tensor:
        encoded = self.encoder.encode(values.tolist())

        return torch.tensor(
            encoded,
            dtype=values.dtype,
            device=values.device,
        )

    def discover(self, tensor: Tensor) -> None:
        unique = tensor.unique().tolist()
        self.encoder.discover(unique)
