import pytest
import torch
from collections.abc import Callable, Sequence

from data.process.encoders import GameStateEncoder, TensorGameStateEncoder


@pytest.fixture
def make_encoder() -> Callable[[dict[int, int]], GameStateEncoder]:
    def create(encodings: dict[int, int]) -> GameStateEncoder:
        def append_encoding(
            keys: Sequence[int] | int, values: Sequence[int] | int
        ) -> None:
            if isinstance(keys, int):
                keys = (keys,)
            if isinstance(values, int):
                values = (values,)

            for k, v in zip(keys, values):
                encodings[k] = v

        return GameStateEncoder(
            fields=("test_field",),
            get_encodings=encodings.copy,
            append_encoding=append_encoding,
        )

    return create


class TestGameStateEncoder:

    def test_when_gap_then_no_overwrite(self, make_encoder) -> None:
        encoder = make_encoder({1234: 1, 5678: 3})

        encoder.discover(9810)
        assert encoder.encode(9810) == 4

    def test_first_value_is_not_zero(
        self, make_encoder: Callable[[dict[int, int]], GameStateEncoder]
    ) -> None:
        encoder = make_encoder({})

        encoder.discover(1234)
        assert encoder.encode(1234) != 0

    def test_existing_value_existing_encoding(
        self,
        make_encoder: Callable[[dict[int, int]], GameStateEncoder],
    ) -> None:
        encoder = make_encoder({1234: 2})

        assert encoder.encode(1234) == 2

    def test_multiple_values_all_encoded(
        self,
        make_encoder: Callable[[dict[int, int]], GameStateEncoder],
    ) -> None:
        encoder = make_encoder({1234: 1})

        encoder.discover([1234, 5678, 9810])
        assert encoder.encode([1234, 5678, 9810]) == [1, 2, 3]

    def test_same_new_value_same_encoding(
        self,
        make_encoder: Callable[[dict[int, int]], GameStateEncoder],
    ) -> None:
        encoder = make_encoder({})

        encoder.discover([1234, 1234])
        assert encoder.encode([1234, 1234]) == [1, 1]

    def test_sequence_with_one_unknown_then_one_zero_encoding(
        self,
        make_encoder: Callable[[dict[int, int]], GameStateEncoder],
    ) -> None:
        encoder = make_encoder({1234: 1})

        assert encoder.encode([1234, 5678]) == [1, 0]


class TestTensorGameStateEncoder:

    @pytest.fixture
    def make_tensor_encoder(
        self,
        make_encoder: Callable[[dict[int, int]], GameStateEncoder],
    ) -> Callable[[dict[int, int]], TensorGameStateEncoder]:
        def create(encodings: dict[int, int]) -> TensorGameStateEncoder:
            return TensorGameStateEncoder(make_encoder(encodings))

        return create

    def test_encode_preserves_shape_dtype(
        self,
        make_tensor_encoder: Callable[[dict[int, int]], TensorGameStateEncoder],
    ) -> None:
        encoder = make_tensor_encoder({123: 1, 456: 2})

        values = torch.tensor(
            [123, 456],
            dtype=torch.int64,
        )

        encoded = encoder.encode(values)

        assert encoded.shape == values.shape
        assert encoded.dtype == values.dtype
        assert encoded.tolist() == [1, 2]
