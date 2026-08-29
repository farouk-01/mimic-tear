import pytest
from typing import Callable

from data.process.encoders import GameStateEncoder


class TestGameStateEncoder:

    @pytest.fixture
    def make_encoder(self) -> Callable[[dict[int, int]], GameStateEncoder]:
        def create(encodings: dict[int, int]) -> GameStateEncoder:
            def append_encoding(key: int, value: int) -> None:
                encodings[key] = value

            return GameStateEncoder(
                get_encodings=encodings.copy,
                append_encoding=append_encoding,
            )

        return create

    def test_when_gap_then_no_overwrite(self, make_encoder) -> None:
        encoder = make_encoder({1234: 1, 5678: 3})

        assert encoder.encode(9810) == 4
