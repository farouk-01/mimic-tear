from abc import ABC, abstractmethod

from torch import Tensor, nn


class Aggregator(nn.Module, ABC):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def forward(self, tokens: Tensor) -> Tensor: ...


class Mean(Aggregator):
    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, input_size)

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens.mean(dim=-2)


class Flatten(Aggregator):
    def __init__(self, input_size: int, num_tokens: int) -> None:
        super().__init__(input_size, input_size * num_tokens)

        self.num_tokens = num_tokens

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens.flatten(start_dim=-2)