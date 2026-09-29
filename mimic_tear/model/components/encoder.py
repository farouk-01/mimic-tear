
from abc import ABC, abstractmethod

from torch import nn, Tensor


class Encoder(nn.Module, ABC):
    def __init__(self, output_size: int) -> None:
        super().__init__()

        self.output_size = output_size

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor: ...


class Embedding(Encoder):
    def __init__(
        self,
        cardinality: int,
        output_size: int,
    ) -> None:
        super().__init__(output_size=output_size)

        self.cardinality = cardinality

        self.embedding = nn.Embedding(
            num_embeddings=cardinality,
            embedding_dim=output_size,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.embedding(inputs)


class ContinuousEncoder(Encoder):
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ) -> None:
        super().__init__(output_size=output_size)
        
        self.input_size = input_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.input_size == 1:
            inputs = inputs.unsqueeze(-1)

        # bool / integer fields (binary, discrete, ordinal) must be float for Linear
        return self.linear(inputs.to(self.linear.weight.dtype))
