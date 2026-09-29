from abc import ABC, abstractmethod

from torch import Tensor, nn
import torch


class Fusion(nn.Module, ABC):
    def __init__(self, input_sizes: tuple[int, ...], output_size: int) -> None:
        super().__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size

    @abstractmethod
    def forward(self, *features: Tensor) -> Tensor: ...


class Concat(Fusion):
    def __init__(self, input_sizes: tuple[int, ...], output_size: int) -> None:
        super().__init__(input_sizes=input_sizes, output_size=output_size)

        if len(input_sizes) == 1 and input_sizes[0] == output_size:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Linear(sum(input_sizes), output_size),
                nn.ReLU(inplace=True),
            )

    def forward(self, *features: Tensor) -> Tensor:
        return self.projection(torch.cat(features, dim=-1))