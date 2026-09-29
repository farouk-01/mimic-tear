from abc import ABC, abstractmethod

from torch import Tensor, nn
import torch


class Temporal[Context](nn.Module, ABC):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def forward(
        self, features: Tensor, state: Context | None = None
    ) -> tuple[Tensor, Context]: ...

    @abstractmethod
    def initial_state(self, batch_size: int, device: torch.device) -> Context: ...

    @abstractmethod
    def detach_state(self, state: Context) -> Context: ...


type LSTMState = tuple[Tensor, Tensor]  # (short term memory, long term memory)


class LSTM(Temporal[LSTMState]):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        batch_first: bool = True,
    ) -> None:
        super().__init__(input_size, hidden_size)

        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        features: Tensor,
        state: LSTMState | None = None,
    ) -> tuple[Tensor, LSTMState]:
        output, (hidden_state, cell_state) = self.lstm(features, state)

        return output, (hidden_state, cell_state)

    def initial_state(self, batch_size: int, device: torch.device) -> LSTMState:
        shape = (self.num_layers, batch_size, self.output_size)

        hidden = torch.zeros(shape, device=device)
        cell = torch.zeros(shape, device=device)

        return hidden, cell

    def detach_state(self, state: LSTMState) -> LSTMState:
        hidden, cell = state

        return hidden.detach(), cell.detach()
