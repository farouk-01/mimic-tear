from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class GameStateTransformConfig:
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None


class GameStateTransform:
    def __init__(
        self,
        *,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
    ) -> None:
        if (mean is None) != (std is None):
            raise ValueError(
                "mean and std must either both be provided or both be None"
            )

        if mean is not None and std is not None:
            if len(mean) != len(std):
                raise ValueError(
                    "mean and std must contain the same number of features"
                )

            if any(value <= 0.0 for value in std):
                raise ValueError(
                    "All standard deviations must be greater than zero"
                )

            self.mean: Tensor | None = torch.tensor(
                mean,
                dtype=torch.float32,
            )

            self.std: Tensor | None = torch.tensor(
                std,
                dtype=torch.float32,
            )
        else:
            self.mean = None
            self.std = None

    def __call__(self, state: Tensor) -> Tensor:
        state = state.to(torch.float32)

        if self.mean is None or self.std is None:
            return state

        if state.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"Expected {self.mean.shape[0]} game-state features, "
                f"received {state.shape[-1]}"
            )

        mean = self.mean.to(
            device=state.device,
            dtype=state.dtype,
        )

        std = self.std.to(
            device=state.device,
            dtype=state.dtype,
        )

        return (state - mean) / std